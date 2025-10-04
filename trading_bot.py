import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, time
import logging
from typing import Dict, List, Optional
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BinanceTradingBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'HYPE/USDT'):
        """
        Inicializar el bot de trading con Binance - Estrategia Wyckoff
        
        Args:
            api_key: API key de Binance
            api_secret: API secret de Binance
            symbol: Par de trading (default: HYPE/USDT)
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        self.symbol = symbol
        self.timeframe = '5m'
        self.candle_limit = 200
        self.stop_loss_percent = 0.02  # 2% stop loss
        self.take_profit_percent = 0.04  # 4% take profit (R:R 1:2)
        
        # Estado del bot
        self.is_running = False
        self.position = None
        self.daily_first_trade = True
        self.last_high = None
        self.current_direction = None
        self.entry_price_target = None
        
        # DataFrame para almacenar velas
        self.df = None
        
    async def fetch_candles(self) -> pd.DataFrame:
        """
        Obtener las últimas 200 velas de 5 minutos
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                limit=self.candle_limit
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error obteniendo velas: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcular indicadores técnicos
        """
        # EMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR para volatilidad
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        return df
    
    def is_long_trading_hours(self) -> bool:
        """
        Verificar si estamos en horario permitido para entradas LONG
        Lunes a Viernes, 9:30 AM - 9:30 PM
        """
        now = datetime.now()
        
        # Verificar que sea lunes a viernes (0=lunes, 6=domingo)
        if now.weekday() >= 5:  # Sábado o domingo
            return False
        
        # Verificar horario 9:30 AM - 9:30 PM
        current_time = now.time()
        start_time = time(9, 30)
        end_time = time(21, 30)
        
        return start_time <= current_time <= end_time
    
    def detect_smc_long(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Detectar señal LONG usando Smart Money Concepts SIMPLIFICADO
        
        1. Barrido de liquidez (sweep) + rechazo fuerte
        2. Estructura alcista (higher lows)
        3. RSI oversold recovery
        """
        if len(df) < lookback + 5:
            return False
        
        recent = df.tail(lookback)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. BARRIDO DE LIQUIDEZ SIMPLE
        recent_low = recent['low'].min()
        
        sweep_conditions = [
            latest['low'] <= recent_low * 1.01,
            (latest['close'] - latest['low']) / (latest['high'] - latest['low'] + 0.0001) > 0.6,
            latest['close'] > latest['open'],
            prev['close'] < prev['open']
        ]
        
        # 2. ESTRUCTURA ALCISTA (Higher Lows)
        last_5_lows = recent.tail(5)['low'].values
        structure_bullish = False
        if len(last_5_lows) >= 3:
            if last_5_lows[-1] > last_5_lows[-3] * 0.995:
                structure_bullish = True
        
        # 3. RSI OVERSOLD RECOVERY
        rsi_recovery = (
            latest['rsi'] > prev['rsi'] and
            latest['rsi'] < 45 and
            latest['rsi'] > 25
        )
        
        # 4. CONFIRMACIONES
        volume_spike = latest['volume'] > recent['volume'].mean() * 1.2
        momentum_positive = latest['macd_hist'] > prev['macd_hist']
        bullish_candle = latest['close'] > latest['open']
        
        # SCORING
        score = 0
        if sum(sweep_conditions) >= 3:
            score += 2
        if structure_bullish:
            score += 2
        if rsi_recovery:
            score += 1
        if volume_spike:
            score += 1
        if momentum_positive:
            score += 1
        if bullish_candle:
            score += 1
        
        return score >= 4
    
    def detect_wyckoff_upthrust(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        Detectar Upthrust de Wyckoff (señal de venta SHORT)
        Upthrust = Precio sube sobre resistencia, luego cae fuertemente
        Lookback reducido a 20 velas para más oportunidades
        """
        if len(df) < lookback + 10:
            return False
        
        recent = df.tail(lookback)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Identificar resistencia
        resistance = recent['high'].max()
        
        # Condiciones para Upthrust (más flexibles):
        upthrust_conditions = [
            # 1. Precio tocó o rompió la resistencia (1% flexible)
            latest['high'] >= resistance * 0.99,
            # 2. Cierre por debajo de la resistencia (rechazo)
            latest['close'] < resistance,
            # 3. Volumen alto (reducido a 1.1x)
            latest['volume'] > recent['volume'].mean() * 1.1,
            # 4. Vela bajista
            latest['close'] < latest['open'],
            # 5. RSI sobrecompra (más flexible: 55)
            latest['rsi'] > 55,
            # 6. Momentum cambiando
            latest['macd_hist'] < prev['macd_hist']
        ]
        
        return sum(upthrust_conditions) >= 4  # 4 de 6 en lugar de 5
    
    def analyze_market_direction(self, df: pd.DataFrame) -> Optional[str]:
        """
        Analizar mercado usando Método Wyckoff
        
        LONGs: Solo Lun-Vie 9:30-11:30
        SHORTs: Sin restricción
        
        Returns:
            'LONG', 'SHORT' o None
        """
        if len(df) < 200:
            return None
        
        # Detectar SMC LONG (simplificado)
        if self.detect_smc_long(df):
            # LONG solo en horario permitido
            if self.is_long_trading_hours():
                logger.info(f"🔵 SMC LONG: Barrido + Estructura + Recovery - Entrando LONG ⏰")
                return 'LONG'
            else:
                now = datetime.now()
                logger.info(f"🟡 Señal SMC LONG detectada pero FUERA de horario")
                logger.info(f"   Hora actual: {now.strftime('%A %H:%M')} | Permitido: Lun-Vie 9:30-21:30")
                return None
        
        # Detectar Upthrust (señal SHORT)
        if self.detect_wyckoff_upthrust(df):
            logger.info(f"🔴 WYCKOFF UPTHRUST detectado - Entrando SHORT")
            return 'SHORT'
        
        return None
    
    def find_last_high(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """
        Encontrar el último máximo significativo
        """
        recent_data = df.tail(lookback)
        return recent_data['high'].max()
    
    def find_last_low(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """
        Encontrar el último mínimo significativo
        """
        recent_data = df.tail(lookback)
        return recent_data['low'].min()
    
    def calculate_entry_price(self, df: pd.DataFrame, direction: str) -> float:
        """
        Calcular precio de entrada basado en la estrategia
        """
        latest = df.iloc[-1]
        
        if direction == 'LONG':
            # Entrada en el siguiente nivel de soporte o breakout
            entry = latest['close'] * 0.9995  # Ligeramente por debajo del precio actual
        else:  # SHORT
            entry = latest['close'] * 1.0005  # Ligeramente por encima del precio actual
        
        return entry
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """
        Calcular stop loss al 2%
        """
        stop_distance = entry_price * self.stop_loss_percent
        
        if direction == 'LONG':
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, direction: str, df: pd.DataFrame) -> float:
        """
        Calcular take profit al 4% (R:R 1:2)
        """
        tp_distance = entry_price * self.take_profit_percent
        
        if direction == 'LONG':
            return entry_price + tp_distance
        else:  # SHORT
            return entry_price - tp_distance
    
    async def place_limit_order(self, direction: str, entry_price: float, amount: float) -> Optional[Dict]:
        """
        Colocar orden límite
        """
        try:
            side = 'buy' if direction == 'LONG' else 'sell'
            
            order = self.exchange.create_limit_order(
                symbol=self.symbol,
                side=side,
                amount=amount,
                price=entry_price
            )
            
            logger.info(f"Orden límite colocada: {direction} @ {entry_price}")
            logger.info(f"Detalles: {order}")
            
            return order
        except Exception as e:
            logger.error(f"Error colocando orden: {e}")
            return None
    
    async def place_stop_loss_order(self, direction: str, stop_price: float, amount: float) -> Optional[Dict]:
        """
        Colocar orden de stop loss
        """
        try:
            side = 'sell' if direction == 'LONG' else 'buy'

            # Binance spot requiere órdenes stop-loss limit
            order = self.exchange.create_order(
                symbol=self.symbol,
                type='stop_loss_limit',
                side=side,
                amount=amount,
                price=stop_price,
                params={'stopPrice': stop_price, 'timeInForce': 'GTC'}
            )

            logger.info(f"Stop loss colocado @ {stop_price}")
            return order
        except Exception as e:
            logger.error(f"Error colocando stop loss: {e}")
            return None
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("Archivo config.json no encontrado. Por favor créalo con tus credenciales de Binance.")
        logger.info(f"Timeframe: {self.timeframe} ({self.timeframe} intervalo)")
        logger.info(json.dumps({
            "api_key": "TU_API_KEY",
            "api_secret": "TU_API_SECRET",
            "symbol": "BTC/USDT",
            "trading_amount": 0.001
        }))
    try:
        bot.run(trading_amount=config.get('trading_amount', 0.001))
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    asyncio.run(main())

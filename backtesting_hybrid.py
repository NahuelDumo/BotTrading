import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, time
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xlsxwriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridTradingBacktest:
    def __init__(self, symbol: str = 'HYPE/USDT', initial_balance: float = 1000):
        """
        Inicializar backtesting - Estrategia Mean Reversion Bidireccional
        LONGs: Mean Reversion en sobreventa
        SHORTs: Mean Reversion en sobrecompra MEJORADO
        
        Args:
            symbol: Par de trading
            initial_balance: Balance inicial en USDT
        """
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        self.symbol = symbol
        self.timeframe = '5m'
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Apalancamiento
        self.leverage = 15
        
        # Stops y TPs asimétricos
        self.stop_loss_percent_long = 0.012   # 1.2% para LONGs
        self.stop_loss_percent_short = 0.015  # 1.5% para SHORTs
        
        self.take_profit_percent_long = 0.03   # 3% para LONGs
        self.take_profit_percent_short = 0.022 # 2.2% para SHORTs (más conservador)
        
        # Límites de tiempo diferenciados
        self.max_candles_long = 60   # 5 horas para LONGs
        self.max_candles_short = 45  # 3.75 horas para SHORTs
        
        # Historial de trades
        self.trades = []
        self.equity_curve = []
        # Estadísticas adicionales
        self.filtered_long_signals = 0  # Señales LONG fuera de horario
        self.filtered_short_signals = 0  # Señales SHORT rechazadas por calidad
        
        # Datos
        self.df = None
        
    def fetch_historical_data(self, days: int = 120) -> pd.DataFrame:
        try:
            # Calcular cuántas velas necesitamos según timeframe
            if self.timeframe == '1m':
                candles_per_day = 1440
            elif self.timeframe == '5m':
                candles_per_day = 288
            elif self.timeframe == '15m':
                candles_per_day = 96
            elif self.timeframe == '1h':
                candles_per_day = 24
            elif self.timeframe == '4h':
                candles_per_day = 6
            else:
                candles_per_day = 288  # Default 5m

            total_candles_needed = days * candles_per_day
            max_per_request = 1000  # Límite típico de Binance para fetch_ohlcv

            logger.info(f"Necesitamos {total_candles_needed} velas de {self.symbol} ({self.timeframe})")

            # Calcular timestamp de inicio (días atrás)
            import time
            
            start_timestamp = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)
            # Mostrar la fecha de inicio en AR
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True).tz_convert('America/Argentina/Buenos_Aires')
            print(f"Fecha de inicio: {start_dt}")
            
            # Si necesitamos menos que el límite por petición, descarga directa
            if total_candles_needed <= max_per_request:
                logger.info(f"Descargando {total_candles_needed} velas en una sola petición...")
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    limit=total_candles_needed,
                    since=start_timestamp
                )

                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                logger.info(f"✅ Datos descargados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
                return df

            # Si necesitamos más de 1000, descargar en chunks desde el pasado
            logger.info(f"⚠️  Necesitamos más de {max_per_request} velas. Descargando en múltiples peticiones...")

            all_data = []
            since = start_timestamp

            while True:
                logger.info(f"   📥 Descargando datos desde timestamp {since}...")

                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    limit=max_per_request,
                    since=since
                )

                if not ohlcv:
                    logger.info(f"   No más datos disponibles")
                    break

                all_data.extend(ohlcv)
                logger.info(f"   Obtenidas {len(ohlcv)} velas en este chunk (total acumulado: {len(all_data)})")

                # Actualizar 'since' para la próxima petición (último timestamp + 1)
                since = ohlcv[-1][0] + 1

                # Si obtuvimos menos velas de las esperadas, probablemente ya no hay más datos
                if len(ohlcv) < max_per_request:
                    logger.info(f"   Chunk final: {len(ohlcv)} < {max_per_request}, terminando descarga")
                    break

                # Pequeña pausa para no saturar la API
                time.sleep(0.5)

                # Limitar para evitar loops infinitos (máximo 100 chunks)
                if len(all_data) >= total_candles_needed * 2:
                    logger.warning(f"   ⚠️  Demasiados datos descargados ({len(all_data)}), deteniendo para evitar loop infinito")
                    break

            # Convertir a DataFrame
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Eliminar duplicados (por si acaso)
            df = df.drop_duplicates(subset=['timestamp'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            logger.info(f"✅ Total descargado: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")

            # Limitar al número de velas solicitadas si obtuvimos más
            if len(df) > total_candles_needed:
                df = df.tail(total_candles_needed)
                logger.info(f"   📊 Limitando a últimas {total_candles_needed} velas")

            return df

        except Exception as e:
            logger.error(f"❌ Error descargando datos: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # ATR
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
    
    def is_long_trading_hours(self, timestamp) -> bool:
        """
        Verificar si el timestamp está en horario permitido para LONGs
        Lunes a Viernes, 8:00 AM - 8:00 PM (12 horas)
        """
        if timestamp.weekday() >= 5:  # Sábado o domingo
            return False
        
        current_time = timestamp.time()
        start_time = time(8, 0)
        end_time = time(20, 0)
        
        return start_time <= current_time <= end_time
    
    def detect_mean_reversion_long(self, idx: int) -> bool:
        """
        Detectar señal LONG usando Mean Reversion MEJORADO (sobreventa)
        
        Mejoras:
        1. Confirmación de volumen
        2. Filtro de tendencia mayor (EMA 200)
        3. RSI más estricto
        """
        if idx < 200:
            return False
        
        current = self.df.iloc[idx]
        prev = self.df.iloc[idx - 1]
        recent = self.df.iloc[max(0, idx-20):idx]

        # Filtros de tendencia mayor para evitar largos en mercados bajistas
        if current['ema_50'] <= current['ema_200']:
            return False
        if len(recent) >= 2 and recent['ema_200'].iloc[-1] <= recent['ema_200'].iloc[0]:
            return False

        # CONDICIÓN CRÍTICA: Precio cerca de soporte (BB lower)
        at_support = current['close'] < current['bb_lower'] * 1.003

        if not at_support:
            return False

        # Confirmación de vela de reversión alcista
        if not (current['close'] > current['open'] and current['close'] >= prev['close']):
            return False

        volume_mean = recent['volume'].mean() if len(recent) > 0 else current['volume']

        # Detectar sobreventa con confirmaciones actualizadas
        oversold_conditions = [
            current['ema_9'] < current['ema_21'],                     # Retroceso de corto plazo
            current['close'] < current['ema_21'],                     # Precio por debajo de EMA media
            current['rsi'] < 35,                                      # RSI en sobreventa
            current['macd'] < current['macd_signal'],                 # MACD aún bajista
            current['macd_hist'] > prev['macd_hist'],                 # Momentum recuperándose
            current['volume'] >= volume_mean * 1.1,                   # Confirmación de volumen
            current['close'] >= current['ema_200'] * 0.99             # Mantenerse cerca de EMA 200
        ]

        return sum(oversold_conditions) >= 5
    
    def detect_mean_reversion_short_improved(self, idx: int) -> bool:
        """
        Detectar señal SHORT MEJORADO - Versión MÁS PERMISIVA
        """
        if idx < 200:
            return False
        
        current = self.df.iloc[idx]
        prev = self.df.iloc[idx - 1]
        recent = self.df.iloc[max(0, idx-20):idx]
        recent_longer = self.df.iloc[max(0, idx-50):idx]
        
        # FILTRO 1: Contexto alcista (MÁS FLEXIBLE)
        uptrend_context = [
            current['ema_9'] > current['ema_21'],
            current['close'] > current['ema_50'],
            recent_longer['close'].iloc[-1] > recent_longer['close'].iloc[0]
        ]
        
        # Solo requiere 1 de 3 (antes era 2 de 3)
        if sum(uptrend_context) < 1:
            self.filtered_short_signals += 1
            return False
        
        # CONDICIÓN CRÍTICA: Precio cerca de BB upper (MÁS PERMISIVO)
        bb_distance = (current['close'] - current['bb_upper']) / current['bb_upper'] * 100
        near_bb_upper = bb_distance > -1.0  # Puede estar hasta 1% debajo (antes -0.5%)
        
        if not near_bb_upper:
            return False
        
        # PATRÓN DE VELA: MÁS FLEXIBLE
        candle_range = current['high'] - current['low']
        if candle_range == 0:
            return False
        
        close_position = (current['close'] - current['low']) / candle_range
        
        # Acepta cualquier vela que no cierre en el 70% superior
        has_bearish_pattern = close_position < 0.7  # Antes era 0.6
        
        if not has_bearish_pattern:
            return False
        
        volume_mean = recent['volume'].mean()
        
        # CONDICIONES MÁS PERMISIVAS
        short_conditions = [
            current['rsi'] > 65,                                      # Bajado de 70
            bb_distance > -0.8,                                       # Más permisivo
            current['close'] > current['ema_9'],                      
            current['macd'] > current['macd_signal'],                 
            current['macd_hist'] < prev['macd_hist'],                 
            current['volume'] > volume_mean * 1.05,                   # Bajado de 1.15
            current['high'] >= recent['high'].tail(10).max(),         # Solo últimas 10 velas
            (current['close'] - current['ema_50']) / current['ema_50'] * 100 > 0.5  # Bajado de 1.0
        ]
        
        score = sum(short_conditions)
        
        # Requiere 4 de 8 (antes era 5 de 8)
        return score >= 4

    def add_short_quality_filter(self, idx: int) -> bool:
        """
        Filtro de calidad SIMPLIFICADO - Menos restrictivo
        """
        if idx < 10:
            return True
        
        recent = self.df.iloc[max(0, idx-10):idx]
        
        # Solo filtrar casos EXTREMOS:
        
        # 1. Cruce alcista muy reciente (solo últimas 3 velas)
        for i in range(max(0, len(recent) - 3), len(recent)):
            if i < len(recent) - 1:
                if recent['ema_9'].iloc[i] < recent['ema_21'].iloc[i] and \
                recent['ema_9'].iloc[i+1] >= recent['ema_21'].iloc[i+1]:
                    self.filtered_short_signals += 1
                    return False
        
        # 2. Solo filtrar volumen EXTREMADAMENTE explosivo
        volumes = recent['volume'].tail(3).values
        if len(volumes) >= 3:
            if volumes[-1] > volumes[-2] * 2.0 and volumes[-2] > volumes[-3] * 2.0:
                self.filtered_short_signals += 1
                return False
        
        # Eliminamos el filtro de mínimos ascendentes (era muy restrictivo)
        
        return True
    
    def analyze_market_direction(self, idx: int) -> str:
        """
        Analizar mercado usando Estrategia Híbrida de Mean Reversion
        
        LONGs: Mean Reversion en sobreventa (Lun-Vie 8:00-20:00)
        SHORTs: Mean Reversion en sobrecompra MEJORADO (24/7)
        
        Args:
            idx: Índice de la vela a analizar
        """
        if idx < 200:
            return None
        
        current_time = self.df.index[idx]
        
        # Detectar Mean Reversion LONG (sobreventa)
        if self.detect_mean_reversion_long(idx):
            if self.is_long_trading_hours(current_time):
                return 'LONG'
            else:
                self.filtered_long_signals += 1
                return None
        
        # Detectar Mean Reversion SHORT MEJORADO (sobrecompra)
        if self.detect_mean_reversion_short_improved(idx) and self.add_short_quality_filter(idx):
            return 'SHORT'
        
        return None
    
    def calculate_stop_loss(self, entry_price: float, direction: str, idx: int = None) -> float:
        """
        Calcular stop loss asimétrico
        LONGs: 1.2%
        SHORTs: 1.5%
        """
        if direction == 'LONG':
            stop_distance = entry_price * self.stop_loss_percent_long
            return entry_price - stop_distance
        else:
            stop_distance = entry_price * self.stop_loss_percent_short
            return entry_price + stop_distance
    
    def calculate_liquidation_price(self, entry_price: float, direction: str) -> float:
        """
        Calcular precio de liquidación aproximado con apalancamiento 15x
        Liquidación ocurre cuando pérdidas = 100% del margen
        """
        liquidation_distance = entry_price * (1 / self.leverage)
        
        if direction == 'LONG':
            return entry_price - liquidation_distance
        else:
            return entry_price + liquidation_distance
    
    def calculate_take_profit(self, entry_price: float, direction: str, 
                            is_first_trade: bool, idx: int) -> float:
        """
        Calcular take profit asimétrico
        LONGs: 3.0%
        SHORTs: 2.2% (más conservador)
        """
        if direction == 'LONG':
            tp_distance = entry_price * self.take_profit_percent_long
            return entry_price + tp_distance
        else:
            tp_distance = entry_price * self.take_profit_percent_short
            return entry_price - tp_distance
    
    def simulate_trade(self, entry_idx: int, direction: str, entry_price: float,
                      stop_loss: float, take_profit: float, position_size: float) -> Dict:
        """
        Simular un trade de FUTUROS desde la entrada hasta la salida
        Incluye verificación de liquidación con apalancamiento 15x
        
        LÍMITE DE TIEMPO diferenciado:
        - LONGs: 60 velas (5 horas)
        - SHORTs: 45 velas (3.75 horas)
        
        Returns:
            Dict con información del trade
        """
        # Calcular precio de liquidación
        liquidation_price = self.calculate_liquidation_price(entry_price, direction)
        
        # Límite de tiempo diferenciado
        max_candles = self.max_candles_long if direction == 'LONG' else self.max_candles_short
        end_idx = min(entry_idx + max_candles, len(self.df) - 1)
        
        # Buscar salida en las siguientes velas
        for i in range(entry_idx + 1, end_idx + 1):
            candle = self.df.iloc[i]
            
            if direction == 'LONG':
                # VERIFICAR LIQUIDACIÓN PRIMERO (más crítico que stop loss)
                if candle['low'] <= liquidation_price:
                    exit_price = liquidation_price
                    exit_reason = 'LIQUIDACIÓN'
                    # En liquidación, pierdes el 100% del margen usado
                    pnl = -(position_size * entry_price / self.leverage)  # Pérdida total del margen
                    return {
                        'entry_time': self.df.index[entry_idx],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'liquidation_price': liquidation_price,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': -100,  # Pérdida total del margen
                        'exit_reason': exit_reason,
                        'duration_candles': i - entry_idx
                    }
                
                # Verificar stop loss
                if candle['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                    pnl = (exit_price - entry_price) * position_size
                    return {
                        'entry_time': self.df.index[entry_idx],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'liquidation_price': liquidation_price,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position_size / self.leverage)) * 100,
                        'exit_reason': exit_reason,
                        'duration_candles': i - entry_idx
                    }
                
                # Verificar take profit
                if candle['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'Take Profit'
                    pnl = (exit_price - entry_price) * position_size
                    return {
                        'entry_time': self.df.index[entry_idx],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'liquidation_price': liquidation_price,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position_size / self.leverage)) * 100,
                        'exit_reason': exit_reason,
                        'duration_candles': i - entry_idx
                    }
            
            else:  # SHORT
                # VERIFICAR LIQUIDACIÓN PRIMERO
                if candle['high'] >= liquidation_price:
                    exit_price = liquidation_price
                    exit_reason = 'LIQUIDACIÓN'
                    pnl = -(position_size * entry_price / self.leverage)
                    return {
                        'entry_time': self.df.index[entry_idx],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'liquidation_price': liquidation_price,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': -100,
                        'exit_reason': exit_reason,
                        'duration_candles': i - entry_idx
                    }
                
                # Verificar stop loss
                if candle['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                    pnl = (entry_price - exit_price) * position_size
                    return {
                        'entry_time': self.df.index[entry_idx],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'liquidation_price': liquidation_price,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position_size / self.leverage)) * 100,
                        'exit_reason': exit_reason,
                        'duration_candles': i - entry_idx
                    }
                
                # Verificar take profit
                if candle['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'Take Profit'
                    pnl = (entry_price - exit_price) * position_size
                    return {
                        'entry_time': self.df.index[entry_idx],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'liquidation_price': liquidation_price,
                        'position_size': position_size,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position_size / self.leverage)) * 100,
                        'exit_reason': exit_reason,
                        'duration_candles': i - entry_idx
                    }
        
        # Si llegamos aquí, el trade alcanzó el límite de tiempo o fin de datos
        # Cerrar al precio actual de la última vela disponible
        last_available_idx = min(end_idx, len(self.df) - 1)
        last_candle = self.df.iloc[last_available_idx]
        exit_price = last_candle['close']
        
        # Determinar razón de salida
        if last_available_idx < len(self.df) - 1:
            # Alcanzó límite de tiempo
            exit_reason = 'Time Limit'
        else:
            # Fin de datos del backtest
            exit_reason = 'End of Data'
        
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
        
        return {
            'entry_time': self.df.index[entry_idx],
            'exit_time': self.df.index[last_available_idx],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'liquidation_price': liquidation_price,
            'position_size': position_size,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * position_size / self.leverage)) * 100,
            'exit_reason': exit_reason,
            'duration_candles': last_available_idx - entry_idx
        }
    
    def run_backtest(self, risk_per_trade: float = 0.10, days: int = 30):
        """
        Ejecutar backtest completo
        
        Args:
            risk_per_trade: Porcentaje del balance a usar por trade (default: 10%)
            days: Días de historia a analizar
        """
        logger.info("=== BACKTEST FUTUROS 15X ===")
        logger.info(f"Margen por trade: {risk_per_trade*100:.1f}%")
        logger.info(f"SL Long: {self.stop_loss_percent_long*100}% | TP Long: {self.take_profit_percent_long*100}%")
        logger.info(f"SL Short: {self.stop_loss_percent_short*100}% | TP Short: {self.take_profit_percent_short*100}%")
        logger.info(f"Límite tiempo Long: {self.max_candles_long} velas | Short: {self.max_candles_short} velas")
        
        # Descargar datos
        self.df = self.fetch_historical_data(days)
        if self.df is None:
            return
        # Calcular indicadores
        logger.info("Calculando indicadores...")
        self.df = self.calculate_indicators(self.df)
        
        # Variables de control
        in_position = False
        daily_first_trade = True
        last_trade_date = None
        
        # Iterar por cada vela
        logger.info("Simulando trades...")
        for i in range(200, len(self.df)):
            current_time = self.df.index[i]
            current_date = current_time.date()
            
            # Resetear daily_first_trade cada nuevo día
            if last_trade_date != current_date:
                daily_first_trade = True
                last_trade_date = current_date
            
            # Solo buscar nuevas señales si no estamos en posición
            if not in_position:
                direction = self.analyze_market_direction(i)
                
                if direction:
                    entry_price = self.df.iloc[i]['close']
                    stop_loss = self.calculate_stop_loss(entry_price, direction, i)
                    
                    # Verificar si es primer trade de la mañana
                    is_morning = current_time.time() < time(12, 0)
                    is_first = daily_first_trade and is_morning
                    
                    take_profit = self.calculate_take_profit(
                        entry_price, direction, is_first, i
                    )
                    
                    # Calcular tamaño de posición para FUTUROS con 15x
                    # Riesgo configurado como porcentaje del balance
                    margin = self.balance * risk_per_trade
                    # Posición total controlada (con apalancamiento)
                    position_value = margin * self.leverage
                    # Cantidad de contratos/monedas
                    position_size = position_value / entry_price
                    
                    # Simular el trade
                    trade = self.simulate_trade(
                        i, direction, entry_price, stop_loss, 
                        take_profit, position_size
                    )
                    
                    # Actualizar balance
                    self.balance += trade['pnl']
                    self.trades.append(trade)
                    
                    # Registrar equity
                    self.equity_curve.append({
                        'time': current_time,
                        'balance': self.balance
                    })
                    
                    # Log del trade
                    logger.info(f"Trade #{len(self.trades)}: {direction} @ {entry_price:.4f} -> "
                              f"{trade['exit_price']:.4f} | P/L: ${trade['pnl']:.2f} "
                              f"({trade['pnl_pct']:.2f}%) | {trade['exit_reason']}")
                    
                    # Marcar que ya no es el primer trade del día
                    if is_first:
                        daily_first_trade = False
                    
                    in_position = False
        
        logger.info("=== BACKTEST COMPLETADO ===")
        self.print_results()
    
    def print_results(self):
        """
        Imprimir resultados del backtest
        """
        if not self.trades:
            logger.info("No se ejecutaron trades en el período analizado")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Métricas generales
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        # Calcular drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['cummax'] = equity_df['balance'].cummax()
        equity_df['drawdown'] = equity_df['balance'] - equity_df['cummax']
        equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['cummax']) * 100
        max_drawdown = equity_df['drawdown_pct'].min()
        
        # Return total
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Imprimir resultados
        print("\n" + "="*60)
        print("RESULTADOS FUTUROS 15X - ESTRATEGIA MEJORADA")
        print("="*60)
        print(f"\n📊 RESUMEN GENERAL")
        print(f"Balance Inicial:        ${self.initial_balance:.2f}")
        print(f"Balance Final:          ${self.balance:.2f}")
        print(f"P/L Total:              ${total_pnl:.2f}")
        print(f"Retorno Total:          {total_return:.2f}%")
        print(f"Max Drawdown:           {max_drawdown:.2f}%")
        
        print(f"\n📈 TRADES")
        print(f"Total:              {total_trades}")
        print(f"Ganadores:          {winning_trades} ({win_rate:.2f}%)")
        print(f"Perdedores:         {losing_trades}")
        print(f"Profit Factor:      {profit_factor:.2f}")
        
        print(f"\n💰 PROMEDIOS")
        print(f"Ganancia Avg:       ${avg_win:.2f}")
        print(f"Pérdida Avg:        ${avg_loss:.2f}")
        print(f"Max Ganancia:       ${max_win:.2f}")
        print(f"Max Pérdida:        ${max_loss:.2f}")
        
        print(f"\n📋 POR DIRECCIÓN")
        for direction in ['LONG', 'SHORT']:
            dir_trades = trades_df[trades_df['direction'] == direction]
            if len(dir_trades) > 0:
                dir_wins = len(dir_trades[dir_trades['pnl'] > 0])
                dir_wr = (dir_wins / len(dir_trades)) * 100
                dir_pnl = dir_trades['pnl'].sum()
                print(f"{direction:5} - {len(dir_trades):2} trades | WR: {dir_wr:5.2f}% | P/L: ${dir_pnl:8.2f}")
        
        print(f"\n🎯 SALIDAS")
        for reason in trades_df['exit_reason'].unique():
            count = len(trades_df[trades_df['exit_reason'] == reason])
            pct = (count / total_trades) * 100
            print(f"{reason:20} - {count:2} ({pct:.2f}%)")
        
        print(f"\n⏰ FILTROS DE SEÑALES")
        
        # Filtro LONG
        long_trades_executed = len(trades_df[trades_df['direction'] == 'LONG'])
        total_long_signals = long_trades_executed + self.filtered_long_signals
        if total_long_signals > 0:
            filter_rate_long = (self.filtered_long_signals / total_long_signals) * 100
            print(f"\nLONG (horario):")
            print(f"  Detectadas: {total_long_signals}")
            print(f"  Ejecutadas: {long_trades_executed} ({100-filter_rate_long:.1f}%)")
            print(f"  Filtradas:  {self.filtered_long_signals} ({filter_rate_long:.1f}%)")
        else:
            print(f"\nLONG: Sin señales detectadas")
        
        # Filtro SHORT
        short_trades_executed = len(trades_df[trades_df['direction'] == 'SHORT'])
        total_short_signals = short_trades_executed + self.filtered_short_signals
        if total_short_signals > 0:
            filter_rate_short = (self.filtered_short_signals / total_short_signals) * 100
            print(f"\nSHORT (calidad):")
            print(f"  Detectadas: {total_short_signals}")
            print(f"  Ejecutadas: {short_trades_executed} ({100-filter_rate_short:.1f}%)")
            print(f"  Filtradas:  {self.filtered_short_signals} ({filter_rate_short:.1f}%)")
        else:
            print(f"\nSHORT: Sin señales detectadas")
        
        print("\n" + "="*60)
        
        # Guardar resultados detallados
        self.save_detailed_results(trades_df)
        self.save_excel_report(trades_df)
    
    def save_detailed_results(self, trades_df: pd.DataFrame):
        """
        Guardar resultados detallados en CSV
        """
        filename = f"backtest_futures_15x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"Resultados guardados en: {filename}")
    
    def save_excel_report(self, trades_df: pd.DataFrame):
        """
        Guardar reporte completo en Excel con múltiples hojas y gráficos
        """
        filename = f"backtest_futures_15x_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        # Crear workbook
        workbook = xlsxwriter.Workbook(filename)
        
        # Formatos
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1
        })
        money_format = workbook.add_format({'num_format': '$#,##0.00'})
        percent_format = workbook.add_format({'num_format': '0.00%'})
        number_format = workbook.add_format({'num_format': '#,##0'})
        
        # Hoja 1: Resumen General
        summary_sheet = workbook.add_worksheet('Resumen General')
        
        # Calcular métricas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        # Calcular drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['cummax'] = equity_df['balance'].cummax()
        equity_df['drawdown'] = equity_df['balance'] - equity_df['cummax']
        equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['cummax']) * 100
        max_drawdown = equity_df['drawdown_pct'].min()
        
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Escribir resumen
        summary_data = [
            ['Métrica', 'Valor'],
            ['Balance Inicial', self.initial_balance],
            ['Balance Final', self.balance],
            ['P/L Total', total_pnl],
            ['Retorno Total', total_return / 100],
            ['Max Drawdown', max_drawdown / 100],
            ['Total Trades', total_trades],
            ['Trades Ganadores', winning_trades],
            ['Trades Perdedores', losing_trades],
            ['Win Rate', win_rate / 100],
            ['Profit Factor', profit_factor],
            ['Ganancia Promedio', avg_win],
            ['Pérdida Promedio', avg_loss],
            ['Max Ganancia', max_win],
            ['Max Pérdida', max_loss]
        ]
        
        for row, data in enumerate(summary_data):
            summary_sheet.write(row, 0, data[0], header_format if row == 0 else None)
            if isinstance(data[1], (int, float)):
                if 'Balance' in data[0] or 'P/L' in data[0] or 'Ganancia' in data[0] or 'Pérdida' in data[0]:
                    summary_sheet.write(row, 1, data[1], money_format)
                elif 'Rate' in data[0] or 'Retorno' in data[0] or 'Drawdown' in data[0]:
                    summary_sheet.write(row, 1, data[1], percent_format)
                else:
                    summary_sheet.write(row, 1, data[1], number_format)
            else:
                summary_sheet.write(row, 1, data[1])
        
        # Hoja 2: Desempeño por Días
        daily_sheet = workbook.add_worksheet('Desempeño por Días')
        
        # Agrupar por fecha
        trades_df['date'] = trades_df['entry_time'].dt.date
        daily_performance = trades_df.groupby('date').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
        }).round(2)
        daily_performance.columns = ['total_trades', 'total_pnl', 'winning_trades', 'losing_trades']
        daily_performance['win_rate'] = (daily_performance['winning_trades'] / daily_performance['total_trades'] * 100).round(2)
        daily_performance = daily_performance.reset_index()
        
        # Escribir headers
        headers = ['Fecha', 'Total Trades', 'P/L Total', 'Trades Ganadores', 'Trades Perdedores', 'Win Rate %']
        for col, header in enumerate(headers):
            daily_sheet.write(0, col, header, header_format)
        
        # Escribir datos
        for row, (_, data) in enumerate(daily_performance.iterrows(), 1):
            daily_sheet.write(row, 0, str(data['date']))
            daily_sheet.write(row, 1, data['total_trades'], number_format)
            daily_sheet.write(row, 2, data['total_pnl'], money_format)
            daily_sheet.write(row, 3, data['winning_trades'], number_format)
            daily_sheet.write(row, 4, data['losing_trades'], number_format)
            daily_sheet.write(row, 5, data['win_rate'] / 100, percent_format)
        
        # Gráfico de P/L diario
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'P/L Diario',
            'categories': f"='Desempeño por Días'!$A$2:$A${len(daily_performance)+1}",
            'values': f"='Desempeño por Días'!$C$2:$C${len(daily_performance)+1}",
        })
        chart.set_title({'name': 'P/L por Día'})
        chart.set_x_axis({'name': 'Fecha'})
        chart.set_y_axis({'name': 'P/L (USDT)'})
        daily_sheet.insert_chart('H2', chart)
        
        # Hoja 3: Desempeño por Horas
        hourly_sheet = workbook.add_worksheet('Desempeño por Horas')
        
        # Agrupar por hora
        trades_df['hour'] = trades_df['entry_time'].dt.hour
        hourly_performance = trades_df.groupby('hour').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
        }).round(2)
        hourly_performance.columns = ['total_trades', 'total_pnl', 'winning_trades', 'losing_trades']
        hourly_performance['win_rate'] = (hourly_performance['winning_trades'] / hourly_performance['total_trades'] * 100).round(2)
        hourly_performance = hourly_performance.reset_index()
        
        # Escribir headers
        headers = ['Hora', 'Total Trades', 'P/L Total', 'Trades Ganadores', 'Trades Perdedores', 'Win Rate %']
        for col, header in enumerate(headers):
            hourly_sheet.write(0, col, header, header_format)
        
        # Escribir datos
        for row, (_, data) in enumerate(hourly_performance.iterrows(), 1):
            hourly_sheet.write(row, 0, data['hour'], number_format)
            hourly_sheet.write(row, 1, data['total_trades'], number_format)
            hourly_sheet.write(row, 2, data['total_pnl'], money_format)
            hourly_sheet.write(row, 3, data['winning_trades'], number_format)
            hourly_sheet.write(row, 4, data['losing_trades'], number_format)
            hourly_sheet.write(row, 5, data['win_rate'] / 100, percent_format)
        
        # Gráfico de P/L por hora
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'P/L por Hora',
            'categories': f"='Desempeño por Horas'!$A$2:$A${len(hourly_performance)+1}",
            'values': f"='Desempeño por Horas'!$C$2:$C${len(hourly_performance)+1}",
        })
        chart.set_title({'name': 'P/L por Hora'})
        chart.set_x_axis({'name': 'Hora del Día'})
        chart.set_y_axis({'name': 'P/L (USDT)'})
        hourly_sheet.insert_chart('H2', chart)
        
        # Hoja 3.1: Desempeño por Día de Semana
        weekday_sheet = workbook.add_worksheet('Desempeño por Día Semana')
        
        # Mapear números de día a nombres
        day_names = {
            0: 'Lunes',
            1: 'Martes', 
            2: 'Miércoles',
            3: 'Jueves',
            4: 'Viernes',
            5: 'Sábado',
            6: 'Domingo'
        }
        
        # Agrupar por día de semana
        trades_df['weekday'] = trades_df['entry_time'].dt.weekday
        weekday_performance = trades_df.groupby('weekday').agg({
            'pnl': ['count', 'sum', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
        }).round(2)
        weekday_performance.columns = ['total_trades', 'total_pnl', 'winning_trades', 'losing_trades']
        weekday_performance['win_rate'] = (weekday_performance['winning_trades'] / weekday_performance['total_trades'] * 100).round(2)
        weekday_performance = weekday_performance.reset_index()
        weekday_performance['day_name'] = weekday_performance['weekday'].map(day_names)
        
        # Escribir headers
        headers = ['Día', 'Total Trades', 'P/L Total', 'Trades Ganadores', 'Trades Perdedores', 'Win Rate %']
        for col, header in enumerate(headers):
            weekday_sheet.write(0, col, header, header_format)
        
        # Escribir datos
        for row, (_, data) in enumerate(weekday_performance.iterrows(), 1):
            weekday_sheet.write(row, 0, data['day_name'])
            weekday_sheet.write(row, 1, data['total_trades'], number_format)
            weekday_sheet.write(row, 2, data['total_pnl'], money_format)
            weekday_sheet.write(row, 3, data['winning_trades'], number_format)
            weekday_sheet.write(row, 4, data['losing_trades'], number_format)
            weekday_sheet.write(row, 5, data['win_rate'] / 100, percent_format)
        
        # Gráfico de P/L por día de semana
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'P/L por Día',
            'categories': f"='Desempeño por Día Semana'!$A$2:$A${len(weekday_performance)+1}",
            'values': f"='Desempeño por Día Semana'!$C$2:$C${len(weekday_performance)+1}",
        })
        chart.set_title({'name': 'P/L por Día de Semana'})
        chart.set_x_axis({'name': 'Día de la Semana'})
        chart.set_y_axis({'name': 'P/L (USDT)'})
        weekday_sheet.insert_chart('H2', chart)
        
        # Hoja 4: Datos para Gráfico de Precios
        chart_data_sheet = workbook.add_worksheet('Datos para Gráfico')
        
        # Preparar datos de velas con indicadores y trades
        chart_df = self.df.copy()
        chart_df = chart_df.reset_index()
        
        # Agregar columnas para trades
        chart_df['trade_entry'] = None
        chart_df['trade_exit'] = None
        chart_df['direction'] = None
        
        # Marcar entradas y salidas
        for trade in self.trades:
            entry_idx = self.df.index.get_loc(trade['entry_time'])
            exit_idx = self.df.index.get_loc(trade['exit_time'])
            
            chart_df.at[entry_idx, 'trade_entry'] = trade['entry_price']
            chart_df.at[exit_idx, 'trade_exit'] = trade['exit_price']
            chart_df.at[entry_idx, 'direction'] = trade['direction']
        
        # Escribir headers
        headers = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'EMA9', 'EMA21', 'EMA50', 'EMA200', 'RSI', 'BB_Upper', 'BB_Lower', 'Trade_Entry', 'Trade_Exit', 'Direction']
        for col, header in enumerate(headers):
            chart_data_sheet.write(0, col, header, header_format)
        
        # Escribir datos (últimas 1000 velas para no hacer el archivo demasiado grande)
        data_to_write = chart_df.tail(1000)
        for row, (_, data) in enumerate(data_to_write.iterrows(), 1):
            chart_data_sheet.write(row, 0, str(data['timestamp']))
            chart_data_sheet.write(row, 1, data['open'])
            chart_data_sheet.write(row, 2, data['high'])
            chart_data_sheet.write(row, 3, data['low'])
            chart_data_sheet.write(row, 4, data['close'])
            chart_data_sheet.write(row, 5, data['volume'], number_format)
            chart_data_sheet.write(row, 6, data.get('ema_9', 0))
            chart_data_sheet.write(row, 7, data.get('ema_21', 0))
            chart_data_sheet.write(row, 8, data.get('ema_50', 0))
            chart_data_sheet.write(row, 9, data.get('ema_200', 0))
            chart_data_sheet.write(row, 10, data.get('rsi', 0))
            chart_data_sheet.write(row, 11, data.get('bb_upper', 0))
            chart_data_sheet.write(row, 12, data.get('bb_lower', 0))
            chart_data_sheet.write(row, 13, data['trade_entry'])
            chart_data_sheet.write(row, 14, data['trade_exit'])
            chart_data_sheet.write(row, 15, data['direction'])
        
        # Hoja 5: Todos los Trades
        trades_sheet = workbook.add_worksheet('Todos los Trades')
        
        # Escribir trades_df
        for col, header in enumerate(trades_df.columns):
            trades_sheet.write(0, col, header, header_format)
        
        for row, (_, data) in enumerate(trades_df.iterrows(), 1):
            for col, value in enumerate(data):
                if pd.isna(value):
                    trades_sheet.write(row, col, '')
                elif isinstance(value, (int, float)) and 'pnl' in trades_df.columns[col]:
                    trades_sheet.write(row, col, value, money_format)
                elif isinstance(value, (int, float)) and 'pct' in trades_df.columns[col]:
                    trades_sheet.write(row, col, value / 100, percent_format)
                elif isinstance(value, (int, float)) and 'price' in trades_df.columns[col]:
                    trades_sheet.write(row, col, value)
                else:
                    trades_sheet.write(row, col, str(value))
        
        workbook.close()
        logger.info(f"Reporte Excel guardado en: {filename}")
    
    def plot_results(self):
        """
        Graficar resultados del backtest
        """
        if not self.trades or not self.equity_curve:
            logger.info("No hay datos para graficar")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfico 1: Curva de Equity
        equity_df = pd.DataFrame(self.equity_curve)
        axes[0].plot(equity_df['time'], equity_df['balance'], label='Balance', linewidth=2, color='#2E86AB')
        axes[0].axhline(y=self.initial_balance, color='gray', linestyle='--', label='Balance Inicial', alpha=0.7)
        axes[0].fill_between(equity_df['time'], self.initial_balance, equity_df['balance'], 
                            where=(equity_df['balance'] >= self.initial_balance), 
                            interpolate=True, alpha=0.3, color='green')
        axes[0].fill_between(equity_df['time'], self.initial_balance, equity_df['balance'], 
                            where=(equity_df['balance'] < self.initial_balance), 
                            interpolate=True, alpha=0.3, color='red')
        axes[0].set_title('Curva de Equity - Estrategia Mean Reversion 15X', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Fecha')
        axes[0].set_ylabel('Balance (USDT)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Distribución de P/L
        trades_df = pd.DataFrame(self.trades)
        colors = ['#06A77D' if x > 0 else '#D62828' for x in trades_df['pnl']]
        axes[1].bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].set_title('P/L por Trade', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Trade #')
        axes[1].set_ylabel('P/L (USDT)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Guardar gráfico
        filename = f"backtest_futures_15x_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en: {filename}")
        
        plt.show()


def main():
    """
    Función principal para ejecutar backtest híbrido mejorado
    
    ⚠️ ADVERTENCIA IMPORTANTE SOBRE APALANCAMIENTO 15X:
    
    Este backtest simula operaciones con apalancamiento 15x.
    
    Configuración:
    - Balance: $1000
    - Margen por trade: 10% = $100
    - Con 15x: Posición de $1,500
    - Precio de liquidación: ~6.67% en tu contra
    
    RIESGO EXTREMO:
    - Una vela del 6.67% en contra = liquidación total del margen
    - Con HYPE/USDT (alta volatilidad), esto es COMÚN
    - Múltiples liquidaciones pueden destruir tu cuenta rápidamente
    
    RECOMENDACIONES:
    1. Considera usar apalancamiento 3-5x para reducir riesgo de liquidación
    2. Usa stop loss más ajustados
    3. Reduce el margen por trade (5-7% en lugar de 10%)
    4. Prueba exhaustivamente en demo antes de usar dinero real
    """
    # Configuración
    symbol = 'HYPE/USDT'
    initial_balance = 10  # Balance inicial en USDT
    risk_per_trade = 0.10  # 10% del capital por trade (margen)
    days = 120  # Días de historia para backtest
    
    print("="*70)
    print("📊 BACKTEST FUTUROS 15X - ESTRATEGIA MEAN REVERSION MEJORADA")
    print("="*70)
    print("Configuración:")
    print("• Symbol: HYPE/USDT")
    print("• Apalancamiento: 15x")
    print("• Margen por trade: 10%")
    print("• Stop Loss Long: 1.2% | Short: 1.5%")
    print("• Take Profit Long: 3.0% | Short: 2.2%")
    print("• Límite tiempo Long: 60 velas (5h) | Short: 45 velas (3.75h)")
    print("• Liquidación: ±6.67%")
    print("")
    print(f"Con ${initial_balance}:")
    print(f"• Margen: ${initial_balance * risk_per_trade}")
    print(f"• Posición: ${initial_balance * risk_per_trade * 15}")
    print("="*70 + "\n")
    
    # Crear instancia de backtest
    backtest = HybridTradingBacktest(symbol=symbol, initial_balance=initial_balance)
    
    # Ejecutar backtest
    backtest.run_backtest(risk_per_trade=risk_per_trade, days=days)
    
    # Graficar resultados
    try:
        backtest.plot_results()
    except Exception as e:
        logger.error(f"Error graficando: {e}")
        logger.info("Los resultados están guardados en CSV")


if __name__ == "__main__":
    main()


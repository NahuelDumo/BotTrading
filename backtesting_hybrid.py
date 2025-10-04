import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, time
import json
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        SHORTs: Mean Reversion en sobrecompra
        
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
        
        self.stop_loss_percent = 0.012
        self.take_profit_percent = 0.03
        
        # Historial de trades
        self.trades = []
        self.equity_curve = []
        # Estadísticas adicionales
        self.filtered_long_signals = 0  # Señales LONG fuera de horario
        
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

            # Si necesitamos más de 500, descargar en chunks desde el pasado
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
    
    def detect_smc_long(self, idx: int, lookback: int = 30) -> bool:
        """
        DEPRECADO - Ahora usamos mean_reversion_long
        """
        return False
    
    def detect_mean_reversion_short(self, idx: int) -> bool:
        """
        Detectar señal SHORT usando Mean Reversion ULTRA MEJORADO (sobrecompra)
        
        Mejoras v2:
        1. RSI más estricto (>72)
        2. Confirmación de vela bajista (rechazo)
        3. Volumen significativo
        4. Precio extendido desde EMA 50
        5. Requiere 7 de 9 condiciones
        """
        if idx < 200:
            return False
        
        current = self.df.iloc[idx]
        prev = self.df.iloc[idx - 1]
        prev2 = self.df.iloc[idx - 2]
        recent = self.df.iloc[max(0, idx-20):idx]
        
        # CONDICIÓN CRÍTICA 1: Precio cerca de resistencia (BB upper)
        at_resistance = current['close'] > current['bb_upper'] * 0.995
        
        if not at_resistance:
            return False
        
        # CONDICIÓN CRÍTICA 2: Vela de rechazo (patrón bajista) - MÁS FLEXIBLE
        # Vela actual debe cerrar en el 50% inferior de su rango O ser bajista
        candle_range = current['high'] - current['low']
        if candle_range > 0:
            close_position = (current['close'] - current['low']) / candle_range
            has_rejection = (close_position < 0.5) or (current['close'] < current['open'])
        else:
            has_rejection = current['close'] < current['open']
        
        if not has_rejection:
            return False
        
        # Detectar sobrecompra con confirmaciones BALANCEADAS
        overbought_conditions = [
            current['ema_9'] > current['ema_21'],          # Tendencia alcista corto plazo
            current['ema_21'] > current['ema_50'],         # Tendencia alcista medio plazo
            current['close'] > current['ema_9'],           # Precio arriba de EMA rápida
            current['rsi'] > 68,                           # RSI alto (sobrecompra)
            current['macd'] > current['macd_signal'],      # MACD alcista
            current['macd_hist'] > prev['macd_hist'],      # Momentum alcista creciendo
            current['volume'] > recent['volume'].mean() * 1.2,  # Volumen presente
            current['close'] > current['ema_50'] * 1.01,   # Precio 1% arriba de EMA 50
            current['high'] > prev['high']                 # Nuevo máximo (exhaustión)
        ]
        
        return sum(overbought_conditions) >= 6  # Balanceado: 6 de 9
    
    def analyze_market_direction(self, idx: int) -> str:
        """
        Analizar mercado usando Estrategia Híbrida de Mean Reversion
        
        LONGs: Mean Reversion en sobreventa (Lun-Vie 8:00-20:00)
        SHORTs: Mean Reversion en sobrecompra (24/7)
        
        Ambas estrategias ahora son simétricas y consistentes
        
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
        
        # Detectar Mean Reversion SHORT (sobrecompra)
        if self.detect_mean_reversion_short(idx):
            return 'SHORT'
        
        return None
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """
        Calcular stop loss al 1.5% (ajustado para futuros con 5x)
        """
        stop_distance = entry_price * self.stop_loss_percent
        
        if direction == 'LONG':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_liquidation_price(self, entry_price: float, direction: str) -> float:
        """
        Calcular precio de liquidación aproximado con apalancamiento 5x
        Liquidación ocurre cuando pérdidas = 100% del margen (20% del precio)
        """
        liquidation_distance = entry_price * (1 / self.leverage)
        
        if direction == 'LONG':
            return entry_price - liquidation_distance  # -20%
        else:
            return entry_price + liquidation_distance  # +20%
    
    def calculate_take_profit(self, entry_price: float, direction: str, 
                            is_first_trade: bool, idx: int) -> float:
        """
        Calcular take profit al 4.5% (R:R 1:3)
        Con apalancamiento 5x, esto representa un movimiento del precio del 4.5%
        pero una ganancia del 22.5% sobre el margen
        """
        tp_distance = entry_price * self.take_profit_percent
        
        if direction == 'LONG':
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance
    
    def simulate_trade(self, entry_idx: int, direction: str, entry_price: float,
                      stop_loss: float, take_profit: float, position_size: float) -> Dict:
        """
        Simular un trade de FUTUROS desde la entrada hasta la salida
        Incluye verificación de liquidación con apalancamiento 5x
        
        LÍMITE DE TIEMPO: Si no alcanza TP/SL en 60 velas (5 horas en 5m),
        cierra el trade automáticamente al precio actual
        
        Returns:
            Dict con información del trade
        """
        # Calcular precio de liquidación
        liquidation_price = self.calculate_liquidation_price(entry_price, direction)
        
        # Límite de tiempo: máximo 60 velas (~5 horas en 5m)
        max_candles = 60
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
                    # Con 5x: pérdida del 1.5% en precio = 7.5% del margen
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
                    # Con 5x: ganancia del 4.5% en precio = 22.5% del margen
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
            # Alcanzó límite de tiempo (60 velas)
            exit_reason = 'Time Limit (5h)'
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
            
        NOTA: Este backtest simula trading SIN apalancamiento.
        Si usas apalancamiento 15x en el exchange:
        - 20% del capital = posición de ~$3000 con $200 (si tienes $1000)
        - El riesgo de liquidación es EXTREMADAMENTE ALTO
        - Una vela del 0.67% en contra = liquidación total
        """
        logger.info("=== BACKTEST FUTUROS 5X ===")
        logger.info("Margen por trade: 20.0%")
        logger.info("SL: 1.5% | TP: 4.5%")
        logger.info("Límite de tiempo: 60 velas (5 horas)")
        
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
                    stop_loss = self.calculate_stop_loss(entry_price, direction)
                    
                    # Verificar si es primer trade de la mañana
                    is_morning = current_time.time() < time(12, 0)
                    is_first = daily_first_trade and is_morning
                    
                    take_profit = self.calculate_take_profit(
                        entry_price, direction, is_first, i
                    )
                    
                    # Calcular tamaño de posición para FUTUROS con 5x
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
                    logger.info(f"Trade #{len(self.trades)}: {direction} @ {entry_price:.2f} -> "
                              f"{trade['exit_price']:.2f} | P/L: ${trade['pnl']:.2f} "
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
        print("RESULTADOS FUTUROS 5X")
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
        
        print(f"\n⏰ FILTRO LONGS")
        long_trades_executed = len(trades_df[trades_df['direction'] == 'LONG'])
        total_long_signals = long_trades_executed + self.filtered_long_signals
        if total_long_signals > 0:
            filter_rate = (self.filtered_long_signals / total_long_signals) * 100
            print(f"Detectadas: {total_long_signals}")
            print(f"Ejecutadas: {long_trades_executed} ({100-filter_rate:.1f}%)")
            print(f"Filtradas:  {self.filtered_long_signals} ({filter_rate:.1f}%)")
        else:
            print(f"Sin señales LONG")
        
        print("\n" + "="*60)
        
        # Guardar resultados detallados
        self.save_detailed_results(trades_df)
    
    def save_detailed_results(self, trades_df: pd.DataFrame):
        """
        Guardar resultados detallados en CSV
        """
        filename = f"backtest_futures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"Guardado: {filename}")
    
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
        axes[0].plot(equity_df['time'], equity_df['balance'], label='Balance', linewidth=2)
        axes[0].axhline(y=self.initial_balance, color='gray', linestyle='--', label='Balance Inicial')
        axes[0].set_title('Curva de Equity - Estrategia Híbrida', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Fecha')
        axes[0].set_ylabel('Balance (USDT)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Distribución de P/L
        trades_df = pd.DataFrame(self.trades)
        colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
        axes[1].bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.6)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_title('P/L por Trade - Estrategia Híbrida', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Trade #')
        axes[1].set_ylabel('P/L (USDT)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        filename = f"backtest_hybrid_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en: {filename}")
        
        plt.show()
    # ==============================
    # DESCARGA DE DATOS COMPLETOS
    # ==============================
    def get_full_historical_klines(symbol, interval, start_str, end_str=None):
        """
        Descarga TODO el histórico de velas entre start_str y end_str,
        iterando de a 1000 velas por request.
        """
        all_klines = []
        start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else int(datetime.now().timestamp() * 1000)


        while True:
            klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_ts,
            limit=1000
            )


            if not klines:
                break


            all_klines.extend(klines)


            last_open_time = klines[-1][0]
            start_ts = last_open_time + 1


            if start_ts >= end_ts:
                break


            time.sleep(0.2)


            return all_klines


def main():
    """
    Función principal para ejecutar backtest híbrido
    
    ⚠️ ADVERTENCIA IMPORTANTE SOBRE APALANCAMIENTO:
    
    Este backtest simula operaciones SIN apalancamiento (spot trading).
    Usa 20% del capital por trade.
    
    Si planeas usar apalancamiento 15x en el exchange:
    - Balance: $1000
    - Capital por trade: 20% = $200
    - Con 15x: Controlarás una posición de $3,000
    - Precio de liquidación: 0.67% en tu contra
    - Con HYPE/USDT (muy volátil), las liquidaciones son MUY comunes
    
    RIESGO REAL: Una sola vela volátil puede liquidar tu cuenta completa.
    """
    # Configuración
    symbol = 'HYPE/USDT'
    initial_balance = 1000  # Balance inicial en USDT
    risk_per_trade = 0.10  # 10% del capital por trade
    days = 215  # Días de historia que quieres probar
    
    print("="*70)
    print("📊 BACKTEST FUTUROS 5X - HYPE/USDT")
    print("="*70)
    print("Configuración:")
    print("• Apalancamiento: 5x")
    print("• Margen por trade: 20%")
    print("• Stop Loss: 1.5% → ~7.5% del margen")
    print("• Take Profit: 4.5% → ~22.5% del margen")
    print("• Límite de tiempo: 60 velas (5 horas)")
    print("• Liquidación: ±20%")
    print("")
    print("Con $1000:")
    print("• Margen: $200")
    print("• Posición: $1,000")
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
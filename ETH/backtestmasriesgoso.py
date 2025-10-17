import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import xlsxwriter

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartMoneyScalpingBacktest:
    def __init__(self, symbol: str, initial_balance: float = 1000.0):
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
        self.symbol = symbol
        self.timeframe = '5m'
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # Parámetros Optimizados para MAYOR FRECUENCIA
        self.structure_lookback = 20
        self.risk_reward_ratio = 2.5
        self.leverage = 20
        self.risk_per_trade_pct = 0.02
        self.max_candles_in_trade = 24

        # Historial
        self.trades = []
        self.equity_curve = [{'time': None, 'balance': self.initial_balance}]
        self.active_trade = None

    def fetch_historical_data(self, days=365*2) -> pd.DataFrame:
        """Descarga datos históricos."""
        try:
            logger.info(f"Descargando datos para {self.symbol} ({self.timeframe}) de los últimos {days} días...")
            limit = 1000
            since = self.exchange.milliseconds() - 86400000 * days
            all_ohlcv = []
            
            while since < self.exchange.milliseconds():
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=limit)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(all_ohlcv) % 20000 == 0: logger.info(f"Descargadas {len(all_ohlcv)} velas...")
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Convertir a UTC y luego a la zona horaria deseada (UTC-4)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Etc/GMT+4')
            df.set_index('timestamp', inplace=True)
            logger.info(f"✅ Descarga completa: {len(df)} velas desde {df.index[0]}.")
            return df
        except Exception as e:
            logger.error(f"❌ Error descargando datos: {e}")
            return None

    def find_patterns(self):
        """Identifica la estructura del mercado, barridos de liquidez y FVGs."""
        logger.info("Identificando patrones SMC (alcistas y bajistas)...")
        
        n = self.structure_lookback
        self.df['min'] = self.df.iloc[argrelextrema(self.df.low.values, np.less_equal, order=n)[0]]['low']
        self.df['max'] = self.df.iloc[argrelextrema(self.df.high.values, np.greater_equal, order=n)[0]]['high']

        self.df['is_fvg_bullish'] = False
        self.df['is_fvg_bearish'] = False

        lows = self.df['low'].values
        highs = self.df['high'].values
        is_fvg_bullish = np.zeros(len(self.df), dtype=bool)
        is_fvg_bearish = np.zeros(len(self.df), dtype=bool)

        for i in range(2, len(self.df)):
            if lows[i] > highs[i-2]: is_fvg_bullish[i-1] = True
            if highs[i] < lows[i-2]: is_fvg_bearish[i-1] = True
        
        self.df['is_fvg_bullish'] = is_fvg_bullish
        self.df['is_fvg_bearish'] = is_fvg_bearish

    def run_backtest(self):
        """Ejecuta el backtest completo."""
        self.df = self.fetch_historical_data()
        if self.df is None: return
        self.find_patterns()

        logger.info("Simulando trades (Longs y Shorts)...")
        i = self.structure_lookback + 2
        while i < len(self.df):
            if self.active_trade:
                exit_info = self.manage_active_trade(i)
                if exit_info:
                    i = exit_info['exit_idx'] + 1
                    continue
            
            if self.check_long_setup(i):
                i += 1
                continue

            if self.check_short_setup(i):
                i += 1
                continue
            i += 1

        logger.info(f"=== BACKTEST COMPLETADO PARA {self.symbol} ===")
        self.print_results()
        self.plot_results()
    
    def check_long_setup(self, i):
        recent_lows = self.df['min'].iloc[i-50:i].dropna()
        if len(recent_lows) < 2 or recent_lows.iloc[-1] >= recent_lows.iloc[-2]: return False
        
        sweep_idx = self.df.index.get_loc(recent_lows.index[-1])
        if i - sweep_idx > 12: return False
        
        fvg_window = self.df.iloc[sweep_idx:i]
        bullish_fvgs = fvg_window[fvg_window['is_fvg_bullish']]
        if not bullish_fvgs.empty:
            fvg_candle_idx = self.df.index.get_loc(bullish_fvgs.index[-1])
            fvg_high = self.df['low'].iloc[fvg_candle_idx + 1]
            if self.df['low'].iloc[i] <= fvg_high:
                self.open_trade(i, 'LONG', fvg_high, recent_lows.iloc[-1])
                return True
        return False

    def check_short_setup(self, i):
        recent_highs = self.df['max'].iloc[i-50:i].dropna()
        if len(recent_highs) < 2 or recent_highs.iloc[-1] <= recent_highs.iloc[-2]: return False

        sweep_idx = self.df.index.get_loc(recent_highs.index[-1])
        if i - sweep_idx > 12: return False

        fvg_window = self.df.iloc[sweep_idx:i]
        bearish_fvgs = fvg_window[fvg_window['is_fvg_bearish']]
        if not bearish_fvgs.empty:
            fvg_candle_idx = self.df.index.get_loc(bearish_fvgs.index[-1])
            fvg_low = self.df['high'].iloc[fvg_candle_idx + 1]
            if self.df['high'].iloc[i] >= fvg_low:
                self.open_trade(i, 'SHORT', fvg_low, recent_highs.iloc[-1])
                return True
        return False

    def open_trade(self, entry_idx, direction, entry_price, liquidity_level):
        """Abre una nueva posición."""
        if direction == 'LONG':
            stop_loss_price = liquidity_level * 0.999
            risk_per_unit = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk_per_unit * self.risk_reward_ratio)
        else:
            stop_loss_price = liquidity_level * 1.001
            risk_per_unit = stop_loss_price - entry_price
            take_profit_price = entry_price - (risk_per_unit * self.risk_reward_ratio)

        if risk_per_unit <= 0: return
        
        capital_to_risk = self.balance * self.risk_per_trade_pct
        position_size = capital_to_risk / risk_per_unit

        self.active_trade = {
            'entry_idx': entry_idx, 'direction': direction, 'entry_price': entry_price,
            'sl_price': stop_loss_price, 'tp_price': take_profit_price,
            'position_size': position_size
        }

    def manage_active_trade(self, current_idx):
        """Gestiona la salida de un trade activo."""
        trade = self.active_trade
        
        for i in range(trade['entry_idx'] + 1, current_idx + 1):
            if i >= len(self.df): return None
            candle = self.df.iloc[i]
            exit_reason, pnl, exit_price = None, 0, 0

            if trade['direction'] == 'LONG':
                if candle['low'] <= trade['sl_price']:
                    exit_reason, exit_price = 'Stop Loss', trade['sl_price']
                elif candle['high'] >= trade['tp_price']:
                    exit_reason, exit_price = 'Take Profit', trade['tp_price']
            else:
                if candle['high'] >= trade['sl_price']:
                    exit_reason, exit_price = 'Stop Loss', trade['sl_price']
                elif candle['low'] <= trade['tp_price']:
                    exit_reason, exit_price = 'Take Profit', trade['tp_price']
            
            if not exit_reason and (i - trade['entry_idx']) >= self.max_candles_in_trade:
                exit_reason, exit_price = 'Time Limit', candle['close']

            if exit_reason:
                pnl = (exit_price - trade['entry_price']) * trade['position_size'] if trade['direction'] == 'LONG' else (trade['entry_price'] - exit_price) * trade['position_size']
                self._log_and_close_trade(trade, i, exit_price, exit_reason, pnl)
                return {'exit_idx': i}
        return None

    def _log_and_close_trade(self, trade, exit_idx, exit_price, exit_reason, pnl):
        """Función auxiliar para registrar un trade con todos los detalles."""
        trade_log = {
            'entry_time': self.df.index[trade['entry_idx']],
            'exit_time': self.df.index[exit_idx],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': exit_price,
            'stop_loss': trade['sl_price'],
            'take_profit': trade['tp_price'],
            'pnl': pnl,
            'exit_reason': exit_reason
        }
        self.balance += pnl
        self.trades.append(trade_log)
        self.equity_curve.append({'time': self.df.index[exit_idx], 'balance': self.balance})
        self.active_trade = None

    def print_results(self):
        """Imprime las métricas y llama a la generación del reporte Excel."""
        if not self.trades:
            logger.warning("No se ejecutaron trades.")
            return

        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        wins = trades_df[trades_df['pnl'] > 0]
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        print("\n" + "="*60)
        print(f"RESULTADOS SMC BIDIRECCIONAL: {self.symbol}")
        print("="*60)
        print(f"Balance Final:      ${self.balance:,.2f} | Retorno: {(self.balance/self.initial_balance - 1)*100:,.2f}%")
        print(f"Total de Trades:    {total_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print("="*60)

        self.save_excel_report(trades_df)

    def plot_results(self):
        """Grafica la curva de equity."""
        if len(self.equity_curve) <= 1: return
        
        equity_df = pd.DataFrame(self.equity_curve).dropna().set_index('time')
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(equity_df.index, equity_df['balance'], label='Balance', color='#0077b6', linewidth=1.5)
        ax.fill_between(equity_df.index, self.initial_balance, equity_df['balance'], where=(equity_df['balance'] >= self.initial_balance), color='#2ca02c', alpha=0.3, interpolate=True)
        ax.set_title(f'Curva de Equity - SMC Bidireccional - {self.symbol}', fontsize=16, fontweight='bold')
        filename = f"equity_curve_SMC_bidirectional_{self.symbol.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300)
        logger.info(f"Gráfico guardado como: {filename}")
        plt.show()

    def save_excel_report(self, trades_df: pd.DataFrame):
        """Guarda un reporte completo en Excel con múltiples hojas y gráficos."""
        filename = f"report_SMC_{self.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        logger.info(f"Generando reporte Excel: {filename}")
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1, 'align': 'center'})
            money_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # --- Hoja 1: Resumen General ---
            summary_sheet = workbook.add_worksheet('Resumen General')
            
            total_trades = len(trades_df)
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] <= 0]
            win_rate = (len(wins) / total_trades) if total_trades > 0 else 0
            
            equity_df = pd.DataFrame(self.equity_curve).dropna().set_index('time')
            if not equity_df.empty:
                equity_df['peak'] = equity_df['balance'].cummax()
                equity_df['drawdown_pct'] = (equity_df['balance'] - equity_df['peak']) / equity_df['peak']
                max_drawdown = equity_df['drawdown_pct'].min()
            else: max_drawdown = 0

            summary_data = {
                "Balance Inicial": self.initial_balance, "Balance Final": self.balance,
                "Retorno Total": (self.balance / self.initial_balance) - 1, "P/L Neto": trades_df['pnl'].sum(),
                "Max Drawdown": max_drawdown, "Total de Trades": total_trades, "Win Rate": win_rate,
                "Profit Factor": wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() != 0 else float('inf'),
                "Ganancia Promedio": wins['pnl'].mean() if len(wins) > 0 else 0,
                "Pérdida Promedio": losses['pnl'].mean() if len(losses) > 0 else 0
            }

            summary_sheet.write_row('A1', ['Métrica', 'Valor'], header_format)
            row = 1
            for key, value in summary_data.items():
                summary_sheet.write(row, 0, key)
                fmt = money_format
                if "Retorno" in key or "Rate" in key or "Drawdown" in key: fmt = percent_format
                elif "Trades" in key or "Factor" in key: fmt = None
                summary_sheet.write(row, 1, value, fmt)
                row += 1
            summary_sheet.set_column('A:A', 20); summary_sheet.set_column('B:B', 15)

            # --- Hoja 2: Todos los Trades ---
            # ## <<< CORRECCIÓN CLAVE PARA EL ERROR DE ZONA HORARIA >>>
            trades_df_excel = trades_df.copy()
            trades_df_excel['entry_time'] = trades_df_excel['entry_time'].dt.tz_localize(None)
            trades_df_excel['exit_time'] = trades_df_excel['exit_time'].dt.tz_localize(None)
            trades_df_excel.to_excel(writer, sheet_name='Todos los Trades', index=False)
            
            # --- Lógica para las demás hojas ---
            if not trades_df.empty:
                # Daily
                daily_sheet = workbook.add_worksheet('Desempeño por Días')
                daily_perf = trades_df.groupby(trades_df['entry_time'].dt.date).agg(total_pnl=('pnl', 'sum')).reset_index()
                daily_sheet.write_row('A1', ['Fecha', 'P/L Total'], header_format)
                for r_idx, row in daily_perf.iterrows():
                    daily_sheet.write(r_idx + 1, 0, row['entry_time'].strftime('%Y-%m-%d')); daily_sheet.write(r_idx + 1, 1, row['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'}); chart.add_series({'name': 'P/L Diario', 'categories': "='Desempeño por Días'!$A$2:$A$" + str(len(daily_perf)+1), 'values': "='Desempeño por Días'!$B$2:$B$" + str(len(daily_perf)+1)}); daily_sheet.insert_chart('D2', chart)

                # Hourly
                hourly_sheet = workbook.add_worksheet('Desempeño por Horas')
                hourly_perf = trades_df.groupby(trades_df['entry_time'].dt.hour).agg(total_pnl=('pnl', 'sum')).reset_index()
                hourly_sheet.write_row('A1', ['Hora', 'P/L Total'], header_format)
                for r_idx, row in hourly_perf.iterrows():
                    hourly_sheet.write(r_idx + 1, 0, row['entry_time']); hourly_sheet.write(r_idx + 1, 1, row['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'}); chart.add_series({'name': 'P/L por Hora', 'categories': "='Desempeño por Horas'!$A$2:$A$" + str(len(hourly_perf)+1), 'values': "='Desempeño por Horas'!$B$2:$B$" + str(len(hourly_perf)+1)}); hourly_sheet.insert_chart('D2', chart)

                # Weekday
                weekday_sheet = workbook.add_worksheet('Desempeño por Día Semana')
                day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                trades_df['weekday'] = trades_df['entry_time'].dt.weekday
                weekday_perf = trades_df.groupby('weekday').agg(total_pnl=('pnl', 'sum')).reindex(range(7)).fillna(0).reset_index()
                weekday_perf['weekday'] = weekday_perf['weekday'].map(lambda x: day_names[x])
                weekday_sheet.write_row('A1', ['Día', 'P/L Total'], header_format)
                for r_idx, row in weekday_perf.iterrows():
                    weekday_sheet.write(r_idx + 1, 0, row['weekday']); weekday_sheet.write(r_idx + 1, 1, row['total_pnl'], money_format)
                chart = workbook.add_chart({'type': 'column'}); chart.add_series({'name': 'P/L por Día', 'categories': "='Desempeño por Día Semana'!$A$2:$A$8", 'values': "='Desempeño por Día Semana'!$B$2:$B$8"}); weekday_sheet.insert_chart('D2', chart)

        logger.info("✅ Reporte Excel guardado.")

if __name__ == "__main__":
    backtest = SmartMoneyScalpingBacktest(symbol='ETH/USDT', initial_balance=1000.0)
    backtest.run_backtest()
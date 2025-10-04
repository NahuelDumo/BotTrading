"""
Test de Diferentes Duraciones Máximas de Trade
================================================
Compara el rendimiento del backtest con diferentes límites de tiempo máximo por trade.

Duraciones a probar:
- 1 hora  = 12 velas de 5m
- 2 horas = 24 velas
- 3 horas = 36 velas
- 5 horas = 60 velas
- 7 horas = 84 velas
- 9 horas = 108 velas
- 12 horas = 144 velas
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, time
import logging
from typing import Dict, List
import matplotlib.pyplot as plt

# Importar la clase de backtest
import sys
sys.path.append('.')
from backtesting_hybrid import HybridTradingBacktest

logging.basicConfig(
    level=logging.WARNING,  # Solo warnings y errores para no saturar output
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeDurationComparison:
    """
    Clase para comparar resultados con diferentes duraciones máximas de trade
    """
    
    def __init__(self, symbol: str = 'HYPE/USDT', initial_balance: float = 1000):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.results = []
        
    def run_test_with_duration(self, max_hours: float, days: int = 120) -> Dict:
        """
        Ejecutar backtest con una duración máxima específica
        
        Args:
            max_hours: Duración máxima en horas
            days: Días de historia
            
        Returns:
            Dict con métricas del backtest
        """
        # Crear instancia de backtest
        backtest = HybridTradingBacktest(
            symbol=self.symbol,
            initial_balance=self.initial_balance
        )
        
        # Modificar el método simulate_trade para usar la duración específica
        max_candles = int(max_hours * 12)  # 12 velas de 5m por hora
        
        # Guardar método original
        original_simulate = backtest.simulate_trade
        
        # Crear wrapper con la duración específica
        def simulate_with_custom_duration(entry_idx, direction, entry_price, 
                                         stop_loss, take_profit, position_size):
            # Calcular precio de liquidación
            liquidation_price = backtest.calculate_liquidation_price(entry_price, direction)
            
            # Límite de tiempo personalizado
            end_idx = min(entry_idx + max_candles, len(backtest.df) - 1)
            
            # Buscar salida en las siguientes velas
            for i in range(entry_idx + 1, end_idx + 1):
                candle = backtest.df.iloc[i]
                
                if direction == 'LONG':
                    # VERIFICAR LIQUIDACIÓN PRIMERO
                    if candle['low'] <= liquidation_price:
                        exit_price = liquidation_price
                        exit_reason = 'LIQUIDACIÓN'
                        pnl = -(position_size * entry_price / backtest.leverage)
                        return {
                            'entry_time': backtest.df.index[entry_idx],
                            'exit_time': backtest.df.index[i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'liquidation_price': liquidation_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': -100.0,
                            'exit_reason': exit_reason,
                            'duration_candles': i - entry_idx
                        }
                    
                    # STOP LOSS
                    if candle['low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'Stop Loss'
                        pnl = (exit_price - entry_price) * position_size
                        return {
                            'entry_time': backtest.df.index[entry_idx],
                            'exit_time': backtest.df.index[i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'liquidation_price': liquidation_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': (pnl / (entry_price * position_size / backtest.leverage)) * 100,
                            'exit_reason': exit_reason,
                            'duration_candles': i - entry_idx
                        }
                    
                    # TAKE PROFIT
                    if candle['high'] >= take_profit:
                        exit_price = take_profit
                        exit_reason = 'Take Profit'
                        pnl = (exit_price - entry_price) * position_size
                        return {
                            'entry_time': backtest.df.index[entry_idx],
                            'exit_time': backtest.df.index[i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'liquidation_price': liquidation_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': (pnl / (entry_price * position_size / backtest.leverage)) * 100,
                            'exit_reason': exit_reason,
                            'duration_candles': i - entry_idx
                        }
                
                else:  # SHORT
                    # VERIFICAR LIQUIDACIÓN PRIMERO
                    if candle['high'] >= liquidation_price:
                        exit_price = liquidation_price
                        exit_reason = 'LIQUIDACIÓN'
                        pnl = -(position_size * entry_price / backtest.leverage)
                        return {
                            'entry_time': backtest.df.index[entry_idx],
                            'exit_time': backtest.df.index[i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'liquidation_price': liquidation_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': -100.0,
                            'exit_reason': exit_reason,
                            'duration_candles': i - entry_idx
                        }
                    
                    # STOP LOSS
                    if candle['high'] >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'Stop Loss'
                        pnl = (entry_price - exit_price) * position_size
                        return {
                            'entry_time': backtest.df.index[entry_idx],
                            'exit_time': backtest.df.index[i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'liquidation_price': liquidation_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': (pnl / (entry_price * position_size / backtest.leverage)) * 100,
                            'exit_reason': exit_reason,
                            'duration_candles': i - entry_idx
                        }
                    
                    # TAKE PROFIT
                    if candle['low'] <= take_profit:
                        exit_price = take_profit
                        exit_reason = 'Take Profit'
                        pnl = (entry_price - exit_price) * position_size
                        return {
                            'entry_time': backtest.df.index[entry_idx],
                            'exit_time': backtest.df.index[i],
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'liquidation_price': liquidation_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'pnl_pct': (pnl / (entry_price * position_size / backtest.leverage)) * 100,
                            'exit_reason': exit_reason,
                            'duration_candles': i - entry_idx
                        }
            
            # Límite de tiempo alcanzado
            last_available_idx = min(end_idx, len(backtest.df) - 1)
            last_candle = backtest.df.iloc[last_available_idx]
            exit_price = last_candle['close']
            
            if last_available_idx < len(backtest.df) - 1:
                exit_reason = f'Time Limit ({max_hours}h)'
            else:
                exit_reason = 'End of Data'
            
            if direction == 'LONG':
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            return {
                'entry_time': backtest.df.index[entry_idx],
                'exit_time': backtest.df.index[last_available_idx],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'liquidation_price': liquidation_price,
                'position_size': position_size,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * position_size / backtest.leverage)) * 100,
                'exit_reason': exit_reason,
                'duration_candles': last_available_idx - entry_idx
            }
        
        # Reemplazar método
        backtest.simulate_trade = simulate_with_custom_duration
        
        # Ejecutar backtest (sin prints)
        print(f"\n⏱️  Probando {max_hours}h máximo por trade...", end=" ")
        
        # Descargar datos
        backtest.df = backtest.fetch_historical_data(days)
        if backtest.df is None:
            return None
        
        # Calcular indicadores
        backtest.df = backtest.calculate_indicators(backtest.df)
        
        # Simular trades
        risk_per_trade = 0.20
        for idx in range(200, len(backtest.df)):
            current = backtest.df.iloc[idx]
            
            # Analizar dirección del mercado
            direction = backtest.analyze_market_direction(idx)
            
            if direction == 'NEUTRAL':
                continue
            
            # Verificar si estamos en un trade activo
            if backtest.trades and backtest.trades[-1].get('exit_time') is None:
                continue
            
            # Calcular posición
            available_capital = backtest.balance * risk_per_trade
            position_size = (available_capital * backtest.leverage) / current['close']
            
            # Calcular SL/TP
            if direction == 'LONG':
                entry_price = current['close']
                stop_loss = entry_price * (1 - backtest.stop_loss_percent)
                take_profit = backtest.calculate_dynamic_take_profit(entry_price, direction, idx)
            else:
                entry_price = current['close']
                stop_loss = entry_price * (1 + backtest.stop_loss_percent)
                take_profit = backtest.calculate_dynamic_take_profit(entry_price, direction, idx)
            
            # Simular trade
            trade_result = backtest.simulate_trade(
                entry_idx=idx,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size
            )
            
            # Actualizar balance
            backtest.balance += trade_result['pnl']
            backtest.trades.append(trade_result)
            backtest.equity_curve.append({
                'time': trade_result['exit_time'],
                'balance': backtest.balance
            })
        
        # Calcular métricas
        if not backtest.trades:
            print("❌ Sin trades")
            return None
        
        trades_df = pd.DataFrame(backtest.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_return = ((backtest.balance - backtest.initial_balance) / backtest.initial_balance) * 100
        
        # Calcular drawdown
        equity_df = pd.DataFrame(backtest.equity_curve)
        equity_df['cummax'] = equity_df['balance'].cummax()
        equity_df['drawdown_pct'] = ((equity_df['balance'] - equity_df['cummax']) / equity_df['cummax']) * 100
        max_drawdown = equity_df['drawdown_pct'].min()
        
        # Profit factor
        total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_losses = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # LONG/SHORT stats
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        long_wr = (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
        short_wr = (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
        
        long_pnl = long_trades['pnl'].sum() if len(long_trades) > 0 else 0
        short_pnl = short_trades['pnl'].sum() if len(short_trades) > 0 else 0
        
        # Razones de salida
        exit_reasons = trades_df['exit_reason'].value_counts()
        time_limit_count = sum(1 for reason in exit_reasons.index if 'Time Limit' in reason)
        time_limit_pct = (time_limit_count / total_trades) * 100 if total_trades > 0 else 0
        
        print(f"✅ Return: {total_return:.2f}% | WR: {win_rate:.2f}%")
        
        return {
            'max_hours': max_hours,
            'max_candles': max_candles,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_balance': backtest.balance,
            'max_drawdown': max_drawdown,
            'total_pnl': total_pnl,
            'long_trades': len(long_trades),
            'long_wr': long_wr,
            'long_pnl': long_pnl,
            'short_trades': len(short_trades),
            'short_wr': short_wr,
            'short_pnl': short_pnl,
            'time_limit_pct': time_limit_pct,
            'avg_trade_pnl': total_pnl / total_trades if total_trades > 0 else 0
        }
    
    def run_comparison(self, durations: List[float] = [1, 2, 3, 5, 7, 9, 12], days: int = 120):
        """
        Ejecutar comparación con múltiples duraciones
        
        Args:
            durations: Lista de duraciones máximas en horas
            days: Días de historia
        """
        print("="*70)
        print("📊 COMPARACIÓN DE DURACIONES MÁXIMAS DE TRADE")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Balance Inicial: ${self.initial_balance}")
        print(f"Período: {days} días")
        print(f"Duraciones a probar: {durations} horas")
        print("="*70)
        
        self.results = []
        
        for duration in durations:
            result = self.run_test_with_duration(duration, days)
            if result:
                self.results.append(result)
        
        # Mostrar comparación
        self.show_comparison()
        
        # Graficar resultados
        self.plot_comparison()
        
        # Guardar resultados
        self.save_results()
    
    def show_comparison(self):
        """
        Mostrar tabla comparativa de resultados
        """
        if not self.results:
            print("\n❌ No hay resultados para comparar")
            return
        
        print("\n" + "="*120)
        print("📊 TABLA COMPARATIVA DE RESULTADOS")
        print("="*120)
        
        # Header
        print(f"{'Duración':>10} | {'Trades':>7} | {'Win Rate':>9} | {'PF':>6} | "
              f"{'Return':>9} | {'Drawdown':>10} | {'LONG WR':>9} | {'SHORT WR':>9} | {'Time Limit':>12}")
        print("-" * 120)
        
        # Datos
        for r in self.results:
            print(f"{r['max_hours']:>8}h | {r['total_trades']:>7} | "
                  f"{r['win_rate']:>8.2f}% | {r['profit_factor']:>6.2f} | "
                  f"{r['total_return']:>8.2f}% | {r['max_drawdown']:>9.2f}% | "
                  f"{r['long_wr']:>8.2f}% | {r['short_wr']:>8.2f}% | "
                  f"{r['time_limit_pct']:>11.2f}%")
        
        print("="*120)
        
        # Mejor configuración
        best_return = max(self.results, key=lambda x: x['total_return'])
        best_wr = max(self.results, key=lambda x: x['win_rate'])
        best_pf = max(self.results, key=lambda x: x['profit_factor'])
        
        print(f"\n🏆 MEJORES CONFIGURACIONES:")
        print(f"   • Mejor Return:       {best_return['max_hours']}h → {best_return['total_return']:.2f}%")
        print(f"   • Mejor Win Rate:     {best_wr['max_hours']}h → {best_wr['win_rate']:.2f}%")
        print(f"   • Mejor Profit Factor: {best_pf['max_hours']}h → {best_pf['profit_factor']:.2f}")
        print("="*120)
    
    def plot_comparison(self):
        """
        Graficar comparación de métricas
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comparación de Duraciones Máximas de Trade', fontsize=16, fontweight='bold')
        
        # 1. Return Total
        axes[0, 0].bar(df['max_hours'].astype(str) + 'h', df['total_return'], color='green', alpha=0.7)
        axes[0, 0].set_title('Return Total (%)')
        axes[0, 0].set_xlabel('Duración Máxima')
        axes[0, 0].set_ylabel('Return %')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Win Rate
        axes[0, 1].bar(df['max_hours'].astype(str) + 'h', df['win_rate'], color='blue', alpha=0.7)
        axes[0, 1].set_title('Win Rate (%)')
        axes[0, 1].set_xlabel('Duración Máxima')
        axes[0, 1].set_ylabel('Win Rate %')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
        
        # 3. Profit Factor
        axes[0, 2].bar(df['max_hours'].astype(str) + 'h', df['profit_factor'], color='purple', alpha=0.7)
        axes[0, 2].set_title('Profit Factor')
        axes[0, 2].set_xlabel('Duración Máxima')
        axes[0, 2].set_ylabel('Profit Factor')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break Even')
        
        # 4. Max Drawdown
        axes[1, 0].bar(df['max_hours'].astype(str) + 'h', df['max_drawdown'], color='red', alpha=0.7)
        axes[1, 0].set_title('Max Drawdown (%)')
        axes[1, 0].set_xlabel('Duración Máxima')
        axes[1, 0].set_ylabel('Drawdown %')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Total Trades
        axes[1, 1].bar(df['max_hours'].astype(str) + 'h', df['total_trades'], color='orange', alpha=0.7)
        axes[1, 1].set_title('Total Trades')
        axes[1, 1].set_xlabel('Duración Máxima')
        axes[1, 1].set_ylabel('Número de Trades')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Time Limit %
        axes[1, 2].bar(df['max_hours'].astype(str) + 'h', df['time_limit_pct'], color='brown', alpha=0.7)
        axes[1, 2].set_title('Trades cerrados por Time Limit (%)')
        axes[1, 2].set_xlabel('Duración Máxima')
        axes[1, 2].set_ylabel('%')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        filename = f"duration_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado: {filename}")
        
        plt.show()
    
    def save_results(self):
        """
        Guardar resultados en CSV
        """
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        filename = f"duration_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Resultados guardados: {filename}")


if __name__ == "__main__":
    # Configuración
    symbol = 'HYPE/USDT'
    initial_balance = 1000
    days = 120  # Días de historia
    
    # Duraciones a probar (en horas)
    durations = [1, 2, 3, 5, 7, 9, 12]
    
    # Crear comparador
    comparison = TradeDurationComparison(
        symbol=symbol,
        initial_balance=initial_balance
    )
    
    # Ejecutar comparación
    comparison.run_comparison(durations=durations, days=days)

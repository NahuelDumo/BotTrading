# 📊 Guía de Backtesting

Guía completa para probar tu estrategia con datos históricos antes de operar en vivo.

## 🎯 ¿Qué es el Backtesting?

El backtesting te permite probar tu estrategia con datos históricos para ver cómo habría funcionado en el pasado. Esto te ayuda a:

- ✅ Validar tu estrategia antes de arriesgar dinero real
- ✅ Optimizar parámetros (stop loss, take profit, indicadores)
- ✅ Entender las métricas de rendimiento
- ✅ Identificar debilidades en la estrategia

## 🚀 Cómo Ejecutar el Backtesting

### Opción 1: Ejecución Básica

```bash
python backtesting.py
```

Esto ejecutará el backtest con configuración por defecto:
- **Par**: BTC/USDT
- **Balance inicial**: $1000 USDT
- **Riesgo por trade**: 2%
- **Período**: Últimos 30 días

### Opción 2: Personalizar Parámetros

Edita `backtesting.py` en la función `main()`:

```python
def main():
    # Configuración personalizada
    symbol = 'ETH/USDT'        # Cambiar par
    initial_balance = 5000     # Balance inicial
    risk_per_trade = 0.01      # 1% de riesgo
    days = 60                  # 60 días de historia
    
    backtest = TradingBacktest(symbol=symbol, initial_balance=initial_balance)
    backtest.run_backtest(risk_per_trade=risk_per_trade, days=days)
    backtest.plot_results()
```

## 📈 Interpretando los Resultados

### Resumen General

```
📊 RESUMEN GENERAL
Balance Inicial:        $1000.00
Balance Final:          $1250.00
P/L Total:              $250.00
Retorno Total:          25.00%
Max Drawdown:           -8.50%
```

**Métricas clave:**
- **P/L Total**: Ganancia o pérdida total en el período
- **Retorno Total**: % de ganancia sobre el capital inicial
- **Max Drawdown**: Mayor caída desde un pico (importante para gestión de riesgo)

### Estadísticas de Trades

```
📈 ESTADÍSTICAS DE TRADES
Total Trades:           45
Trades Ganadores:       28 (62.22%)
Trades Perdedores:      17 (37.78%)
Profit Factor:          1.85
```

**Interpretación:**
- **Win Rate > 50%**: Buena señal (pero no es todo)
- **Profit Factor > 1**: Rentable (>1.5 es excelente)
- **Profit Factor < 1**: Perdiendo dinero, revisar estrategia

### Promedios

```
💰 PROMEDIOS
Ganancia Promedio:      $25.50
Pérdida Promedio:       $-15.20
Mejor Trade:            $85.00
Peor Trade:             $-48.50
```

**Lo ideal:**
- Ganancia promedio > Pérdida promedio (en valor absoluto)
- Esto permite tener win rate < 50% y aún ser rentable

### Breakdown por Dirección

```
📋 BREAKDOWN POR DIRECCIÓN
LONG  - Trades:  28 | Win Rate: 64.29% | P/L: $  180.50
SHORT - Trades:  17 | Win Rate: 58.82% | P/L: $   69.50
```

**Análisis:**
- ¿Una dirección es más rentable?
- ¿Deberías enfocarte solo en LONG o SHORT?

### Razones de Salida

```
🎯 RAZONES DE SALIDA
Take Profit     -  28 trades (62.22%)
Stop Loss       -  15 trades (33.33%)
End of Data     -   2 trades (4.44%)
```

**Lo ideal:**
- Más salidas por Take Profit que por Stop Loss
- Si muchos Stop Loss, revisar precio de entrada o SL

## 📊 Gráficos Generados

El backtesting genera automáticamente:

### 1. Curva de Equity
Muestra cómo evoluciona tu balance en el tiempo.

**Señales positivas:**
- ✅ Tendencia ascendente constante
- ✅ Drawdowns pequeños y rápida recuperación

**Señales negativas:**
- ❌ Tendencia descendente
- ❌ Grandes drawdowns con recuperación lenta

### 2. Distribución de P/L
Muestra ganancia/pérdida de cada trade individual.

**Lo que buscas:**
- ✅ Más barras verdes (ganancias) que rojas (pérdidas)
- ✅ Barras verdes más grandes que las rojas

## 📁 Archivos Generados

Después de ejecutar el backtesting, se crean:

1. **`backtest_results_YYYYMMDD_HHMMSS.csv`**
   - Detalle completo de cada trade
   - Puedes abrirlo en Excel para análisis más profundo

2. **`backtest_chart_YYYYMMDD_HHMMSS.png`**
   - Gráficos de equity y P/L
   - Para reportes o análisis visual

## 🔧 Optimización de Estrategia

### 1. Ajustar Stop Loss

En `backtesting.py`, línea ~32:

```python
self.stop_loss_pips = 485  # Prueba con 300, 400, 600, etc.
```

Ejecuta múltiples backtests con diferentes valores y compara resultados.

### 2. Ajustar Condiciones de Entrada

En el método `analyze_market_direction()`:

```python
# Cambiar de >= 5 a >= 4 o >= 6
if sum(long_conditions) >= 5:  # Más estricto = 6, Más flexible = 4
    return 'LONG'
```

### 3. Ajustar Risk/Reward

En el método `calculate_take_profit()`:

```python
# Para trades no-primeros, cambiar multiplicador
tp = entry_price + (stop_distance * 2)  # Prueba con 1.5, 2.5, 3
```

### 4. Cambiar Timeframe

```python
self.timeframe = '15m'  # Prueba 15m, 30m, 1h, etc.
```

**Nota**: Para otros timeframes, ajusta el límite de velas en `fetch_historical_data()`.

## 📝 Mejores Prácticas

### 1. Test con Suficiente Data
- Mínimo: 30 días
- Recomendado: 60-90 días
- Óptimo: 6-12 meses (si disponible)

### 2. Test en Diferentes Condiciones
- Mercado alcista (bull market)
- Mercado bajista (bear market)
- Mercado lateral (sideways)

### 3. Out-of-Sample Testing
1. Divide tus datos: 70% para optimizar, 30% para validar
2. Optimiza en el 70%
3. Valida que funcione en el 30% restante

### 4. Walk-Forward Testing
Test de forma continua en períodos sucesivos para evitar overfitting.

## ⚠️ Advertencias Importantes

### El Pasado No Garantiza el Futuro
- Un backtest positivo NO garantiza ganancias futuras
- Las condiciones del mercado cambian constantemente

### Evita el Overfitting
- No optimices demasiado para datos históricos específicos
- Busca robustez, no perfección en el pasado

### Considera Costos Reales
El backtest actual NO incluye:
- Comisiones de exchange
- Slippage (diferencia entre precio esperado y ejecutado)
- Spreads bid/ask

**Para incluir comisiones**, modifica en `simulate_trade()`:

```python
# Agregar después de calcular pnl
commission = (entry_price * position_size) * 0.001  # 0.1% comisión
pnl -= (commission * 2)  # Entrada + Salida
```

### Liquidez y Tamaño de Posición
- El backtest asume que siempre puedes entrar/salir a los precios deseados
- En la práctica, órdenes grandes pueden mover el mercado

## 🎓 Métricas Avanzadas

### Sharpe Ratio
Mide retorno ajustado por riesgo (no implementado, pero puedes agregarlo):

```python
sharpe = (mean_return - risk_free_rate) / std_deviation_returns
```

**Interpretación:**
- \> 1: Bueno
- \> 2: Muy bueno  
- \> 3: Excelente

### Sortino Ratio
Similar a Sharpe pero solo considera volatilidad negativa.

### Maximum Consecutive Losses
Cuántas pérdidas seguidas puedes tener (psicológicamente importante).

## 🔄 Workflow Recomendado

1. **Backtest Inicial**
   ```bash
   python backtesting.py
   ```

2. **Analizar Resultados**
   - Revisar métricas clave
   - Identificar debilidades

3. **Optimizar**
   - Ajustar un parámetro a la vez
   - Documentar cambios

4. **Re-test**
   - Ejecutar nuevo backtest
   - Comparar con resultados anteriores

5. **Validar**
   - Test en diferentes períodos
   - Test con diferentes pares

6. **Paper Trading**
   - Antes de dinero real, opera en simulado

7. **Live con Capital Pequeño**
   - Empieza con cantidades mínimas
   - Valida en condiciones reales

## 📞 Troubleshooting

### "No se ejecutaron trades"
- El período puede no tener señales válidas
- Condiciones muy estrictas
- Prueba con más días o ajusta condiciones

### "Todos los trades son pérdidas"
- Revisa la lógica de entrada/salida
- Verifica que los indicadores estén correctos
- Puede ser un mal período para la estrategia

### Error descargando datos
- Verifica conexión a internet
- MEXC puede tener límites de rate
- Reduce el número de días

## 💡 Próximos Pasos

Después de un backtest exitoso:

1. **Documenta tus hallazgos** en `estrategia.md`
2. **Prueba en paper trading** (si disponible)
3. **Empieza con capital real mínimo** usando `trading_bot.py`
4. **Monitorea y compara** resultados reales vs backtest

---

**Recuerda: El backtesting es una herramienta, no una garantía. Opera con responsabilidad. 📊**

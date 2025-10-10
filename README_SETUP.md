# 🤖 Bot de Trading - Guía de Configuración Completa

## 📋 Resumen

Este bot de trading automatizado opera en **Binance Futures** con apalancamiento 15x, implementando una estrategia híbrida de Mean Reversion. Incluye:

- ✅ **Notificaciones de Telegram** en tiempo real
- ✅ **Reportes en Excel** automáticos
- ✅ **Horarios de trading** configurados (LONG: Lun-Vie 8:00-20:00)
- ✅ **Gestión de riesgo** con SL, TP y liquidación
- ✅ **Logs detallados** de todas las operaciones

---

## 🚀 Instalación Rápida

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar credenciales

**Opción A: Copiar el ejemplo**
```bash
copy config.example.json config.json
```

**Opción B: Crear manualmente `config.json`**
```json
{
  "api_key": "TU_API_KEY_BINANCE",
  "api_secret": "TU_API_SECRET_BINANCE",
  "telegram_token": "7738692166:AAHPBKWUVM8gcYldRojnFCWnOKwidfXL2cY",
  "telegram_chat_id": "6670799737",
  "symbol": "HYPE/USDT",
  "initial_balance": 1000,
  "leverage": 15,
  "risk_per_trade": 0.10,
  "refresh_seconds": 60
}
```

### 3. Iniciar conversación con el bot
1. Abre Telegram
2. Busca: **@TradingIAFUTURES_bot**
3. Presiona **Start** o envía `/start`

### 4. Probar notificaciones (opcional)
```bash
python test_telegram.py
```

### 5. Ejecutar el bot
```bash
python trading_bot.py
```

---

## 📱 Telegram - Notificaciones

### Bot configurado
- **Nombre**: Trading IA
- **Usuario**: @TradingIAFUTURES_bot
- **Token**: `7738692166:AAHPBKWUVM8gcYldRojnFCWnOKwidfXL2cY`
- **Tu Chat ID**: `6670799737`

### Tipos de notificaciones

#### ✅ Apertura de posición
```
✅ POSICIÓN ABIERTA

📊 Par: HYPE/USDT
📈 Dirección: LONG
💰 Precio entrada: $44.5550
📏 Tamaño: 168.5393
💵 Margen usado: $100.00
🎯 Take Profit: $45.8916 (+$45.00)
🛑 Stop Loss: $44.0204
⚠️ Liquidación: $41.5883
⏰ Hora: 2025-10-10 15:30:25
```

#### 🟢/🔴 Cierre de posición
```
🟢 POSICIÓN CERRADA

📊 Par: HYPE/USDT
📈 Dirección: LONG
💰 Entrada: $44.5550
💰 Salida: $45.8916
💵 P/L: $45.00 (45.00%)
📝 Razón: Take Profit
💼 Balance: $1045.00
⏰ Duración: 35 velas
```

#### 📊 Resumen diario
```
📊 RESUMEN DEL DÍA

📈 Total trades: 12
✅ Ganadores: 7
❌ Perdedores: 5
🎯 Win Rate: 58.3%
💵 P/L Total: $125.50
💼 Balance: $1125.50
📈 Retorno: 12.55%
```

---

## 📊 Excel - Reportes Automáticos

### Archivo generado
Cada vez que inicias el bot se crea:
```
trading_live_20251010_155524.xlsx
```

### Hoja 1: Trades
Cada operación registra:
- Fecha y Hora de Entrada
- Fecha y Hora de Salida
- Dirección (LONG/SHORT)
- Precio Entrada
- Precio Salida
- Stop Loss
- Take Profit
- Precio de Liquidación
- Tamaño de Posición
- Margen Usado
- **P/L USD** (verde si ganancia ✅, rojo si pérdida ❌)
- P/L %
- Razón de Salida
- Duración en velas
- Balance actualizado

### Hoja 2: Resumen
Métricas en tiempo real:
- Balance Inicial
- Balance Actual
- P/L Total
- Retorno %
- Total Trades
- Trades Ganadores
- Trades Perdedores
- Win Rate %
- Última Actualización

### Características
- ✅ **Actualización automática** al cerrar cada posición
- ✅ **Formato profesional** con colores y estilos
- ✅ **Fechas en hora Argentina** (UTC-3)
- ✅ Puedes abrirlo mientras el bot corre (modo solo lectura)

---

## 🎯 Estrategia de Trading

### Señales LONG (Mean Reversion en sobreventa)
- **Horario**: Lunes a Viernes, 8:00 - 20:00 (Argentina)
- **Condiciones**: 
  - Precio cerca de Banda de Bollinger inferior
  - RSI < 35 (sobreventa)
  - Confirmación de volumen
  - Tendencia mayor alcista (EMA 50 > EMA 200)
- **Stop Loss**: 1.2%
- **Take Profit**: 3.0%
- **Límite de tiempo**: 60 velas (5 horas)

### Señales SHORT (Mean Reversion en sobrecompra)
- **Horario**: 24/7
- **Condiciones**:
  - Precio cerca de Banda de Bollinger superior
  - RSI > 65 (sobrecompra)
  - Contexto alcista (evita tendencias bajistas)
  - Patrón de vela bajista
- **Stop Loss**: 1.5%
- **Take Profit**: 2.2%
- **Límite de tiempo**: 45 velas (3.75 horas)

### Gestión de riesgo
- **Apalancamiento**: 15x
- **Riesgo por trade**: 10% del balance (configurable)
- **Liquidación**: Calculada automáticamente
- **Máximo 1 posición** abierta a la vez

---

## 📝 Archivos generados

### Durante la ejecución
1. **`trading_live_YYYYMMDD_HHMMSS.xlsx`** - Reporte Excel
2. **`trading_bot.log`** - Log detallado de todas las operaciones
3. **Notificaciones Telegram** - En tiempo real

### Archivos de configuración
- **`config.json`** - Tu configuración (NO subir a GitHub)
- **`config.example.json`** - Plantilla de ejemplo

---

## 🔒 Seguridad

### ⚠️ IMPORTANTE
- **NUNCA** compartas tu `config.json`
- **NUNCA** subas tus API keys a GitHub
- El archivo `config.json` está en `.gitignore`
- Usa `config.example.json` como plantilla

### Permisos de API de Binance
El bot necesita:
- ✅ **Lectura** de datos de mercado
- ✅ **Trading** (para órdenes simuladas)
- ❌ **NO necesita** permisos de retiro

---

## 🧪 Testing

### Probar Telegram
```bash
python test_telegram.py
```
Deberías recibir un mensaje en Telegram confirmando que funciona.

### Ejecutar backtest
```bash
python backtesting_hybrid.py
```
Genera reportes históricos para validar la estrategia.

---

## 📞 Soporte

### Logs
Revisa `trading_bot.log` para ver:
- Señales detectadas
- Posiciones abiertas/cerradas
- Errores de conexión
- Actualizaciones de mercado

### Problemas comunes

**1. Error de Telegram**
- Verifica que iniciaste conversación con @TradingIAFUTURES_bot
- Confirma que el Chat ID es correcto: `6670799737`

**2. Error de Binance API**
- Verifica tus credenciales en `config.json`
- Confirma que las API keys tienen permisos de trading
- Revisa que no estés en una región restringida

**3. Excel no se actualiza**
- Cierra el archivo Excel si lo tienes abierto
- Verifica que tienes permisos de escritura en la carpeta

---

## 📈 Próximos pasos

1. ✅ Instalar dependencias
2. ✅ Configurar `config.json` con tus credenciales
3. ✅ Iniciar conversación con el bot de Telegram
4. ✅ Probar notificaciones con `test_telegram.py`
5. ✅ Ejecutar el bot con `python trading_bot.py`
6. ✅ Monitorear en Telegram y Excel

---

## 🎉 ¡Listo!

Tu bot está configurado y listo para operar. Recibirás notificaciones en Telegram y todos los trades se guardarán automáticamente en Excel.

**¡Buena suerte con el trading! 🚀**

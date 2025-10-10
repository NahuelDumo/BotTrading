# 📱 Configuración de Telegram y Excel para el Bot de Trading

## 🤖 Bot de Telegram ya creado
- **Nombre**: Trading IA
- **Usuario**: @TradingIAFUTURES_bot
- **Token**: `7738692166:AAHPBKWUVM8gcYldRojnFCWnOKwidfXL2cY`

## 📋 Cómo obtener tu Chat ID

### Opción 1: Usando @userinfobot (Más fácil)
1. Abre Telegram
2. Busca el bot **@userinfobot**
3. Inicia una conversación con `/start`
4. El bot te enviará tu **Chat ID** (un número como `123456789`)
5. Copia ese número

### Opción 2: Usando @RawDataBot
1. Abre Telegram
2. Busca el bot **@RawDataBot**
3. Envía cualquier mensaje
4. El bot responderá con información en JSON
5. Busca el campo `"id"` dentro de `"from"` o `"chat"`
6. Ese es tu Chat ID

### Opción 3: Método manual
1. Abre tu bot @TradingIAFUTURES_bot
2. Envía el comando `/start`
3. Abre en tu navegador: `https://api.telegram.org/bot7738692166:AAHPBKWUVM8gcYldRojnFCWnOKwidfXL2cY/getUpdates`
4. Busca `"chat":{"id":XXXXXXX}` en la respuesta
5. Ese número es tu Chat ID

## ⚙️ Configuración del bot

1. **Copia el archivo de ejemplo**:
   ```bash
   cp config.example.json config.json
   ```

2. **Edita `config.json`** con tus datos:
   ```json
   {
     "api_key": "TU_API_KEY_BINANCE",
     "api_secret": "TU_API_SECRET_BINANCE",
     "telegram_token": "7738692166:AAHPBKWUVM8gcYldRojnFCWnOKwidfXL2cY",
     "telegram_chat_id": "TU_CHAT_ID_AQUI",
     "symbol": "HYPE/USDT",
     "initial_balance": 1000,
     "leverage": 15,
     "risk_per_trade": 0.10,
     "refresh_seconds": 60
   }
   ```

3. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecuta el bot**:
   ```bash
   python trading_bot.py
   ```

## 📨 Tipos de notificaciones

El bot enviará mensajes de Telegram para:

### ✅ Apertura de posición
- Par
- Dirección (LONG/SHORT)
- Precio de entrada
- Tamaño de posición
- Margen usado
- Take Profit objetivo
- Stop Loss
- Precio de liquidación
- Hora de entrada

### 🔴/🟢 Cierre de posición
- Par
- Dirección
- Precio de entrada y salida
- P/L en USD y porcentaje
- Razón de salida (TP, SL, Liquidación, Time Limit)
- Balance actualizado
- Duración en velas

### 📊 Resumen diario (automático)
- Total de trades del día
- Trades ganadores y perdedores
- Win Rate
- P/L total del día
- Balance actual
- Retorno acumulado

## 🔒 Seguridad

- **NUNCA** compartas tu `config.json` (contiene tus API keys)
- El archivo `config.json` está en `.gitignore` para evitar subirlo a GitHub
- Usa `config.example.json` como plantilla
- El token de Telegram ya está incluido en el ejemplo

## 🧪 Probar notificaciones

Para probar que las notificaciones funcionan, puedes usar este script:

```python
import asyncio
from telegram import Bot

async def test_telegram():
    bot = Bot(token="7738692166:AAHPBKWUVM8gcYldRojnFCWnOKwidfXL2cY")
    chat_id = "TU_CHAT_ID"  # Reemplaza con tu chat ID
    
    await bot.send_message(
        chat_id=chat_id,
        text="✅ <b>Prueba exitosa!</b>\n\nLas notificaciones de Telegram están funcionando correctamente.",
        parse_mode='HTML'
    )
    print("Mensaje enviado!")

asyncio.run(test_telegram())
```

Guarda esto como `test_telegram.py` y ejecútalo para verificar que todo funciona.

## 📊 Reportes en Excel

El bot genera automáticamente un archivo Excel cada vez que se inicia con el formato:
```
trading_live_YYYYMMDD_HHMMSS.xlsx
```

### Contenido del Excel

#### Hoja 1: Trades
Cada trade se registra con:
- **Fecha y Hora de Entrada**
- **Fecha y Hora de Salida**
- **Dirección** (LONG/SHORT)
- **Precio de Entrada**
- **Precio de Salida**
- **Stop Loss**
- **Take Profit**
- **Precio de Liquidación**
- **Tamaño de Posición**
- **Margen Usado**
- **P/L en USD** (verde si ganancia, rojo si pérdida)
- **P/L en %**
- **Razón de Salida** (TP, SL, Liquidación, Time Limit)
- **Duración en velas**
- **Balance actualizado**

#### Hoja 2: Resumen
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

### Actualización automática
- El Excel se actualiza **automáticamente** cada vez que se cierra una posición
- No necesitas hacer nada manualmente
- Puedes abrir el archivo mientras el bot está corriendo (en modo solo lectura)
- Las fechas y horas están en **hora de Argentina (UTC-3)**

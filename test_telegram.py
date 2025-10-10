"""Script para probar las notificaciones de Telegram."""

import asyncio
from telegram import Bot
from telegram.error import TelegramError

TELEGRAM_TOKEN = "7738692166:AAHPBKWUVM8gcYldRojnFCWnOKwidfXL2cY"
TELEGRAM_CHAT_ID = "6670799737"

async def test_telegram():
    """Probar conexión con Telegram."""
    
    chat_id = TELEGRAM_CHAT_ID
    
    print(f"\n🔄 Enviando mensaje de prueba al chat ID: {chat_id}...")
    
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        
        # Enviar mensaje de prueba
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "✅ <b>¡Prueba exitosa!</b>\n\n"
                "Las notificaciones de Telegram están funcionando correctamente.\n\n"
                "📱 Bot: @TradingIAFUTURES_bot\n"
                "🤖 Ahora puedes usar este Chat ID en tu config.json"
            ),
            parse_mode='HTML'
        )
        
        print("✅ ¡Mensaje enviado exitosamente!")
        print(f"\n📋 Usa este Chat ID en tu config.json:")
        print(f'   "telegram_chat_id": "{chat_id}"')
        
    except TelegramError as e:
        print(f"❌ Error de Telegram: {e}")
        print("\n💡 Posibles causas:")
        print("   1. El Chat ID es incorrecto")
        print("   2. No has iniciado conversación con @TradingIAFUTURES_bot")
        print("   3. El bot no tiene permisos para enviarte mensajes")
        print("\n🔧 Solución:")
        print("   1. Abre Telegram")
        print("   2. Busca @TradingIAFUTURES_bot")
        print("   3. Presiona 'Start' o envía /start")
        print("   4. Vuelve a ejecutar este script")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST DE NOTIFICACIONES DE TELEGRAM")
    print("=" * 60)
    print(f"\n📱 Bot: @TradingIAFUTURES_bot")
    print(f"💬 Chat ID: {TELEGRAM_CHAT_ID}")
    print("\n" + "=" * 60 + "\n")
    
    asyncio.run(test_telegram())

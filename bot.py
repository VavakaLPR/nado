import logging
import os
import re
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BINANCE_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"


def normalize_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.strip().upper()
    symbol = re.sub(r"[^A-Z0-9]", "", symbol)
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    return symbol


async def fetch_funding_data(symbol: str) -> Optional[dict]:
    params = {"symbol": symbol}
    timeout = aiohttp.ClientTimeout(total=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(BINANCE_PREMIUM_INDEX_URL, params=params) as response:
            if response.status != 200:
                logger.warning("Binance API error status=%s symbol=%s", response.status, symbol)
                return None
            data = await response.json()
            if isinstance(data, dict) and data.get("code"):
                logger.warning("Binance API business error symbol=%s payload=%s", symbol, data)
                return None
            return data


def format_funding_message(data: dict) -> str:
    symbol = data.get("symbol", "N/A")
    mark_price = data.get("markPrice", "N/A")
    index_price = data.get("indexPrice", "N/A")
    funding_rate_raw = data.get("lastFundingRate", "0")
    next_funding_time_ms = data.get("nextFundingTime", 0)

    try:
        funding_rate_pct = float(funding_rate_raw) * 100
        funding_rate_text = f"{funding_rate_pct:.4f}%"
    except (TypeError, ValueError):
        funding_rate_text = "N/A"

    if isinstance(next_funding_time_ms, (int, float)) and next_funding_time_ms > 0:
        from datetime import datetime, timezone

        next_funding_dt = datetime.fromtimestamp(next_funding_time_ms / 1000, tz=timezone.utc)
        next_funding_text = next_funding_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        next_funding_text = "N/A"

    return (
        f"<b>{symbol}</b>\n"
        f"Funding rate: <b>{funding_rate_text}</b>\n"
        f"Mark price: <code>{mark_price}</code>\n"
        f"Index price: <code>{index_price}</code>\n"
        f"Next funding: <b>{next_funding_text}</b>"
    )


async def send_funding(update: Update, symbol_input: str) -> None:
    symbol = normalize_symbol(symbol_input)
    if len(symbol) < 6 or len(symbol) > 20:
        await update.message.reply_text("Некорректный тикер. Пример: BTC или BTCUSDT")
        return

    data = await fetch_funding_data(symbol)
    if not data:
        await update.message.reply_text(
            "Не удалось получить данные. Проверь тикер (пример: BTC, ETH, SOL) и попробуй снова."
        )
        return

    msg = format_funding_message(data)
    await update.message.reply_text(msg, parse_mode="HTML")


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет. Отправь /fund <тикер> (например /fund BTC)\n"
        "Или просто отправь тикер сообщением: ETH"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды:\n"
        "/fund <тикер> - показать funding rate\n"
        "/help - помощь\n\n"
        "Примеры:\n"
        "/fund BTC\n"
        "/fund ETHUSDT"
    )


async def fund_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Укажи тикер. Пример: /fund BTC")
        return

    await send_funding(update, context.args[0])


async def plain_text_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return
    await send_funding(update, text)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN. Добавь его в .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("fund", fund_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text_ticker))

    logger.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

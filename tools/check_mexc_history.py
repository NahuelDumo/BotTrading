import os
from pathlib import Path
from typing import List

import ccxt
import pandas as pd

ENV_FILENAME = "BotTrading.env"
DEFAULT_SYMBOL = "HYPE/USDT"
DEFAULT_TIMEFRAME = "5m"
DEFAULT_MAX_DAYS = 60
DEFAULT_MARKET_TYPE = "spot"
DEFAULT_LIMIT = 500
MAX_LIMIT = 1000


def load_local_env() -> None:
    """Carga un archivo .env si existe, sin sobrescribir variables ya definidas."""

    env_path = Path(__file__).resolve().parents[1] / ENV_FILENAME
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def timeframe_to_milliseconds(exchange: ccxt.Exchange, timeframe: str) -> int:
    seconds = exchange.parse_timeframe(timeframe)
    if seconds is None:
        raise ValueError(f"No se pudo interpretar el timeframe '{timeframe}'")
    return int(seconds * 1000)


def fetch_ohlcv_with_pagination(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    days_back: int,
    market_type: str,
    limit: int,
) -> List[List[float]]:
    """Solicita velas iterando con 'since' para cubrir el rango solicitado."""

    timeframe_ms = timeframe_to_milliseconds(exchange, timeframe)
    now_ms = exchange.milliseconds()
    target_since = now_ms - days_back * 24 * 60 * 60 * 1000

    limit = max(1, min(limit, MAX_LIMIT))
    end = now_ms
    all_candles: List[List[float]] = []
    seen_timestamps = set()

    while end > target_since:
        window_start = max(target_since, end - timeframe_ms * limit)
        params: dict[str, int] = {"endTime": end}

        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=window_start,
            limit=limit,
            params=params,
        )
        if not batch:
            break

        for candle in batch:
            ts = int(candle[0])
            if ts < target_since or ts in seen_timestamps:
                continue
            seen_timestamps.add(ts)
            all_candles.append(candle)

        earliest = int(batch[0][0])
        if earliest == end:
            break

        end = earliest - timeframe_ms

        if earliest <= target_since:
            break

    all_candles.sort(key=lambda item: item[0])
    return all_candles


def main() -> None:
    load_local_env()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Faltan las variables BINANCE_API_KEY y/o BINANCE_API_SECRET. "
            "Defínelas en el entorno o en un archivo .env (no lo compartas públicamente)."
        )

    symbol = os.getenv("BINANCE_SYMBOL", DEFAULT_SYMBOL)
    timeframe = os.getenv("BINANCE_TIMEFRAME", DEFAULT_TIMEFRAME)
    market_type = os.getenv("BINANCE_MARKET_TYPE", DEFAULT_MARKET_TYPE).lower()
    if market_type not in {"spot", "future"}:
        raise ValueError("BINANCE_MARKET_TYPE debe ser 'spot' o 'future'")

    max_days_env = os.getenv("BINANCE_MAX_DAYS", str(DEFAULT_MAX_DAYS))
    try:
        days_back = max(1, int(max_days_env))
    except ValueError as exc:
        raise ValueError(
            "La variable BINANCE_MAX_DAYS debe ser un entero positivo"
        ) from exc

    limit_env = os.getenv("BINANCE_LIMIT", str(DEFAULT_LIMIT))
    try:
        request_limit = max(1, int(limit_env))
    except ValueError as exc:
        raise ValueError("BINANCE_LIMIT debe ser un entero positivo") from exc

    exchange = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": market_type,
            },
        }
    )

    print(
        f"Solicitando velas de {symbol} ({market_type}) en {timeframe} hasta {days_back} días atrás "
        f"(lotes de {request_limit})...\n"
    )

    try:
        candles = fetch_ohlcv_with_pagination(
            exchange,
            symbol,
            timeframe,
            days_back,
            market_type,
            request_limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Error al solicitar velas: {type(exc).__name__}: {exc}") from exc

    if not candles:
        print("No se recibieron velas para los parámetros indicados.")
        return

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)

    first_dt = df["datetime"].iloc[0]
    last_dt = df["datetime"].iloc[-1]

    print(f"Total de velas recibidas: {len(df)}")
    print(f"Primera vela: {first_dt} (timestamp {df['timestamp'].iloc[0]})")
    print(f"Última vela:  {last_dt} (timestamp {df['timestamp'].iloc[-1]})")
    print(f"Rango cubierto: {last_dt - first_dt}")

    print("\nÚltimas 5 velas:")
    print(
        df.tail(5)[
            ["datetime", "open", "high", "low", "close", "volume"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

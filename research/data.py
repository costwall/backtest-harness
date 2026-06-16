"""
research/data.py
================
Каноничные загрузчики данных для харнесса. Все возвращают pandas DataFrame с
UTC DatetimeIndex (sorted, без дублей) и КАНОНИЧНОЙ схемой колонок, чтобы сигналы
не зависели от того, откуда пришли данные.

Каноничные колонки (присутствует подмножество в зависимости от источника):
    open, high, low, close, volume, volume_delta, atr_14,
    open_interest, lsr, top_lsr, funding_rate

Источники:
  liquidity_bot (внешний проект, ТОЛЬКО ЧТЕНИЕ):
    - load_deriv_5m(symbol) — *_advanced_5m.csv  (5-мин, длинная история OHLC;
                              OI/LSR появляются позже начала OHLC → NaN в начале)
    - load_deriv_1m(symbol) — *_history.parquet  (1-мин, live, + funding_rate)
  наши коллекторы (forex_bot):
    - load_depth(pair)      — data/kraken/kraken_depth.db
    - load_quotes(symbol)   — data/quotes/etoro_quotes.db
  forex OHLC:
    - load_forex_ohlc(symbol) — liquidity_bot/data/Forex/*_full_2025_2026.csv
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

# External derivatives feed (OHLC + OI + LSR + funding). Bring your own data:
# point EXTERNAL_DATA_DIR at a folder with the documented file layout. Default: ./data/external.
EXTERNAL_DATA = Path(os.environ.get("EXTERNAL_DATA_DIR", "data/external"))
LIQUIDITY_BOT_DATA = EXTERNAL_DATA  # backwards-compatible alias used below
# Local collector DBs (relative to repo root). Optional — only for the depth/quotes loaders.
_ROOT = Path(__file__).resolve().parent.parent
KRAKEN_DEPTH_DB = _ROOT / "data" / "kraken" / "kraken_depth.db"
ETORO_QUOTES_DB = _ROOT / "data" / "quotes" / "etoro_quotes.db"

# Маппинг исходных имён → каноничные.
_RENAME_5M = {
    "global_long_short_ratio": "lsr",
    "top_long_short_ratio": "top_lsr",
}
_RENAME_1M = {
    "long_short_ratio": "lsr",
    "top_trader_account_ratio": "top_lsr",
}
_CANONICAL = [
    "open", "high", "low", "close", "volume", "volume_delta",
    "atr_14", "open_interest", "lsr", "top_lsr", "funding_rate",
]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Привести к каноничному виду: только известные колонки, sorted UTC index, dedup."""
    keep = [c for c in _CANONICAL if c in df.columns]
    df = df[keep].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_deriv_5m(symbol: str) -> pd.DataFrame:
    """5-мин advanced-датасет liquidity_bot (OHLC + OI + LSR + volume_delta + atr).

    БЕЗ funding_rate (его нет в 5m-файлах — только в 1m parquet).
    OI/LSR пустые в ранней части истории (фичи начали собирать позже OHLC).
    """
    path = LIQUIDITY_BOT_DATA / f"{symbol}_advanced_5m.csv"
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.index.name = "ts"
    df = df.rename(columns=_RENAME_5M)
    return _finalize(df)


def load_deriv_1m(symbol: str) -> pd.DataFrame:
    """1-мин live-история liquidity_bot (close + funding + OI + LSR + volume_delta).

    Короткая (~дни), зато с funding_rate. Cadence ~60с.
    """
    path = LIQUIDITY_BOT_DATA / f"{symbol}_history.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df["ts"], utc=True)
    df.index.name = "ts"
    df = df.rename(columns=_RENAME_1M)
    return _finalize(df)


def load_funding(symbol: str) -> pd.DataFrame:
    """Мультирежимный funding+цена с Binance (8h-сетка, ~2019→now). Кэш parquet.

    Источник для решающего теста funding через разные режимы (bull+bear), а не
    11 дней одного режима из liquidity_bot. См. research/binance_backfill.py.
    """
    from research.binance_backfill import build_funding_df
    return build_funding_df(symbol)


def load_cot(name: str) -> pd.DataFrame:
    """COT-позиционирование + цена (форекс/золото), недельная сетка. Кэш parquet.
    name из research.cot_backfill.MARKETS (GOLD/EURUSD/GBPUSD/USDJPY). См. модуль.
    """
    from research.cot_backfill import build_cot_df
    return build_cot_df(name)


def load_forex_ohlc(symbol: str) -> pd.DataFrame:
    """Форекс OHLC 2025-2026 из liquidity_bot/data/Forex/ (только цена)."""
    path = LIQUIDITY_BOT_DATA / "Forex" / f"{symbol}_full_2025_2026.csv"
    df = pd.read_csv(path)
    # Найти временную колонку гибко (timestamp ms / date / time).
    tcol = next((c for c in df.columns if c.lower() in ("timestamp", "date", "time", "datetime")), df.columns[0])
    if pd.api.types.is_numeric_dtype(df[tcol]):
        df.index = pd.to_datetime(df[tcol], unit="ms", utc=True)
    else:
        df.index = pd.to_datetime(df[tcol], utc=True)
    df.index.name = "ts"
    df.columns = [c.lower() for c in df.columns]
    return _finalize(df)


def load_depth(pair: str) -> pd.DataFrame:
    """Стакан Kraken из нашего коллектора (mid/microprice/imbalance/spread)."""
    con = sqlite3.connect(KRAKEN_DEPTH_DB)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM depth WHERE pair = ? ORDER BY ts", con, params=(pair,)
        )
    finally:
        con.close()
    df.index = pd.to_datetime(df["ts"], utc=True)
    df.index.name = "ts"
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_quotes(symbol: str) -> pd.DataFrame:
    """eToro котировки из нашего коллектора (bid/ask/mid/spread)."""
    con = sqlite3.connect(ETORO_QUOTES_DB)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM quotes WHERE symbol = ? ORDER BY ts", con, params=(symbol,)
        )
    finally:
        con.close()
    df.index = pd.to_datetime(df["ts"], utc=True)
    df.index.name = "ts"
    return df[~df.index.duplicated(keep="last")].sort_index()

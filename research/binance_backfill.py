"""
research/binance_backfill.py
============================
Бэкфилл публичной истории Binance USDⓈ-M futures для мультирежимного теста:
  - funding rate (/fapi/v1/fundingRate) — с инцепшена контракта (BTCUSDT ~2019-09)
  - 8h klines (/fapi/v1/klines) — цена на тех же 8-часовых границах, что funding

Публичные эндпоинты, БЕЗ auth (CLAUDE.md: Binance read-only). Корп-сеть капризна →
session + browser-UA + retry + пэйсинг. Всё кэшируется в data/research_cache/*.parquet,
чтобы повторные прогоны были бесплатны (feedback-reusable-research-base).

build_funding_df(symbol) → DataFrame[UTC index 8h, columns: close, funding_rate].
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

FAPI = "https://fapi.binance.com"
_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / "data" / "research_cache"
INCEPTION_MS = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (research-backfill)"})
    return s


def _get(s: requests.Session, path: str, params: dict, retries: int = 5):
    """GET с экспоненциальным retry (корп-сеть рвёт соединения)."""
    last = None
    for attempt in range(retries):
        try:
            r = s.get(FAPI + path, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:  # ConnectionReset и пр.
            last = e
        time.sleep(1.0 * (attempt + 1))
    raise last


def fetch_funding(symbol: str) -> pd.DataFrame:
    """Все funding-точки symbol от инцепшена. Пагинация по startTime, limit=1000."""
    s = _session()
    rows, start = [], INCEPTION_MS
    while True:
        batch = _get(s, "/fapi/v1/fundingRate",
                     {"symbol": symbol, "startTime": start, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        last_t = batch[-1]["fundingTime"]
        if len(batch) < 1000:
            break
        start = last_t + 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df.set_index("fundingTime")[["funding_rate"]].sort_index()


def fetch_klines_8h(symbol: str) -> pd.DataFrame:
    """8h-klines symbol от инцепшена. Пагинация по startTime, limit=1500."""
    s = _session()
    rows, start = [], INCEPTION_MS
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while start < now_ms:
        batch = _get(s, "/fapi/v1/klines",
                     {"symbol": symbol, "interval": "8h",
                      "startTime": start, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < 1500:
            break
        start = last_open + 1
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=[
        "openTime", "open", "high", "low", "close", "volume",
        "closeTime", "qav", "trades", "tbav", "tqav", "ignore"])
    df["ts"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df.set_index("ts")[["close"]].sort_index()


def build_funding_df(symbol: str, force: bool = False) -> pd.DataFrame:
    """funding + 8h-цена на одной 8-часовой сетке. Кэш в data/research_cache/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"funding_aligned_{symbol}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    f_cache = CACHE / f"funding_{symbol}.parquet"
    k_cache = CACHE / f"klines8h_{symbol}.parquet"
    funding = pd.read_parquet(f_cache) if (f_cache.exists() and not force) else fetch_funding(symbol)
    if not f_cache.exists() or force:
        funding.to_parquet(f_cache)
    klines = pd.read_parquet(k_cache) if (k_cache.exists() and not force) else fetch_klines_8h(symbol)
    if not k_cache.exists() or force:
        klines.to_parquet(k_cache)

    # funding-времена (00/08/16:00 UTC) совпадают с 8h openTime → join по индексу
    df = klines.join(funding, how="inner").dropna()
    df.to_parquet(out)
    return df

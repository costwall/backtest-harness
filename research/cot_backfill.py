"""
research/cot_backfill.py
========================
Бэкфилл COT-позиционирования (CFTC) + цены (Yahoo) для форекс+золото. Аналог
funding-теста, но для рынков с НИЗКИМИ костами и НЕДЕЛЬНЫМ горизонтом (ретейл-friendly).

Приор: экстремум спекулятивного (non-commercial) позиционирования → разворот цены
(классический COT-контртренд). Альтернатива — momentum позиционирования.

Источники (оба достижимы с корп-сети, FRED/Stooq — нет):
  - CFTC Socrata Legacy Futures-Only (6dca-aqww): недельный net non-commercial,
    золото с 1986, EUR с 2000. report_date = вторник (дата данных).
  - Yahoo v8 chart: дневная цена. GC=F (COMEX-золото, совпадает с COT-рынком),
    EURUSD=X и т.д. Берём close на дату COT (ffill ближайший торговый день).

build_cot_df(name) → DataFrame[weekly index, close, net_noncomm, pct_net, oi].
Кэш data/research_cache/. Знак сигнала ориентирует costwall по train.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CFTC = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / "data" / "research_cache"

# COT публикуется в ПЯТНИЦУ по данным на ВТОРНИК (~3 дня лага). Вход строго по
# цене релиза, иначе look-ahead (торговля на неопубликованных данных).
RELEASE_LAG_DAYS = 3

# friendly name -> (CFTC LIKE-паттерн, Yahoo symbol). LIKE ловит переименования
# (BRITISH POUND / ...STERLING; AUSTRALIAN DOLLAR / DOLLARS), XRATE/INDEX исключаются.
MARKETS = {
    "GOLD":   ("GOLD - COMMODITY EXCHANGE%", "GC=F"),
    "EURUSD": ("EURO FX - CHICAGO MERCANTILE%", "EURUSD=X"),
    "GBPUSD": ("BRITISH POUND%CHICAGO MERCANTILE%", "GBPUSD=X"),
    "USDJPY": ("JAPANESE YEN - CHICAGO MERCANTILE%", "JPY=X"),
    "AUDUSD": ("AUSTRALIAN DOLLAR%CHICAGO MERCANTILE%", "AUDUSD=X"),
    "USDCAD": ("CANADIAN DOLLAR - CHICAGO MERCANTILE%", "CAD=X"),
    "USDCHF": ("SWISS FRANC - CHICAGO MERCANTILE%", "CHF=X"),
    "NZDUSD": ("NEW ZEALAND DOLLAR%CHICAGO MERCANTILE%", "NZDUSD=X"),
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (research-cot)"})
    return s


def _get(s, url, params, retries=5):
    last = None
    for a in range(retries):
        try:
            r = s.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            last = e
        time.sleep(1.0 * (a + 1))
    raise last


def fetch_cot(pattern: str) -> pd.DataFrame:
    """Недельный COT по LIKE-паттерну рынка: net non-commercial, net %OI, OI."""
    s = _session()
    rows = _get(s, CFTC, {
        "$where": (f"market_and_exchange_names like '{pattern}' "
                   "and market_and_exchange_names not like '%XRATE%' "
                   "and market_and_exchange_names not like '%INDEX%'"),
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": 60000,
    })
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize("UTC")
    out = pd.DataFrame(index=idx)
    fl = lambda c: pd.to_numeric(df[c], errors="coerce").values
    out["net_noncomm"] = fl("noncomm_positions_long_all") - fl("noncomm_positions_short_all")
    out["pct_net"] = fl("pct_of_oi_noncomm_long_all") - fl("pct_of_oi_noncomm_short_all")
    out["oi"] = fl("open_interest_all")
    out.index.name = "ts"
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_yahoo(symbol: str) -> pd.Series:
    """Дневной close с Yahoo (вся история)."""
    s = _session()
    j = _get(s, YAHOO + symbol, {"period1": 0, "period2": int(time.time()), "interval": "1d"})
    res = j["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    close = res["indicators"]["quote"][0]["close"]
    px = pd.Series(close, index=ts, name="close").dropna()
    # нормализуем индекс к датам (Yahoo даёт intraday-ts для дневных баров)
    px.index = px.index.normalize()
    return px[~px.index.duplicated(keep="last")].sort_index()


def build_cot_df(name: str, force: bool = False) -> pd.DataFrame:
    """COT + цена на недельной COT-сетке. Кэш в data/research_cache/."""
    if name not in MARKETS:
        raise KeyError(f"{name} не в MARKETS: {list(MARKETS)}")
    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = CACHE / f"cot_aligned_{name}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    pattern, ysym = MARKETS[name]
    cot = fetch_cot(pattern)
    px = fetch_yahoo(ysym)

    # ФИКС look-ahead: вход по цене ДНЯ РЕЛИЗА (вторник+3=пятница), а не вторника.
    # bfill → первый торговый день >= релиза (никогда раньше публикации).
    release = (cot.index + pd.Timedelta(days=RELEASE_LAG_DAYS)).normalize()
    px_at = px.reindex(px.index.union(release)).bfill().reindex(release)
    df = cot.copy()
    df.index = release
    df.index.name = "ts"
    df["close"] = px_at.values
    df = df.dropna(subset=["close", "pct_net"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(out_path)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["GOLD", "EURUSD"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for n in args.names:
        df = build_cot_df(n, force=args.force)
        print(f"{n}: {len(df)} weekly rows, {df.index.min().date()} -> {df.index.max().date()}, "
              f"pct_net last={df['pct_net'].iloc[-1]:.1f}")

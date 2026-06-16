"""
research/btc_regimes.py
=======================
Задание: часовой BTC за макс. историю → дневная средняя → крупные ралли/обвалы
с таймстампами стартов → привязка событий. Цель — оценить, есть ли рабочий алгоритм.

Данные: Binance spot 1h (BTCUSDT, с листинга 2017-08), кэш parquet.
Регимы: ZigZag по дневной средней — пивоты подтверждаются разворотом > threshold%.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SPOT = "https://api.binance.com/api/v3/klines"
CACHE = Path(__file__).resolve().parent.parent / "data" / "research_cache"


def fetch_1h(symbol="BTCUSDT", force=False) -> pd.DataFrame:
    """Часовой spot OHLC с листинга. Кэш."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"spot1h_{symbol}.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0"})
    rows, start = [], 0
    now = int(time.time() * 1000)
    while start < now:
        ok = False
        for a in range(5):
            try:
                r = s.get(SPOT, params={"symbol": symbol, "interval": "1h",
                                        "startTime": start, "limit": 1000}, timeout=30)
                if r.status_code == 200:
                    batch = r.json(); ok = True; break
            except Exception:
                pass
            time.sleep(1.0 * (a + 1))
        if not ok or not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    df = pd.DataFrame(rows, columns=["openTime", "open", "high", "low", "close", "volume",
                                     "closeTime", "qav", "trades", "tb", "tq", "ig"])
    df.index = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df[["open", "high", "low", "close", "volume"]]
    df.to_parquet(p)
    return df


def zigzag(daily: pd.Series, threshold=0.20) -> list[dict]:
    """ZigZag-пивоты (mode-based): в 'up' трекаем максимум, при падении > threshold от
    него фиксируем максимум-пивот и идём в 'down'; зеркально. Возвращает леги."""
    t = daily.index; p = daily.values; n = len(p)
    piv = [(t[0], p[0])]
    mode = "up"; ext_i = 0
    for i in range(1, n):
        if mode == "up":
            if p[i] > p[ext_i]:
                ext_i = i
            elif p[i] <= p[ext_i] * (1 - threshold):
                piv.append((t[ext_i], p[ext_i])); mode = "down"; ext_i = i
        else:
            if p[i] < p[ext_i]:
                ext_i = i
            elif p[i] >= p[ext_i] * (1 + threshold):
                piv.append((t[ext_i], p[ext_i])); mode = "up"; ext_i = i
    piv.append((t[ext_i], p[ext_i]))
    legs = []
    for (t0, p0), (t1, p1) in zip(piv, piv[1:]):
        if p1 == p0 or t1 == t0:
            continue
        legs.append({"kind": "rally" if p1 > p0 else "crash",
                     "start": t0, "end": t1, "start_px": round(p0), "end_px": round(p1),
                     "pct": round((p1 / p0 - 1) * 100, 1),
                     "days": (t1 - t0).days})
    return legs


def _equity_metrics(daily_ret: np.ndarray) -> tuple[float, float, float]:
    """CAGR%, Sharpe, MaxDD% из ряда дневных доходностей."""
    eq = np.cumprod(1 + daily_ret)
    yrs = len(daily_ret) / 365.0
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else float("nan")
    vol = daily_ret.std() * np.sqrt(365)
    sharpe = (daily_ret.mean() * 365) / vol if vol > 0 else float("nan")
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return round(cagr, 1), round(sharpe, 2), round(dd, 0)


def causal_trend_follow(daily: pd.Series, threshold=0.20, cost_bps=40.0,
                        long_only=False) -> dict:
    """ПРИЧИННОЕ трендследование: позиция меняется на ПОДТВЕРЖДЕНИИ разворота (цена ушла
    threshold от экстремума — известно в реальном времени). long_only=True → в даунтренде
    кэш (без шорта). Считает equity-кривую и сравнивает с buy&hold по CAGR/Sharpe/MaxDD."""
    p = daily.values; n = len(p)
    ret = np.diff(p) / p[:-1]                      # дневные доходности (n-1)
    mode = "up"; ext = p[0]; pos = 0
    pos_arr = np.zeros(n - 1); flips = 0
    for i in range(1, n):
        flip = None
        if mode == "up":
            if p[i] > ext: ext = p[i]
            elif p[i] <= ext * (1 - threshold): flip = -1; mode = "down"; ext = p[i]
        else:
            if p[i] < ext: ext = p[i]
            elif p[i] >= ext * (1 + threshold): flip = 1; mode = "up"; ext = p[i]
        if flip is not None:
            newpos = max(flip, 0) if long_only else flip
            if newpos != pos: flips += 1
            pos = newpos
        # CAUSAL: позиция, решённая на баре i, применяется к СЛЕДУЮЩЕй доходности (i->i+1)
        if i <= n - 2:
            pos_arr[i] = pos
    strat = pos_arr * ret
    strat[1:] -= (np.abs(np.diff(pos_arr)) * cost_bps / 1e4)   # кост на смену позиции
    cagr, sharpe, dd = _equity_metrics(strat)
    bh_cagr, bh_sharpe, bh_dd = _equity_metrics(ret)
    return {"thr": threshold, "mode": "long-only" if long_only else "L/S", "flips": flips,
            "CAGR%": cagr, "Sharpe": sharpe, "MaxDD%": dd,
            "bh_CAGR%": bh_cagr, "bh_Sharpe": bh_sharpe, "bh_MaxDD%": bh_dd}


if __name__ == "__main__":
    df = fetch_1h("BTCUSDT")
    # ТОРГУЕМАЯ цена = дневной close (.last()), НЕ .mean() — среднюю исполнить нельзя,
    # и она сглаживает ряд, льстя трендоследованию (меньше ложных флипов).
    daily = df["close"].resample("1D").last().dropna()
    print(f"BTC 1h: {len(df)} bars {df.index.min().date()}..{df.index.max().date()} "
          f"| daily {len(daily)} дней")
    legs = zigzag(daily, threshold=0.20)
    print(f"\n=== Крупные леги (ZigZag 20%), {len(legs)} штук ===")
    print(f"{'kind':6} {'start':12} {'end':12} {'pct':>7} {'days':>5} {'start_px':>9} {'end_px':>9}")
    for L in legs:
        print(f"{L['kind']:6} {str(L['start'].date()):12} {str(L['end'].date()):12} "
              f"{L['pct']:>6}% {L['days']:>5} {L['start_px']:>9} {L['end_px']:>9}")
    print("\n=== Причинное трендследование vs buy&hold (BTC cost 40bp) ===")
    for thr in (0.10, 0.15, 0.20, 0.30):
        for lo in (False, True):
            print(" ", causal_trend_follow(daily, threshold=thr, long_only=lo))

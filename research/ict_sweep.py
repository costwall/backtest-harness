"""
research/ict_sweep.py
=====================
Механический бэктест ICT/SMC-сетапа по правилам пользователя (шорт; лонг зеркально):
  1. SWEEP    — high пробивает недавний swing-high (вынос ликвидности сверху).
  2. MSS      — затем close уходит ниже предыдущего swing-low (слом структуры вниз).
  3. FVG      — в импульсе вниз 3-свечной разрыв: low[a] > high[c] (зона [high[c], low[a]]).
  ENTRY       — откат вверх в FVG → шорт у нижней кромки FVG (high[c]).
  STOP        — выше sweep-high. TARGET — 1:rr от риска.

ЧЕСТНАЯ ОГОВОРКА: механизация ICT = десятки степеней свободы (swing-lookback, окно MSS,
мин. размер FVG, глубина отката, размещение стопа). Здесь — разумные дефолты; дискреционная
версия может отличаться. Тест событийный (не AUC): n сделок, win-rate, EV/сделку net of cost,
profit-factor, t. Non-overlap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _swings(high, low, k):
    """Фрактальные swing-high/low: экстремум среди k баров слева и справа."""
    n = len(high)
    sh = np.zeros(n, bool); sl = np.zeros(n, bool)
    for i in range(k, n - k):
        if high[i] == max(high[i - k:i + k + 1]) and high[i] > high[i - 1]:
            sh[i] = True
        if low[i] == min(low[i - k:i + k + 1]) and low[i] < low[i - 1]:
            sl[i] = True
    return sh, sl


def backtest(df: pd.DataFrame, *, k=2, mss_window=15, retrace_window=20, max_hold=40,
             rr=2.0, cost_bps=2.0, min_fvg_bps=1.0) -> dict:
    """Прогон сетапа на OHLC. df: open/high/low/close. Возвращает статистику сделок."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    n = len(df)
    sh, sl = _swings(h, l, k)
    # последний подтверждённый swing-high/low до бара i
    last_sh = np.full(n, np.nan); last_sl = np.full(n, np.nan)
    cur_h = cur_l = np.nan
    for i in range(n):
        if sh[i]: cur_h = h[i]
        if sl[i]: cur_l = l[i]
        last_sh[i] = cur_h; last_sl[i] = cur_l

    trades = []
    i = k + 1
    while i < n - 3:
        # ШОРТ-ветка: sweep выше недавнего swing-high
        if not np.isnan(last_sh[i - 1]) and h[i] > last_sh[i - 1]:
            swept_high = h[i]
            prior_low = last_sl[i - 1]
            if np.isnan(prior_low):
                i += 1; continue
            # MSS: close ниже prior_low в пределах окна
            m = None
            for j in range(i + 1, min(i + 1 + mss_window, n)):
                if c[j] < prior_low:
                    m = j; break
            if m is None:
                i += 1; continue
            # FVG (bearish) между i и m: low[a] > high[a+2]
            fvg = None
            for a in range(i, m - 1):
                if l[a] - h[a + 2] > min_fvg_bps / 1e4 * c[a]:
                    fvg = (h[a + 2], l[a], a + 2)   # (низ, верх, индекс)
            if fvg is None:
                i = m + 1; continue
            fvg_lo, fvg_hi, fvg_i = fvg
            # ENTRY: откат вверх в FVG (high достигает fvg_lo) в пределах окна
            entry = stop = tgt = None; ej = None
            for j in range(max(m, fvg_i) + 1, min(max(m, fvg_i) + 1 + retrace_window, n)):
                if h[j] >= fvg_lo:
                    entry = fvg_lo
                    stop = swept_high
                    risk = stop - entry
                    if risk <= 0:
                        entry = None; break
                    tgt = entry - rr * risk
                    ej = j; break
            if entry is None or ej is None:
                i = m + 1; continue
            # симуляция шорта от входа
            outcome = None
            for j in range(ej, min(ej + max_hold, n)):
                if h[j] >= stop:                       # стоп (консервативно первым)
                    outcome = -(stop - entry) / entry; break
                if l[j] <= tgt:
                    outcome = (entry - tgt) / entry; break
            if outcome is None:
                outcome = (entry - c[min(ej + max_hold - 1, n - 1)]) / entry
            trades.append(outcome * 1e4 - cost_bps)
            i = ej + 1
            continue
        i += 1

    t = np.array(trades, dtype=float)
    nt = len(t)
    if nt == 0:
        return {"n": 0}
    wins = t[t > 0]; losses = t[t < 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    sd = t.std(ddof=1) if nt > 1 else float("nan")
    tstat = t.mean() / (sd / np.sqrt(nt)) if sd and sd > 0 else float("nan")
    return {"n": nt, "win_rate": round((t > 0).mean(), 3), "ev_bps": round(t.mean(), 2),
            "profit_factor": round(pf, 2), "t": round(tstat, 2), "total_bps": round(t.sum(), 0)}


def _run_forex(names, cost, **kw):
    from research.session_forex import fetch_1h
    for nm in names:
        df = fetch_1h(nm)
        r = backtest(df, cost_bps=cost, **kw)
        print(f"  FX1h {nm:8s} {r}")


def _run_crypto(names, cost, **kw):
    from research.data import load_deriv_5m
    for nm in names:
        df = load_deriv_5m(nm)
        r = backtest(df, cost_bps=cost, **kw)
        print(f"  5m {nm:10s} {r}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx", nargs="*", default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    ap.add_argument("--crypto", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--rr", type=float, default=2.0)
    args = ap.parse_args()
    print(f"ICT sweep+MSS+FVG, k={args.k} rr={args.rr}")
    print("=== Forex 1h (cost 2bp) ===")
    _run_forex(args.fx, 2.0, k=args.k, rr=args.rr)
    print("=== Crypto 5m (cost 40bp) ===")
    _run_crypto(args.crypto, 40.0, k=args.k, rr=args.rr)

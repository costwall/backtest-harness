"""
research/session_forex.py
=========================
Тест session/time-of-day эффекта на форексе — единственная НЕпротестированная часть
дискретного плана пользователя (вход на европейском открытии после утреннего анализа).

План (в местном CEST=UTC+2): анализ 7-9 утра → вход после 9 → 1:2 брекет.
В UTC: анализ-окно 05-07, ВХОД 07:00 (европейское открытие), выход к концу дня (~20:00).

Две проверки:
  1. AUC — даёт ли session-сигнал направление лучше монетки (07→20 forward).
  2. Брекет-симуляция — ТОЧНО план юзера: вход 07 по сигналу, стоп −R, тейк +2R (1:2),
     выход в EOD если не сработало. По часовым hi/lo. win-rate + EV/сделку net of cost.
     Конфликт «стоп и тейк в одном баре» → считаем стоп первым (консервативно).

Данные: Yahoo 1h OHLC (~2.8 года), кэш parquet. Сигналы causal (только прошлое/текущее).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research.costwall import _auc
from research.cot_backfill import CACHE

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
# Yahoo-символы форекса
SYMBOLS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X"}


def fetch_1h(name: str, force: bool = False) -> pd.DataFrame:
    """Часовой OHLC с Yahoo (~730d). Кэш parquet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"fx1h_{name}.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0"})
    for a in range(4):
        try:
            r = s.get(YAHOO + SYMBOLS[name], params={"range": "730d", "interval": "1h"}, timeout=30)
            if r.status_code == 200:
                j = r.json()["chart"]["result"][0]
                q = j["indicators"]["quote"][0]
                df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                                  index=pd.to_datetime(j["timestamp"], unit="s", utc=True)).dropna()
                df.to_parquet(p)
                return df
        except Exception:
            time.sleep(1.5)
    raise RuntimeError(f"Yahoo 1h fail {name}")


def build_sessions(name: str, *, on_start=0, on_end=5, analysis_start=5,
                   entry_h=7, exit_h=20) -> pd.DataFrame:
    """Дневная session-таблица: ночной диапазон, утренний моментум, вход 07, путь до EOD."""
    df = fetch_1h(name)
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    rows = []
    for d, g in df.groupby("date"):
        on = g[(g.hour >= on_start) & (g.hour < on_end)]
        am = g[(g.hour >= analysis_start) & (g.hour < entry_h)]
        ent = g[g.hour == entry_h]
        intraday = g[(g.hour >= entry_h) & (g.hour < exit_h)]
        if on.empty or am.empty or ent.empty or len(intraday) < 3:
            continue
        prev = rows[-1]["eod_close"] if rows else np.nan
        rows.append({
            "date": d,
            "on_hi": on["high"].max(), "on_lo": on["low"].min(),
            "morning_ret": am["close"].iloc[-1] / am["open"].iloc[0] - 1.0,
            "entry": ent["open"].iloc[0],
            "eod_close": intraday["close"].iloc[-1],
            "fwd_ret": intraday["close"].iloc[-1] / ent["open"].iloc[0] - 1.0,
            "path_hi": list(intraday["high"].values),
            "path_lo": list(intraday["low"].values),
            "path_close": list(intraday["close"].values),
            "prev_close": prev,
        })
    return pd.DataFrame(rows).set_index("date")


# ── сигналы (балл в момент входа 07:00, causal) ──────────────────────────────
def sig_morning_mom(t: pd.DataFrame) -> pd.Series:
    """Моментум утреннего окна 05-07 → продолжение."""
    return t["morning_ret"]

def sig_overnight_break(t: pd.DataFrame) -> pd.Series:
    """Пробой ночного диапазона на входе: +1 выше hi, -1 ниже lo, иначе 0."""
    up = (t["entry"] > t["on_hi"]).astype(float)
    dn = (t["entry"] < t["on_lo"]).astype(float)
    return up - dn

def sig_trend_prevclose(t: pd.DataFrame) -> pd.Series:
    """Гэп к вчерашнему закрытию (тренд)."""
    return t["entry"] / t["prev_close"] - 1.0

SIGNALS = {"morning_mom": sig_morning_mom, "overnight_break": sig_overnight_break,
           "trend_prevclose": sig_trend_prevclose}


def eval_auc(t: pd.DataFrame, sig: pd.Series, cost_bps: float, train_frac=0.5) -> dict:
    """AUC + простой net (07→20 directional) на test-половине."""
    v = sig.notna() & t["fwd_ret"].notna() & np.isfinite(sig)
    s, r = sig[v].values, t["fwd_ret"][v].values
    n = len(s); split = int(n * train_frac)
    if split < 30 or n - split < 30:
        return {"auc": float("nan"), "net_bps": float("nan"), "n": n}
    sign = 1 if np.corrcoef(s[:split], r[:split])[0, 1] >= 0 else -1
    ste, rte = s[split:] * sign, r[split:]
    auc = _auc(ste, (rte > 0).astype(int))
    # вход по знаку сигнала (только ненулевые), net = dir*ret - cost
    take = ste != 0
    net = (np.sign(ste[take]) * rte[take] * 1e4 - cost_bps).mean() if take.sum() else float("nan")
    return {"auc": round(auc, 4), "net_bps": round(net, 2), "n_test": len(ste), "n_signal": int(take.sum())}


def eval_bracket(t: pd.DataFrame, sig: pd.Series, *, stop_bps: float, rr: float, cost_bps: float) -> dict:
    """ТОЧНАЯ симуляция плана: вход 07 по знаку сигнала, стоп −stop, тейк +rr*stop, EOD-выход.
    По часовым hi/lo. Возвращает win-rate, EV/сделку net, t."""
    sf = stop_bps / 1e4
    tf = rr * stop_bps / 1e4
    pnls = []
    for i in range(len(t)):
        sv = sig.iloc[i]
        if not np.isfinite(sv) or sv == 0:
            continue
        d = int(np.sign(sv))
        e = t["entry"].iloc[i]
        hi, lo, cl = t["path_hi"].iloc[i], t["path_lo"].iloc[i], t["path_close"].iloc[i]
        stop = e * (1 - d * sf); tgt = e * (1 + d * tf)
        outcome = None
        for h, l in zip(hi, lo):
            hit_stop = (l <= stop) if d > 0 else (h >= stop)
            hit_tgt = (h >= tgt) if d > 0 else (l <= tgt)
            if hit_stop:                       # стоп первым при конфликте
                outcome = -stop_bps; break
            if hit_tgt:
                outcome = rr * stop_bps; break
        if outcome is None:                    # EOD-выход
            outcome = d * (cl[-1] / e - 1) * 1e4
        pnls.append(outcome - cost_bps)
    pnls = np.array(pnls, dtype=float)
    n = len(pnls)
    if n == 0:
        return {"n": 0}
    mean = pnls.mean()
    sd = pnls.std(ddof=1) if n > 1 else float("nan")
    tstat = mean / (sd / np.sqrt(n)) if sd and sd > 0 else float("nan")
    return {"n": n, "win_rate": round((pnls > 0).mean(), 3),
            "ev_bps": round(mean, 2), "t": round(tstat, 2)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["EURUSD", "GBPUSD"])
    ap.add_argument("--cost", type=float, default=2.0, help="round-trip bps (форекс)")
    ap.add_argument("--stop", type=float, default=20.0, help="стоп в bps (1:2 → тейк 2x)")
    ap.add_argument("--rr", type=float, default=2.0)
    args = ap.parse_args()
    for name in args.names:
        t = build_sessions(name)
        print(f"\n===== {name}: {len(t)} торговых дней ({t.index.min().date()}..{t.index.max().date()}) =====")
        for sn, fn in SIGNALS.items():
            sig = fn(t)
            a = eval_auc(t, sig, args.cost)
            b = eval_bracket(t, sig, stop_bps=args.stop, rr=args.rr, cost_bps=args.cost)
            print(f"  {sn:16s} AUC={a.get('auc')} net07-20={a.get('net_bps')}bp | "
                  f"BRACKET n={b.get('n')} win={b.get('win_rate')} EV={b.get('ev_bps')}bp t={b.get('t')}")

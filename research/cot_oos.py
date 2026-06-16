"""
research/cot_oos.py
===================
Проспективный forward-OOS для EURUSD-COT (единственная зацепка, пережившая
pre-registered кросс-инструмент тест в изоляции; claim по backtest нельзя —
см. память cot-positioning-forex). Цель: разрешить её на данных, которых НЕ БЫЛО
в момент перебора параметров.

Дисциплина:
  1. freeze() ОДИН раз фиксирует правило (знак + порог |cot_z|), вычисленные на
     ИСТОРИИ <= даты заморозки, + frozen-параметры (cot_z, w104, H4, q0.65, cost5).
     Записывает freeze_date. После заморозки НИЧЕГО не подкручиваем.
  2. evaluate() каждую неделю пересобирает df, применяет ЗАМОРОЖЕННОЕ правило к
     релизам с entry_date > freeze_date, считает net по завершённым сделкам
     (entry+H недель <= последняя цена), пишет леджер. Non-overlap для статистики.

Это НЕ backtest: до накопления n>=30 завершённых OOS-сделок вердикт = ACCUMULATING.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.cot_backfill import build_cot_df, CACHE
from research.signals.cot import cot_z

PARAMS = dict(signal="cot_z", window=104, horizon=4, quantile=0.65, cost_bps=5.0)


def _freeze_path(name: str) -> Path:
    return CACHE / f"cot_oos_freeze_{name}.json"


def _ledger_path(name: str) -> Path:
    return CACHE / f"cot_oos_ledger_{name}.csv"


def freeze(name: str = "EURUSD", force: bool = False) -> dict:
    """Зафиксировать правило по истории <= сегодня. Один раз. Повторно — только force."""
    fp = _freeze_path(name)
    if fp.exists() and not force:
        return json.loads(fp.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    df = build_cot_df(name, force=True)
    sig = cot_z(df, window=PARAMS["window"])
    fwd = df["close"].shift(-PARAMS["horizon"]) / df["close"] - 1.0
    v = sig.notna() & fwd.notna() & np.isfinite(sig)
    s, r = sig[v], fwd[v]
    sign = 1 if s.corr(r) >= 0 else -1
    threshold = float(s.abs().quantile(PARAMS["quantile"]))
    frozen = {
        "name": name,
        "freeze_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sign": sign,
        "threshold": threshold,
        "n_history": int(len(s)),
        "last_history_date": df.index.max().strftime("%Y-%m-%d"),
        **PARAMS,
    }
    fp.write_text(json.dumps(frozen, indent=2))
    return frozen


@dataclass
class OOSReport:
    name: str
    freeze_date: str
    n_signals_oos: int      # сигналов после заморозки
    n_completed: int        # из них завершённых (горизонт прошёл), non-overlap
    net_mean_bps: float
    t_stat: float
    hit_rate: float
    verdict: str


def evaluate(name: str = "EURUSD") -> OOSReport:
    """Применить замороженное правило к OOS-релизам, обновить леджер, вернуть отчёт."""
    fr = freeze(name)                      # вернёт существующий freeze (не пересоздаёт)
    H = fr["horizon"]
    df = build_cot_df(name, force=True)    # свежие данные
    freeze_ts = pd.Timestamp(fr["freeze_date"], tz="UTC")

    sig = cot_z(df, window=fr["window"]) * fr["sign"]   # ориентированный сигнал
    close = df["close"].astype(float)
    last_date = df.index.max()

    rows = []
    for i, (ts, sv) in enumerate(sig.items()):
        if ts <= freeze_ts or not np.isfinite(sv) or abs(sv) < fr["threshold"]:
            continue
        direction = int(np.sign(sv))
        entry = float(close.iloc[i])
        exit_i = i + H
        completed = exit_i < len(df)
        exit_price = float(close.iloc[exit_i]) if completed else np.nan
        net = (direction * (exit_price / entry - 1.0) * 1e4 - fr["cost_bps"]) if completed else np.nan
        rows.append({
            "entry_date": ts.strftime("%Y-%m-%d"), "direction": direction,
            "entry": entry, "exit_date": df.index[exit_i].strftime("%Y-%m-%d") if completed else "",
            "exit": exit_price, "net_bps": round(net, 2) if completed else np.nan,
            "completed": completed,
        })

    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger.to_csv(_ledger_path(name), index=False)

    comp = ledger[ledger["completed"]] if not ledger.empty else ledger
    # non-overlap: брать каждую сделку, отстоящую от прошлой взятой на >= H релизов
    keep_idx = []
    last = -10**9
    for k, row in enumerate(rows):
        if row["completed"]:
            if k - last >= H:
                keep_idx.append(k); last = k
    nets = np.array([rows[k]["net_bps"] for k in keep_idx], dtype=float)
    n_comp = len(nets)
    if n_comp == 0:
        net_mean = t = hit = float("nan")
    else:
        net_mean = float(nets.mean())
        sd = nets.std(ddof=1) if n_comp > 1 else float("nan")
        t = float(net_mean / (sd / np.sqrt(n_comp))) if sd and sd > 0 else float("nan")
        hit = float((nets > 0).mean())

    if n_comp < 30:
        verdict = "ACCUMULATING"
    elif net_mean > 0 and not np.isnan(t) and t >= 2.0:
        verdict = "CONFIRMED"
    elif net_mean <= 0:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return OOSReport(
        name=name, freeze_date=fr["freeze_date"],
        n_signals_oos=len(rows), n_completed=n_comp,
        net_mean_bps=round(net_mean, 2) if not np.isnan(net_mean) else float("nan"),
        t_stat=round(t, 2) if not np.isnan(t) else float("nan"),
        hit_rate=round(hit, 3) if not np.isnan(hit) else float("nan"),
        verdict=verdict,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="EURUSD")
    ap.add_argument("--freeze", action="store_true", help="зафиксировать правило (один раз)")
    ap.add_argument("--force-freeze", action="store_true")
    args = ap.parse_args()
    if args.freeze or args.force_freeze:
        fr = freeze(args.name, force=args.force_freeze)
        print(f"FROZEN {args.name}: freeze_date={fr['freeze_date']} sign={fr['sign']} "
              f"threshold={fr['threshold']:.3f} n_history={fr['n_history']} "
              f"(last_history {fr['last_history_date']})")
    rep = evaluate(args.name)
    print(f"\nOOS {rep.name} (frozen {rep.freeze_date}): "
          f"signals={rep.n_signals_oos} completed(non-overlap)={rep.n_completed} "
          f"net={rep.net_mean_bps}bp t={rep.t_stat} hit={rep.hit_rate} -> {rep.verdict}")

"""
research/vol_managed.py
=======================
Vol-managed exposure (Moreira-Muir 2017), ретейл-спот версия. НЕ directional:
не угадываем сторону, а сайзим лонг инверсно прошлой реализованной волатильности.
Приор: vol форкастится (vol clustering), direction — нет ([[research-harness-deriv-firstpass]]).

Вопрос: бьёт ли vol-managed обычный buy-and-hold по Sharpe/MaxDD **net of costs**?
Главный риск — turnover: при whipsaw волатильности вес дёргается → rebalancing costs
(спот Kraken taker ~26bp) съедают улучшение. Поэтому net считается честно.

Causal: вес[t] = clip(target_vol / realized_vol[t-1], 0, w_max). На споте w_max=1.0
(плечо ретейлу недоступно → можно только СНИЖАТЬ экспозицию, не наращивать). Левередж
(w_max>1) — отдельный сценарий «если бы были фьючи».

Оговорки: cash при снижении веса = 0 доходности (без risk-free); идеальное исполнение
по close; без проскальзывания сверх cost_bps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.binance_backfill import build_funding_df

TRADING_DAYS = 365  # крипта торгуется 24/7/365


@dataclass
class PerfStats:
    name: str
    years: float
    cagr_pct: float
    vol_pct: float
    sharpe: float
    max_dd_pct: float
    avg_weight: float
    turnover_yr: float
    total_cost_pct: float


def _metrics(name: str, daily_ret: pd.Series, weight: pd.Series | None,
             cost_pct_total: float, years: float) -> PerfStats:
    eq = (1 + daily_ret.fillna(0)).cumprod()
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else float("nan")
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily_ret.mean() * TRADING_DAYS) / vol if vol > 0 else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    if weight is not None:
        avg_w = float(weight.mean())
        turn_yr = float(weight.diff().abs().sum() / years)
    else:
        avg_w, turn_yr = 1.0, 0.0
    return PerfStats(
        name=name, years=round(years, 2), cagr_pct=round(cagr * 100, 2),
        vol_pct=round(vol * 100, 2), sharpe=round(sharpe, 3),
        max_dd_pct=round(dd * 100, 2), avg_weight=round(avg_w, 3),
        turnover_yr=round(turn_yr, 2), total_cost_pct=round(cost_pct_total, 2),
    )


def run(symbol: str = "BTCUSDT", *, vol_window: int = 20, target_vol: float = 0.60,
        w_max: float = 1.0, cost_bps: float = 26.0, band: float = 0.0) -> dict:
    """Бэктест vol-managed vs buy&hold на дневных данных. target_vol — годовая (0.60=60%).
    band — no-trade band: не менять вес, пока |target_w - текущий| <= band (режет turnover)."""
    df = build_funding_df(symbol)                 # 8h close, ~2019->now (из кэша)
    price = df["close"].resample("1D").last().dropna()
    r = price.pct_change()
    years = (price.index[-1] - price.index[0]).days / 365.25

    # реализованная годовая vol на ПРОШЛОМ окне
    rv = r.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(TRADING_DAYS)
    # целевой вес, решённый по info до t-1 (causal): сдвигаем rv на 1
    w_target = (target_vol / rv.shift(1)).clip(0, w_max).fillna(0.0)
    if band > 0:
        # держим прежний вес, пока целевой не уйдёт дальше band (режет churn)
        vals = w_target.values
        held = np.zeros_like(vals)
        cur = 0.0
        for i, t in enumerate(vals):
            if abs(t - cur) > band:
                cur = t
            held[i] = cur
        w = pd.Series(held, index=w_target.index)
    else:
        w = w_target

    # доходность стратегии: вес дня применяется к доходности дня
    strat_gross = w * r
    turnover = w.diff().abs().fillna(w.abs())
    cost = turnover * cost_bps / 1e4
    strat_net = strat_gross - cost
    total_cost_pct = float(cost.sum() * 100)

    bh = _metrics(f"{symbol} buy&hold", r, None, 0.0, years)
    vm = _metrics(f"{symbol} vol-managed (net)", strat_net, w, total_cost_pct, years)
    vm_gross = _metrics(f"{symbol} vol-managed (gross)", strat_gross, w, 0.0, years)
    return {"buy_hold": bh, "vm_net": vm, "vm_gross": vm_gross,
            "params": dict(vol_window=vol_window, target_vol=target_vol,
                           w_max=w_max, cost_bps=cost_bps)}


def _print(res: dict) -> None:
    p = res["params"]
    print(f"\nparams: window={p['vol_window']}d target_vol={p['target_vol']:.0%} "
          f"w_max={p['w_max']} cost={p['cost_bps']}bp")
    print(f"{'strategy':32s} {'CAGR%':>7} {'vol%':>6} {'Sharpe':>7} "
          f"{'MaxDD%':>7} {'avgW':>5} {'turn/y':>6} {'cost%':>6}")
    for key in ("buy_hold", "vm_gross", "vm_net"):
        s = res[key]
        print(f"{s.name:32s} {s.cagr_pct:>7} {s.vol_pct:>6} {s.sharpe:>7} "
              f"{s.max_dd_pct:>7} {s.avg_weight:>5} {s.turnover_yr:>6} {s.total_cost_pct:>6}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--target", type=float, default=0.60)
    ap.add_argument("--wmax", type=float, default=1.0)
    ap.add_argument("--cost", type=float, default=26.0)
    ap.add_argument("--band", type=float, default=0.0)
    args = ap.parse_args()
    _print(run(args.symbol, vol_window=args.window, target_vol=args.target,
               w_max=args.wmax, cost_bps=args.cost, band=args.band))

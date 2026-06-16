"""
research/funding_carry.py
=========================
Тест funding CARRY — НЕ directional. Идея: держать дельта-нейтральную пару
(short perp + long spot при funding>0) и СОБИРАТЬ funding каждые 8ч. Цена
захеджирована (long spot gain = short perp loss), P&L ≈ накопленный funding − costs.

Это единственный funding-угол, не закрытый directional-тестом
([[research-harness-deriv-firstpass]]: funding directional = NO_EDGE мощно).

Модель (causal, без look-ahead):
  - funding[i] известен в момент i. Позиция, установленная по info<=i, собирает
    funding на барах i+1...  (collect funding для позиции, ВОШЕДШЕЙ в бар).
  - Гистерезис: войти когда funding>entry_thr, держать пока funding>exit_thr.
  - allow_negative=True → симметрично собирать и отрицательный funding
    (long perp + short spot; ритейлу труднее шортить спот — по умолчанию False).
  - Транзакционный кост: вход (0→in) = cost_bps/2, выход (in→0) = cost_bps/2,
    флип = cost_bps. cost_bps — round-trip ПАРЫ (спот+перп) в bps.

ЧЕСТНЫЕ оговорки (модель оптимистична):
  - идеальный хедж (игнор basis-слиппеджа на входе/выходе);
  - доходность на NOTIONAL, не на капитале (нужен капитал на ОБЕ ноги → реальный
    ROE ~вдвое ниже);
  - funding settled retroactively — собираем со следующего бара (не текущего).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.binance_backfill import build_funding_df


@dataclass
class CarryResult:
    symbol: str
    cost_bps: float
    entry_thr: float
    exit_thr: float
    allow_negative: bool
    years: float
    gross_funding_pct: float    # накопленный собранный funding, % notional
    total_cost_pct: float
    net_pct: float              # gross - cost, % notional (кумулятивно)
    net_annual_pct: float       # net_pct / years
    n_cycles: int
    avg_hold_days: float
    pct_time_in: float
    by_year: dict               # год -> net % за год
    verdict: str


def simulate(
    df: pd.DataFrame,
    *,
    cost_bps: float,
    entry_thr: float = 0.0001,
    exit_thr: float = 0.0,
    allow_negative: bool = False,
    symbol: str = "BTCUSDT",
) -> CarryResult:
    f = df["funding_rate"].values.astype(float)
    years_idx = df.index.year.values
    n = len(f)
    half = cost_bps / 1e4 / 2.0

    pos = 0                       # 0 flat, +1 short-perp(сбор funding>0), -1 наоборот
    income = 0.0
    cost = 0.0
    in_steps = 0
    n_cycles = 0
    holds: list[int] = []
    cur_hold = 0
    yr_income: dict = {}
    yr_cost: dict = {}

    for i in range(n):
        yr = int(years_idx[i])
        # 1) собрать funding для позиции, вошедшей В этот бар (установлена на i-1)
        if pos == 1:
            income += f[i]; yr_income[yr] = yr_income.get(yr, 0.0) + f[i]
            in_steps += 1; cur_hold += 1
        elif pos == -1:
            income += -f[i]; yr_income[yr] = yr_income.get(yr, 0.0) - f[i]
            in_steps += 1; cur_hold += 1

        # 2) решить новую позицию по f[i] (causal), эффект — со следующего бара
        if pos == 1:
            new = 1 if f[i] > exit_thr else 0
        elif pos == -1:
            new = -1 if f[i] < -exit_thr else 0
        else:
            new = 0
        if new == 0:
            if f[i] > entry_thr:
                new = 1
            elif allow_negative and f[i] < -entry_thr:
                new = -1

        # 3) транзакционные косты на смену состояния
        if new != pos:
            if pos == 0 and new != 0:               # вход
                cost += half; yr_cost[yr] = yr_cost.get(yr, 0.0) + half
                n_cycles += 1; cur_hold = 0
            elif pos != 0 and new == 0:             # выход
                cost += half; yr_cost[yr] = yr_cost.get(yr, 0.0) + half
                holds.append(cur_hold)
            else:                                   # флип = выход+вход
                cost += 2 * half; yr_cost[yr] = yr_cost.get(yr, 0.0) + 2 * half
                holds.append(cur_hold); n_cycles += 1; cur_hold = 0
            pos = new
    if pos != 0:
        holds.append(cur_hold)

    years = (df.index[-1] - df.index[0]).days / 365.25
    gross = income * 100
    tcost = cost * 100
    net = gross - tcost
    by_year = {
        y: round((yr_income.get(y, 0.0) - yr_cost.get(y, 0.0)) * 100, 2)
        for y in sorted(set(int(x) for x in years_idx))
    }
    net_annual = net / years if years else float("nan")
    avg_hold_days = (np.mean(holds) * 8 / 24) if holds else 0.0

    if net_annual <= 0:
        verdict = "DEAD"
    elif net_annual < 5:
        verdict = "MARGINAL"
    else:
        verdict = "POSITIVE"

    return CarryResult(
        symbol=symbol, cost_bps=cost_bps, entry_thr=entry_thr, exit_thr=exit_thr,
        allow_negative=allow_negative, years=round(years, 2),
        gross_funding_pct=round(gross, 2), total_cost_pct=round(tcost, 2),
        net_pct=round(net, 2), net_annual_pct=round(net_annual, 2),
        n_cycles=n_cycles, avg_hold_days=round(avg_hold_days, 2),
        pct_time_in=round(in_steps / n, 3), by_year=by_year, verdict=verdict,
    )


def _print(r: CarryResult) -> None:
    print(f"\n{r.symbol} carry | cost={r.cost_bps}bp entry={r.entry_thr} "
          f"exit={r.exit_thr} neg={r.allow_negative}")
    print(f"  span {r.years}y | gross={r.gross_funding_pct}% cost={r.total_cost_pct}% "
          f"net={r.net_pct}% -> net/yr={r.net_annual_pct}%  [{r.verdict}]")
    print(f"  cycles={r.n_cycles} avg_hold={r.avg_hold_days}d time_in={r.pct_time_in*100:.0f}%")
    print(f"  by year: {r.by_year}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--costs", nargs="+", type=float, default=[40, 65, 90])
    ap.add_argument("--entry", type=float, default=0.0001)
    ap.add_argument("--exit", type=float, default=0.0)
    ap.add_argument("--neg", action="store_true", help="собирать и отрицательный funding")
    args = ap.parse_args()

    df = build_funding_df(args.symbol)
    print(f"{args.symbol}: {len(df)} funding-points, "
          f"{df.index.min().date()} -> {df.index.max().date()}, "
          f"mean funding={df['funding_rate'].mean()*100:.4f}%/8h")
    for c in args.costs:
        _print(simulate(df, cost_bps=c, entry_thr=args.entry,
                        exit_thr=args.exit, allow_negative=args.neg, symbol=args.symbol))

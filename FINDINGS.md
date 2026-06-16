# Findings — what rigorous testing actually returns

A sample of results produced with this harness. The point isn't a winning strategy —
it's that each verdict is *trustworthy*: causal, cost-aware, power-gated, and audited for
the bugs that make most backtests lie. Two cases below show the discipline in action.

---

## Case 1 — A "0.64 AUC edge" that was really noise

A funding-rate signal on BTC looked promising on ~11 days of data:
`AUC 0.64` at a multi-day horizon, positive net. Tempting to trade.

**The discipline:** before believing it, two checks.
1. **Effective sample.** AUC was computed on overlapping bars — the *independent* sample
   was `n_eff = n_test / horizon ≈ 9`, not thousands. At n_eff = 9, an AUC of 0.64 is
   well inside one standard error of a coin flip.
2. **Multi-regime backfill.** Binance serves funding history to 2019. Re-run on **6.75
   years across bull and bear** (4,150 points):

| horizon | n_eff | AUC | net (bps) | verdict |
|---|---|---|---|---|
| 8h | 2,076 | 0.504 | −34 | NO_EDGE |
| 1d | 691 | 0.519 | −33 | NO_EDGE |
| 3d | 230 | 0.529 | −21 | NO_EDGE |

The 0.64 **collapsed to ~0.51** once the sample was adequate and spanned multiple regimes.
It was a regime-bias artifact, not alpha. Confirmed `NO_EDGE` market-wide (ETH, SOL, BNB,
XRP all agree). **A weak result on a short, one-regime window is not evidence — it's the
null hypothesis wearing a costume.**

---

## Case 2 — Finding the bug that made the backtest lie

A causal trend-following test on BTC regimes (2017–2026) first reported
**CAGR 474%, Sharpe 3.58** vs buy & hold's 0.84. Too good — so it got audited, not shipped.

- **Bug 1 (look-ahead):** the position decided at bar *i* was being credited with the
  return *into* bar *i* — i.e. earning the very move that triggered the signal. Fixed:
  the position applies to the *next* return. Result dropped to Sharpe ~1.0.
- **Bug 2 (smoothing):** the signal ran on the daily *mean* price — which can't be traded
  and smooths away false flips, flattering a trend follower. Re-run on the tradeable
  daily **close**: the remaining "edge" collapsed.

**Honest result on tradeable prices:** long-only trend-following ≈ buy & hold on
return/Sharpe (0.55–0.86 vs 0.80); the long/short version loses. The only robust effect is
a modest drawdown reduction. **What a client is really paying for is someone who catches
these two bugs — because they are the difference between a 474% fantasy and the truth.**

---

## The ledger (breadth)

Every hypothesis tested, with its verdict, is kept in an append-only ledger. A sample:

| Hypothesis | Data | Verdict |
|---|---|---|
| Funding direction | Binance 2019–2026, 5 symbols | NO_EDGE (well-powered) |
| Funding carry | Binance 2019–2026 | regime relic, decayed |
| COT positioning | CFTC + Yahoo, decades, 7 FX + gold | factor not confirmed (2/7, pre-registered) |
| Vol-managed exposure | BTC/ETH daily | de-risk only, no alpha |
| Intraday session timing | 4 FX pairs, 2.8y | NO_EDGE, EV < 0 |
| ICT / SMC (sweep+MSS+FVG) | 4 FX pairs + crypto | not profitable, robust across params |
| BTC trend-following | 2017–2026 hourly | ≈ buy & hold (after bug fixes) |

Nine hypotheses, honestly adjudicated. A few remain under **prospective** out-of-sample
tracking — because the only honest way to resolve a borderline case is forward, not by
re-fitting the past.

---

*Want your trading idea tested this way — before you risk money on it?*
**costwall@gmx.de**

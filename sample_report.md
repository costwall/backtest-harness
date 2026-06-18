# Strategy Edge Report — Mean-Reversion (z-score), EUR/USD

**Prepared by costwall** · 2026-06-18 · Package: Quick Edge Check

> **What this report is.** An honest, costs-included test of whether your strategy
> has a tradeable edge — and if not, *exactly what stands between it and one*. A
> red verdict is not a dead end: it means you won't lose your deposit on this as-is,
> plus a map of what would have to change. We sell the truth and the map, not a stamp.

---

## 1. Scope & Assumptions (what we fixed)

| Item | Value |
|---|---|
| Instrument | EUR/USD |
| Timeframe | H1 (signal lookback 48 bars = ~2 days) |
| Holding horizon | 24 bars = 24 hours |
| Round-trip cost assumed | 3 bps (retail forex realistic) |
| Validation | causal 50/50 split + rolling walk-forward (18 folds) |
| **Test type** | ONE pre-specified strategy → N=1, fair test, NO multiple-testing penalty |

> Note: timeframe and signal lookback are separate axes; cost is the binding wall
> for marginal edges (see breakeven below).

---

## 2. Verdict

# NO_EDGE

**Plain English:** After costs this mean-reversion rule is a coin flip — and the
faint tilt it does show is not statistically distinguishable from luck. The
in-sample "profit" disappears out-of-sample. Do not trade it as-is.

| Verdict | Meaning |
|---|---|
| TRADEABLE | Net edge survives costs, is significant, and holds out-of-sample. |
| SUB_COST | Real predictive signal — but your costs eat it. Fixable (see §4). |
| INCONCLUSIVE | Not enough independent data yet to decide. |
| **NO_EDGE** ← | No directional skill beyond a coin flip. Do not trade this as-is. |

---

## 3. Evidence

| Metric | Value | Reads as |
|---|---|---|
| AUC (directional skill) | 0.514 | ≈ coin flip (0.50); below the 0.55 skill line |
| Net per trade | −1.14 bps | loses after costs |
| **Breakeven cost (`gross_bps`)** | **1.86 bps** | lives only if round-trip cost < 1.86 bps |
| t-stat | −0.35 | not significant (need ≥2) |
| Permutation p (is the signal *real*?) | 0.30 | **> 0.05 → direction is NOT distinguishable from luck** |
| Sub-period consistency | 0.50 | profitable in only half the periods |
| Max drawdown | 323 bps | worst peak-to-trough |
| Walk-forward | WF_DEGRADED | IS net +3.3 bps → **OOS net −2.9 bps** (flips to loss) |
| Deflated Sharpe | N/A | single pre-specified strategy — no search to deflate |

---

## 4. Revival Map — what would make this tradeable

- **Is the signal even real?** Permutation **p = 0.30** → the direction is **not
  statistically distinguishable from luck**. This is the core stop sign: there is no
  faint-but-real edge here to rescue, unlike a SUB_COST case.
- **Cost wall.** Breakeven is **1.86 bps** round-trip; you assumed 3 bps. Even a
  broker under 1.86 bps wouldn't save it — the signal itself isn't significant.
- **Walk-forward tells the real story.** In-sample the threshold finds a +3.3 bps
  "profit", but out-of-sample it flips to −2.9 bps. That gap **is** the overfit: the
  in-sample gain was the entry threshold fitting noise, not a repeatable edge.
- **Nearest it came.** On an H4 timeframe AUC rose to ~0.56 — but on a tiny sample
  (14–30 trades) that does not survive deflation; not a real lead.
- **Data sufficiency.** 113 trades / 182 independent observations — adequate. The
  verdict is a real "no", not a "not enough data".

**Bottom line:** don't trade this as-is — you'd be trading an in-sample illusion.
We just saved you the drawdown. If you have a *different* idea you believe in, bring
it: a real edge passes this same stack (we calibrate on it).

---

## 5. Variation Matrix — what else we could test (and what it costs)

*Each axis you add is another trial. More variations = more work (price) AND a
higher significance bar (we deflate for the search — that's the honest part).*

| Axis | Tested here | Alternative | What it pulls | + tests |
|---|---|---|---|---|
| Timeframe | H1 | + H4, D1 | re-scale horizon to real time; each TF a separate run | ×2 |
| Holding horizon | 24h | sweep 6h / 24h / 72h | 2nd degree of freedom → DSR deflation applies | ×3 |
| Cost model | flat 3 bps | vol-scaled slippage | breakeven vs slippage in fast moves | ×3 |
| Validation | 50/50 + walk-forward | — (already included) | — | — |
| Instrument | EUR/USD | + GBP/USD, USD/JPY | does the edge generalize or is it one-off? | ×2 |

> Pick what's worth it. We'll quote the added work before running anything.

---

## 6. Honest disclaimer

Most retail strategies are observed in-sample or cherry-picked, and **most fail an
honest out-of-sample, costs-included test** — that's the base rate, not a judgment
of your idea. Our value is telling you the truth *before* the market does, and
showing you exactly where the line is. Methodology: cost-wall (AUC + net + breakeven),
Deflated Sharpe (de Prado), circular-shift permutation, sub-period consistency,
rolling walk-forward. Open harness: github.com/costwall/backtest-harness.

---

## 7. Important — not financial advice

This report is a statistical analysis of historical data — **not financial,
investment, or trading advice, and not a recommendation to trade.**

- **A verdict is not a guarantee.** TRADEABLE (or any result) does **not** guarantee
  future profit. Past statistical behavior does not predict future results; markets
  change regime, and live execution (slippage, fills, latency, spread, gaps) differs
  from any backtest.
- **Your decisions, your risk.** All trading decisions, position sizing, risk
  management, and any resulting profits or losses are **solely your responsibility.**
- **Analysis only.** costwall does not manage funds, place trades, or advise whether
  you should trade. Never risk capital you cannot afford to lose.

By using this report you agree that costwall accepts **no liability** for any losses
or damages arising from your use of, or reliance on, this analysis.

*— costwall · honest, costs-included strategy validation · costwall@gmx.de*

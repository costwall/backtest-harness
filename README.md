# costwall — honest backtesting & quant-research harness

**A research framework that tells you whether a trading signal actually has an edge —
net of costs, free of look-ahead bias.**

Most retail backtests lie to themselves: look-ahead leaks, overfit parameters, costs
ignored, a lucky curve mistaken for skill. This harness is built to do the opposite —
to *reject* false edges quickly and cheaply, and to say "no" with evidence. Every
hypothesis runs through the same **cost wall**:

> Does the signal predict direction better than a coin flip (AUC), **and** does the
> predicted move beat round-trip costs — on a causal, out-of-sample, adequately-powered
> sample?

A new idea becomes a ~10-line signal function and one command; the verdict lands in an
append-only ledger so dead ends are never re-explored.

📄 **See [FINDINGS.md](FINDINGS.md)** for two worked examples — including a "0.64 AUC edge"
that was really noise, and the look-ahead bug that turned a fake 474% CAGR into the truth.

---

## Why it's different (the methodology is the product)

- **Causal train/test split** — sign and entry threshold are fixed on the *train* half
  only; metrics are measured out-of-sample. No fitting on the test set.
- **Effective-sample gating (`n_eff`)** — AUC on overlapping bars has far fewer
  independent observations than rows. The harness reports `n_eff = n_test / horizon`
  and marks underpowered cells `INCONCLUSIVE` instead of falsely `NO_EDGE`/`TRADEABLE`.
- **Significance gate** — nothing is stamped `TRADEABLE` without `t ≥ 2`. This stops
  noise (e.g. a `t = 1.6` cell) from masquerading as an edge.
- **Pre-registration** — cross-instrument tests fix parameters *before* the run, with a
  written pass/fail rule, to defeat best-of-many selection.
- **Look-ahead audits** — release-lag handling for lagged data (e.g. COT publishes
  Friday on Tuesday positions), and explicit checks that caught real off-by-one and
  smoothing artifacts in development.
- **Prospective forward-OOS** — for borderline single-instrument survivors, a frozen
  rule accumulates *genuinely* out-of-sample results over time (`cot_oos.py`).

## What it has tested (and mostly rejected — honestly)

| Hypothesis | Data | Verdict |
|---|---|---|
| Funding-rate direction | Binance, 2019–2026, 5 symbols | `NO_EDGE` (well-powered) |
| Funding carry (delta-neutral) | Binance, 2019–2026 | regime relic, decayed |
| Order-flow / volume-delta | 1.4 yr 5m | `NO_EDGE` |
| COT positioning | CFTC + Yahoo, decades, 7 FX + gold | factor not confirmed (2/7) |
| Vol-managed exposure | BTC/ETH daily | de-risk only, no alpha |
| Intraday session timing | 4 FX pairs, 2.8 yr | `NO_EDGE`, EV < 0 |
| ICT / SMC (sweep + MSS + FVG) | 4 FX pairs + crypto | not profitable, robust |
| BTC trend-following on regimes | 2017–2026 hourly | ≈ buy & hold on tradeable price |

The value isn't a magic strategy — it's a framework rigorous enough that its "no" is
trustworthy, and disciplined enough to find the rare "maybe" without fooling itself.

## Components

```
research/
  costwall.py        # the evaluator: AUC, move-vs-cost, net, n_eff, t-gate, verdicts
  run.py             # CLI: symbol × signal × horizon → ledger
  signals/           # pluggable signal library (pure df -> Series)
  binance_backfill.py# public Binance funding + klines (no API key)
  cot_backfill.py    # public CFTC COT + Yahoo price alignment (release-lagged)
  session_forex.py   # intraday session / breakout study (Yahoo 1h)
  btc_regimes.py     # ZigZag regime map + causal trend-following
  funding_carry.py   # delta-neutral carry simulation
  vol_managed.py     # volatility-targeted exposure (Moreira–Muir)
  cot_oos.py         # frozen-rule prospective out-of-sample tracker
```

All data sources are **public and key-free** (CFTC, Binance public endpoints, Yahoo).

## Quick start

```bash
pip install -r requirements.txt

python -m research.btc_regimes                       # BTC regime map 2017–2026 + trend-follow
python -m research.funding_carry --symbol BTCUSDT    # funding carry across regimes
python -m research.session_forex --names EURUSD GBPUSD
python -m research.run --symbol BTCUSDT --source funding --signals funding_z --horizons 1 3 9 --cost 40
```

## Available for work

Rigorous strategy validation · market-data pipelines & collectors · trading automation
and dashboards. If you want to know whether your idea actually has an edge — before you
risk money on it — that's exactly what this framework is for.

**Contact:** costwall@gmx.de

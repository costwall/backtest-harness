"""
research/run.py
===============
CLI харнесса: символ × сигнал × горизонт → cost-wall вердикт → RESULTS.md.

Примеры:
  python -m research.run --symbol BTCUSDT --source 5m \
      --signals lsr_extreme smart_money oi_roc cvd \
      --horizons 12 48 288 --cost 40

  python -m research.run --symbol BTCUSDT --source 1m \
      --signals funding_z --horizons 60 240 --cost 40

Горизонт H — в БАРАХ источника (5m: 12=1ч, 48=4ч, 288=1д; 1m: 60=1ч, 1440=1д).
cost — round-trip в bps (Kraken крипта ~32-52). --no-save не писать в RESULTS.md.
"""
from __future__ import annotations

import argparse

from research import data as datamod
from research.costwall import append_results, evaluate
from research.signals import REGISTRY

_LOADERS = {
    "5m": datamod.load_deriv_5m,
    "1m": datamod.load_deriv_1m,
    "funding": datamod.load_funding,   # мультирежим Binance 8h (~2019→now)
    "cot": datamod.load_cot,           # COT позиционирование + цена, недельный (форекс/золото)
}


def main() -> None:
    ap = argparse.ArgumentParser(description="cost-wall harness")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--source", choices=list(_LOADERS), default="5m")
    ap.add_argument("--signals", nargs="+", required=True,
                    help=f"из {list(REGISTRY)}")
    ap.add_argument("--horizons", nargs="+", type=int, required=True)
    ap.add_argument("--cost", type=float, default=40.0, help="round-trip bps")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--quantile", type=float, default=0.80, help="порог входа |signal|")
    ap.add_argument("--window", type=int, default=None,
                    help="override окна сигнала (в барах источника); иначе дефолт сигнала")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    df = _LOADERS[args.source](args.symbol)
    have = [c for c in df.columns]
    print(f"{args.symbol} [{args.source}] rows={len(df)} "
          f"span={df.index.min()} -> {df.index.max()}")
    print(f"cols={have}\n")

    results = []
    for sig_name in args.signals:
        if sig_name not in REGISTRY:
            print(f"  ! неизвестный сигнал: {sig_name}")
            continue
        sig = REGISTRY[sig_name](df, window=args.window) if args.window else REGISTRY[sig_name](df)
        for H in args.horizons:
            res = evaluate(
                df, sig, horizon=H, cost_bps=args.cost,
                label=f"{args.symbol}:{sig_name}", source=args.source,
                train_frac=args.train_frac, trade_quantile=args.quantile,
            )
            results.append(res)
            print(
                f"  {sig_name:12s} H={H:<4d} n={res.n_total:<6d} "
                f"n_eff={res.n_eff:<5d} trades={res.n_trades:<5d} AUC={res.auc_test} "
                f"move_med={res.move_bps_med}bp net={res.net_mean_bps}bp "
                f"t={res.net_t_stat} drift={res.drift_bps} -> {res.verdict}"
            )

    if not args.no_save and results:
        append_results(results)
        print(f"\nwritten {len(results)} rows -> research/RESULTS.md")


if __name__ == "__main__":
    main()

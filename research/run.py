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
from research.costwall import append_results, evaluate, vol_scaled_cost
from research.signals import REGISTRY

_LOADERS = {
    "5m": datamod.load_deriv_5m,
    "1m": datamod.load_deriv_1m,
    "funding": datamod.load_funding,   # мультирежим Binance 8h (~2019→now)
    "cot": datamod.load_cot,           # COT позиционирование + цена, недельный (форекс/золото)
    "forex": datamod.load_forex_ohlc,  # форекс OHLC H1 2025-2026 (EURUSD/USDJPY)
}


def main() -> None:
    ap = argparse.ArgumentParser(description="cost-wall harness")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--source", choices=list(_LOADERS), default="5m")
    ap.add_argument("--signals", nargs="+", required=True,
                    help=f"из {list(REGISTRY)}")
    ap.add_argument("--horizons", nargs="+", type=int, required=True)
    ap.add_argument("--cost", type=float, default=40.0, help="round-trip bps (база)")
    ap.add_argument("--slip-coef", nargs="+", type=float, default=[0.0],
                    help="vol-slippage: cost=база+coef*vol_bps. Несколько → свип "
                         "(breakeven). 0 → плоский кост (легаси, побитово). Анкер k≈0.05-0.1")
    ap.add_argument("--vol-window", type=int, default=20,
                    help="окно vol-прокси (ATR/std) для vol-slippage, в барах")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--quantile", type=float, default=0.80, help="порог входа |signal|")
    ap.add_argument("--window", type=int, default=None,
                    help="override окна сигнала (в барах источника); иначе дефолт сигнала")
    ap.add_argument("--resample", type=str, default=None, metavar="RULE",
                    help="агрегировать в старший ТФ перед прогоном (pandas: 1h/4h/1d). "
                         "Форекс не 24/7 → выходные бины отбрасываются")
    ap.add_argument("--dsr", action="store_true",
                    help="batch Deflated Sharpe: дефлировать ЛУЧШИЙ Sharpe всего грида "
                         "(всех signal×horizon×slip) по числу испытаний N (de Prado)")
    ap.add_argument("--permute", type=int, default=0, metavar="B",
                    help="permutation p-value для net (B circular-shift перестановок); "
                         "диагностика рядом с t, вердикт не меняет. 0 = выкл. На больших "
                         "данных (H=12, 150k баров) бери B поменьше (50-100) — O(n) на перестановку")
    ap.add_argument("--perm-seed", type=int, default=0, help="seed permutation (воспроизводимость)")
    ap.add_argument("--wf", action="store_true",
                    help="rolling Walk-Forward (отдельный вердикт, реоптимизация порога). "
                         "ОПЦИЯ, НЕ смешивается с 50/50 split")
    ap.add_argument("--wf-train", type=int, default=1000, help="размер train-окна WFA (бары)")
    ap.add_argument("--wf-test", type=int, default=250, help="размер test-окна WFA (бары)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    df = _LOADERS[args.source](args.symbol)
    src_tag = args.source
    if args.resample:
        df = datamod.resample_ohlc(df, args.resample)
        src_tag = f"{args.source}@{args.resample}"   # ТФ в метке источника
    have = [c for c in df.columns]
    print(f"{args.symbol} [{src_tag}] rows={len(df)} "
          f"span={df.index.min()} -> {df.index.max()}")
    print(f"cols={have}\n")

    results = []
    for sig_name in args.signals:
        if sig_name not in REGISTRY:
            print(f"  ! неизвестный сигнал: {sig_name}")
            continue
        sig = REGISTRY[sig_name](df, window=args.window) if args.window else REGISTRY[sig_name](df)
        for H in args.horizons:
            for slip in args.slip_coef:
                if slip == 0.0:
                    cseries, src = None, src_tag            # плоский кост (легаси)
                else:
                    cseries = vol_scaled_cost(
                        df, base_cost_bps=args.cost,
                        slip_coef=slip, vol_window=args.vol_window,
                    )
                    src = f"{src_tag}/vol(s={slip:g},w={args.vol_window})"
                res = evaluate(
                    df, sig, horizon=H, cost_bps=args.cost,
                    label=f"{args.symbol}:{sig_name}", source=src,
                    train_frac=args.train_frac, trade_quantile=args.quantile,
                    cost_series=cseries,
                )
                results.append(res)
                print(
                    f"  {sig_name:12s} H={H:<4d} slip={slip:<4g} "
                    f"n_eff={res.n_eff:<5d} trades={res.n_trades:<5d} AUC={res.auc_test} "
                    f"move_med={res.move_bps_med}bp gross={res.gross_bps}bp "
                    f"eff_cost={res.cost_bps}bp net={res.net_mean_bps}bp "
                    f"t={res.net_t_stat} maxDD={res.max_dd_bps}bp "
                    f"sub_pos={res.sub_pos} -> {res.verdict}"
                )
                if args.permute > 0:
                    from research.permutation import permutation_pvalue
                    pr = permutation_pvalue(
                        df, sig, horizon=H, cost_bps=args.cost,
                        label=f"{args.symbol}:{sig_name}", source=src,
                        train_frac=args.train_frac, trade_quantile=args.quantile,
                        cost_series=cseries, n_perm=args.permute, seed=args.perm_seed,
                    )
                    print(
                        f"        perm B={pr.n_perm} obs_net={pr.obs_net_bps}bp "
                        f"null_mean={pr.null_mean_bps}bp null_p95={pr.null_p95_bps}bp "
                        f"p={pr.p_value} -> {pr.verdict}"
                    )
            if args.wf:    # отдельно от slip-цикла: WFA на плоском косте, свой вердикт
                from research.walk_forward import walk_forward
                wf = walk_forward(
                    df, sig, horizon=H, cost_bps=args.cost,
                    label=f"{args.symbol}:{sig_name}", source=src_tag,
                    train_size=args.wf_train, test_size=args.wf_test,
                )
                print(
                    f"        WFA folds={wf.n_folds} oos_trades={wf.oos_n_trades} "
                    f"oos_AUC={wf.oos_auc} oos_net={wf.oos_net_bps}bp oos_t={wf.oos_t} "
                    f"is_net={wf.is_net_bps}bp wf_eff={wf.wf_efficiency} -> {wf.verdict}"
                )

    if args.dsr and results:
        from research.dsr import deflated_sharpe
        rep = deflated_sharpe(results)
        print(
            f"\n[DSR batch] trials N={rep.n_trials} best={rep.best_label} "
            f"H={rep.best_horizon} SR*={rep.sr_best} SR0(exp_max)={rep.sr_expected_max} "
            f"T*={rep.t_best} DSR={rep.dsr} -> {rep.verdict}"
        )

    if not args.no_save and results:
        append_results(results)
        print(f"\nwritten {len(results)} rows -> research/RESULTS.md")


if __name__ == "__main__":
    main()

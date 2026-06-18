"""
research/walk_forward.py
========================
Rolling Walk-Forward Analysis — ОПЦИЯ с ОТДЕЛЬНЫМ вердиктом (НЕ смешивать с 50/50
causal-split из costwall.evaluate; решено с юзером). Проверяет: держится ли edge при
ПОВТОРНОЙ реоптимизации параметра сквозь время (защита от distribution shift, bias №4).

Механика (Pardo / de Prado):
  Скользящие окна: train[i, i+train_size) → test[i+train_size, +test_size), шаг = test_size
  (OOS-куски НЕ перекрываются). В каждом окне:
    1. Знак ориентируется по train-корреляции (как в evaluate).
    2. РЕОПТИМИЗАЦИЯ ПОРОГА: из сетки квантилей берём тот, что даёт лучший train-net
       (одна степень свободы — порог входа; горизонт H фиксирован параметром).
    3. Выбранный порог применяется на следующий test-кусок → OOS-сделки.
  Все OOS-куски склеиваются → общий OOS net / AUC / t.

WF-efficiency = mean(OOS net) / mean(IS net). Близко к 1 → edge держится вне выборки;
сильно ниже 0.5 → подгонка под train (IS красив, OOS разваливается). Считается только
при IS net > 0 (иначе деление бессмысленно — сигнал убыточен даже in-sample).

Вердикт (ОТДЕЛЬНЫЙ от costwall; планка ROBUST НЕ ниже costwall TRADEABLE):
  WF_INCONCLUSIVE : фолдов < 3 ИЛИ OOS-сделок < MIN_TRADES ИЛИ AUC=nan
  WF_ROBUST       : OOS net>0 И AUC>0.55 И oos_t>=2 И WF-efficiency>=0.5
                    (без t/AUC шум при дешёвом косте давал бы ложный ROBUST)
  WF_DEGRADED     : иначе (OOS развалился, незначим или efficiency низкая)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.costwall import AUC_MIN, MIN_T, MIN_TRADES, _auc, _nonoverlap_mask

WF_MIN_FOLDS = 3
WF_EFF_MIN = 0.5


@dataclass
class WFReport:
    label: str
    source: str
    horizon: int
    n_folds: int
    oos_n_trades: int
    oos_net_bps: float      # средний net OOS-сделки (склейка всех test-кусков)
    oos_auc: float          # directional skill на склеенном OOS
    oos_t: float            # t-стат OOS net
    is_net_bps: float       # средний IS net по фолдам (при выбранном пороге)
    wf_efficiency: float    # oos_net / is_net (робастность реоптимизации)
    verdict: str


def _net_for_threshold(s_o: np.ndarray, r: np.ndarray, thr: float,
                       horizon: int, cost) -> tuple[np.ndarray | None, int]:
    """Net непересекающихся сделок выше порога. cost — скаляр или per-bar массив."""
    take = np.abs(s_o) >= thr
    take = _nonoverlap_mask(take, horizon)
    n = int(take.sum())
    if n == 0:
        return None, 0
    dir_taken = np.sign(s_o[take])
    c = cost[take] if isinstance(cost, np.ndarray) else cost
    net = dir_taken * r[take] * 1e4 - c
    return net, n


def walk_forward(
    df: pd.DataFrame,
    signal_raw: pd.Series,
    *,
    horizon: int,
    cost_bps: float,
    label: str,
    source: str,
    train_size: int = 1000,
    test_size: int = 250,
    quantiles: tuple[float, ...] = (0.70, 0.80, 0.90),
    cost_series: pd.Series | None = None,
) -> WFReport:
    """Rolling WFA с реоптимизацией порога. См. модульный docstring."""
    close = df["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0
    valid = signal_raw.notna() & fwd.notna() & np.isfinite(signal_raw)
    if cost_series is not None:
        cost_series = cost_series.reindex(df.index)
        valid = valid & cost_series.notna()
    s = signal_raw[valid].astype(float).values
    r = fwd[valid].astype(float).values
    cv = cost_series[valid].astype(float).values if cost_series is not None else None
    n = len(s)

    is_net_chunks: list[np.ndarray] = []   # train-net выбранного порога (pooled, как OOS)
    oos_net_chunks: list[np.ndarray] = []
    oos_s_chunks: list[np.ndarray] = []
    oos_r_chunks: list[np.ndarray] = []
    folds = 0
    start = 0
    while start + train_size + test_size <= n:
        tr = slice(start, start + train_size)
        te = slice(start + train_size, start + train_size + test_size)
        s_tr, r_tr = s[tr], r[tr]
        s_te, r_te = s[te], r[te]
        c = np.corrcoef(s_tr, r_tr)[0, 1]
        sign = 1 if (np.isnan(c) or c >= 0) else -1
        s_tr_o, s_te_o = s_tr * sign, s_te * sign
        cost_tr = cv[tr] if cv is not None else cost_bps
        cost_te = cv[te] if cv is not None else cost_bps

        # реоптимизация порога: argmax train-net по сетке квантилей
        best_is = best_thr = best_net_tr = None
        for q in quantiles:
            thr = float(np.nanquantile(np.abs(s_tr_o), q))
            net_tr, nt = _net_for_threshold(s_tr_o, r_tr, thr, horizon, cost_tr)
            if net_tr is None or nt < MIN_TRADES // 2:   # не выбирать порог по 3 удачным
                continue
            m = float(net_tr.mean())
            if best_is is None or m > best_is:
                best_is, best_thr, best_net_tr = m, thr, net_tr
        if best_thr is None:
            start += test_size
            continue

        net_te, _ = _net_for_threshold(s_te_o, r_te, best_thr, horizon, cost_te)
        is_net_chunks.append(best_net_tr)   # pooled trade-weighted (согласовано с OOS)
        if net_te is not None:
            oos_net_chunks.append(net_te)
        oos_s_chunks.append(s_te_o)     # знак уже применён → корректный склеенный AUC
        oos_r_chunks.append(r_te)
        folds += 1
        start += test_size

    nan = float("nan")
    if folds < WF_MIN_FOLDS:
        return WFReport(label, source, horizon, folds, 0, nan, nan, nan, nan, nan,
                        "WF_INCONCLUSIVE")

    oos_net = np.concatenate(oos_net_chunks) if oos_net_chunks else np.array([])
    oos_n = len(oos_net)
    oos_s_all = np.concatenate(oos_s_chunks)
    oos_r_all = np.concatenate(oos_r_chunks)
    oos_auc = _auc(oos_s_all, (oos_r_all > 0).astype(int))

    if oos_n >= 2:
        m = float(oos_net.mean())
        sd = oos_net.std(ddof=1)
        oos_t = float(m / (sd / np.sqrt(oos_n))) if sd > 0 else nan
    else:
        m = oos_t = nan

    is_all = np.concatenate(is_net_chunks) if is_net_chunks else np.array([])
    is_mean = float(is_all.mean()) if len(is_all) else nan      # pooled, как OOS
    wf_eff = float(m / is_mean) if (is_mean and is_mean > 0 and not np.isnan(m)) else nan

    # Гейт WF_ROBUST НЕ ниже costwall TRADEABLE: net>0 И значим (t) И skill (AUC) И
    # держится OOS (efficiency). Без t/AUC шум при дешёвом косте даёт ложный ROBUST.
    if oos_n < MIN_TRADES or np.isnan(oos_auc):
        verdict = "WF_INCONCLUSIVE"
    elif (not np.isnan(m) and m > 0 and oos_auc > AUC_MIN
          and not np.isnan(oos_t) and oos_t >= MIN_T
          and not np.isnan(wf_eff) and wf_eff >= WF_EFF_MIN):
        verdict = "WF_ROBUST"
    else:
        verdict = "WF_DEGRADED"

    return WFReport(
        label=label, source=source, horizon=horizon, n_folds=folds,
        oos_n_trades=oos_n,
        oos_net_bps=round(m, 3) if not np.isnan(m) else nan,
        oos_auc=round(oos_auc, 4) if not np.isnan(oos_auc) else nan,
        oos_t=round(oos_t, 2) if not np.isnan(oos_t) else nan,
        is_net_bps=round(is_mean, 3),
        wf_efficiency=round(wf_eff, 3) if not np.isnan(wf_eff) else nan,
        verdict=verdict,
    )

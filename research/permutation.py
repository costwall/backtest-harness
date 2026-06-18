"""
research/permutation.py
=======================
Permutation significance для net_mean — distribution-free p-value, УВАЖАЮЩИЙ
автокорреляцию (в отличие от t-гейта). Диагностика (решено с юзером 2026-06-18):
репортится РЯДОМ с t, вердикт НЕ меняет (остаётся воспроизводимым, не зависит от
RNG/seed). Параллель DSR — отдельный честный инструмент, без двойного счёта.

Зачем не t: t-стат на непересекающихся сделках предполагает iid-нормальность. AUC же
считается на ПЕРЕКРЫВАЮЩИХСЯ барах (автокорреляция), и сделочные net'ы фат-тейлы/skew.
Permutation строит нулевое распределение статистики напрямую из данных.

Нуль (H0): «сигнал не выровнен с будущими доходностями». Реализация — CIRCULAR SHIFT
сигнала на случайный оффсет: ломает выравнивание signal↔returns, но СОХРАНЯЕТ автокорр.
обоих рядов (обычный shuffle её разрушил бы → слишком оптимистичный p). Каждый сдвинутый
сигнал прогоняется через ТОТ ЖЕ evaluate (split, ориентация знака на train, порог,
непересечение, кост) — нуль испытывает идентичный пайплайн, включая свободу выбора знака.

p = (1 + #{null_net >= obs_net}) / (1 + B)   — one-sided (Phipson–Smyth, p>0 всегда;
obs = член группы при offset=0, под H0 все ротации обмениваемы; NaN ротируются ВМЕСТЕ
с сигналом → obs и нули делят один NaN-паттерн). Решающее правило: где t-гейт и p
расходятся — верить p (t предполагает iid-нормаль, perm уважает автокорр/fat-tails).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.costwall import MIN_TRADES, evaluate

MIN_VALID_PERM = 30      # меньше валидных перестановок → INCONCLUSIVE
P_SIGNIF = 0.05


@dataclass
class PermReport:
    label: str
    source: str
    horizon: int
    n_perm: int            # валидных перестановок (net не NaN)
    obs_net_bps: float     # наблюдённый net_mean
    null_mean_bps: float   # средний net под H0 (контроль)
    null_p95_bps: float    # 95-й перцентиль null-net (порог «случайности»)
    p_value: float         # доля null >= obs (one-sided)
    verdict: str           # SIGNIF / NS / INCONCLUSIVE


def permutation_pvalue(
    df: pd.DataFrame,
    signal_raw: pd.Series,
    *,
    horizon: int,
    cost_bps: float,
    label: str,
    source: str,
    train_frac: float = 0.5,
    trade_quantile: float = 0.80,
    cost_series: pd.Series | None = None,
    n_perm: int = 500,
    seed: int = 0,
) -> PermReport:
    """Эмпирический p-value для net_mean через circular-shift нуль. См. docstring."""
    def _net(sig: pd.Series) -> float:
        return evaluate(
            df, sig, horizon=horizon, cost_bps=cost_bps, label=label, source=source,
            train_frac=train_frac, trade_quantile=trade_quantile, cost_series=cost_series,
        ).net_mean_bps

    obs = evaluate(
        df, signal_raw, horizon=horizon, cost_bps=cost_bps, label=label, source=source,
        train_frac=train_frac, trade_quantile=trade_quantile, cost_series=cost_series,
    )
    nan = float("nan")
    if obs.n_trades < MIN_TRADES or math.isnan(obs.net_mean_bps):
        return PermReport(label, source, horizon, 0, obs.net_mean_bps,
                          nan, nan, nan, "INCONCLUSIVE")

    vals = signal_raw.values
    n = len(vals)
    lo = max(horizon, 1)
    if n - 2 * lo <= 1:
        return PermReport(label, source, horizon, 0, obs.net_mean_bps,
                          nan, nan, nan, "INCONCLUSIVE")
    rng = np.random.default_rng(seed)
    offsets = rng.integers(lo, n - lo, size=n_perm)

    nulls = []
    for off in offsets:
        shifted = pd.Series(np.roll(vals, int(off)), index=signal_raw.index)
        nv = _net(shifted)
        if not math.isnan(nv):
            nulls.append(nv)
    nulls = np.asarray(nulls, dtype=float)
    if len(nulls) < MIN_VALID_PERM:
        return PermReport(label, source, horizon, len(nulls), obs.net_mean_bps,
                          nan, nan, nan, "INCONCLUSIVE")

    ge = int((nulls >= obs.net_mean_bps).sum())
    p = (1 + ge) / (1 + len(nulls))
    verdict = "SIGNIF" if (p < P_SIGNIF and obs.net_mean_bps > 0) else "NS"
    return PermReport(
        label=label, source=source, horizon=horizon, n_perm=len(nulls),
        obs_net_bps=round(obs.net_mean_bps, 3),
        null_mean_bps=round(float(nulls.mean()), 3),
        null_p95_bps=round(float(np.percentile(nulls, 95)), 3),
        p_value=round(p, 4), verdict=verdict,
    )

"""
research/test_permutation.py
============================
Тесты circular-shift permutation. Свойства: настоящий предиктор значим (obs на вершине
нулевого распределения → p минимален), независимый шум — нет; p воспроизводим по seed.

Запуск:  python -m pytest research/test_permutation.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.permutation import permutation_pvalue


def _predictive_case(n=3000, alpha=0.001, seed=1):
    """signal[t] предсказывает знак fwd[t] (H=1): close[t+1]/close[t]-1 = alpha·sign(sig)+шум."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    step = alpha * np.sign(sig) + rng.normal(0, alpha * 0.5, n)   # сигнал → знак след. бара
    close = np.empty(n)
    close[0] = 100.0
    for t in range(n - 1):
        close[t + 1] = close[t] * (1 + step[t])
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx)
    return df, pd.Series(sig, index=idx)


def _noise_case(n=3000, seed=2):
    """signal независим от доходностей (отдельный поток RNG)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx)
    sig = pd.Series(np.random.default_rng(seed + 999).normal(size=n), index=idx)
    return df, sig


def test_real_predictor_is_significant():
    df, sig = _predictive_case()
    pr = permutation_pvalue(df, sig, horizon=1, cost_bps=1.0, label="t", source="s",
                            n_perm=300, seed=0)
    assert pr.verdict == "SIGNIF"
    assert pr.p_value < 0.05
    assert pr.obs_net_bps > pr.null_p95_bps        # наблюдённый выше 95-го перцентиля шума


def test_pure_noise_not_significant():
    df, sig = _noise_case()
    pr = permutation_pvalue(df, sig, horizon=12, cost_bps=40.0, label="t", source="s",
                            n_perm=300, seed=0)
    assert pr.verdict == "NS"
    assert pr.p_value > 0.05


def test_pvalue_bounds_and_reproducible():
    df, sig = _noise_case()
    a = permutation_pvalue(df, sig, horizon=12, cost_bps=40.0, label="t", source="s",
                           n_perm=200, seed=7)
    b = permutation_pvalue(df, sig, horizon=12, cost_bps=40.0, label="t", source="s",
                           n_perm=200, seed=7)
    assert 0.0 < a.p_value <= 1.0
    assert a.p_value == b.p_value                  # тот же seed → тот же p

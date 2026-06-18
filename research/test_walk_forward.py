"""
research/test_walk_forward.py
=============================
Тесты rolling WFA. Свойства: настоящий предиктор держится OOS (WF_ROBUST, efficiency>0),
шум разваливается (DEGRADED/INCONCLUSIVE), мало данных → INCONCLUSIVE.

Запуск:  python -m pytest research/test_walk_forward.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.walk_forward import walk_forward


def _predictive(n=6000, alpha=0.0015, seed=1):
    """signal[t] предсказывает знак fwd[t] (H=1) во ВСЁМ ряду → edge держится OOS."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    step = alpha * np.sign(sig) + rng.normal(0, alpha * 0.5, n)
    close = np.empty(n)
    close[0] = 100.0
    for t in range(n - 1):
        close[t + 1] = close[t] * (1 + step[t])
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx)
    return df, pd.Series(sig, index=idx)


def _noise(n=6000, seed=2):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx)
    sig = pd.Series(np.random.default_rng(seed + 9).normal(size=n), index=idx)
    return df, sig


def test_real_predictor_robust_oos():
    df, sig = _predictive()
    wf = walk_forward(df, sig, horizon=1, cost_bps=1.0, label="t", source="s",
                      train_size=800, test_size=200)
    assert wf.n_folds >= 3
    assert wf.verdict == "WF_ROBUST"
    assert wf.oos_net_bps > 0
    assert wf.wf_efficiency > 0          # OOS не развалился относительно IS


def test_noise_not_robust():
    df, sig = _noise()
    wf = walk_forward(df, sig, horizon=12, cost_bps=40.0, label="t", source="s",
                      train_size=800, test_size=200)
    assert wf.verdict in ("WF_DEGRADED", "WF_INCONCLUSIVE")
    assert not (wf.verdict == "WF_ROBUST")


def test_noise_cheap_cost_not_false_robust():
    """Шум при ДЕШЁВОМ косте (форекс-режим): даже если OOS net случайно >0 — НЕ ROBUST
    без значимости (t/AUC-гейт). Защита от ложного ROBUST в честном продукте."""
    df, sig = _noise(seed=5)
    wf = walk_forward(df, sig, horizon=12, cost_bps=0.0, label="t", source="s",
                      train_size=800, test_size=200)
    assert wf.verdict != "WF_ROBUST"     # шум не проходит, даже при cost=0


def test_insufficient_folds_inconclusive():
    df, sig = _noise(n=1200)
    wf = walk_forward(df, sig, horizon=12, cost_bps=40.0, label="t", source="s",
                      train_size=1000, test_size=250)   # помещается < 3 фолдов
    assert wf.verdict == "WF_INCONCLUSIVE"

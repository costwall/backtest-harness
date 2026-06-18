"""
research/test_costwall.py
=========================
Регресс-тесты cost-wall оценщика. Главный инвариант трека A (vol-slippage):

  slip_coef=0  ⇒  vol-scaled путь ПОБИТОВО совпадает с плоским костом.

Без этого «честный кост» нечестен: нельзя добавлять проскальзывание, пока
сигнал не умрёт. Плоская модель должна быть точным частным случаем vol-модели.

Запуск:  python -m pytest research/test_costwall.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.costwall import evaluate, vol_scaled_cost


def _toy_ohlc(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.002, n)
    close = 100 * np.cumprod(1 + ret)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    hi = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    df = pd.DataFrame({"open": close, "high": hi, "low": lo, "close": close}, index=idx)
    sig = pd.Series(rng.normal(0, 1, n), index=idx)
    return df, sig


def test_slip0_identical_to_flat():
    """slip_coef=0 не должен менять ни выборку, ни net, ни gross."""
    df, sig = _toy_ohlc()
    flat = evaluate(df, sig, horizon=12, cost_bps=40, label="t", source="s")
    cs0 = vol_scaled_cost(df, base_cost_bps=40, slip_coef=0.0, vol_window=20)
    vol0 = evaluate(df, sig, horizon=12, cost_bps=40, label="t", source="s", cost_series=cs0)
    assert flat.n_trades == vol0.n_trades
    assert flat.net_mean_bps == vol0.net_mean_bps
    assert flat.gross_bps == vol0.gross_bps
    assert flat.cost_bps == vol0.cost_bps


def test_gross_is_breakeven_cost():
    """gross_bps = net + плоский кост (breakeven round-trip cost)."""
    df, sig = _toy_ohlc()
    r = evaluate(df, sig, horizon=12, cost_bps=40, label="t", source="s")
    assert abs(r.gross_bps - (r.net_mean_bps + 40)) < 1e-6


def test_slip_raises_cost_same_sample():
    """slip>0: тот же набор сделок, но эффективный кост выше и net ниже."""
    df, sig = _toy_ohlc()
    flat = evaluate(df, sig, horizon=12, cost_bps=40, label="t", source="s")
    cs = vol_scaled_cost(df, base_cost_bps=40, slip_coef=0.5, vol_window=20)
    vol = evaluate(df, sig, horizon=12, cost_bps=40, label="t", source="s", cost_series=cs)
    assert vol.n_trades == flat.n_trades          # warmup→base, выборка не плывёт
    assert vol.cost_bps > flat.cost_bps
    assert vol.net_mean_bps < flat.net_mean_bps


def test_maxdd_nonnegative_and_deepened_by_cost():
    """MaxDD >= 0; рост костов (vol-slippage) не уменьшает просадку net-кривой."""
    df, sig = _toy_ohlc()
    flat = evaluate(df, sig, horizon=12, cost_bps=40, label="t", source="s")
    cs = vol_scaled_cost(df, base_cost_bps=40, slip_coef=0.5, vol_window=20)
    vol = evaluate(df, sig, horizon=12, cost_bps=40, label="t", source="s", cost_series=cs)
    assert flat.max_dd_bps >= 0.0
    assert vol.max_dd_bps >= flat.max_dd_bps     # выше кост → глубже (или равная) просадка


def test_subperiod_consistency_all_positive():
    """Сигнал со стабильно положительным net во всех периодах → sub_pos=1.0."""
    # close растёт когда sig>0 на след. баре → последовательный edge во всём test-периоде
    rng = np.random.default_rng(3)
    n = 4000
    sig = rng.normal(size=n)
    step = 0.002 * np.sign(sig) + rng.normal(0, 0.0005, n)
    close = np.empty(n)
    close[0] = 100.0
    for t in range(n - 1):
        close[t + 1] = close[t] * (1 + step[t])
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"close": close}, index=idx)
    r = evaluate(df, pd.Series(sig, index=idx), horizon=1, cost_bps=1.0, label="t", source="s")
    assert r.sub_pos == 1.0                  # все 4 чанка прибыльны


def test_subperiod_one_chunk_carries():
    """Edge только в первой четверти сделок → sub_pos = 0.25 (один период вынес)."""
    n = 4000
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(4)
    sig = rng.normal(size=n)
    # предсказательная связь ТОЛЬКО в первой трети ряда, дальше — чистый шум
    step = rng.normal(0, 0.0005, n)
    first = n // 3
    step[:first] += 0.003 * np.sign(sig[:first])
    close = np.empty(n)
    close[0] = 100.0
    for t in range(n - 1):
        close[t + 1] = close[t] * (1 + step[t])
    df = pd.DataFrame({"close": close}, index=idx)
    r = evaluate(df, pd.Series(sig, index=idx), horizon=1, cost_bps=1.0, label="t", source="s")
    assert 0.0 <= r.sub_pos <= 0.5           # большинство периодов НЕ прибыльны


def test_vol_cost_is_causal():
    """cost[t] не должен зависеть от будущих баров (сдвиг будущего не меняет cost[t])."""
    df, sig = _toy_ohlc()
    base = vol_scaled_cost(df, base_cost_bps=40, slip_coef=0.5, vol_window=20)
    df2 = df.copy()
    df2.iloc[2000:] *= 1.5            # возмущаем ТОЛЬКО будущее (с бара 2000)
    pert = vol_scaled_cost(df2, base_cost_bps=40, slip_coef=0.5, vol_window=20)
    # cost на барах входа < 2000 не должен измениться (vol через shift(1), окно в прошлом)
    assert np.allclose(base.values[:1999], pert.values[:1999], equal_nan=True)

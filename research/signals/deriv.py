"""
research/signals/deriv.py
=========================
Сигналы из деривативного позиционирования Binance (liquidity_bot):
funding_rate, open_interest, long/short ratio (global+top), volume_delta.

Экономический приор (почему это НЕ та же стена, что order-flow):
  - Медленные сигналы (funding 8ч, LSR 5мин, OI минуты) → горизонт часы-дни →
    предсказываемое движение многопроцентное, costs 32-52bp = малая доля.
  - extreme funding / crowded LSR → mean-reversion (толпа на одной стороне).
  - top-trader vs global divergence → «умные деньги» против толпы.
  - OI rate-of-change → подтверждение/исчерпание тренда.

ВСЕ функции считают балл строго на прошлом (rolling). Знак ориентирует costwall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_z(s: pd.Series, window: int) -> pd.Series:
    """Causal z-score: (x - rolling_mean) / rolling_std, по прошлому окну."""
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    return (s - mean) / std.replace(0.0, np.nan)


def funding_z(df: pd.DataFrame, window: int = 480) -> pd.Series:
    """Z-score funding_rate. Приор: высокий funding = перекос в лонг = ждём вниз
    (costwall сам развернёт знак по train). window в барах."""
    if "funding_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _rolling_z(df["funding_rate"].astype(float), window)


def lsr_extreme(df: pd.DataFrame, window: int = 288) -> pd.Series:
    """Z-score global long/short ratio. Приор: толпа перекошена → контртренд."""
    if "lsr" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _rolling_z(df["lsr"].astype(float), window)


def smart_money(df: pd.DataFrame, window: int = 288) -> pd.Series:
    """Дивергенция top-trader vs global crowd: z(top_lsr) - z(global_lsr).
    Приор: топ-трейдеры против толпы = идём за топами."""
    if "top_lsr" not in df.columns or "lsr" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _rolling_z(df["top_lsr"].astype(float), window) - _rolling_z(df["lsr"].astype(float), window)


def oi_roc(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """Rate-of-change open interest (causal). Приор: рост OI = приток позиций →
    подтверждение направления (costwall свяжет с движением)."""
    if "open_interest" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    oi = df["open_interest"].astype(float)
    return oi.pct_change(window)


def cvd(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Кумулятивная дельта объёма (rolling sum volume_delta), z-score.
    Приор: устойчивое давление покупателей/продавцов → продолжение."""
    if "volume_delta" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    roll = df["volume_delta"].astype(float).rolling(window, min_periods=window // 2).sum()
    return _rolling_z(roll, window * 5)

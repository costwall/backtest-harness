"""
research/signals/cot.py
=======================
Сигналы из COT-позиционирования (df с колонкой pct_net = net non-commercial как %OI).
Все causal (rolling по прошлому). Знак ориентирует costwall по train.

Приоры:
  - cot_z / cot_index: ЭКСТРЕМУМ позиционирования → контртренд (толпа спекулянтов
    перекошена → разворот).
  - cot_chg: MOMENTUM позиционирования (приток/отток спека) → подтверждение тренда.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cot_z(df: pd.DataFrame, window: int = 156) -> pd.Series:
    """Z-score net %OI по rolling-окну (156 нед ≈ 3 года). Экстремум → контртренд."""
    s = df["pct_net"].astype(float)
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    return (s - mean) / std.replace(0.0, np.nan)


def cot_index(df: pd.DataFrame, window: int = 156) -> pd.Series:
    """COT index: перцентиль текущего net в rolling-окне, центрирован к [-0.5,0.5].
    +0.5 = исторический максимум спек-лонга (crowded), -0.5 = максимум шорта."""
    s = df["pct_net"].astype(float)
    pr = s.rolling(window, min_periods=window // 2).apply(
        lambda a: (a <= a[-1]).mean(), raw=True)
    return pr - 0.5


def cot_chg(df: pd.DataFrame, window: int = 4) -> pd.Series:
    """Изменение net %OI за window недель (momentum позиционирования)."""
    return df["pct_net"].astype(float).diff(window)

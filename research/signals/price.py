"""
research/signals/price.py
=========================
Чисто ЦЕНОВЫЕ сигналы (только OHLC, без deriv/flow) — чтобы харнесс мог гонять
клиентские стратегии и проверять тезис «старший ТФ / дешёвый форекс чище».

⚠ Контекст честности (см. память):
  - `mom` (моментум/тренд) — directional на цене, УЖЕ мёртв 4 методами на крипте
    ([[project-status]] «НЕ переоткрывать RSI/directional на OHLC»). Здесь допустим
    ТОЛЬКО как узкое НОВОЕ условие: дешёвый-по-спреду форекс / старший ТФ
    ([[wma-macd-rejected]], [[intraday-mtf-rejected]]) — не свежая price-TA охота.
  - `zscore` (mean-reversion) — защитимее: эффект РЕАЛЕН но крошечный (~0.52, ниже
    порога 0.55, [[meanrev-faint-subthreshold]]). Не удивляться повтору.

Главный readout эксперимента — AUC-vs-ТФ (directional skill), НЕ net: старший ТФ и
дешёвый кост улучшают только cost-wall, а не skill. Сдвиг AUC с ~0.5 = реальный лид;
лучший net при AUC≈0.5 = просто меньше cost-drag.

ВСЕ функции считают балл строго на прошлом (rolling). Знак ориентирует costwall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mom(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """Моментум: доходность за прошедшие `window` баров. Приор: тренд продолжается
    (costwall развернёт знак по train, если это контртренд). window в барах ТФ."""
    close = df["close"].astype(float)
    return close / close.shift(window) - 1.0


def zscore(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """Mean-reversion: causal z-score цены относительно скользящего среднего.
    Приор: |z| велик → растяжка от среднего → ждём возврат (знак — на train)."""
    close = df["close"].astype(float)
    mean = close.rolling(window, min_periods=window // 2).mean()
    std = close.rolling(window, min_periods=window // 2).std()
    return (close - mean) / std.replace(0.0, np.nan)

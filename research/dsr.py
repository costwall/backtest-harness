"""
research/dsr.py
===============
Deflated Sharpe Ratio (Bailey & López de Prado, 2014) — BATCH-уровневая защита от
data-mining'а. Отвечает на ОДИН вопрос: ты перебрал N конфигураций и взял лучшую по
Sharpe — значим ли этот максимум С ПОПРАВКОЙ на то, что выбирал из N?

Почему batch, а не per-row (решено с юзером 2026-06-18): DSR дефлирует МАКСИМУМ Sharpe
по числу испытаний. Это selection-метрика на НАБОРЕ, не на одной строке. Вешать её
per-row + наш t-гейт/n_eff/pre-reg = двойной штраф за multiple-testing. Поэтому DSR —
отдельный инструмент для exploratory-фазы (свип signal×horizon×slip×quantile в одном
прогоне run.py), pre-reg+t-гейт остаются для финального claim.

Метод (de Prado):
  SR_i           = t_i / sqrt(T_i)          — Sharpe на наблюдение для конфигурации i
  N              = число валидных конфигураций (испытаний)
  Var(SR)        = кросс-секционная дисперсия оценок SR_i по N
  SR0 (порог)    = sqrt(Var(SR)) · [(1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e))]   γ=0.5772…
                   — ОЖИДАЕМЫЙ максимум Sharpe под H0 (все истинные SR=0) при N испытаниях
  DSR            = Φ( (SR*-SR0)·sqrt(T*-1) / sqrt(1 - skew*·SR* + ((kurt*-1)/4)·SR*²) )
                   где * — параметры ЛУЧШЕЙ конфигурации (max SR_i)

DSR — вероятность (0..1), что истинный Sharpe лучшей конфигурации > 0 после дефляции.
DSR ≥ 0.95 → лучший результат грида переживает поправку на размер перебора.

ЧТО СЧИТАЕТСЯ ИСПЫТАНИЕМ N (важно — иначе category error): N = число SELECTION-измерений,
среди которых ты ВЫБИРАЕШЬ лучшее (signal × horizon × quantile × window). Варианты slip
(vol-slippage) — это cost-stress, НЕ selection: больший slip строго снижает net на тех же
сделках ⇒ slip>0 НИКОГДА не argmax, его нельзя «выбрать». Поэтому slip-варианты схлопываются
(берём наименее-стресснутый = max SR на ячейку) перед подсчётом N. Иначе N раздут, а бар SR0
искажён (Var(SR) эндогенна, направление искажения даже не чистое).

Оговорка: перекрывающиеся горизонты на одном сигнале коррелированы ⇒ эффективное N < сырого N
(допущение независимости испытаний у de Prado здесь приближённое — известное ограничение).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from research.costwall import CostWallResult

_GAMMA = 0.5772156649015329          # Эйлер–Маскерони
_N = NormalDist()                    # стандартная нормаль (cdf / inv_cdf из stdlib)
DSR_MIN = 0.95                       # порог значимости лучшего результата грида


@dataclass
class DSRReport:
    n_trials: int            # N валидных конфигураций в гриде
    best_label: str          # лучшая по Sharpe конфигурация
    best_source: str
    best_horizon: int
    sr_best: float           # SR на наблюдение лучшей конфигурации
    sr_expected_max: float   # ожидаемый макс Sharpe под H0 при N испытаниях (SR0)
    t_best: int              # число сделок лучшей конфигурации
    dsr: float               # вероятность истинный SR>0 после дефляции
    verdict: str             # PASS_DSR / FAIL_DSR / INCONCLUSIVE


def _sr_per_obs(r: CostWallResult) -> float | None:
    """SR на наблюдение. Предпочитает неокруглённый r.sharpe; иначе t/sqrt(T)."""
    if not math.isnan(r.sharpe):
        return r.sharpe
    if r.n_trades < 2 or r.net_t_stat is None or math.isnan(r.net_t_stat):
        return None
    return r.net_t_stat / math.sqrt(r.n_trades)


def _selection_cell(r: CostWallResult) -> tuple:
    """Ключ selection-ячейки: source без slip-суффикса '/vol(...)'. Варианты slip —
    cost-stress, не отдельные испытания, поэтому схлопываются в одну ячейку."""
    base_source = r.source.split("/vol")[0]
    return (r.label, base_source, r.horizon)


def deflated_sharpe(results: list[CostWallResult]) -> DSRReport:
    """Дефлировать лучший Sharpe грида по числу испытаний. См. модульный docstring."""
    pairs = [(r, _sr_per_obs(r)) for r in results]
    valid = [(r, sr) for r, sr in pairs if sr is not None]
    # схлопнуть slip-варианты: на selection-ячейку оставить max SR (наименьший кост)
    best_per_cell: dict[tuple, tuple] = {}
    for r, sr in valid:
        key = _selection_cell(r)
        if key not in best_per_cell or sr > best_per_cell[key][1]:
            best_per_cell[key] = (r, sr)
    valid = list(best_per_cell.values())
    n = len(valid)
    if n < 2:
        return DSRReport(n, "", "", 0, float("nan"), float("nan"), 0,
                         float("nan"), "INCONCLUSIVE")  # нет дисперсии испытаний

    srs = [sr for _, sr in valid]
    mean_sr = sum(srs) / n
    var_sr = sum((x - mean_sr) ** 2 for x in srs) / (n - 1)   # кросс-секц. дисперсия SR
    best, sr_best = max(valid, key=lambda t: t[1])

    if var_sr <= 0:
        return DSRReport(n, best.label, best.source, best.horizon, round(sr_best, 4),
                         float("nan"), best.n_trades, float("nan"), "INCONCLUSIVE")

    # SR0: ожидаемый максимум из N независимых нулевых испытаний (de Prado)
    sr0 = math.sqrt(var_sr) * (
        (1 - _GAMMA) * _N.inv_cdf(1 - 1.0 / n)
        + _GAMMA * _N.inv_cdf(1 - 1.0 / (n * math.e))
    )

    # знаменатель: поправка на не-нормальность net-распределения лучшей конфигурации
    denom = math.sqrt(max(
        1e-12,
        1 - best.skew * sr_best + ((best.kurt - 1) / 4.0) * sr_best ** 2,
    ))
    dsr = _N.cdf((sr_best - sr0) * math.sqrt(best.n_trades - 1) / denom)

    verdict = "PASS_DSR" if dsr >= DSR_MIN else "FAIL_DSR"
    return DSRReport(
        n_trials=n, best_label=best.label, best_source=best.source,
        best_horizon=best.horizon, sr_best=round(sr_best, 4),
        sr_expected_max=round(sr0, 4), t_best=best.n_trades,
        dsr=round(dsr, 4), verdict=verdict,
    )

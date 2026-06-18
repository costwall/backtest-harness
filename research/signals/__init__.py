"""
research/signals/ — реестр сигналов-кандидатов.

Каждый сигнал = чистая функция (df, **params) -> pd.Series, считающая БАЛЛ
строго на прошлом (rolling/expanding, никакого .shift(-k) внутрь). Знак не важен:
costwall ориентирует его по train. Возвращай NaN там, где данных не хватает.
"""
from research.signals import deriv, cot, price

# имя -> функция. CLI выбирает по имени.
REGISTRY = {
    "funding_z": deriv.funding_z,
    "lsr_extreme": deriv.lsr_extreme,
    "smart_money": deriv.smart_money,
    "oi_roc": deriv.oi_roc,
    "cvd": deriv.cvd,
    "cot_z": cot.cot_z,
    "cot_index": cot.cot_index,
    "cot_chg": cot.cot_chg,
    "mom": price.mom,           # ценовой моментум/тренд (только OHLC)
    "zscore": price.zscore,     # ценовой mean-reversion (только OHLC)
}

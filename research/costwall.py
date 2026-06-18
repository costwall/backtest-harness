"""
research/costwall.py
====================
ЕДИНСТВЕННЫЙ оценщик «есть ли торгуемый edge». Один вопрос ко всем сигналам:
предсказывает ли сигнал знак будущего движения (AUC) И бьёт ли это движение
round-trip costs (net > 0)?

Методология (anti-bias, CLAUDE.md §bias-prevention):
  1. fwd_return[t] = close[t+H]/close[t]-1  — будущее ТОЛЬКО как метка.
  2. Сигнал считается строго на прошлом (за это отвечает функция сигнала).
  3. Causal split: первые train_frac — train, остаток — test (по времени, не shuffle).
  4. Знак сигнала ориентируется по КОРРЕЛЯЦИИ НА TRAIN (не на test) — иначе
     утечка: «развернём знак как удобно» = подгонка под тест.
  5. Порог входа |signal| берётся как квантиль |signal| НА TRAIN.
  6. Net P&L считается на НЕПЕРЕСЕКАЮЩИХСЯ сделках (шаг H) — перекрытие окон
     раздувает значимость (автокорреляция), занижает честный t-stat.
  7. drift_bps (безусловное среднее fwd_ret на test) — контроль: не ловим ли мы
     просто бычий/медвежий дрейф вместо edge.

Эффективная выборка (КРИТИЧНО): AUC считается на ПЕРЕКРЫВАЮЩИХСЯ барах, поэтому
число независимых наблюдений ≈ n_test / horizon = n_eff, а НЕ n_test. AUC=0.64 на
n_eff=9 и AUC=0.49 на n_eff=12000 — разные эпистемические состояния. Поэтому
n_eff < MIN_EFF → INCONCLUSIVE (недостаточно данных), не путать с NO_EDGE
(definitively dead на мощной выборке).

Вердикт (порядок проверки важен):
  INCONCLUSIVE : auc=nan ИЛИ n_eff < MIN_EFF  (недостаточно независимых наблюдений)
  NO_EDGE      : auc_test <= AUC_MIN           (мощно: направление = монетка)
  TRADEABLE    : auc_test > AUC_MIN  И  net_mean_bps > 0  И  n_trades >= MIN_TRADES
  SUB_COST     : auc_test > AUC_MIN  но net_mean_bps <= 0 (сигнал реален, costs съедают)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AUC_MIN = 0.55          # порог «направление лучше монетки» (как в meanrev-памяти)
MIN_TRADES = 30         # ниже — net-выборка несерьёзна
MIN_EFF = 30            # n_eff = n_test//horizon ниже этого → INCONCLUSIVE (не NO_EDGE)
MIN_T = 2.0             # гейт значимости net: без него TRADEABLE штампуется из шума (t=1.64)
SUBPERIODS = 4          # K хронологических чанков сделок для проверки консистентности («кварталы»)
RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


@dataclass
class CostWallResult:
    label: str
    source: str
    horizon: int
    cost_bps: float
    n_total: int          # валидных (signal & fwd не NaN) баров
    n_test: int           # баров в test-половине
    n_eff: int            # ~независимых наблюдений = n_test // horizon (мощность AUC)
    n_trades: int         # непересекающихся сделок выше порога на test
    auc_test: float       # directional skill на test (0.5 = монетка)
    sign: int             # +1/-1 ориентация, выбранная на train
    move_bps_med: float   # медиана |fwd_ret| на сделках (bps)
    move_bps_p90: float
    gross_bps: float      # средний gross = dir*fwd_ret (bps), БЕЗ костов;
    #                     # = breakeven round-trip cost (сигнал жив пока cost < gross)
    net_mean_bps: float   # средний net на сделку = dir*fwd_ret - cost (bps)
    net_t_stat: float     # t-стат net (грубая значимость)
    drift_bps: float      # безусловный средний fwd_ret на test (контроль дрейфа)
    max_dd_bps: float     # глубочайшая peak→trough просадка equity-кривой net-сделок (bps)
    verdict: str
    skew: float = 0.0     # асимметрия net-распределения сделок (вход для batch-DSR)
    kurt: float = 3.0     # куртозис net (raw, нормаль=3; вход для batch-DSR)
    sharpe: float = float("nan")  # SR на наблюдение = net_mean/sd (неокруглён, для DSR)
    sub_pos: float = float("nan")  # доля из K=4 хронологич. чанков сделок с net>0
    #                              # (консистентность: 1.0=все периоды плюс, 0.25=один вынес)

    def as_row(self) -> dict:
        return asdict(self)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC через ранги (Mann-Whitney U). Без sklearn-зависимости.

    labels: 1 = «движение вверх». Возвращает P(score(up) > score(down))."""
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # средний ранг для ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _vol_bps(df: pd.DataFrame, window: int) -> pd.Series:
    """Каузальный прокси волатильности в bps цены (rolling).

    Если есть high/low — ATR-стиль (mean True Range за window, в bps close).
    Иначе — rolling-std close-to-close доходностей. Возвращает Series по индексу df
    (NaN в начале, до накопления окна).
    """
    close = df["close"].astype(float)
    if "high" in df.columns and "low" in df.columns:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        tr_bps = (tr / close) * 1e4
        return tr_bps.rolling(window).mean()
    ret = close.pct_change()
    return ret.rolling(window).std() * 1e4


def vol_scaled_cost(
    df: pd.DataFrame,
    *,
    base_cost_bps: float,
    slip_coef: float,
    vol_window: int = 20,
) -> pd.Series:
    """Per-bar round-trip cost (bps), каузально: cost[t] = base + slip_coef * vol_bps[t].

    Идея (OptimalAd7967, трек A): плоский cost_bps занижает проскальзывание в быстрых
    движениях — заполнение тем хуже, чем выше волатильность в момент входа. Косты
    растут с недавней реализованной волатильностью, которую мы знаем каузально.

    `.shift(1)` — строго прошлая инфа (волатильность через t-1), чтобы исключить любую
    внутрибаровую утечку на баре входа. slip_coef=0 → константа base → точная копия
    плоской модели (регресс-безопасно).

    `slip_coef * vol_bps` — это ВЕСЬ round-trip-довесок проскальзывания (одно слагаемое,
    не сумма двух плеч). Практический анкер: one-way ≈ 0.1·ATR ⇒ round-trip ≈ 0.2·ATR,
    т.е. slip_coef≈0.2 — грубый «реалистичный» центр свипа. Полностью честная версия
    заряжала бы ВЫХОД по vol бара t+H отдельно — это ДОПОЛНИТЕЛЬНЫЙ кост (не
    look-ahead-оптимизм), опциональная полировка.

    ⚠ slip_coef НЕ переносим между источниками: ATR/range систематически больше
    close-to-close std, и бары разных ТФ двигаются по-разному (на наших данных slip=0.1
    даёт +2bp на 5m-ATR, но +0.5bp на 1m-std). Калибровать К заново под КАЖДЫЙ датасет
    (включая клиентский), никогда не тащить значение со стороны.

    Warmup-бары (до накопления окна) заполняются base_cost_bps (нет vol-инфы → флэт),
    чтобы выборка совпадала с плоской моделью и vol-свип был ПРЯМО сравним с базой.
    """
    vol_bps = _vol_bps(df, vol_window).shift(1)
    return (base_cost_bps + slip_coef * vol_bps).fillna(base_cost_bps)


def _nonoverlap_mask(take: np.ndarray, horizon: int) -> np.ndarray:
    """Жадно проредить True-позиции так, чтобы между взятыми было >= horizon баров."""
    out = np.zeros_like(take, dtype=bool)
    last = -10**9
    for i in np.flatnonzero(take):
        if i - last >= horizon:
            out[i] = True
            last = i
    return out


def evaluate(
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
) -> CostWallResult:
    """Прогнать сигнал через cost-wall. df должен содержать 'close'.

    cost_series — опциональный per-bar round-trip cost (bps), выровненный по df.index
    (см. vol_scaled_cost). Если задан — кост каждой сделки берётся по бару входа, а в
    результат пишется реализованный СРЕДНИЙ эффективный кост (cost_bps). Если None —
    плоский cost_bps (легаси-поведение, побитово неизменно).
    """
    close = df["close"].astype(float)
    fwd = close.shift(-horizon) / close - 1.0      # будущая доходность (метка)

    valid = signal_raw.notna() & fwd.notna() & np.isfinite(signal_raw)
    if cost_series is not None:
        cost_series = cost_series.reindex(df.index)
        valid = valid & cost_series.notna()
    s = signal_raw[valid].astype(float)
    r = fwd[valid].astype(float)
    cv = cost_series[valid].astype(float).values if cost_series is not None else None
    n_total = len(s)
    if n_total < 4 * MIN_TRADES:
        return _degenerate(label, source, horizon, cost_bps, n_total)

    split = int(n_total * train_frac)
    s_tr, r_tr = s.iloc[:split], r.iloc[:split]
    s_te, r_te = s.iloc[split:], r.iloc[split:]

    # (4) ориентация знака по train-корреляции
    c = np.corrcoef(s_tr.values, r_tr.values)[0, 1]
    sign = 1 if (np.isnan(c) or c >= 0) else -1
    s_te_o = s_te.values * sign

    # (3) AUC на test
    auc = _auc(s_te_o, (r_te.values > 0).astype(int))

    # (5) порог входа — квантиль |signal| на train
    thr = float(np.nanquantile(np.abs(s_tr.values), trade_quantile))
    take = np.abs(s_te_o) >= thr
    # (6) непересекающиеся сделки
    take = _nonoverlap_mask(take, horizon)

    ret_taken = r_te.values[take]
    dir_taken = np.sign(s_te_o[take])
    n_trades = int(take.sum())

    eff_cost = cost_bps   # эффективный кост для отчёта (перезапишется если cost_series)
    skew = 0.0
    kurt = 3.0
    sharpe = float("nan")
    sub_pos = float("nan")
    if n_trades == 0:
        move_med = move_p90 = gross_mean = net_mean = t_stat = max_dd = float("nan")
    else:
        move = np.abs(ret_taken) * 1e4
        move_med = float(np.median(move))
        move_p90 = float(np.percentile(move, 90))
        gross = dir_taken * ret_taken * 1e4                  # bps gross на сделку
        gross_mean = float(gross.mean())
        if cv is not None:
            cost_taken = cv[split:][take]                    # кост по бару входа
            eff_cost = float(cost_taken.mean())
        else:
            cost_taken = cost_bps
        net = gross - cost_taken                             # bps net на сделку
        net_mean = float(net.mean())
        sd = net.std(ddof=1) if n_trades > 1 else float("nan")
        t_stat = float(net_mean / (sd / np.sqrt(n_trades))) if sd and sd > 0 else float("nan")
        sharpe = float(net_mean / sd) if sd and sd > 0 else float("nan")   # неокруглён
        # MaxDD: equity = cumsum net-bps по хронологии непересекающихся сделок (фикс-юнит
        # на сделку), глубочайший peak→trough в bps. net учитывает кост → vol-slippage
        # углубляет просадку. Prepend 0: первая же убыточная сделка считается просадкой.
        eq = np.concatenate(([0.0], np.cumsum(net)))
        max_dd = float((np.maximum.accumulate(eq) - eq).max())
        # skew/kurt net-распределения — для поправки на не-нормальность в batch-DSR
        if n_trades > 2:
            z = net - net_mean
            var = float((z ** 2).mean())
            if var > 0:
                skew = float((z ** 3).mean() / var ** 1.5)
                kurt = float((z ** 4).mean() / var ** 2)
        # консистентность: K хронологич. чанков сделок, доля с net>0 (один ли период вынес)
        if n_trades >= SUBPERIODS:
            chunks = np.array_split(net, SUBPERIODS)        # порядок = время (test по индексу)
            sub_pos = float(np.mean([c.mean() > 0 for c in chunks]))

    drift = float(r_te.mean() * 1e4)

    n_eff = len(s_te) // horizon       # независимых наблюдений для AUC (не n_test!)
    if np.isnan(auc) or n_eff < MIN_EFF:
        verdict = "INCONCLUSIVE"       # недостаточно мощности — не путать с NO_EDGE
    elif auc <= AUC_MIN:
        verdict = "NO_EDGE"
    elif n_trades >= MIN_TRADES and net_mean > 0 and not np.isnan(t_stat) and t_stat >= MIN_T:
        verdict = "TRADEABLE"   # net>0 И мощно И значимо (t>=2)
    else:
        verdict = "SUB_COST"    # сигнал/net есть, но не значим/мало сделок

    return CostWallResult(
        label=label, source=source, horizon=horizon, cost_bps=round(eff_cost, 2),
        n_total=n_total, n_test=len(s_te), n_eff=n_eff, n_trades=n_trades,
        auc_test=round(auc, 4) if not np.isnan(auc) else float("nan"),
        sign=sign,
        move_bps_med=round(move_med, 2) if not np.isnan(move_med) else float("nan"),
        move_bps_p90=round(move_p90, 2) if not np.isnan(move_p90) else float("nan"),
        gross_bps=round(gross_mean, 3) if not np.isnan(gross_mean) else float("nan"),
        net_mean_bps=round(net_mean, 3) if not np.isnan(net_mean) else float("nan"),
        net_t_stat=round(t_stat, 2) if not np.isnan(t_stat) else float("nan"),
        drift_bps=round(drift, 2),
        max_dd_bps=round(max_dd, 1) if not np.isnan(max_dd) else float("nan"),
        verdict=verdict,
        skew=round(skew, 4), kurt=round(kurt, 4), sharpe=sharpe,
        sub_pos=round(sub_pos, 2) if not np.isnan(sub_pos) else float("nan"),
    )


def _degenerate(label, source, horizon, cost_bps, n_total) -> CostWallResult:
    nan = float("nan")
    return CostWallResult(
        label=label, source=source, horizon=horizon, cost_bps=cost_bps,
        n_total=n_total, n_test=0, n_eff=0, n_trades=0, auc_test=nan, sign=0,
        move_bps_med=nan, move_bps_p90=nan, gross_bps=nan, net_mean_bps=nan,
        net_t_stat=nan, drift_bps=nan, max_dd_bps=nan, verdict="INSUFFICIENT_DATA",
    )


_RESULTS_HEADER = (
    "| date | label | source | H | cost_bps | n_total | n_eff | n_trades | auc_test | "
    "move_med | move_p90 | net_bps | t | drift | verdict | gross_bps | max_dd_bps | sub_pos |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def append_results(results: list[CostWallResult]) -> None:
    """Дописать строки в RESULTS.md (вечный реестр; создаёт шапку при первом разе)."""
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "# RESULTS — реестр проверок сигналов против cost-wall\n\n"
            "Append-only. Каждая строка = один сигнал × источник × горизонт. "
            "НЕ переоткрывать NO_EDGE/SUB_COST без новой гипотезы.\n\n"
            + _RESULTS_HEADER,
            encoding="utf-8",
        )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    lines = []
    for x in results:
        lines.append(
            f"| {now} | {x.label} | {x.source} | {x.horizon} | {x.cost_bps:g} | "
            f"{x.n_total} | {x.n_eff} | {x.n_trades} | {x.auc_test} | {x.move_bps_med} | "
            f"{x.move_bps_p90} | {x.net_mean_bps} | {x.net_t_stat} | {x.drift_bps} | "
            f"**{x.verdict}** | {x.gross_bps} | {x.max_dd_bps} | {x.sub_pos} |\n"
        )
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.writelines(lines)

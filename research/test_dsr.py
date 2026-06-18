"""
research/test_dsr.py
====================
Тесты batch Deflated Sharpe. Главное анти-data-mining свойство: ПЕРЕБОР ШУМА не должен
проходить DSR (лучший из N шумовых ≈ ожидаемому максимуму под H0 → DSR≈0.5), а реальный
выброс далеко над шумом — проходит.

Запуск:  python -m pytest research/test_dsr.py -q
"""
from __future__ import annotations

from research.costwall import CostWallResult
from research.dsr import deflated_sharpe


def _res(t_stat: float, n_trades: int, label: str = "x",
         source: str = "s") -> CostWallResult:
    """Минимальный результат с заданными t/T (SR=t/sqrt(T)); прочее не влияет на DSR."""
    return CostWallResult(
        label=label, source=source, horizon=12, cost_bps=40,
        n_total=n_trades * 12, n_test=n_trades * 12, n_eff=n_trades, n_trades=n_trades,
        auc_test=0.55, sign=1, move_bps_med=10.0, move_bps_p90=20.0, gross_bps=1.0,
        net_mean_bps=1.0, net_t_stat=t_stat, drift_bps=0.0, max_dd_bps=10.0,
        verdict="x", skew=0.0, kurt=3.0,
    )


def _noise_grid(n: int) -> list[CostWallResult]:
    """N шумовых конфигураций: t в детерминированном симметричном разбросе, mean≈0."""
    return [_res(((i % 7) - 3) * 0.1, 100, f"noise{i}") for i in range(n)]


def test_pure_noise_does_not_pass_dsr():
    """Перебор 20 шумовых конфигураций — лучший НЕ проходит дефляцию."""
    rep = deflated_sharpe(_noise_grid(20))
    assert rep.verdict == "FAIL_DSR"
    assert rep.dsr < 0.95


def test_real_outlier_passes_dsr():
    """Один настоящий выброс (SR=0.8) над 20 шумом → переживает поправку на N."""
    grid = _noise_grid(20) + [_res(8.0, 100, "real")]
    rep = deflated_sharpe(grid)
    assert rep.best_label == "real"
    assert rep.verdict == "PASS_DSR"
    assert rep.dsr >= 0.95


def test_more_trades_same_sharpe_more_significant():
    """При ФИКСИРОВАННОМ SR (и том же гриде) больше сделок → выше DSR.

    SR_best, набор SR испытаний и SR0 неизменны; растёт только sqrt(T*-1) в числителе —
    тот же edge на большем числе независимых сделок значимее. (Прямая монотонность; рост
    числа ИСПЫТАНИЙ N немонотонен из-за эндогенной Var(SR) — поэтому тестируем T, не N.)
    """
    sr = 0.3
    noise = _noise_grid(10)
    few = deflated_sharpe([_res(sr * (50 ** 0.5), 50, "best")] + noise)
    many = deflated_sharpe([_res(sr * (500 ** 0.5), 500, "best")] + noise)
    assert abs(few.sr_best - many.sr_best) < 1e-9           # тот же Sharpe
    assert few.sr_expected_max == many.sr_expected_max      # тот же бар
    assert many.dsr > few.dsr


def test_slip_variants_collapse_to_one_trial():
    """slip-варианты (cost-stress) одной ячейки не считаются отдельными испытаниями.

    4 сигнала × (flat + vol-вариант) = 8 результатов, но N=4 selection-ячеек; лучший —
    flat (выше SR), не vol-стресс.
    """
    grid = []
    for i in range(4):
        grid.append(_res(2.0 + i, 100, f"sig{i}", source="5m"))                  # flat
        grid.append(_res(1.0 + i, 100, f"sig{i}", source="5m/vol(s=0.1,w=20)"))  # стресс
    rep = deflated_sharpe(grid)
    assert rep.n_trials == 4                       # 8 результатов → 4 испытания
    assert rep.best_source == "5m"                 # выбран flat, не vol-вариант


def test_single_trial_inconclusive():
    """N<2 — нет дисперсии испытаний, дефлировать нечем."""
    rep = deflated_sharpe([_res(5.0, 100)])
    assert rep.verdict == "INCONCLUSIVE"

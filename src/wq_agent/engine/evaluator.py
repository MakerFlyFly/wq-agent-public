from __future__ import annotations

from ..config import Settings
from ..models import BacktestResult, QualityGrade


class AlphaEvaluator:
    def __init__(self, settings: Settings):
        self.min_fitness = settings.MIN_FITNESS
        self.min_sharpe = settings.MIN_SHARPE
        self.max_turnover = settings.MAX_TURNOVER
        self.min_returns = settings.MIN_RETURNS

    def evaluate(self, result: BacktestResult) -> QualityGrade:
        if result.fitness is None:
            return QualityGrade.REJECT

        fitness = result.fitness
        sharpe = result.sharpe or 0.0
        turnover = result.turnover or 1.0
        returns = result.returns or 0.0

        if fitness >= self.min_fitness and sharpe >= self.min_sharpe:
            if turnover <= self.max_turnover and returns >= self.min_returns:
                return QualityGrade.HIGH

        if fitness >= self.min_fitness * 0.7:
            if sharpe >= self.min_sharpe * 0.5:
                return QualityGrade.MEDIUM

        if fitness >= self.min_fitness * 0.4:
            return QualityGrade.LOW

        return QualityGrade.REJECT

    def filter_high_quality(
        self,
        results: list[BacktestResult],
        min_grade: QualityGrade = QualityGrade.HIGH,
    ) -> list[BacktestResult]:
        grade_order = {
            QualityGrade.HIGH: 4,
            QualityGrade.MEDIUM: 3,
            QualityGrade.LOW: 2,
            QualityGrade.REJECT: 1,
        }
        min_level = grade_order[min_grade]
        return [r for r in results if r.grade and grade_order.get(r.grade, 0) >= min_level]

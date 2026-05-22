from __future__ import annotations

import asyncio

from loguru import logger

from ..config import Settings
from ..db import Database
from ..models import AlphaRecord, AlphaStatus, BacktestResult, QualityGrade
from ..wq.client import WQClient
from .evaluator import AlphaEvaluator


class BacktestEngine:
    def __init__(self, wq: WQClient, db: Database, settings: Settings):
        self.wq = wq
        self.db = db
        self.settings = settings
        self.evaluator = AlphaEvaluator(settings)

    async def backtest_batch(
        self,
        alpha_ids: list[int],
        max_concurrent: int | None = None,
    ) -> list[BacktestResult]:
        semaphore = asyncio.Semaphore(max_concurrent or self.settings.WQ_MAX_CONCURRENT)
        results: list[BacktestResult] = []

        async def _run_one(alpha_id: int) -> BacktestResult | None:
            async with semaphore:
                alpha = await self.db.get_alpha(alpha_id)
                if not alpha:
                    logger.warning(f"Alpha {alpha_id} not found")
                    return None
                return await self._backtest_single(alpha)

        tasks = [_run_one(aid) for aid in alpha_ids]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Backtest error: {item}")
            elif item is not None:
                results.append(item)

        return results

    async def backtest_expressions(
        self,
        expressions: list[tuple[int, str]],
        max_concurrent: int | None = None,
    ) -> list[BacktestResult]:
        semaphore = asyncio.Semaphore(max_concurrent or self.settings.WQ_MAX_CONCURRENT)
        results: list[BacktestResult] = []

        async def _run_one(alpha_id: int, expression: str) -> BacktestResult | None:
            async with semaphore:
                return await self._backtest_expression(alpha_id, expression)

        tasks = [_run_one(aid, expr) for aid, expr in expressions]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Backtest error: {item}")
            elif item is not None:
                results.append(item)

        return results

    async def _backtest_single(self, alpha: AlphaRecord) -> BacktestResult | None:
        await self.db.update_alpha_status(alpha.id, AlphaStatus.BACKTESTING)
        result = await self._backtest_expression(alpha.id, alpha.expression)
        if result and result.grade != QualityGrade.REJECT:
            await self.db.update_alpha_status(alpha.id, AlphaStatus.EVALUATED)
        else:
            await self.db.update_alpha_status(alpha.id, AlphaStatus.FAILED)
        return result

    async def _backtest_expression(self, alpha_id: int, expression: str) -> BacktestResult | None:
        logger.info(f"Backtesting alpha {alpha_id}: {expression[:60]}...")

        submit_result = await self.wq.submit_simulation(expression)
        if submit_result.get("status") == "error":
            msg = submit_result.get("message", "")
            logger.error(f"Simulation submit failed for alpha {alpha_id}: {msg}")
            return None

        progress_url = submit_result["progress_url"]
        poll_result = await self.wq.poll_simulation(progress_url)

        if poll_result.get("status") != "complete":
            logger.error(f"Simulation failed for alpha {alpha_id}: {poll_result.get('message')}")
            return None

        alpha_data = poll_result.get("alpha_data", {})
        is_data = alpha_data.get("is", {})
        wq_alpha_id = poll_result.get("alpha_id")

        backtest = BacktestResult(
            alpha_id=alpha_id,
            region=self.settings.WQ_REGION,
            universe=self.settings.WQ_UNIVERSE,
            delay=self.settings.WQ_DELAY,
            neutralization=self.settings.WQ_NEUTRALIZATION,
            sharpe=is_data.get("sharpe"),
            turnover=is_data.get("turnover"),
            fitness=is_data.get("fitness"),
            returns=is_data.get("returns"),
            checks=is_data.get("checks"),
            wq_alpha_id=wq_alpha_id,
        )

        backtest.grade = self.evaluator.evaluate(backtest)
        await self.db.insert_backtest_result(backtest)

        grade_str = backtest.grade.value if backtest.grade else "unknown"
        fitness_str = f"{backtest.fitness:.4f}" if backtest.fitness is not None else "N/A"
        logger.info(f"Alpha {alpha_id}: fitness={fitness_str}, grade={grade_str}")

        if backtest.grade == QualityGrade.HIGH:
            await self.db.update_alpha_status(alpha_id, AlphaStatus.HIGH_QUALITY)

        return backtest

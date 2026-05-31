from __future__ import annotations
from wq_agent.config import Settings
import pytest
from wq_agent.db import Database


def test_self_corr_settings_defaults():
    s = Settings(_env_file=None)
    assert s.SELF_CORR_THRESHOLD == 0.7
    assert s.SELF_CORR_SHARPE_MARGIN == 0.10
    assert s.SELF_CORR_MIN_OVERLAP == 60


@pytest.mark.asyncio
async def test_pnl_cache_round_trip(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        assert await db.get_cached_pnl(1) is None
        await db.upsert_pnl(1, "WQ123", ["2020-01-02", "2020-01-03"], [0.1, -0.2])
        got = await db.get_cached_pnl(1)
        assert got == (["2020-01-02", "2020-01-03"], [0.1, -0.2])
        # upsert overwrites
        await db.upsert_pnl(1, "WQ123", ["2020-01-02"], [0.5])
        assert await db.get_cached_pnl(1) == (["2020-01-02"], [0.5])
    finally:
        await db.close()

from __future__ import annotations


def parse_pnl_response(data: dict) -> tuple[list[str], list[float]]:
    """WQ PnL recordset（累计 PnL）→（日期, 每日收益）。

    假设 records 为 [[date, cumulative_pnl], ...]。容错：跳过缺值/格式错误的行。
    每日收益 = 累计 PnL 的逐日差分（首日丢弃）。
    """
    records = data.get("records") or []
    dates: list[str] = []
    cum: list[float] = []
    for rec in records:
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            continue
        d, v = rec[0], rec[1]
        if v is None:
            continue
        try:
            cum.append(float(v))
        except (TypeError, ValueError):
            continue
        dates.append(str(d))
    daily = [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    return dates[1:], daily


def pearson(a: list[float], b: list[float]) -> float:
    """Pearson 相关系数。长度 < 2 或任一方零方差 → 0.0。"""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    va = sum(x * x for x in da)
    vb = sum(y * y for y in db)
    if va <= 0 or vb <= 0:
        return 0.0
    return num / ((va ** 0.5) * (vb ** 0.5))


def align(
    dates_a: list[str], ra: list[float],
    dates_b: list[str], rb: list[float],
) -> tuple[list[float], list[float]]:
    """按日期取重叠段，返回两条对齐向量（按日期排序）。"""
    ma = dict(zip(dates_a, ra))
    mb = dict(zip(dates_b, rb))
    common = sorted(set(ma) & set(mb))
    return [ma[d] for d in common], [mb[d] for d in common]

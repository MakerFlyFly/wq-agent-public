# wq-agent

WorldQuant Alpha 生成与回测 Agent Harness。

通过 LLM（Kimi / DeepSeek）、模板与因子挖掘三种策略批量生成 alpha 表达式，调用 WorldQuant Brain Simulator 进行回测，并按 fitness / Sharpe / turnover / returns 阈值自动评级、入库 SQLite。

## 功能特性

- **三种生成策略**
  - `llm` — 调用 LLM（默认 Kimi `kimi-k2.6`，可切 DeepSeek）生成符合 FastExpr 语法的表达式
  - `template` — 基于 `templates/alpha_templates.yaml` 的模板组合
  - `factor_mining` — 因子挖掘式遍历
- **WQ Brain 客户端**：自动登录、拉取 datafields/operators、并发提交 simulation、轮询结果
- **回测评估**：按 `MIN_FITNESS / MIN_SHARPE / MAX_TURNOVER / MIN_RETURNS` 划分 HIGH / MEDIUM / LOW / REJECT
- **持久化**：所有 alpha 表达式与回测结果落地到 `wq_agent.db`（SQLite, aiosqlite）
- **Rich 终端界面**：高质量 alpha 表格展示，分级统计

## 安装

需要 Python 3.11+。

```bash
# 创建虚拟环境并安装
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 配置

复制 `.env.example` 为 `.env` 并填入凭证：

```bash
cp .env.example .env
```

关键变量：

| 变量 | 说明 |
| --- | --- |
| `WQ_USERNAME` / `WQ_PASSWORD` | WorldQuant Brain 账号 |
| `LLM_PROVIDER` | `kimi` 或 `deepseek` |
| `KIMI_API_KEY` / `DEEPSEEK_API_KEY` | LLM 服务密钥 |
| `WQ_REGION` / `WQ_UNIVERSE` / `WQ_DELAY` / `WQ_NEUTRALIZATION` | 回测参数 |
| `MIN_FITNESS` / `MIN_SHARPE` / `MAX_TURNOVER` / `MIN_RETURNS` | 评级阈值 |
| `WQ_MAX_CONCURRENT` | Simulation 并发数 |

## 使用

安装后会注册 `wq-agent` 命令：

```bash
# 全流程：生成 → 回测 → 评估 → 展示
wq-agent run --strategy llm --count 18 --batches 1

# 仅生成（不回测）
wq-agent generate --strategy template -n 20 --no-backtest

# 对待回测的 alpha 跑回测
wq-agent backtest --pending --concurrent 5
wq-agent backtest --ids 1,2,3

# 查看高质量 alpha
wq-agent list --quality high --min-fitness 0.6

# 统计概览
wq-agent status
```

加 `-v / --verbose` 输出 DEBUG 级日志，日志同时写入 `wq_agent.log`。

## 项目结构

```
src/wq_agent/
├── cli.py              # Typer CLI 入口（generate / backtest / list / run / status）
├── config.py           # pydantic-settings 配置加载
├── models.py           # AlphaRecord / BacktestResult / 枚举
├── db.py               # aiosqlite 持久化层
├── agent/
│   └── orchestrator.py # 主流程编排
├── generator/          # 三种 alpha 生成策略
│   ├── llm.py
│   ├── template.py
│   └── factor.py
├── llm/                # LLM 适配（Kimi / DeepSeek）
├── wq/                 # WQ Brain 客户端 + 鉴权
└── engine/
    ├── backtest.py     # Simulation 提交与轮询
    └── evaluator.py    # 多指标评级
templates/
└── alpha_templates.yaml
```

## 开发

```bash
ruff check src tests
pytest
```

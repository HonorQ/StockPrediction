# StockPrediction

A股量化选股与每日推荐系统。覆盖「行情抓取 → 因子计算 → 综合评分 → ML 辅助 → 风控过滤 → HTML 报告 → 结果反馈」的完整链路，支持盘前、盘中、午后三个运行时点，并提供回测、因子 IC 研究、模型训练与影子追踪工具。

> 仅供研究与学习使用，不构成任何投资建议。

## 核心特性

- **三时点运行**：07:00 盘前全量计算、09:25 盘中修正、14:40 午后最终推荐（覆盖同名盘前报告）。
- **统一评分链路**：回测与实盘共用 `StockScorer`（子评分 → 预期扣分 → 截面归一化），避免两套逻辑漂移导致回测失真。
- **分层因子体系**：技术面、资金行为、筹码分布、基本面、情绪周期、事件驱动、板块轮动、相对强度等数十个因子。
- **多重风控**：涨停/跌停口径统一、高位筹码与量价背离排除、开盘跳空风险、盘中跳水保护。
- **数据驱动上线**：新因子须经全市场截面 IC → 十分位分层 → 单边组合模拟 → 影子运行，结论以 `reports/market_ic_verdict.json` 为准。
- **可观测性**：每日 HTML 报告、结构化日志、心跳文件与任务异常横幅。
- **影子追踪隔离**：独立数据库、独立日志、独立产物，不参与生产推荐。

## 架构分层

依赖方向单向，禁止反向引用：

```
根目录入口 run_*.py         参数解析 / 配置校验 / 心跳 / 模式分发
        ↓
src/engine                  评分 scorer / 回测 backtest / 反馈 feedback_loop / 报告 reporter / 风控
        ↓
src/factors                 纯因子计算，输入 K 线与行情字典，不访问数据库和网络
        ↓
src/data                    行情 / 资金流 / 新闻 / 行业映射 / SQLite 存储
        ↓
src/utils                   日志 / 心跳 / 代理管理 等无业务语义工具
```

- 因子层禁止 `import requests / baostock / sqlite3`，数据由上层注入。
- 跨层复用函数放在 `src/utils` 或对应业务模块，入口文件只做 re-export 兼容旧引用。

## 目录结构

```
StockPrediction/
├── run_daily.py             每日入口（--premarket / --intraday / --afternoon / --factor-ic）
├── run_backtest.py          Walk-Forward 回测 + 反馈循环
├── train_ml.py              ML 模型训练（factor_snapshots + T+3 收益标签）
├── run_batch_training.py    批量训练/权重优化
├── config.yaml              权重、分级阈值、过滤条件、关键词、全球市场映射
├── src/
│   ├── pipeline/            盘前/盘中/午后编排与数据准备
│   ├── engine/              评分、回测、ML、风控、反馈、报告
│   ├── factors/             各类选股因子
│   ├── data/                数据抓取、行业映射、SQLite 访问
│   ├── monitor/             实时监控
│   └── utils/               日志、心跳、代理、通用工具
├── scripts/                 补数据、研究、验证、影子追踪等独立脚本
├── tests/                   离线回归测试（不依赖外部数据源）
├── cache/                   K 线缓存（不入库）
├── data/                    SQLite 数据库（不入库）
├── models/                  ML 模型与元数据
└── reports/                 每日报告与研究产物（不入库）
```

## 快速开始

### 环境要求

- Python 3.10+（当前开发环境 3.13）
- Windows（计划任务脚本为 `.bat` / `.ps1`；核心逻辑跨平台）

### 安装

```powershell
git clone https://github.com/HonorQ/StockPrediction.git
cd StockPrediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 准备数据

K 线缓存是一切研究的地基。首次运行或长期未运行时，先补齐缓存：

```powershell
python -u scripts\backfill_missing_klines.py
```

> 该脚本使用 akshare/东财源，全市场约 5200 只股票约 17 分钟，结束时生成 `cache/_code_index.json`。
> 注意：脚本会跳过已有 >= 200 行的股票（"加满即冻结"），跨交易日推进前需先刷新缓存尾部。

### 运行

```powershell
# 盘前：全量因子计算与候选池产出
python run_daily.py --premarket

# 盘中：基于集合竞价与实时行情快速修正
python run_daily.py --intraday

# 午后：结合全天数据给出最终推荐
python run_daily.py --afternoon
```

无参数时默认 `--premarket`。产物位于 `reports/daily/YYYY-MM-DD.html`。

## 常用命令

| 用途 | 命令 |
|---|---|
| 每日推荐（盘前/盘中/午后） | `python run_daily.py --premarket` 等 |
| 因子 IC 快照 | `python run_daily.py --factor-ic --lookback 60` |
| Walk-Forward 回测 | `python run_backtest.py --start 2026-06-29 --end 2026-07-22` |
| 回测并强制刷新 K 线 | `python run_backtest.py --refresh` |
| ML 模型训练 | `python train_ml.py` |
| 重建全市场面板与 IC 结论 | `python scripts\rebuild_market_panel.py` |
| 十分位分层检验 | `python scripts\quantile_analysis.py` |
| 单边组合模拟 | `python scripts\portfolio_simulator.py --top-n 50 --hold 10` |
| 影子追踪（72.5 档） | `python scripts\shadow_rebound_track.py --report` |
| 每周健康检查 | `python scripts\weekly_health_check.py` |

## 配置说明

主要配置集中在 `config.yaml`：

- `weights`：各评分维度权重，启动时校验总和与维度完整性，不一致直接退出。
- `score_tiers`：信号强度、情绪周期、ML 概率、护城河、反弹持续性等分级阈值。
- `filter`：推荐数量、最低分、快速打分加成、开盘跳空与筹码风控参数。
- `keywords` / `global_markets`：新闻事件关键词与外围市场映射。
- `config/policy_chain.json`、`config/concept_map.json`：政策链与概念映射。

修改阈值后须同步跑全量回测验证，避免"调参幻觉"。

## 质量门禁

改动 `src/engine` 或 `src/data` 后必须执行：

```powershell
python -m pytest tests/ -q
ruff check .
```

- 测试必须离线可跑（网络调用需 mock），不依赖 baostock/东财可用性。
- 修一个 bug 补一个最小复现测试（`tests/test_bugfixes.py` 风格）。
- 收益口径全项目唯一：`1.0` 表示 `1%`；涨跌停幅度统一使用 `get_limit_pct`，禁止硬编码。

## 运维

计划任务建议时点：

| 时间 | 任务 | 入口 |
|---|---|---|
| 07:00 | 盘前计算 | `run_premarket.bat` |
| 09:25 | 盘中修正 | `run_daily.py --intraday` |
| 14:40 | 午后最终推荐 | `run_daily.py --afternoon` |
| 16:40 | 影子追踪（隔离） | `run_shadow_rebound.bat` |
| 每周末 | 健康检查/胜率监控 | `run_weekly_monitor.bat` |

- 心跳：`logs/heartbeat.json`，异常会在下次报告横幅提示。
- 备份：`backup.ps1` + `register_backup_task.ps1`。
- 推送失败时先排查系统残留代理（`127.0.0.1:7897`），可用 `git -c http.proxy="" push`。

## 数据与入库说明

- `cache/`、`data/*.db`、`logs/`、`reports/`、`backups/` 均不入库。
- `models/*.pkl` 建议使用 Git LFS 跟踪，`*.meta.json` 正常提交。
- K 线缓存只保留股票（`60/00/30/68` 开头），指数与 ETF 存放在 `cache/klines_nonstock/`。

## 研究纪律

因子有效性结论以全市场每日截面的 `reports/market_ic_verdict.json` 为准，候选池小样本存在幸存者偏差。新因子上线路径：

```
面板 IC → 十分位分层 → 单边组合模拟 → 影子运行 → 转正
```

跳步必须在 PR 或文档中说明理由。详细工程规范见 [AGENTS.md](AGENTS.md)。

## 免责声明

本项目为个人量化研究工程，数据来自公开网络接口，可能存在缺失、延迟或错误。任何输出结果均不构成投资建议，据此操作风险自负。

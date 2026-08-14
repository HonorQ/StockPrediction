# 反馈训练 Walk-Forward 实施计划

## 摘要

实现"逐日 walk-forward 反馈训练"系统：以 6/29 数据荐股为起点，每日滚动累积归因反馈，对**每只股票**单独维护 21 维因子权重；6/29→7/31 跑完后，继续滚动跑 2026/01~06 全年每个交易日。T+N 反馈窗口采用 d1/d3/d5/d10/d14 多窗口（T+14 为主信号）。

## 用户决策（已确认）

| 决策点 | 选择 |
|---|---|
| 权重粒度 | **个股因子权重** — 每只股票独立维护 21 维因子权重 |
| 观察方式 | **滚动累积反馈** — 每天对已过 T+N 的历史推荐做归因+IC 更新 |
| 训练范围 | **7月→2026 全年** — 先 6/29~7/31 walk-forward，再续跑 2026/01~06 |
| 因子口径 | **K 线可重建因子** — 扩展 run_backtest.py 简化版到生产 scorer.get_sub_scores 口径 |
| T+N 窗口 | **多窗口+T+14** — d1/d3/d5/d10/d14 全用，T+14 为主信号 |

## 当前状态分析

### 已有可复用资产

- [run_backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py) — 简化 walk-forward 框架（baostock 预取 K 线 + 逐日切片），但因子硬编码、feedback_loop 误用 `datetime.now()`
- [src/engine/feedback_loop.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/feedback_loop.py) — 完整归因+IC+调权+ML 重训流程，但权重是 **factor-level 全局**，且时间硬编码
- [src/engine/factor_attribution.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/factor_attribution.py) — 单条归因 + 聚合统计，可直接复用
- [src/engine/scorer.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py) — `get_sub_scores` 21 维子评分 + `batch_normalize` 截面 Z-Score
- [src/data/database.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py) — 完整 schema：recommendations/factor_snapshots/factor_attribution/ic_history/weight_history/model_versions
- [src/engine/ml_selector.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/ml_selector.py) — `FactorOptimizer` IC/ICIR/权重优化

### 现有缺口

1. **无 per-stock 权重表** — 只有全局 factor-level 权重（weight_history）
2. **recommendations 表无 d14_return 列** — 只到 d10
3. **FeedbackLoop 时间硬编码** — `_run_ic_analysis` / `_adjust_weights` 用 `datetime.now()`，回测时会指向真实今天
4. **run_backtest.py 因子简化版与生产口径不一致** — 硬编码 21 维占位值，没用 `scorer.get_sub_scores`
5. **无 walk-forward 滚动反馈主入口** — 现有 run_backtest.py 末尾一次性调用 feedback_loop，不是逐日滚动

## 实施步骤

### Step 1: 数据库 schema 扩展

**文件**: [src/data/database.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py)

1. **recommendations 表迁移**：`_migrate_schema` 中幂等添加 `d14_return REAL` 列
2. **新增 `stock_factor_weights` 表**（核心）：
   ```sql
   CREATE TABLE IF NOT EXISTS stock_factor_weights (
       code TEXT NOT NULL,
       factor_name TEXT NOT NULL,
       weight REAL NOT NULL,
       samples INTEGER DEFAULT 0,        -- 该股该因子累计归因样本数
       accuracy REAL DEFAULT 0,           -- 该股该因子历史命中率
       last_updated TEXT,
       PRIMARY KEY (code, factor_name)
   )
   ```
3. **新增 `stock_weight_history` 表**（追踪变化）：
   ```sql
   CREATE TABLE IF NOT EXISTS stock_weight_history (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       trade_date TEXT NOT NULL,
       code TEXT NOT NULL,
       factor_name TEXT NOT NULL,
       old_weight REAL,
       new_weight REAL,
       reason TEXT,
       created_at TEXT DEFAULT (datetime('now'))
   )
   ```
4. **新增 CRUD 方法**：
   - `get_stock_weights(code) -> Dict[str, float]` — 返回该股票 21 维权重，缺失时返回 None（调用方 fallback 到全局）
   - `update_stock_weight(code, factor, new_weight, samples, accuracy, trade_date, reason)` — UPSERT + 写历史
   - `get_all_stock_weights() -> Dict[str, Dict[str, float]]` — 批量加载用于回测
   - `save_d14_performance(trade_date, code, d14_return)` — 更新 d14_return
5. **修改 `update_performance`**：增加 d14 参数

### Step 2: 新增 StockWeightManager

**新文件**: `src/engine/stock_weight_manager.py`

职责：管理每只股票的 21 维因子权重，基于该股票历史归因数据调整权重。

```python
class StockWeightManager:
    def __init__(self, db, global_weights: Dict[str, float]):
        self.db = db
        self.global_weights = global_weights  # config.yaml 的全局权重作为 fallback

    def get_weights(self, code: str) -> Dict[str, float]:
        """获取该股票的因子权重，缺失时 fallback 到全局权重"""

    def get_weights_batch(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """批量加载（回测用，避免 N+1 查询）"""

    def adjust_weights(self, code: str, trade_date: str,
                       attribution_records: List[Dict]) -> List[Dict]:
        """基于该股票的归因记录调整其 21 维因子权重。
        - 准确率高的因子 → 权重提升
        - 失败率高的因子 → 权重降低
        - 单次调整幅度上限 MAX_WEIGHT_CHANGE=0.02
        - 样本数 < MIN_SAMPLES (3) 时不调整
        返回变更列表
        """
```

**调整规则**（参考现有 feedback_loop._adjust_weights）：
- `wrong_rate >= 0.5` → 权重 × 0.5
- `wrong_rate >= 0.3` 且 `accuracy < 0.4` → 权重 × 0.7
- `accuracy > 0.7` 且 `samples >= 5` → 权重 × 1.1（强化好因子）
- 每次调整后归一化保证 21 维权重之和 = 1

### Step 3: FeedbackLoop 改造支持 trade_date

**文件**: [src/engine/feedback_loop.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/feedback_loop.py)

1. **`run_all(today: str)` 已支持**，但内部方法需改造：
   - `_run_ic_analysis(today)` — 把 `datetime.now()` 改为 `today`，IC 衰减对比窗口用 `today - 14d` 而非 `now - 14d`
   - `_adjust_weights(today)` — 时间戳用 `today`
   - `_maybe_retrain_model(today)` — 重训练间隔检查用 `today` 而非 `now`
2. **新增 `_adjust_stock_weights(today)`**：调用 StockWeightManager 对当天有新归因数据的股票批量调权
3. **`run_all` 流程新增一步**：在全局 `_adjust_weights` 之后调用 `_adjust_stock_weights`
4. **新增参数** `skip_ml: bool = False`（回测时通常不重训 ML，避免污染生产模型）

### Step 4: 因子计算扩展到生产口径

**新文件**: `src/engine/backtest_factors.py`

职责：基于历史 K 线重建生产 scorer.get_sub_scores 口径的 21 维子评分（不依赖新闻/资金流等无法回溯的数据）。

```python
def calc_sub_scores_from_kline(code: str, name: str,
                                kline_rows: List[List]) -> Dict[str, float]:
    """从 baostock K 线行 (date,close,volume,turn,pctChg) 计算 21 维 sub_scores。
    只保留可从 K 线重建的因子：
      - trend / strategy / chip / relative_strength / ashare_specific
      - quant_behavior / capital_behavior / deep_research
      - oversold_rebound
    无法重建的（news/policy_event/institutional/dragon_tiger/national_team/
    margin_trading/shareholder/lockup_risk/research/leader/sector_rotation/
    fundamental）置为 50（中性）
    """
```

**实现策略**：
- 直接复用 [src/factors/technical.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/technical.py) 的 `TechnicalFactors.calculate_all()`（输入 pandas DataFrame）
- 复用 [src/factors/strategies.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/strategies.py) 的 `StockStrategies.get_triggered_strategies()`
- 复用 [src/factors/chip.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/chip.py) 的 `ChipDistribution.calculate()`
- 复用 [src/factors/relative_strength.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/relative_strength.py)
- 复用 [src/factors/ashare_specific.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/ashare_specific.py) 的 `AShareCharacteristics.calculate_all()`
- 复用 [src/factors/quant_behavior.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/quant_behavior.py)
- 复用 [src/factors/capital_behavior.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/capital_behavior.py)
- 复用 [src/factors/deep_research.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors/deep_research.py)（部分依赖 news_list，回测时传空列表）
- 最后调用 `StockScorer(weights).get_sub_scores(factors)` 得到 21 维 sub_scores

baostock K 线 → pandas DataFrame 转换格式与 [src/data/market_data.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/market_data.py) `get_stock_history` 输出对齐（列名 "收盘"/"成交量"/"换手率"/"涨跌幅"）。

### Step 5: 主入口 run_feedback_training.py

**新文件**: `run_feedback_training.py`

```python
"""
反馈训练 Walk-Forward 主入口
阶段1: 6/29 → 7/31 (A 股交易日)
阶段2: 2026/01 ~ 2026/06 (A 股交易日)
每日流程:
  1. 计算当日所有候选股的 21 维 sub_scores
  2. 加载每只股票的 stock_factor_weights（fallback 全局）
  3. 用 per-stock 权重做截面评分 → 选 top N 推荐
  4. 写 recommendations + factor_snapshots
  5. 对所有已过 T+N (N=1,3,5,10,14) 的历史推荐计算实际收益并更新
  6. 跑 FeedbackLoop.run_all(today): 归因 + IC + 全局调权 + 个股调权
  7. 每 7 天触发一次 ML 重训练（可选）
"""

DATES_PHASE1 = [...]  # 6/29 ~ 7/31 A 股交易日
DATES_PHASE2 = [...]  # 2026/01 ~ 2026/06 A 股交易日

def run_phase(dates: List[str], phase_name: str):
    # 1. 一次性预取所有股票 K 线 (start=dates[0]-60d, end=dates[-1]+14d)
    # 2. 逐日 walk-forward:
    for today in dates:
        # 2a. 计算因子 (调用 backtest_factors.calc_sub_scores_from_kline)
        # 2b. 加载 per-stock 权重 (StockWeightManager.get_weights_batch)
        # 2c. 截面评分: 用每只股票自己的权重做加权
        #     score_i = sum(sub_scores_i[factor] * stock_weights_i[factor])
        # 2d. 选 top 10 推荐, 写 recommendations
        # 2e. 对历史推荐补算 T+1/T+3/T+5/T+10/T+14 收益
        # 2f. FeedbackLoop.run_all(today, skip_ml=False if 每7天 else True)
        # 2g. StockWeightManager.adjust_weights(每个有新归因的股票)
    # 3. 输出阶段报告

if __name__ == "__main__":
    run_phase(DATES_PHASE1, "2026年7月")
    run_phase(DATES_PHASE2, "2026年上半年")
```

**关键设计点**：

1. **截面评分用 per-stock 权重**：
   - 不能直接用 `StockScorer.batch_normalize`（它用全局权重）
   - 改造：对每只股票 `score_i = 50 + sum(z_scores_i[factor] * stock_weights_i[factor]) * (50/3)`
   - z_scores 仍按截面计算（保持截面可比性），但加权用每只股票自己的权重

2. **T+N 收益计算**：
   - 复用 run_backtest.py 的 `future_returns()` 逻辑
   - 扩展 offset 列表：`[(1,"d1_return"),(3,"d3_return"),(5,"d5_return"),(10,"d10_return"),(14,"d14_return")]`
   - 只对 `trade_date + N <= 当前回测日` 的推荐补算（避免未来函数）

3. **K 线预取范围**：
   - 阶段1: start=2026-04-01（6/29 前 60 个交易日）, end=2026-08-14（7/31 后 14 个交易日）
   - 阶段2: start=2025-10-01, end=2026-07-14

4. **日期列表获取**：用 baostock `query_all_stock(date)` 返回的股票状态判断当日是否为交易日，或直接 `query_trade_dates`

### Step 6: 报告输出

**新文件**: `src/engine/feedback_reporter.py`（或直接在 run_feedback_training.py 内实现）

每阶段输出 markdown 报告到 `reports/feedback_training_<phase>.md`，内容：
- 阶段汇总：总推荐数、胜率、平均 T+14 收益、最佳/最差股票
- 个股权重变化 Top10：权重调整幅度最大的股票
- 因子有效性：21 维因子在该阶段的整体 IC/命中率
- ML 模型版本：训练样本数、accuracy、top 特征
- 推荐表现明细：每天 top10 的 T+1/T+3/T+5/T+10/T+14 收益表

## 假设与决策

### 假设
1. baostock 能稳定提供 2026/01~07 的历史 K 线（已验证 run_backtest.py 在用）
2. 生产 [src/factors/](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors) 下的因子类能接受 baostock 转换后的 DataFrame（列名对齐后应可）
3. 全 A 股约 4500 只，预取 K 线 + 逐日切片计算约 10-15 分钟/阶段（参考 run_backtest.py 实测）

### 决策
1. **不修改 run_backtest.py** — 保留作为简单回测参考，新功能在新文件实现
2. **不修改 run_daily.py** — 生产流程不动，反馈训练是独立离线流程
3. **不修改生产 ML 模型** — 反馈训练用独立 model 路径 `models/stock_selector_feedback.pkl`，避免污染生产模型
4. **K 线可重建的因子置 50（中性）** — 不强行造数据，符合用户"禁止 mock 数据"规则
5. **T+14 为主信号但 d1/d3/d5/d10 也写入** — 多窗口归因更全面，主信号在调权时权重更高
6. **股票池**：复用 run_backtest.py 的 `get_stocks()` 过滤逻辑（排除 ST/北交所/科创/指数）
7. **每日推荐数**：保持 10 只（与 run_backtest.py 一致），便于对比

## 验证步骤

### 单元验证
1. `StockWeightManager.get_weights()` 对新股票返回全局权重
2. `adjust_weights()` 在样本不足时不调整
3. 权重调整后 21 维之和 = 1（归一化正确）
4. `calc_sub_scores_from_kline()` 对单只股票输出 21 维有效分数
5. d14_return 字段读写正确

### 集成验证
1. **小样本跑通**：先只用 10 只股票、3 个交易日（6/29、6/30、7/1）跑完整流程，验证：
   - 推荐写入正确
   - T+1/T+3 收益补算正确
   - 归因记录写入 factor_attribution
   - stock_factor_weights 表有更新
2. **阶段1 跑通**：6/29~7/31 全量跑，输出报告
3. **阶段2 跑通**：2026/01~06 全量跑，输出报告
4. **结果合理性检查**：
   - 推荐胜率应在 45-65% 之间（过低说明因子失效，过高可能未来函数）
   - 个股权重应有变化（不是所有股票都一样）
   - IC 应有正有负（不是全部为 0）

### 命令
```bash
# 小样本验证
python run_feedback_training.py --smoke-test

# 阶段1
python run_feedback_training.py --phase 1

# 阶段2
python run_feedback_training.py --phase 2

# 全量
python run_feedback_training.py --all
```

## 文件清单

### 新建
- `run_feedback_training.py` — 主入口
- `src/engine/stock_weight_manager.py` — 个股因子权重管理器
- `src/engine/backtest_factors.py` — K 线重建 21 维 sub_scores
- `reports/feedback_training_2026_07.md` — 阶段1 报告（运行后生成）
- `reports/feedback_training_2026_ytd.md` — 阶段2 报告（运行后生成）

### 修改
- [src/data/database.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py) — schema 迁移 + 新增 CRUD
- [src/engine/feedback_loop.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/feedback_loop.py) — trade_date 参数化 + 调用 stock weight manager + skip_ml 参数

### 不修改
- [run_daily.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_daily.py) — 生产流程不动
- [run_backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py) — 保留作简单回测参考
- [config.yaml](file:///c:/Users/EDY/Desktop/StockPrediction/config.yaml) — 全局权重作为 fallback 不动
- [src/engine/scorer.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py) — get_sub_scores 直接复用
- [src/factors/](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors) — 因子类直接复用

## 风险与缓解

1. **风险：baostock 限流导致 K 线预取慢**
   - 缓解：复用 run_backtest.py 的 10 线程池 + 一次性预取全阶段数据
2. **风险：生产因子类对 baostock 数据格式不兼容**
   - 缓解：在 backtest_factors.py 中做格式转换层，列名对齐到 "收盘"/"成交量"/"换手率"/"涨跌幅"
3. **风险：per-stock 权重在小样本下过拟合**
   - 缓解：MIN_SAMPLES=3 才调权 + 单次调整幅度 ≤ 0.02 + 归一化保证和=1
4. **风险：T+14 数据不足（7/31 后 14 天的数据可能没有）**
   - 缓解：阶段1 K 线预取 end 设为 8/14，保证 7/31 推荐的 T+14 可计算

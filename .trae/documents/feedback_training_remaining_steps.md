# 反馈训练 Walk-Forward 剩余步骤实施计划（Step 4 验证 + Step 5 + Step 6）

## 摘要

延续已批准的 `feedback_training_walkforward_plan.md`。Step 1-3 已完成（schema 扩展、StockWeightManager、FeedbackLoop 改造），Step 4 文件已创建（`backtest_factors.py`）但未验证。本计划覆盖剩余工作：
- **Step 4 验证**：运行 `_test_step4.py` 确认 K 线重建 21 维 sub_scores 工作正常，删除临时测试文件
- **Step 5 主入口**：新建 `run_feedback_training.py`，实现 walk-forward 滚动反馈训练（阶段1：6/29~7/31，阶段2：2026/01~06）
- **Step 6 报告 + 验证**：生成 markdown 报告 + 小样本验证

## 当前状态分析

### 已完成（Step 1-3 + Step 4 文件创建）

| 模块 | 状态 | 关键内容 |
|---|---|---|
| [src/data/database.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py) | ✅ 完成 | `stock_factor_weights` 表、`stock_weight_history` 表、`d14_return` 列迁移、6 个 CRUD 方法（`get_stock_weights`/`get_stock_weight_stats`/`get_all_stock_weights`/`update_stock_weight`/`get_stock_weight_history`/`get_recent_attribution_for_stock`） |
| [src/engine/stock_weight_manager.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/stock_weight_manager.py) | ✅ 完成 | `StockWeightManager` 类：内存缓存 + 批量加载 + 归一化 + 调权规则（wrong_rate≥0.5→×0.5；wrong_rate≥0.3且acc<0.4→×0.7；acc>0.7且samples≥5→×1.1） |
| [src/engine/feedback_loop.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/feedback_loop.py) | ✅ 完成 | `run_all(today, skip_ml)` 参数化、`_run_attribution`/`_run_ic_analysis`/`_adjust_weights`/`_build_ic_dataset`/`_maybe_retrain_model` 全部支持 today、新增 `_adjust_stock_weights(today)`、`_get_future_return` 新增 d14_return |
| [src/engine/backtest_factors.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest_factors.py) | ✅ 文件已建 | `baostock_rows_to_dataframe`/`calc_factors_from_kline`/`calc_sub_scores_from_kline`/`slice_kline_by_date` |
| [_test_step4.py](file:///c:/Users/EDY/Desktop/StockPrediction/_test_step4.py) | ⚠️ 待运行 | 临时验证脚本，运行后需删除 |

### 现有缺口（本计划要解决的）

1. **Step 4 未验证**：`backtest_factors.py` 已创建但 `_test_step4.py` 未运行，无法确认 9 维可重建因子 + 12 维中性 = 21 维齐全
2. **Step 5 主入口缺失**：无 `run_feedback_training.py`，无法执行 walk-forward 滚动反馈训练
3. **Step 6 报告 + 小样本验证缺失**：无报告生成逻辑，无端到端冒烟测试

## 关键设计决策（基于 Phase 1 探索）

### 决策 1：per-stock 截面评分实现方式

**问题**：[StockScorer.batch_normalize](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py#L92) 用单一全局 weights 对所有股票做 Z-Score 加权，不适用于 per-stock 权重。

**方案**：复用 `batch_normalize` 的**截面 Z-Score 计算逻辑**（含截尾 + 正交化），但**加权阶段用 per-stock 权重**：

```python
def score_cross_sectional_with_per_stock_weights(
    candidates: List[Dict],            # 每项含 sub_scores, stock_weights
    global_weights: Dict[str, float],  # 用于确定 dims 和正交化
) -> None:
    """对每个 candidate 就地写入 'score' 和 'z_scores'。
    Z-Score 用全局截面计算（保持截面可比性），加权用 per-stock 权重。
    """
    # 1. 复用 batch_normalize 的截尾 + 正交化（用 global_weights 选 dims）
    # 2. 对每个 candidate i:
    #    z_total_i = sum(z_scores_i[factor] * candidate_i.stock_weights[factor])
    #    score_i = 50 + z_total_i * (50/3)
    # 3. 写入 candidate['score'] 和 candidate['z_scores']
```

**理由**：
- Z-Score 必须用全局截面计算（否则不同股票不可比）
- 加权用 per-stock 权重（每只股票因子有效性指纹不同）
- 保持与生产 `batch_normalize` 一致的截尾 + 正交化逻辑

### 决策 2：日期列表获取方式

**方案**：硬编码 A 股交易日列表（参考 [run_backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py#L15) 的 `DATES` 风格）。

- **阶段1**（6/29~7/31）：复用 run_backtest.py 已有的 6/29~7/22，扩展到 7/31
- **阶段2**（2026/01~06）：硬编码 2026/01-04~06-30 之间的 A 股交易日（约 120 天）

**理由**：
- 与 run_backtest.py 风格一致
- baostock `query_trade_dates` 需要登录态，硬编码更简单稳定
- 节假日已在列表中过滤

### 决策 3：ML 重训练策略

**方案**：回测全程 `skip_ml=True`（不重训 ML 模型）。

**理由**：
- 反馈训练的核心是 per-stock 因子权重调整，不依赖 ML
- 避免污染生产 ML 模型（plan 文件决策4）
- 加速回测（ML 训练是性能瓶颈）
- 如用户后续需要 ML 重训练，可在 `--phase 1` 跑完后单独执行

### 决策 4：T+N 收益补算时机

**方案**：每日 walk-forward 中，对所有 `trade_date + N <= today` 的历史推荐补算 T+N 收益。

**实现**：扩展 [run_backtest.future_returns](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py#L98) 的 offset 列表：
```python
offsets = [(1,"d1_return"),(3,"d3_return"),(5,"d5_return"),
           (10,"d10_return"),(14,"d14_return")]
```

**理由**：
- 多窗口归因更全面（d1/d3/d5/d10/d14）
- T+14 为主信号（plan 决策5）
- 避免未来函数：只补算 `trade_date + N <= today` 的推荐

### 决策 5：股票池

**方案**：复用 [run_backtest.get_stocks](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py#L40)（baostock `query_all_stock` + 过滤 ST/北交所/科创/指数）。

**理由**：plan 决策6 已明确。

## 实施步骤

### Step 4 验证：运行测试 + 清理

**操作**：
1. 在项目根目录运行：`python _test_step4.py`
2. 验证输出：
   - DataFrame shape = (30, 9)，columns 含 "日期/开盘/收盘/最高/最低/成交量/成交额/换手率/涨跌幅"
   - 因子计算：`macd_golden_cross`/`ma_bullish`/`rsi_6`/`strategies_triggered`/`strategy_count`/`chip_profit_ratio`/`quant_score`/`capital_behavior`/`deep_research_score`/`oversold_rebound_score` 有非 None 值
   - 21 维 sub_scores 齐全：缺失维度 = []
   - 9 维可重建因子有真实值（非 50），12 维中性因子 = 50
   - 切片测试：截止 2026-06-15 = 15 行
3. 验证通过后**删除** `_test_step4.py`（用户规则：保持文件整洁）

**验证失败的回退**：
- 若因子类导入失败 → 检查 [src/factors/](file:///c:/Users/EDY/Desktop/StockPrediction/src/factors) 下模块路径
- 若 9 维可重建因子值为 50 → 检查 `calc_factors_from_kline` 的因子类调用是否抛异常被吞掉
- 若维度数 ≠ 21 → 检查 `ALL_FACTORS` 列表与 `StockScorer.get_sub_scores` 输出对齐

### Step 5：新建 run_feedback_training.py

**新文件**: `c:\Users\EDY\Desktop\StockPrediction\run_feedback_training.py`

**结构**：

```python
"""
反馈训练 Walk-Forward 主入口
阶段1: 6/29 → 7/31 (A 股交易日)
阶段2: 2026/01 ~ 2026/06 (A 股交易日)
每日流程:
  1. 计算当日所有候选股的 21 维 sub_scores (backtest_factors.calc_sub_scores_from_kline)
  2. 加载每只股票的 stock_factor_weights (StockWeightManager.get_weights_batch)
  3. 截面 Z-Score 归一化 + per-stock 权重加权 → 选 top 10 推荐
  4. 写 recommendations + factor_snapshots
  5. 对所有已过 T+N (N=1,3,5,10,14) 的历史推荐补算实际收益
  6. FeedbackLoop.run_all(today, skip_ml=True): 归因 + IC + 全局调权 + 个股调权
"""
import sys, os, time, json, argparse
import numpy as np, pandas as pd, baostock as bs
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

sys.path.insert(0, '.'); sys.path.insert(0, 'src')

from src.data.database import StockDatabase
from src.engine.stock_weight_manager import StockWeightManager, _normalize, ALL_FACTORS
from src.engine.feedback_loop import FeedbackLoop
from src.engine.backtest_factors import calc_sub_scores_from_kline, slice_kline_by_date
from src.engine.scorer import StockScorer

# === 阶段日期列表（硬编码 A 股交易日） ===
DATES_PHASE1 = [
    '2026-06-29','2026-06-30','2026-07-01','2026-07-02','2026-07-03',
    '2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10',
    '2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17',
    '2026-07-20','2026-07-21','2026-07-22','2026-07-23','2026-07-24',
    '2026-07-27','2026-07-28','2026-07-29','2026-07-30','2026-07-31',
]
DATES_PHASE2 = [
    # 2026/01
    '2026-01-05','2026-01-06','2026-01-07','2026-01-08','2026-01-09',
    '2026-01-12','2026-01-13','2026-01-14','2026-01-15','2026-01-16',
    # ... (2026/01~06 全部 A 股交易日，约 120 天，需在实现时补全)
]
```

**核心函数（按文件内顺序）**：

#### 5.1 `fetch_kline(code, start, end)` / `prefetch(stocks, start, end)` / `get_stocks(date)`
**直接复用** [run_backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py#L20) 同名函数，无需改动。

#### 5.2 `compute_sub_scores(date, stocks, kc) -> List[Dict]`
**职责**：对当日所有候选股计算 21 维 sub_scores。

```python
def compute_sub_scores(date, stocks, kc):
    """对每只股票切片 K 线并计算 21 维 sub_scores。
    返回 [{code, name, price, sub_scores, factors_dict}, ...]
    """
    results = []
    for code, name in stocks:
        raw = kc.get(code)
        if not raw or len(raw) < 20:
            continue
        # 切片到当日（避免未来函数）
        rows = slice_kline_by_date(raw, date)
        if len(rows) < 20:
            continue
        sub_scores = calc_sub_scores_from_kline(rows)
        if not sub_scores or len(sub_scores) != 21:
            continue
        # 当前价 = 切片最后一行的 close
        price = float(rows[-1][1])
        results.append({
            "code": code.replace("sh.", "sh").replace("sz.", "sz"),
            "raw_code": code,  # 保留 baostock 格式用于 future_returns
            "name": name,
            "price": price,
            "sub_scores": sub_scores,
        })
    return results
```

#### 5.3 `score_cross_sectional_per_stock(candidates, global_weights) -> None`
**职责**：截面 Z-Score 归一化 + per-stock 权重加权（决策1）。

**实现要点**：
- 复用 [StockScorer.batch_normalize](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py#L92) 的截尾 + 正交化逻辑（用 global_weights 选 dims）
- **关键差异**：加权阶段用 `candidate['stock_weights']`（per-stock），而非全局 weights
- 就地写入 `candidate['score']` 和 `candidate['z_scores']`

```python
def score_cross_sectional_per_stock(candidates, global_weights):
    dims = [k for k in global_weights.keys() if global_weights.get(k, 0) > 0]
    if not dims or not candidates:
        return
    
    # 1. 收集原始值
    raw = {dim: [] for dim in dims}
    for c in candidates:
        ss = c.get("sub_scores", {})
        for dim in dims:
            raw[dim].append(ss.get(dim, 50))
    
    # 2. 截尾处理（复用 batch_normalize 逻辑）
    for dim in dims:
        series = pd.Series(raw[dim])
        raw[dim] = series.clip(
            series.quantile(0.05), series.quantile(0.95)
        ).tolist()
    
    # 3. 因子正交化（复用 batch_normalize 逻辑）
    if len(candidates) >= 10:
        for target, regressor in StockScorer.ORTHOGONALIZE_PAIRS:
            if target not in raw or regressor not in raw:
                continue
            t_vals, r_vals = raw[target], raw[regressor]
            n = len(t_vals)
            mu_t = sum(t_vals) / n
            mu_r = sum(r_vals) / n
            cov_tr = sum((t - mu_t)*(r - mu_r) for t, r in zip(t_vals, r_vals)) / n
            var_r = sum((r - mu_r)**2 for r in r_vals) / n
            if var_r < 1e-6:
                continue
            beta = cov_tr / var_r
            raw[target] = [t - beta * (r - mu_r) for t, r in zip(t_vals, r_vals)]
    
    # 4. Z-Score + per-stock 加权
    for i, c in enumerate(candidates):
        z_total = 0.0
        z_scores = {}
        stock_weights = c.get("stock_weights", global_weights)  # fallback 全局
        for dim in dims:
            vals = raw[dim]
            mu = sum(vals) / len(vals)
            sigma = (sum((v - mu)**2 for v in vals) / len(vals)) ** 0.5
            z = 0.0 if sigma < 1e-6 else max(-3.0, min(3.0, (vals[i] - mu) / sigma))
            z_scores[dim] = round(z, 3)
            z_total += z * stock_weights.get(dim, 0)
        score = 50 + z_total * (50 / 3)
        c["score"] = round(max(0, min(100, score)), 1)
        c["z_scores"] = z_scores
```

#### 5.4 `update_future_returns(db, kc, today) -> int`
**职责**：对所有 `trade_date + N <= today` 的历史推荐补算 T+N 收益。

**实现**：扩展 [run_backtest.future_returns](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py#L98)：
- offset 列表：`[(1,"d1_return"),(3,"d3_return"),(5,"d5_return"),(10,"d10_return"),(14,"d14_return")]`
- 只补算 `trade_date + N <= today` 的推荐（避免未来函数）
- 跳过已有收益的推荐（幂等）

```python
def update_future_returns(db, kc, today):
    """补算历史推荐的 T+N 收益（N=1,3,5,10,14）。
    只处理 trade_date + N <= today 的推荐，避免未来函数。
    """
    offsets = [(1,"d1_return"),(3,"d3_return"),(5,"d5_return"),
               (10,"d10_return"),(14,"d14_return")]
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    count = 0
    
    for rec in db.get_recent_recommendations(days=90):
        date_str = rec.get("trade_date", "")
        code = rec.get("code", "")
        if not date_str or not code:
            continue
        trade_dt = datetime.strptime(date_str, "%Y-%m-%d")
        raw_code = code.replace("sh","sh.").replace("sz","sz.")
        rows = kc.get(raw_code)
        if not rows:
            continue
        entry = rec.get("current_price", 0)
        if entry <= 0:
            continue
        future = [r for r in rows if r[0] > date_str]
        
        for off, col in offsets:
            # 未来函数检查
            if (trade_dt + timedelta(days=off)).strftime("%Y-%m-%d") > today:
                continue
            # 幂等：已有值则跳过
            if rec.get(col) is not None:
                continue
            if len(future) >= off:
                ep = float(future[off-1][1])
                if ep > 0:
                    db.update_performance(date_str, code, **{col: round((ep/entry-1)*100, 2)})
                    count += 1
    return count
```

#### 5.5 `run_phase(dates, phase_name, db, stock_mgr, global_weights) -> Dict`
**职责**：执行一个阶段的 walk-forward 训练。

```python
def run_phase(dates, phase_name, db, stock_mgr, global_weights):
    print("="*60)
    print(f"  Walk-Forward: {dates[0]} → {dates[-1]} ({len(dates)}天) [{phase_name}]")
    print("="*60)
    
    # 1. 获取股票名单（用阶段最后一天的状态）
    all_stocks = get_stocks(dates[-1])
    print(f"  候选: {len(all_stocks)} 只")
    
    # 2. 预取 K 线（start=阶段首-60交易日, end=阶段尾+14交易日）
    start = (datetime.strptime(dates[0], "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    end = (datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=21)).strftime("%Y-%m-%d")
    kc = prefetch(all_stocks, start, end)
    
    # 3. 预热 per-stock 权重缓存
    stock_mgr.warmup_cache()
    
    # 4. 初始化 FeedbackLoop（传入 stock_weight_manager）
    fb = FeedbackLoop({}, stock_weight_manager=stock_mgr)
    
    # 5. 逐日 walk-forward
    daily_results = []
    for i, today in enumerate(dates):
        print(f"\n  [{i+1}/{len(dates)}] {today}:", end="", flush=True)
        
        # 5a. 计算 21 维 sub_scores
        candidates = compute_sub_scores(today, all_stocks, kc)
        if not candidates:
            print(" 无数据"); continue
        
        # 5b. 批量加载 per-stock 权重
        codes = [c["code"] for c in candidates]
        weights_by_code = stock_mgr.get_weights_batch(codes)
        for c in candidates:
            c["stock_weights"] = weights_by_code.get(c["code"], global_weights)
        
        # 5c. 截面评分（per-stock 权重加权）
        score_cross_sectional_per_stock(candidates, global_weights)
        
        # 5d. 选 top 10 + 写 recommendations + factor_snapshots
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top10 = candidates[:10]
        for rank, c in enumerate(top10, 1):
            db.save_recommendation(
                today, c["code"], c["name"], c["score"], c["price"],
                rank=rank, mode="feedback_training"
            )
            db.save_factor_snapshot(
                today, c["code"], c["name"], c["score"],
                sub_scores=c["sub_scores"],
                factors=json.dumps({"z_scores": c.get("z_scores", {})}),
            )
        
        # 5e. 补算历史推荐 T+N 收益
        n_updated = update_future_returns(db, kc, today)
        
        # 5f. 反馈循环（归因 + IC + 全局调权 + 个股调权，skip_ml=True）
        fb_result = fb.run_all(today=today, skip_ml=True)
        
        # 5g. 个股权重缓存失效（下次循环重新加载）
        # 注意：adjust_weights 已更新缓存，这里无需 invalidate
        
        daily_results.append({
            "date": today,
            "n_candidates": len(candidates),
            "top10_score": round(sum(c["score"] for c in top10)/10, 1),
            "n_returns_updated": n_updated,
            "n_attributions": fb_result.get("n_attributions", 0),
            "n_stock_weight_changes": sum(len(v) for v in (fb_result.get("stock_weight_changes") or {}).values()),
        })
        print(f" 候选{len(candidates)}→推荐10, 收益更新{n_updated}, 归因{fb_result.get('n_attributions',0)}")
    
    return {"phase": phase_name, "dates": dates, "daily": daily_results}
```

#### 5.6 `main()` 命令行入口

```python
def main():
    parser = argparse.ArgumentParser(description="反馈训练 Walk-Forward")
    parser.add_argument("--smoke-test", action="store_true", help="小样本验证（10只×3天）")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="运行指定阶段")
    parser.add_argument("--all", action="store_true", help="运行全部两阶段")
    args = parser.parse_args()
    
    # 初始化
    bs.login()
    db = StockDatabase()
    
    # 加载全局权重（config.yaml）
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    global_weights = config.get("weights", {})
    # 保证 21 维齐全
    for f in ALL_FACTORS:
        global_weights.setdefault(f, 0.0)
    global_weights = _normalize(global_weights)
    
    stock_mgr = StockWeightManager(db, global_weights)
    
    if args.smoke_test:
        # 小样本：10 只股票 × 3 天（6/29, 6/30, 7/1）
        smoke_dates = DATES_PHASE1[:3]
        # 覆盖 get_stocks 只返回前 10 只
        _orig_get_stocks = get_stocks
        globals()["get_stocks"] = lambda d: _orig_get_stocks(d)[:10]
        run_phase(smoke_dates, "冒烟测试", db, stock_mgr, global_weights)
        globals()["get_stocks"] = _orig_get_stocks
    elif args.phase == 1:
        run_phase(DATES_PHASE1, "2026年7月", db, stock_mgr, global_weights)
    elif args.phase == 2:
        run_phase(DATES_PHASE2, "2026年上半年", db, stock_mgr, global_weights)
    elif args.all:
        run_phase(DATES_PHASE1, "2026年7月", db, stock_mgr, global_weights)
        run_phase(DATES_PHASE2, "2026年上半年", db, stock_mgr, global_weights)
    else:
        parser.print_help()
    
    bs.logout()
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
```

#### 5.7 DATES_PHASE2 补全

在实现时，需要在 `DATES_PHASE2` 中补全 2026/01~06 的 A 股交易日（约 120 天）。可参考 A 股 2026 年节假日安排：
- 元旦：1/1-1/4（休市）
- 春节：2/16-2/22（休市）
- 清明：4/4-4/6（休市）
- 劳动节：5/1-5/5（休市）
- 端午：6/19-6/21（休市）

**实现策略**：先用 baostock `query_trade_dates` 一次性拉取 2026/01~06 的交易日列表，硬编码到 DATES_PHASE2。或直接参考节假日手动列出。

### Step 6：报告输出 + 小样本验证

**报告生成函数**（直接在 `run_feedback_training.py` 内实现，不新建文件）：

#### 6.1 `generate_report(db, phase_result, output_path)`

```python
def generate_report(db, phase_result, output_path):
    """生成 markdown 报告到 reports/feedback_training_<phase>.md"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. 阶段汇总
    daily = phase_result["daily"]
    total_recs = len(daily) * 10
    
    # 2. 推荐表现（T+14 主信号）
    recs = db.get_recent_recommendations(days=90)
    recs_in_phase = [r for r in recs if phase_result["dates"][0] <= r.get("trade_date","") <= phase_result["dates"][-1]]
    d14_recs = [r for r in recs_in_phase if r.get("d14_return") is not None]
    d14_wins = sum(1 for r in d14_recs if r["d14_return"] > 0)
    avg_d14 = sum(r["d14_return"] for r in d14_recs) / len(d14_recs) if d14_recs else 0
    
    # 3. 个股权重变化 Top10
    all_codes = set(r["code"] for r in recs_in_phase)
    weight_changes = []
    for code in all_codes:
        history = db.get_stock_weight_history(code, days=90)
        for h in history:
            weight_changes.append({
                "code": code, "factor": h["factor_name"],
                "old": h["old_weight"], "new": h["new_weight"],
                "change": (h["new_weight"] or 0) - (h["old_weight"] or 0),
                "reason": h["reason"],
            })
    weight_changes.sort(key=lambda x: abs(x["change"]), reverse=True)
    
    # 4. 因子归因统计
    attr_stats = db.get_attribution_stats(days=90)
    
    # 5. 写入 markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 反馈训练报告 - {phase_result['phase']}\n\n")
        f.write(f"**阶段**: {phase_result['dates'][0]} → {phase_result['dates'][-1]} ({len(daily)}天)\n\n")
        f.write(f"## 阶段汇总\n\n")
        f.write(f"- 总推荐: {total_recs} 只\n")
        f.write(f"- 有 T+14 收益: {len(d14_recs)} 只\n")
        f.write(f"- T+14 胜率: {d14_wins}/{len(d14_recs)} = {d14_wins/len(d14_recs)*100:.1f}%\n" if d14_recs else f"- T+14 胜率: N/A\n")
        f.write(f"- 平均 T+14 收益: {avg_d14:.2f}%\n\n")
        
        f.write(f"## 每日推荐明细\n\n")
        f.write(f"| 日期 | 候选数 | Top10均分 | 收益更新 | 归因数 | 个股调权数 |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for d in daily:
            f.write(f"| {d['date']} | {d['n_candidates']} | {d['top10_score']} | {d['n_returns_updated']} | {d['n_attributions']} | {d['n_stock_weight_changes']} |\n")
        
        f.write(f"\n## 个股权重变化 Top10\n\n")
        f.write(f"| 股票 | 因子 | 旧权重 | 新权重 | 变化 | 原因 |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for w in weight_changes[:10]:
            f.write(f"| {w['code']} | {w['factor']} | {w['old']:.4f} | {w['new']:.4f} | {w['change']:+.4f} | {w['reason']} |\n")
        
        f.write(f"\n## 因子归因统计\n\n")
        if attr_stats and attr_stats.get("factor_accuracy"):
            f.write(f"| 因子 | 命中 | 失败 | 准确率 |\n")
            f.write(f"|---|---|---|---|\n")
            for factor, data in sorted(attr_stats["factor_accuracy"].items(), key=lambda x: x[1]["correct"]+x[1]["wrong"], reverse=True):
                total = data["correct"] + data["wrong"]
                acc = data["correct"] / total if total > 0 else 0
                f.write(f"| {factor} | {data['correct']} | {data['wrong']} | {acc*100:.1f}% |\n")
        else:
            f.write(f"无归因数据\n")
    
    print(f"  报告已生成: {output_path}")
```

#### 6.2 集成到 `run_phase` 末尾

在 `run_phase` 函数末尾调用：
```python
# 在 run_phase return 之前
report_path = f"reports/feedback_training_{phase_name}.md"
generate_report(db, {"phase": phase_name, "dates": dates, "daily": daily_results}, report_path)
```

#### 6.3 小样本验证流程

运行 `python run_feedback_training.py --smoke-test`，验证：
1. **推荐写入**：3 天 × 10 只 = 30 条 recommendations，`mode='feedback_training'`
2. **T+N 收益补算**：6/29 推荐的 T+1 收益在 6/30 当天被补算
3. **归因记录**：factor_attribution 表有写入（6/30 起对 6/29 推荐做归因）
4. **个股权重更新**：stock_factor_weights 表有记录（样本数 ≥ 3 后才调权，所以 3 天可能不足，需观察）
5. **报告生成**：`reports/feedback_training_冒烟测试.md` 文件存在且内容合理

## 假设与决策（本计划新增）

### 假设
1. baostock 能提供 2026/01~07 的历史 K 线数据（[run_backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py) 已验证可用）
2. `backtest_factors.calc_sub_scores_from_kline` 输出 21 维齐全（Step 4 验证后确认）
3. [db.save_factor_snapshot](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py) 接受 `sub_scores` 和 `factors` 参数（需在实现时确认签名）
4. [db.save_recommendation](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py) 接受 `mode` 参数（[run_backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py#L130) 已用 `mode="backtest"`，应支持）

### 决策
1. **per-stock 截面评分**：Z-Score 全局计算 + per-stock 权重加权（决策1）
2. **日期列表**：硬编码 A 股交易日（决策2）
3. **ML 重训练**：回测全程 `skip_ml=True`（决策3）
4. **T+N 补算**：扩展 offset 到 d1/d3/d5/d10/d14，未来函数检查（决策4）
5. **股票池**：复用 run_backtest.get_stocks（决策5）
6. **报告生成**：在 `run_feedback_training.py` 内实现，不新建文件（保持文件整洁）
7. **小样本验证**：10 只股票 × 3 天（6/29, 6/30, 7/1），用真实 baostock 数据

## 验证步骤

### Step 4 验证（单文件模块）
```bash
python _test_step4.py
```
**预期输出**：
- DataFrame shape: (30, 9)
- 因子数 ≥ 10（9 个因子类输出）
- 21 维 sub_scores 齐全，缺失维度 = []
- 9 维可重建因子有真实值（非 50），12 维中性 = 50
- 切片测试通过

**通过后**：删除 `_test_step4.py`

### Step 5 验证（小样本）
```bash
python run_feedback_training.py --smoke-test
```
**预期**：
- 3 天 walk-forward 跑通，无异常
- recommendations 表新增 30 条 `mode='feedback_training'` 记录
- factor_snapshots 表新增 30 条记录
- 6/30 当天补算 6/29 推荐的 d1_return
- 7/1 当天补算 6/29 推荐的 d3_return 和 6/30 的 d1_return
- factor_attribution 表有归因记录
- `reports/feedback_training_冒烟测试.md` 生成

### Step 6 验证（阶段1）
```bash
python run_feedback_training.py --phase 1
```
**预期**：
- 25 天 walk-forward 跑通（约 15-30 分钟）
- recommendations 表新增 250 条记录
- T+14 胜率在 45-65% 之间
- 个股权重有变化（stock_weight_history 表非空）
- `reports/feedback_training_2026年7月.md` 生成

### Step 6 验证（阶段2）
```bash
python run_feedback_training.py --phase 2
```
**预期**：
- ~120 天 walk-forward 跑通（约 60-120 分钟）
- `reports/feedback_training_2026年上半年.md` 生成

## 文件清单

### 新建
- `run_feedback_training.py` — walk-forward 主入口（Step 5）
- `reports/feedback_training_冒烟测试.md` — 小样本报告（运行后生成）
- `reports/feedback_training_2026年7月.md` — 阶段1 报告（运行后生成）
- `reports/feedback_training_2026年上半年.md` — 阶段2 报告（运行后生成）

### 删除
- `_test_step4.py` — Step 4 验证后删除

### 不修改
- [src/data/database.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py) — Step 1 已完成
- [src/engine/stock_weight_manager.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/stock_weight_manager.py) — Step 2 已完成
- [src/engine/feedback_loop.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/feedback_loop.py) — Step 3 已完成
- [src/engine/backtest_factors.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest_factors.py) — Step 4 已完成（待验证）
- [run_backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_backtest.py) — 保留作参考
- [run_daily.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_daily.py) — 生产流程不动
- [config.yaml](file:///c:/Users/EDY/Desktop/StockPrediction/config.yaml) — 全局权重作 fallback
- [src/engine/scorer.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py) — 复用 batch_normalize 逻辑

## 风险与缓解

1. **风险：`backtest_factors.calc_sub_scores_from_kline` 输出维度不全**
   - 缓解：Step 4 验证脚本检查 21 维齐全，缺失维度打印出来
   - 回退：检查 `calc_factors_from_kline` 中各因子类是否抛异常被吞掉

2. **风险：per-stock 截面评分在大样本下性能问题**
   - 缓解：每天约 4500 只候选，截面 Z-Score 计算复杂度 O(n×21)，可接受
   - 监控：smoke-test 先验证 10 只×3 天性能

3. **风险：DATES_PHASE2 补全工作量大**
   - 缓解：用 baostock `query_trade_dates` 一次性拉取 2026/01~06 交易日列表

4. **风险：baostock 限流导致 K 线预取慢**
   - 缓解：复用 run_backtest.py 的 10 线程池 + 一次性预取全阶段数据

5. **风险：小样本 3 天不足以触发个股权重调整（MIN_SAMPLES=3）**
   - 缓解：smoke-test 主要验证流程跑通，个股权重调整在 phase 1 完整跑时验证
   - 可选：smoke-test 时临时把 MIN_SAMPLES 降到 1（不推荐，会污染验证）

6. **风险：`save_factor_snapshot` / `save_recommendation` 签名不匹配**
   - 缓解：Step 5 实现时先读取 [database.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py) 确认方法签名

## 实施顺序

1. **Step 4 验证**：运行 `_test_step4.py` → 通过则删除 → 标记 Step 4 完成
2. **Step 5 实现**：新建 `run_feedback_training.py`
   - 先确认 `db.save_factor_snapshot` / `db.save_recommendation` / `db.update_performance` 签名
   - 补全 `DATES_PHASE2`（用 baostock `query_trade_dates`）
   - 实现 `fetch_kline`/`prefetch`/`get_stocks`（复用 run_backtest.py）
   - 实现 `compute_sub_scores`（调用 backtest_factors）
   - 实现 `score_cross_sectional_per_stock`（复用 batch_normalize 逻辑）
   - 实现 `update_future_returns`（扩展 offset）
   - 实现 `run_phase`（主循环）
   - 实现 `generate_report`
   - 实现 `main`（argparse）
3. **Step 5 验证**：`python run_feedback_training.py --smoke-test`
4. **Step 6 验证**：`python run_feedback_training.py --phase 1` → `--phase 2`
5. **最终**：返回完成报告给用户

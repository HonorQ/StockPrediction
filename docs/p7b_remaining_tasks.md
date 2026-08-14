# P7B 终审计划

> 版本：v9（最终版）
> 基于：10轮评审 + 终审评审（发现关键遗漏）

---

## 一、现状摘要

### 核心Bug

| 路径 | 传入数据 | 期望数据 | 影响 |
|------|---------|---------|------|
| `run_feedback_training.py:488` | `sub_scores` ("trend","strategy") | `factors` ("rsi_6","rsi_14") | 回测ML融合无效 |
| `feedback_loop.py:252` | `sub_scores` (维度名) | `factors` (特征名) | 训练数据45特征全=50 |
| `ml_selector.py:408,612` | `sub_scores` (维度名) | `factors` (特征名) | 影子ML训练同样无效 |

### 生产代码状态

`run_daily.py:1625` ✅ **已正确**：`ml.predict(c.get("factors", {}))` — 传的是原始因子

### 对比基线

| 运行 | 状态 | 说明 |
|------|------|------|
| A2 (α=1.0, 无ML) | ⏳ 运行中 | 不受bug影响，结果有效 |
| P7B Phase 1 (α=0.8) | ❌ 作废 | ML评分=50常量，结果无效 |

---

## 二、修复计划（3个文件7处修改）

### 修改清单（3个文件7处）

```python
# === 文件1: run_feedback_training.py (3处) ===

# 修改A: compute_sub_scores() — 同时返回原始因子
# 在 calc_sub_scores_from_kline(rows) 后添加
factors = calc_factors_from_kline(rows)
results.append({
    "code": ..., "sub_scores": sub_scores,
    "factors": factors,  # ← 新增
})

# 修改B: ML推理 — 改传 factors
pred = ml_selector.predict(c.get("factors", {}))

# 修改C: save_factor_snapshot — 存原始因子
factors=c.get("factors", {})


# === 文件2: feedback_loop.py (2处) ===

# 修改D: _build_ic_dataset — 读 factors 替代 sub_scores
for snap in snapshots:
    row = {"trade_date": snap["trade_date"], "stock_code": snap["code"]}
    factors = snap.get("factors", {})
    if not factors or "z_scores" in factors:
        continue  # 安全跳过未回填的快照
    for key, value in factors.items():
        row[key] = value
    ...


# === 文件3: ml_selector.py (3处) ===

# 修改E: _extract_features._v() — 防护非标量值
# 添加 isinstance 检查避免 float(dict) 崩溃
def _v(key):
    v = factors.get(key, 50)
    if isinstance(v, (list, dict, set, tuple)):
        return 50.0
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 50.0
    return float(v)

# 修改F: _load_factor_data — 读 factors 替代 sub_scores
factors = snap.get("factors", {})
if not factors or "z_scores" in factors:
    continue
row.update(factors)

# 修改G: train_shadow_ml — 读 factors 替代 sub_scores
factors = snap.get("factors", {})
if not factors or "z_scores" in factors:
    continue
row.update(factors)
```

---

## 三、执行顺序

```
① [立即]   修复代码（3个文件7处）
② [立即]   预验证（mock测试特征提取 + _v防护验证）
③ [A2完成] 立即保存A2报告（重命名以防被覆盖）：
              copy reports\feedback_training_2026年7月.md reports\A2_baseline.md
④ [新进程] 回填旧因子快照（关键步骤）：
              python backfill_factor_snapshots.py
              （遍历6600行已有snapshot，从kline缓存计算原始因子并更新factors列）
⑤ [新进程] 清理旧模型 + model_versions：
              del models\stock_selector.pkl
              del models\stock_selector_meta.json
              python -c "import sqlite3; c=sqlite3.connect('data/stock_history.db'); c.execute('DELETE FROM model_versions'); c.commit()"
⑥ [新进程] 重跑Phase 1：
              python run_feedback_training.py --phase 1 --ml-alpha 0.8 --force
⑦           对比新Phase 1 vs A2_baseline.md → 决策
```

---

## 四、验收标准

| 场景 | 新Phase1 T+5 vs A2 T+5 | 结论 | 行动 |
|------|-----------------------|------|------|
| ✅ ML有效 | 高 ≥5pp | ML有真实信号 | 模型文件已就位，run_daily.py重启即生效 |
| ⚠️ ML微弱 | 高 2-5pp | 信号弱 | 调参或增特征后重测 |
| ❌ ML无效 | ≈ 或 < | ML无价值 | 保持skip_ml=True，放弃P7B ML融合 |

---

## 五、预验证脚本（步骤②）

修复代码后，立即执行预验证确认特征提取正确：

```bash
python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from src.engine.ml_selector import MLStockSelector
s = MLStockSelector()
raw = {'rsi_6': 65, 'rsi_14': 58, 'macd_golden_cross': 1, 'ma_bullish': 1,
       'breakout_20d_high': 0, 'volume_price_up': 1, 'strategy_count': 3,
       'chip_profit_ratio': 0.6, 'chip_concentration_70': 0.7, 'chip_above_peak': 0.3,
       'pe_score': 55, 'pb_score': 48, 'market_cap_score': 62, 'turnover_score': 45,
       'inst_holding_count': 15, 'inst_holding_ratio': 0.4,
       'relative_strength_5d': 2.5, 'relative_strength_20d': 1.8,
       'oversold_rebound_score': 30, 'rotation_score': 60, 'tech_sector_strength': 55,
       'news_score': 45, 'expectation_score': 50, 'mean_reversion_score': 40,
       'volatility_score': 55, 'position_score': 65, 'quant_score': 50,
       'margin_score': 45, 'shareholder_score': 50, 'lockup_score': 40,
       'research_score': 55, 'lh_score': 50, 'leader_score': 60,
       'limit_up_20d': 2, 'hot_money_score': 55}
feats = s._extract_features(raw).flatten()
non_default = [x for x in feats if abs(float(x) - 50) > 0.001]
print(f'Features: {len(feats)}, Varying: {len(non_default)}')
print('PASS' if len(non_default) > 30 else 'FAIL - features not varying')
"
```

---

## 六、附录：回填脚本（步骤④）

修复后执行此脚本，将已有 factor_snapshots 的 `factors` 列从 `{"z_scores":...}` 更新为原始因子 dict：

新建 `backfill_factor_snapshots.py`：

```python
import sys, os, json, sqlite3, pickle, hashlib, logging
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from src.data.database import StockDatabase
from src.engine.backtest_factors import calc_factors_from_kline, slice_kline_by_date

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "klines")
CACHE_KEY = "phase2_full"

def cache_path(code):
    return os.path.join(CACHE_DIR, f"{CACHE_KEY}_{hashlib.md5(code.encode()).hexdigest()[:16]}.pkl")

def load_klines(code):
    cp = cache_path(code)
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)
    return None

db = StockDatabase()
conn = sqlite3.connect(db.db_path)
snapshots = conn.execute(
    "SELECT id, trade_date, code FROM factor_snapshots WHERE trade_date >= '2026-04-01' ORDER BY trade_date"
).fetchall()
print(f"Found {len(snapshots)} snapshots to backfill")

count, errors = 0, 0
for sid, tdate, code in snapshots:
    raw_code = code[:2] + '.' + code[2:] if '.' not in code else code
    rows = load_klines(raw_code)
    if not rows or len(rows) < 20:
        errors += 1; continue
    sliced = slice_kline_by_date(rows, tdate)
    if len(sliced) < 20:
        errors += 1; continue
    raw_factors = calc_factors_from_kline(sliced)
    if raw_factors:
        conn.execute("UPDATE factor_snapshots SET factors=? WHERE id=?", (json.dumps(raw_factors), sid))
        count += 1
    if count % 500 == 0:
        conn.commit()
conn.commit()
print(f"Updated: {count}, Skipped: {errors}")
conn.close()
```

---

## 七、已确认无需修改

| 组件 | 状态 |
|------|------|
| `run_daily.py` 生产ML推理 | ✅ 传factors，路径正确 |
| A2基线 (α=1.0) | ✅ 不依赖ML，结果有效 |
| 因子评分系统 (sub_scores) | ✅ 不受影响 |
| 日志系统 (_DualOutput) | ✅ 已修复 |

---

## 八、时间预估

| 步骤 | 耗时 |
|------|------|
| 修复代码 | ~10 min |
| 预验证 | ~2 min |
| A2剩余时间 | ~15 min |
| 回填旧因子快照 | ~10 min |
| 清理旧模型 | ~1 min |
| 重跑Phase 1 (25天) | ~30-60 min |
| **总计** | **~1.5-2小时** |

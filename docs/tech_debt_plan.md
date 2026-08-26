# AGENTS.md 技术债偿还方案（2026-08-26）

> 本文件记录剩余 7 个超长函数的具体拆分方案，供新会话执行。
> 每项包含：当前行数、目标、具体切割点、验证方式。
> 完成后请删除对应条目并在 commit 中注明。

---

## 快速参考

- 拆分工具: `C:\Users\EDY\AppData\Local\Temp\opencode\func_splitter.py`
  （支持 ctx 模式函数；普通函数需用 `extract_loop2.py` 的自由变量分析法）
- 验证命令: `python -m pytest tests/ -q && python -m ruff check .`
- 规则来源: `AGENTS.md` 第二节

---

## 剩余 7 个超标函数

### 1. `_pm_p2s1` — 126 行（超 6 行）
- **文件**: `src/pipeline/premarket_finalize.py`
- **难度**: ★☆☆☆☆
- **方案**: 与 `_pm_quick_topn/_pm_deep_loop` 相同——压缩 unpack 排版
  （5名/行→10名/行），可省约 12 行。参考已完成的
  `compact_v2.py` 脚本逻辑（在 Temp 目录，如已清理可重写）。
- **注意**: 该函数有重复的 docstring（`"""阶段4之1"""` 出现两次），
  顺手清理。

### 2. `run_intraday` — 204 行
- **文件**: `src/pipeline/intraday.py`
- **难度**: ★★☆☆☆
- **方案**: 两刀切分：
  - 刀1: 竞价快照落库 + 崩盘检测段提取为 `_id_snapshot_and_crash(ctx)`
  - 刀2: 预警输出+候选刷新段提取为 `_id_alerts_and_rescore(ctx)`
- **切割点**（语句索引）: 找 `[1/1]` print 和 `竞价快照落库` 注释行
- **验证**: `python -m pytest tests/ -q`；下次 09:25 计划任务实战

### 3. `train` — 183 行
- **文件**: `src/engine/ml_selector.py`
- **难度**: ★★☆☆☆
- **方案**: 三段切分：
  - `_train_prepare`: 特征矩阵构建 + 时序切分 + embargo
  - `_train_search`: LightGBM + RandomForest 超参搜索（含 _val_auc）
  - `_train_evaluate`: 测试集评估 + feature_importance + 版本保存
- **注意**: 这是普通函数无 ctx，切出的段需要显式参数传递或用类化改造。
  建议 MLStockSelector 类加一个 `self._train_state` dict 传递中间状态。

### 4. `_build_factor_rows` — 300 行
- **文件**: `src/engine/reporter.py`
- **难度**: ★★★☆☆
- **方案**: 函数内部已有清晰的类别分组注释。按因子族提取为独立方法：
  - `_factor_rows_tech(factors, add)` → 技术面(MACD/均线/量价/突破)
  - `_factor_rows_sentiment(factors, add)` → 情绪面(涨停/热点/游资)
  - `_factor_rows_fundamental(factors, add)` → 基本面(PE/PB/市值/机构)
  - `_factor_rows_event(factors, add)` → 事件面(政策/新闻/解禁/量化)
  - 主函数保留 `add()` 定义 + 依次调用四个子方法 + 返回 rows
- **切割点**: 按 `add(` 调用的类别分界线，每类约 50~80 行

### 5. `_init_db` — 302 行
- **文件**: `src/data/db/base.py`
- **难度**: ★★☆☆☆
- **方案**: 单个 with 块内全部 CREATE TABLE/INDEX。按表域拆为子函数，
  每个接收 conn 参数：
  - `_create_recommendations_tables(conn)`
  - `_create_factor_tables(conn)`
  - `_create_paper_trade_tables(conn)`
  - `_create_weight_ic_tables(conn)`
  - `_create_kline_auction_tables(conn)`
  - `_migrate_schema(conn)` 也同理拆分
- **切割点**: 每组 CREATE TABLE + INDEX 之间有空行分隔
- **注意**: `_init_db` 内的 `conn.execute("PRAGMA ...")` 三行留在最前

### 6. `_calc_one_stock` — 268 行
- **文件**: `src/pipeline/premarket_deep.py`
- **难度**: ★★★★☆
- **方案**: 单票因子计算流水线。按因子族提取为4个子方法：
  - `_calc_tech_block(ctx, cand, history)` → TechnicalFactors + strategies + chip (~80行)
  - `_calc_flow_block(ctx, cand, flow_data)` → 资金流 + money_flow_factor (~40行)
  - `_calc_event_block(ctx, cand, stock_data, news_list)` → 政策/新闻/宏观/量化行为 (~80行)
  - `_calc_composite_block(ctx, all_factors)` → 合成评分 + deep_research (~60行)
  主循环调用顺序不变：tech → flow → event → composite
- **风险**: 因子间有少量依赖（如 chip_score 被 deep_research 引用），
  需确保 all_factors dict 在各块之间正确传递。
- **建议**: 用 AST 自由变量分析法（同 extract_loop2.py 模式）确定各段的
  输入输出变量集合。

### 7. `run_afternoon` — 367 行
- **文件**: `src/pipeline/afternoon.py`
- **难度**: ★★★☆☆
- **方案**: 四段切分（同 premarket 的 ctx 模式）：
  - `_am_init(ctx)`: 组件实例化 + 缓存载入 + 告警检查 (L39~92)
  - `_am_rescore(ctx)`: 逐票修正评分循环 (L95~248, 含114行大循环)
    - 大循环可再提取为 `_am_rescore_one(ctx, c, idx)` 
  - `_am_filter_persist(ctx)`: 健康度降级 + 筛选 + DB保存 + 开仓 (L254~330)
  - `_am_report(ctx)`: 报告生成 + 总结输出 (L331~末尾)
- **注意**: afternoon.py 没有 Ctx 类，需从 common.py 导入或复制。
  run_afternoon 当前是普通函数（config 为参数），
  需改为 `ctx = {"config": config}` 开头。

---

## 建议执行顺序

```
第一批(简单): #1 _pm_p2s1 → #2 run_intraday → #3 train     [半天]
第二批(中等): #5 _init_db → #7 run_afternoon                [半天]
第三批(复杂): #4 _build_factor_rows → #6 _calc_one_stock    [一天]
```

## 完成标准

每个函数修复后必须满足：
1. `python -c "import ast; ast.parse(open(文件).read())"` 语法通过
2. `python -m ruff check .` 无新增错误
3. `python -m pytest tests/ -q` 全绿
4. 函数 ≤120 行（unpack 机械代码可豁免但需注释说明）
5. 相关计划任务下次触发时正常运行

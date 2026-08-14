# ML 训练与因子优化计划

> 编制时间:2026-07-31
> 编制背景:paper_trades 9 笔平均亏损 -7.47%、胜率 11.1%,需建立科学的因子优化闭环提升策略表现。
> 适用范围:本项目(StockPrediction)的因子权重调优与 ML 模型训练。
> 版本:v5(2026-07-31 终稿,落实第四轮评审意见)

---

## 一、目标

建立**可验证、防过拟合**的因子权重优化闭环,回答三个问题:
1. 哪些因子真正有预测能力?(IC/ICIR)
2. 在不同市场环境下表现如何?(Regime 分层)
3. 调整后的权重在样本外是否稳定?(滚动验证)

**禁止目标**:用历史推荐结果反复调参追求"历史胜率漂亮"——这是 data snooping,实盘必失效。

---

## 二、现状盘点

### 2.1 数据资产(截至 2026-07-31)

| 数据 | 规模 | 时间范围 | 评估 |
|---|---|---|---|
| recommendations 表 | 5952 行 / 276 只 | 2026-01-05 ~ 07-31 | 7 个月,样本偏少 |
| factor_snapshots 表 | 6330 行 / 307 只 | 2026-01-05 ~ 07-31 | 因子截面快照 |
| factor_attribution 表 | 101 万行 / 589 只 | 2026-01-05 ~ 07-27 | 归因数据充足 |
| d1/d3/d5/d10/d14_return | 全部回填 | 截至 2026-06-30 | T+N 真实收盘价,口径可信 |
| paper_trades | 11 笔 / 9 只 | 2026-07-23 ~ 07-31 | 含止损/止盈,样本严重不足 |
| ic_history | 6828 行 / 14+ 因子 | 2026-06-29 ~ 07-31 | 已有 IC 基础数据 |
| **原始 K线** | **不落库** | — | 每次实时拉取,无法离线全市场回测 |

> 说明:[factor_snapshots 表](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L93) 同时保存原始因子 JSON(`factors` 列)和 24 维子评分独立列(`sub_trend`、`sub_capital_behavior` 等,见 [database.py:30 SUB_SCORE_DIMS](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L30))。IC 计算应**优先基于 24 维 `sub_*` 独立列**,这样与 config.yaml 权重一一对应,可直接指导权重调整;原始因子 JSON 仅作辅助诊断(如 IC 异常时下钻到具体指标)。

### 2.2 已有能力

| 能力 | 位置 | 状态 |
|---|---|---|
| T+N 真实收盘价回填 | run_daily.py:1316-1340 | ✅ |
| weight_history 落库 | [src/data/database.py:198](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L198), feedback_loop.py:399 | ✅ |
| 防前视 today 参数 | feedback_loop.py:52 | ✅ |
| 日频 Rank IC/ICIR 计算 | [src/engine/ml_selector.py:308](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/ml_selector.py#L308) | ✅(按日分组 spearman) |
| BacktestEngine.run_walk_forward | [src/engine/backtest.py:31](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py#L31) | ⚠️ 实为滚动窗口回测,无训练集调参 |
| AttributionAggregator 因子归因 | [src/engine/factor_attribution.py:120](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/factor_attribution.py#L120) | ✅(仅全量,无 regime 分组) |

### 2.3 关键缺口

1. **按推荐日遍历+胜率统计脚本**:不存在
2. **因子×时间段二维聚合**:AttributionAggregator 未实现
3. **BacktestEngine 因子覆盖不全**:[src/engine/backtest.py:343](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py#L343) 的 `_calc_factors` 缺 capital_behavior / ashare_specific / moat 等因子族,违反规则 §6
4. **Regime 字段未入 recommendations 表**:无法按市场状态分层归因
5. **原始 K线不落库**:无法做需要重算因子的全市场回测
6. **walk-forward 缺少训练集权重拟合**:[src/engine/backtest.py:31](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py#L31) 仅滚动用固定 weights 回测,未在 train 窗口内优化权重

### 2.4 数据范围硬约束

- **近半年可做**(2026-01 ~ 06,d3_return 已回填)
- **近一年做不到**:数据只有 7 个月
- **近一年半做不到**:无数据
- **7 月数据 d3_return 未回填**:T+3 还未到或 premarket 中断
- **滚动验证窗口有限**:按 train=4 月 / test=1 月,1~6 月仅能得到 **2 个验证窗口**(1-4→5、2-5→6),统计显著性极弱

---

## 三、方法论选型

### 3.1 放弃方案:遍历历史推荐 + 调权重 + 看胜率提升

**放弃理由(5 个硬伤)**:

1. **Data Snooping**:用历史推荐结果调权重,再用同一批数据验证,等于用测试集训练 → 过拟合,实盘必失效
2. **胜率非正确目标**:胜率 65% + 盈亏比 0.5 = 亏钱;胜率 50% + 盈亏比 2.0 = 赚钱。应优化期望收益 α = 胜率×平均盈利 − 败率×平均亏损
3. **选择偏差**:recommendations 只存被推荐的票,调权重后新选入的票无历史收益可比,等于只看幸存者
4. **T+N ≠ 实盘**:T+3 固定持仓收益与实盘止损/止盈/到期平仓收益不同,调出来的权重实盘未必有效
5. **样本不足**:7 个月 × 每天 2 只 ≈ 300 条,切分后每段几十条,统计不显著

### 3.2 推荐方案:IC + 滚动验证 + Regime 分层

| 方法 | 用途 | 落地难度 | 优先级 |
|---|---|---|---|
| **IC/ICIR 分析** | 客观判断因子有效性,不带调参偏差 | 低(已有功能) | P0 立即做 |
| **分层回测** | 验证因子单调性(Q1~Q5 收益是否递增) | 中 | P1 |
| **Regime 分层归因** | 看因子在不同市场环境的表现 | 中(需加 regime 字段) | P1 |
| **滚动 IC 加权稳健性检验** | 用滚动窗口 IC 生成动态权重,检验稳定性 | 中 | P2 |
| **Walk-Forward 验证** | 训练集拟合权重,验证集看真实表现,防过拟合 | 高(需补齐因子+实现拟合) | P2(可选升级) |
| **paper_trades 实盘归因** | 含止损/止盈的真实收益归因 | 低(等样本) | P3(等 50+ 笔) |

**选型理由**:
- IC 是业界标准,不受样本量限制(每只股票每天的因子值都是样本)
- 滚动验证比单一切口更能检验权重稳定性
- Regime 分层比日历切分更有信号(同 regime 内因子表现一致)
- **当前代码的 `run_walk_forward` 并非真正的 walk-forward**,阶段 3 默认采用轻量的"滚动 IC 加权稳健性检验",若后续数据充足再升级为完整 walk-forward

---

## 四、分阶段实施计划

### 阶段 1:IC 分析(P0,立即可做,脚本开发 30-60 分钟 + 跑数 15 分钟)

**目标**:客观判断因子的有效性,不带调参偏差。

**动作**:

1. 先跑现有原始因子 IC(作下钻诊断用):
```bash
python run_daily.py --factor-ic
```
> 注意:当前 `run_factor_ic`([run_daily.py:1878](file:///c:/Users/EDY/Desktop/StockPrediction/run_daily.py#L1878))通过 [FactorOptimizer.load_from_database](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/ml_selector.py#L415) 读取 `factor_snapshots.factors` JSON 列(原始因子如 `atr_14`、`boll_width`、`momentum_60d`),输出**原始因子 IC**,不是 24 维 sub_* IC。这部分输出仅用于异常下钻。

2. **新写 `analyze_sub_ic.py`**(放项目根目录,与 `analyze_db_data.py` 等脚本一致),专门对 `factor_snapshots` 表的 24 个 `sub_*` 独立列(`sub_trend`、`sub_capital_behavior` 等,见 [SUB_SCORE_DIMS](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L30))计算日频 Rank IC/ICIR。24 维 IC 直接对应 config.yaml 权重,是权重调整的依据。

**口径说明**:
- IC 类型:**日频截面 Rank IC**(spearman),复用 [FactorOptimizer.calculate_ic_by_day](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/ml_selector.py#L501) 逻辑
- 收益口径:future_return 优先用**路径 A**(`factor_snapshots` JOIN `recommendations.d3_return`),与 IC 计算样本口径一致;对未推荐但入 snapshots 的票(`d3_return IS NULL`)直接剔除,在报告中标注有效样本量。路径 B(实时拉 T+3 收盘价)作为后备,仅在样本过少时启用,且需注意并发限制(规则 §4)
- **IC 计算对象**:
  - 24 维 `sub_*` 列(主,用于权重调整)
  - 原始因子(辅,仅用于 sub_* IC 异常时下钻诊断)
- **无需手工映射**:`sub_*` 独立列已通过 [_migrate_schema](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L318) 落库,`analyze_sub_ic.py` 直接遍历 `SUB_SCORE_DIMS` 即可

**输出**:
- `reports/factor_ic_*.md`:24 维子评分的 IC/ICIR/样本日数 + 原始因子下钻表(仅异常子评分)
- 阈值(按绝对值判断,基于日频 Rank IC):\|IC\|>0.03 值得注意,\|IC\|>0.05 有效,\|IC\|>0.1 强;\|ICIR\|>0.5 可用,>1.0 优秀
- IC 显著为负的子评分应考虑反向使用或剔除

**验收**:
- 产出 24 维子评分 IC 排名表
- 标记"有效因子 / 噪声因子 / 反向因子"三类
- **IC 不达标时的处理原则**:噪声因子降权至 ≤0.01;反向因子改为负向权重或从组合中剔除;有效因子保持/小幅提升权重,但单次调整幅度 ≤3pp

**风险**:无,纯只读分析。

---

### 阶段 2:Regime 分层归因(P1,需开发,耗时 2-3 小时)

**目标**:看因子在 bull / bear / crash / neutral 四种市场环境下的胜率差异。

**前置改动**:
1. recommendations 表加 `regime` 字段:
   - 在 [src/data/database.py:68](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L68) 建表语句中新增 `regime TEXT`
   - 在 [src/data/database.py:318 _migrate_schema](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L318) 中幂等补列,兼容旧库
2. [run_daily.py](file:///c:/Users/EDY/Desktop/StockPrediction/run_daily.py) 在保存 recommendations 时把 `_regime_info["regime"]` 写入该字段
3. 扩展 [AttributionAggregator.get_factor_stats](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/factor_attribution.py#L119) 加 regime 二级分组

**动作**:
- 新写 `evaluate_recs.py` 脚本(放在项目根目录,与现有 `analyze_training_data.py` / `analyze_db_data.py` 等分析脚本一致)
- 遍历 recommendations,关联 regime 字段
- 输出"因子 × regime"二维胜率/期望收益表

**输出**:
- `reports/regime_factor_attribution.md`
- 例:动量因子在 bull regime 胜率 65%,在 bear regime 胜率 30% → 考虑动态加权

**验收**:
- 产出二维归因表
- 识别"全 regime 有效"vs"仅特定 regime 有效"的因子

**风险**:
- 旧 recommendations 无 regime 字段,需回算。
- **前视风险**:[MarketRegimeFilter.assess_regime](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/market_regime.py#L42) 使用了当日 `sentiment`(涨停/跌停/炸板数)和指数涨跌,这些数据在盘前 entry 时刻无法完全获得。回算时必须使用 **T-1 日收盘后可得的 regime**,不能用 T 日盘中/收盘数据,否则违反规则 §1。
- **数据可获取性风险**:
  - 上证指数/深证成指/创业板指历史日线可通过 `MarketDataFetcher.get_stock_history` 获取,MA20 趋势可回算 ✅
  - 北向资金:需确认 akshare 是否提供历史日级北向净流入接口,若不可得则回算时该信号退化
  - **T-1 日涨跌停数(sentiment)**:live 端通过热点/概念数据实时计算,历史每日涨跌停数需额外拉取(可考虑 `ak.stock_zh_a_spot_em` 历史快照或从全市场 K 线按涨跌停阈值统计)。**阶段 2 启动前必须先验证该数据可获取性**:若历史 sentiment 拉不到,回算 regime 退化为"指数涨跌 + MA20 趋势"两项信号,判定精度低于 live,需在报告中明确标注
  - 外盘数据:历史外盘日涨跌可获取(全球指数日线),通常可得

---

### 阶段 3:滚动验证(P2,需较大开发)

**目标**:检验权重调整方案在样本外的稳定性,防止过拟合。

> 拆分为 3a(因子补齐)和 3b(滚动验证)两个子阶段,3a 是 3b 的硬前置。预计总耗时 1-2 个工作日(因子补齐可能因外部数据源依赖而延长)。

#### 阶段 3a:BacktestEngine 因子补齐(硬前置)

**目标**:让回测的因子计算与 live 完全对齐,消除两套代码漂移(规则 §6)。

**动作**:
- 改造 [src/engine/backtest.py:343 `_calc_factors`](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py#L343),补齐 capital_behavior / ashare_specific / ashare_extended / phase2_market_intel / moat 等因子族
- 逐个因子族确认 live 端的输入参数(资金流、龙虎榜、机构调研等),处理异常降级(规则 §4)
- **先按 [SUB_SCORE_DIMS 24 维](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L30) 逐个标注数据源类型**(不要按文档枚举为准),分两类验证:
  - **纯 K 线因子**(基于行情 K 线即可计算,如 trend / strategy / chip / ashare_specific / relative_strength / oversold_rebound / quant_behavior / leader / moat 等):回测与 live 在相同输入下 sub_scores 应一致
  - **外部数据因子**(依赖资金流/龙虎榜/机构调研/新闻/融资融券/股东/限售/研报/政策/国家队等外部源,如 capital_behavior / dragon_tiger / institutional / news / margin_trading / shareholder / lockup_risk / research / deep_research / policy_event / national_team / sector_rotation / holder_concentration / circulate_holder / fundamental 等):回测时历史外部数据可能缺失,需降级处理,记录降级率,不强求数值完全一致
  - 上述分类为初步参考,**阶段 3a 启动时必须逐个对照 [StockScorer.get_sub_scores](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py#L55) 确认每个子评分的真实数据源依赖**,以代码为准

**验收**:
- BacktestEngine `_calc_factors` 输出的 sub_scores 维度 = 24,与 [StockScorer.get_sub_scores](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py#L55) 一致
- 纯 K 线因子:抽样 5 只股票 × 5 个日期,sub_scores 与 live StockScorer 差异 ≤0.01(输入完全相同)
- 外部数据因子:记录"历史数据可得率"(可拉到历史数据的日期占比),降级时返回 None/默认值不崩溃

**风险**:
- 部分因子族依赖外部数据源(资金流、龙虎榜),回测时可能拿不到历史数据,需降级处理
- 工作量可能超出预期,若 8-28 前未完成,3b 顺延

#### 阶段 3b:滚动 IC 加权稳健性检验

**前置**:3a 完成。

**明确当前 `run_walk_forward` 的真实行为**:[src/engine/backtest.py:31](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py#L31) 目前只是用**固定 weights** 在滚动 test 窗口上回测,train 窗口内**没有权重拟合步骤**。本阶段默认采用轻量方案 A;若数据/资源允许,可升级为方案 B。

##### 方案 A(默认):滚动 IC 加权稳健性检验

**动作**:
```
[lookback 4个月] → [test 1个月] → 滑动
  计算ICIR加权      用该权重回测(不调参)
```
- 1-4 月计算各因子 ICIR → 生成 IC 加权 weights → 5 月回测
- 2-5 月计算 ICIR → 生成 weights → 6 月回测
- 滚动产出"训练窗口 IC 加权 weights"与"测试窗口表现"的对应关系

**权重生成规则**(示例):
```
w_i ∝ max(0, ICIR_i) / Σ max(0, ICIR_j)
```
仅对 IC>0 且 ICIR>0 的因子赋权;IC 为负的因子反向或置 0。

> 注意:纯 ICIR 归一化会抹平 config.yaml 中"大类平衡"的先验结构(如基本面 30% vs K 线 12%)。若希望保留大类结构,可改为"ICIR 调整 × 先验权重"形式(`w_i ∝ prior_i × max(0, ICIR_i) / Σ`),具体形式在阶段 3b 实施时与用户确认。

**退化策略**:若某窗口内**所有因子 ICIR≤0**(IC 信号失效),该窗口不应用 IC 加权,改为沿用 config.yaml 默认权重,并在报告中标记为"IC 信号失效窗口"。若连续 2 个窗口失效,触发决策点 3(回滚/暂停调整)。

**输出**:
- `reports/rolling_ic_weight_result.md`
- 每个窗口的 IC 加权 weights vs 测试窗口期望收益 α、胜率、最大回撤
- 窗口间 weights 的稳定系数(如权重的 mean/std)

**验收**:
- 测试窗口期望收益 α 的 **95% 置信区间下界** 不显著为负
- 各窗口 weights 的变异系数(CV)不过高(如 <0.5),否则说明 IC 信号不稳定
- 不追求"胜率 >50%"硬阈值(样本不足时无统计意义)

##### 方案 B(升级):真正 Walk-Forward 验证

**触发条件**:数据累积 ≥12 个月,或用户明确要求

**前置新增代码**:
- 在 [src/engine/backtest.py](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py) 中扩展 `run_walk_forward`,在每个 train 窗口内调用优化器拟合 weights(如 ICIR 加权、风险平价、或带约束的凸优化)
- test 窗口只能用 train 窗口产出的 weights,严禁用未来信息

**输出**:
- `reports/walk_forward_result.md`
- 每个窗口的训练集表现 vs 验证集表现
- 验证集 α 不显著低于训练集

**验收**:
- 验证集表现不显著低于训练集(过拟合检测)
- 验证集期望收益为正

**共同风险**:
- 因子补齐工作量中等,需逐个确认 live 因子族的输入参数
- 滚动验证慢(每个窗口都重算因子),预计 30-60 分钟/次
- **选择偏差**:无论方案 A 还是 B,当前只能基于已推荐票池做回测(原始 K线不落库),结论不能直接外推到全市场。需在报告中明确标注此限制。

---

### 阶段 4:paper_trades 实盘归因(P3,等样本积累)

**目标**:用含止损/止盈的真实平仓收益做归因,最接近实盘。

**触发条件**:paper_trades 累积 ≥ 50 笔 **且** 二项分布 95% 置信区间宽度 ≤ ±15%(预计 2-3 个月)

**动作**:
- 扩展 AttributionAggregator 支持 paper_trades 数据源
- 按 exit_reason(stop_loss / take_profit / trailing_stop / crash_stop / hold_expire,见 [run_daily.py:229-278](file:///c:/Users/EDY/Desktop/StockPrediction/run_daily.py#L229-L278) 与 [run_daily.py:413](file:///c:/Users/EDY/Desktop/StockPrediction/run_daily.py#L413))分组归因
- 关联 entry 时的因子分,看哪类因子容易导致 stop_loss / trailing_stop

**输出**:
- `reports/paper_trades_attribution.md`
- "因子 X 高分 → stop_loss 概率 Y%"的映射

**验收**:
- 识别"高分但易止损"的因子(假信号)
- 识别"低分但常止盈"的因子(被低估)

---

## 五、不做清单(Explicit Non-Goals)

| 不做的事 | 理由 |
|---|---|
| 遍历历史推荐调权重看胜率 | Data snooping,过拟合 |
| 用 T+N 固定收益作为唯一优化目标 | 与实盘止损逻辑不符 |
| 自动改写 config.yaml 权重 | 规则 §2 要求权重改动需用户确认 |
| 全市场 K线落库 | 存储成本高,当前无强需求 |
| 频繁调参 | 每次调整后至少跑 2 周再评估 |
| ML 模型凌驾多因子框架 | 规则 §6:ML 仅作辅助信号 |

---

## 六、风险与约束

### 6.1 数据约束

- **样本量**:7 个月数据,切分后每段几十到几百条,统计显著性有限
- **d3_return 缺口**:7 月数据未回填,需 premarket 持续运行补齐
- **K线不落库**:无法做需要重算因子的全市场回测,只能基于已推荐的票,存在选择偏差
- **滚动验证窗口少**:按 4+1 月切分,1~6 月仅 2 个验证窗口,不足以做严格统计推断

### 6.2 过拟合风险

- 阶段 1"计算 IC"纯客观指标,无过拟合风险;但**若据此调整 config.yaml 权重,属于轻度 data snooping**(用全样本决策再用于未来),需满足:① 单次调整幅度 ≤3pp;② 调整结果**必须在阶段 3b 滚动验证中复核**,若验证集 α 显著为负则回退;③ 阶段 1 的调权是临时方案,最终权重以阶段 3b 滚动验证背书的版本为准
- 阶段 2(Regime 归因)有轻度风险,只看不调可规避
- 阶段 3(滚动 IC 加权)是中度风险,需重点观察"窗口间权重稳定性"而非单窗口胜率
- **红线**:任何调参都必须用样本外表现背书,不能只看训练集/同一批数据

### 6.3 工程约束

- BacktestEngine 因子覆盖不全,阶段 3 前必须补齐(规则 §6)
- 所有路径不得写死绝对路径(规则 §7)
- 异常降级是硬性要求(规则 §4):归因/回测失败不得导致主流程崩溃
- Regime 回算必须使用 T-1 日收盘后可得数据,禁止用 T 日盘中/收盘数据(规则 §1)

---

## 七、时间线

| 时间点 | 里程碑 | 产出 | 前置依赖 |
|---|---|---|---|
| 2026-08-01 | 阶段 1 完成 | 24 维子评分 IC 排名表 | 无 |
| 2026-08-07 | paper_trades 修复验证完成 | 确认入场点过滤生效 | 无 |
| 2026-08-14 | 阶段 2 完成(Regime 归因) | 因子×regime 二维表 | 阶段 1 |
| 2026-08-21 | 阶段 3a 完成(因子补齐) | BacktestEngine 与 live 因子对齐 | 阶段 1 |
| 2026-08-28 | 阶段 3b 完成(滚动 IC 加权稳健性检验) | 验证窗口表现报告 | 3a |
| 2026-09-30 | paper_trades ≥ 50 笔 | 阶段 4 启动 | paper_trades 累积 |
| 2026-10-15 | 阶段 4 完成 | 实盘归因报告 | 3b、≥50 笔 |

> 若 3a 因子补齐工作延期,3b 及后续阶段顺延;若 3a 在 8-21 前完成,可提前启动 3b。

---

## 八、决策点与回滚机制

执行过程中需要用户确认的决策:
1. **阶段 1 后**:是否根据 IC 排名调整 config.yaml 权重?(规则 §2 要求确认)
2. **阶段 2 后**:是否启用"按 regime 动态加权"?(架构改动较大)
3. **阶段 3 后**:验证集表现若不达标,是否回退权重?
4. **阶段 4 后**:是否将 paper_trades 归因结果作为 ML 模型的**样本权重/label 调整**,而非直接替代多因子评分?(规则 §6:ML 仅作辅助信号)

**回滚预案**:
- 每次调整 config.yaml 权重前,自动写入 `weight_history` 表(已有)
- **触发条件**(同时满足才触发,避免噪声误触发):
  - 调整后 2 周内新增 paper_trades ≥ 20 笔 **且** paper_trades α 下降 >3pp;或
  - 连续 4 周周度 α 为负
- 回滚目标:回滚至最近一次"阶段 3b 验证集 α > 0"对应的 `weight_history` 记录;若阶段 3b 尚未执行,回滚至本次调整前的权重快照
- 禁止直接回滚到"直觉权重",必须以可验证的历史记录为锚点

---

## 九、附录:方法对比

| 维度 | 遍历推荐调胜率(放弃) | IC + 滚动验证(推荐) |
|---|---|---|
| 过拟合风险 | 高(data snooping) | 中(训练/验证分离,但验证窗口少) |
| 样本量要求 | 高(每段需几百条) | 低(IC 用全部样本;滚动验证窗口仍偏少) |
| 优化目标 | 胜率(错误) | 期望收益 / IC(正确) |
| 选择偏差 | 有(只看幸存者) | 仍有(基于已推荐票池),已标注 |
| 实盘一致性 | 差(T+N ≠ 实盘) | 较好(滚动验证模拟实盘调参) |
| 业界认可 | 否 | 是 |

---

## 十、关键代码引用

- IC 计算:[src/engine/ml_selector.py:308](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/ml_selector.py#L308)
- BacktestEngine 滚动回测(当前无训练集拟合):[src/engine/backtest.py:31](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py#L31)
- BacktestEngine 因子计算(待补齐):[src/engine/backtest.py:343](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/backtest.py#L343)
- MarketRegimeFilter  regime 判定:[src/engine/market_regime.py:42](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/market_regime.py#L42)
- 24 维子评分定义:[src/engine/scorer.py:55](file:///c:/Users/EDY/Desktop/StockPrediction/src/engine/scorer.py#L55)
- 数据库 schema 与迁移:[src/data/database.py:68](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L68)、[src/data/database.py:318](file:///c:/Users/EDY/Desktop/StockPrediction/src/data/database.py#L318)

---

*本计划遵循[项目规则](file:///c:/Users/EDY/Desktop/StockPrediction/.trae/rules/stock_prediction.md) §6(回测与 ML 的诚实性),禁止 mock/假数据,禁止权重前视,回测必须与 live 共用同一套因子计算。Regime 回算必须使用 T-1 日可得数据,禁止引入前视偏差(规则 §1)。权重调整需用户确认(规则 §2),异常降级是硬性要求(规则 §4),路径不得写死(规则 §7)。*

# StockPrediction 项目规则（个人开发 / 个人使用）

> 适用范围：本仓库（Python 量化股票预测系统）。个人使用，不追求企业级规范，但必须守住量化系统的几条底线。
> 通用编码规范见 user_rules（中文回复、先做计划、先查现有实现、不覆盖 .env、禁止 mock 等），本文件只补充**本项目特有**约束，不重复通用原则。

## 1. 红线：禁止未来函数（look-ahead bias）—— 最高优先
量化系统最大的坑是用未来数据算过去收益。以下必须严格禁止：
- **收益/回测/表现计算**：一律用 T+N 日**真实收盘价**回看（参考 `run_daily.py::_fetch_close_price_on_date`），不得用当日实时价反推历史收益。
- **因子计算**：因子值只能用 `entry_date` 及之前的数据；滑动窗口只能 `history.iloc[:i+1]` 这种截至当日的窗口，不得引用 entry 之后的 K 线、资金流、新闻。
- **反馈循环 / 归因**：`feedback_loop.py`、`factor_attribution.py` 必须基于真实 T+N 收益，不得用标签泄漏或未来信号。
- 新增任何因子或指标前，先自问“这个值在实盘 entry 那一刻拿得到吗？”拿不到就禁止写入评分。

## 2. 配置驱动，参数单一来源
- 权重、筛选条件、技术参数、交易参数、外盘/新闻关键词都集中在 `config.yaml`，**不要硬编码**到函数里。
- 在 `config.yaml` 的 `weights` 增删/调整权重后，**必须同步** `src/engine/scorer.py::StockScorer` 的默认 `weights` dict 与 `get_sub_scores`（两处保持一致，避免配置缺失时行为漂移）。
- **不得擅自覆盖 `config.yaml` 的权重**：除非用户明确确认要调整。

## 3. 新增因子 checklist（先做查重）
新增因子前先搜索 `src/factors/`（已有 technical / capital_behavior / ashare_specific / ashare_extended / phase2_market_intel / moat 等大量因子），确认无近似实现再动手。完成后逐项核对：
- [ ] 类放 `src/factors/`，方法返回 `dict`，key 命名与现有因子一致
- [ ] 在 `StockScorer.get_sub_scores` 注册子评分
- [ ] 在 `config.yaml.weights` 增加对应权重 key
- [ ] 检查是否与高相关因子重复，必要时补 `StockScorer.ORTHOGONALIZE_PAIRS` 残差化
- [ ] 跑 `python run_daily.py --factor-ic` 看 IC/ICIR，用客观数据决定权重
- [ ] 同步 `run_intraday` / `run_afternoon` 读取的缓存字段（见第 5 条）

## 4. 数据源与异常降级
- 所有行情/资金流/新闻统一走 `src/data/` 封装（`MarketDataFetcher` / `MoneyFlowFetcher` / `NewsFetcher` 等），**不要绕过封装直连东财等域名**。
- 网络 I/O 受数据源限流约束，目前安全并发上限为 **8**（`run_daily.py::_prefetch_history_and_flow`），调整前需做健康测试。
- 代理绕过已通过 `NO_PROXY` 设置；新增数据源若也会被系统代理干扰，要补进 `NO_PROXY` 列表。
- 新增外部源先接入 `src/data/source_health.py` 健康自检。
- **异常降级是硬性要求**：外部依赖失败必须 catch 后降级（返回 None / 用缓存），不得因单个因子或数据源挂掉导致整轮崩溃。

## 5. 运行模式契约（premarket / intraday / afternoon）
- `premarket` 是全量计算入口，写入 `cache/premarket_<date>.json` 候选池缓存。
- `intraday`（9:25）**只读缓存 + 刷实时行情做修正**，不得触发重计算/重网络取数。
- `afternoon`（13:00）可补资金流/午间新闻，但不得重跑盘前因子。
- 改 `premarket` 写入缓存的字段结构时，**必须同步** `run_intraday` / `run_afternoon` 读取侧，否则盘中模式会 KeyError 或拿到 None；改了结构要删除当日 `cache/premarket_<date>.json` 重跑以防脏缓存。

## 6. 回测与 ML 的诚实性
- **禁止 mock / 假数据**：与 user_rules 一致，生产/开发环境不得制造假行情、假收益、假推荐；评价指标准能量化复现。（单测可用受控合成数据，但必须显式标注为测试数据。）
- **回测前视偏差**：权重分两层——`config.yaml` 是**静态先验**（不被反馈循环改写），反馈循环的调权结果落库（全局 `weight_history` 表 + 个股权重表），随时间累积。回测**不能**用“已知未来归因/IC”调出的权重跑历史；回测模式必须给 `feedback_loop` / `stock_weight_manager` 传 `today`，只引用截至该日的调权记录，禁止权重前视（`feedback_loop` 已支持 `today` 参数）。
- **回测与实盘一致**：回测逻辑必须与 live 共用同一套 `StockScorer` 和因子计算，禁止两套代码漂移导致回测漂亮、实盘失灵。
- **ML 时序验证**：`ml_selector.train` 在传入 `dates` 时走时序验证（按 date 排序前 80% 训练、后 20% 测试）。务必保证 `train_ml.py` 正确传入 `dates` 且行数一致，**禁止落到 `random_split` 降级分支**。ML 仅作辅助信号（当前午后模式低权重），不得凌驾于多因子框架之上。

## 7. 持久化、路径与产物
- **路径禁止写死**：所有数据/缓存/日志路径从 `ROOT`（或 `config.yaml`）相对取得，不得写死 `C:\Users\...` 等绝对路径，保证换机可跑。
- **改 schema 要兼容**：改动 `database.py` 表结构时，要考虑旧 `data/*.db` 兼容或显式迁移，不要手动改 db 文件。
- **产物去向**：日志 → `logs/`（按日期+模式轮转，摘要存 `daily_summary.log`）；报告 → `reports/`（`reporter.py`）；数据 → `data/*.db`；缓存 → `cache/`（按日期命名，由运行模式自动覆盖）。
- **免责声明**：所有对外报告/推荐必须标注“仅供研究，不构成投资建议”。

## 8. 改动影响范围
- 因子 key 改名/新增，全局搜索 `all_factors["..."]` 引用点（scorer、报告、缓存、intraday/afternoon 读取），保证四处一致。
- 坚持“先查现有实现、小步改动、优先简单方案”，不为减行数强行拆文件/类；只有真实重复 ≥2 处或职责确实耦合难懂才抽公共逻辑。

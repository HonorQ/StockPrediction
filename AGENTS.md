# StockPrediction 项目规则

> 本文件是 AI 助手与本仓库开发者的共同约定。任何代码修改前先阅读本规则。
> 制定时间: 2026-08-26。规则的每一条几乎都对应一个曾经真实踩过的坑。

---

## 一、语言约定

- 注释、日志、commit message、文档一律使用中文。
- commit 格式: 一行中文主题 + 空行 + 要点列表(可选)。

## 二、文件组织(硬性)

### 行数预算
| 层级 | 软上限 | 硬上限 |
|---|---|---|
| 入口脚本(根目录 run_*.py) | 500 | **800** |
| 业务模块(src/**) | 600 | **1000** |
| 单个函数 | — | **120**(评审线, 见函数规范) |

文件级超硬上限必须拆分; 超过软上限应在下次触碰该文件时顺手拆分。
当前欠账(2026-08-26): run_daily.py 2273行 / database.py 1365行 / reporter.py 902行,
逐步偿还, 新代码严禁再加重。

### 分层与依赖方向(单向,禁止反向)

```
根目录入口(run_*.py 只做: 参数解析/配置校验/心跳/模式分发)
    ↓
src/engine (评分scorer/回测backtest/反馈feedback_loop/风控strategy_health/报告reporter)
    ↓
src/factors (纯因子计算, 输入K线/行情dict, 不碰数据库和网络)
    ↓
src/data (行情market_data/资金流money_flow/新闻news_fetcher/存储database)
    ↓
src/utils (helpers/log/heartbeat/proxy_manager 无业务语义的工具)
```

- 因子层禁止 import requests/baostock/sqlite —— 数据由上层注入。
- 跨层复用的函数放 utils 或对应业务模块, **禁止留在入口文件**(入口只 re-export 兼容旧引用)。
- 一个模块只回答一个问题: "chip分布怎么算" 和 "chip异常怎么排除" 是两个模块的事。

### 函数规范
- 单一职责: 一个函数只做一个阶段的事(取数→算因子→评分→落库 各自独立可测)。
- **单函数 120 行是评审线, 不是强制线**: 触发即评审, 不为拆而拆。满足任一豁免条件
  可保持长函数:
  - 线性数据流/单入口单出口/无深层嵌套 → 用 `# === 阶段N: xxx ===` 注释分节即可;
  - 拆分需引入 >6 个变量的状态容器或 ctx 键包 → 说明耦合是本质的, 保持内聚。
- **必须拆的是结构信号, 与行数无关**: 嵌套深度>3、同逻辑重复出现、
  修一处需理解全函数、分支组合爆炸。规则惩罚"深而绕", 不是"长而直"。
- **拆分收益原则**: 只有当独立可测/复用/命名自解释三者居其一时才拆;
  禁止拆出只有一个调用点、纯凑行数的转发函数(62变量unpack的教训)。
- **ctx.unpack 键清单超过 ~10 个即坏味道**: 应改为显式传依赖对象
  (dataclass 或直接传 fetcher/scorer 实例); 字符串键无静态检查,
  拼错静默返回 None 比长函数更危险。
- 流水线上下文确需跨阶段传递时用 dataclass/dict 显式传递, 禁止闭包隐式共享;
  原"超过5个局部状态应拆"条款废止——状态数量本身不构成拆分理由(2026-08-26修订)。
- 可选参数一律给默认值并写进 docstring; 布尔开关用显式参数而非环境变量(回测对比除外)。

## 三、编码红线(每条都是血泪史)

1. **收益口径全项目唯一**: 百分数(0.01 = 0.01%? 不, 1.0 = 1%)。新代码涉及收益必须
   复用现有函数或在 docstring 声明口径。历史上三种口径混存导致 ML 标签全部失效。
2. **涨跌停幅度禁止硬编码**: 一律 `from src.utils.helpers import get_limit_pct`。
   A股有 10/20/30/5% 四种口径。
3. **禁止裸 `except Exception: pass`**: 至少 `logger.warning`。静默吞错曾让 NameError
   伪装成"网络失败"潜伏半年。
4. **因子/信号的打分方向必须有数据出处**: 在 docstring 注明依据
   (如 `reports/market_ic_verdict.json` 的 IC/t值)。凭直觉定方向已被大样本推翻过
   (quant_behavior 事件)。
5. **上线任何选股信号前必须跑单边组合模拟**
   (`scripts/portfolio_simulator.py`): 分层多空价差 ≠ 多头可提取 alpha,
   空头端信号在 A股不可交易(2026-08 alpha_overlay 翻车事件)。
6. **SQLite**: 写操作走 `db._connection()`(自带重试/关闭); 值一律参数化;
   幂等写入用唯一索引 + OR REPLACE, 不用 SELECT-then-INSERT。
7. **多进程共享连接禁止**: baostock 全局 socket 不支持并发(实测断连+解码错),
   预取用串行或进程隔离。
8. **网络调用必须有超时**: requests 显式 timeout=8~10; 入口脚本设
   `socket.setdefaulttimeout(20)`。baostock 会内部吞超时, 挂死时优先怀疑它。
9. **pandas 大 DataFrame 不要 Set-Content/内联python -c 处理**, 写临时脚本文件执行。

## 四、测试与质量门禁

- 改动 engine/data 层后必须: `python -m pytest tests/ -q` 全绿 + `ruff check .` 无错。
- 回归测试跟着修复走: 修一个 bug 补一个最小复现测试(tests/test_bugfixes.py 风格)。
- 测试必须离线可跑(mock 网络), 不允许测试依赖 baostock/东财可用性。
- 往类中间插入模块级函数会把类打断——新模块级函数追加到文件末尾。

## 五、数据与研究纪律

- K线缓存(cache/klines, cache/bt_kline_*)是一切研究的地基, 断点安全, 定期补齐:
  `python -u scripts\fetch_1year_data.py`
- 因子有效性结论以 `reports/market_ic_verdict.json`(全市场每日截面)为准;
  候选池小样本(284只)存在幸存者偏差, 结论只能作参考。
- 新因子上线路径: 面板IC → 十分位分层(scripts/quantile_analysis.py)
  → 单边组合模拟 → 影子运行 → 转正。跳步需要说明理由。
- 回测与实盘共用同一评分链路(batch_normalize/overlay/guard),
  两套逻辑漂移 = 回测作废。

## 六、运维

- 计划任务三个时点: 07:00 premarket / 09:25 intraday / 14:40 afternoon(最终推荐)。
- 心跳: logs/heartbeat.json; 异常会在下次报告横幅出现(check_task_alerts)。
- 报告产物 reports/daily/YYYY-MM-DD.html, 午后版覆盖同名盘前版(设计如此)。
- 推送失败先查系统残留代理(127.0.0.1:7897): `git -c http.proxy="" push`。

## 七、已知技术债(按危害排序, 领取任务前先看这里)

> 2026-08-26 偿还记录: run_daily.py 2273→128行(pipeline/包), database.py 1365→db/组合壳,
> reporter.py 902→788(CSS外移), _pm_* 巨型键包全部清除(62键→按需≤19),
> premarket_finalize 双跑编排/config污染/计时丢失三处缺陷修复,
> 归因时序衰减合并为 feedback_loop.recency_decay 单一口径,
> 阈值配置化(score_tiers config.yaml段 + load_score_tiers加载器),
> market_data 975→526行(拆出 industry_mapper/global_market_fetcher/kline_fetcher),
> ml_selector 928→475行(拆出 factor_optimizer), reporter 910→605行(拆出 report_styles),
> backtest 868→714行(拆出 risk_control), market.py _KLINE_COL_MAP bug修复。

- [ ] 换手率备源: 腾讯/新浪K线接口无换手率字段, 依赖 baostock 逐股补充
      (cache/turnover 7天缓存+10秒超时兜底, 首次冷启动/缓存过期仍有延迟,
      根因是数据源限制非代码缺陷, 标记为"已缓解")
- [ ] database.py repository 层: db/ Mixin 拆分已完成, 统一 repository 抽象仍缺

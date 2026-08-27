# AGENTS.md 技术债偿还方案（2026-08-26）

> 本文件记录剩余 7 个超长函数的拆分方案与执行结果。
> ~~原计划: 拆分剩余超标函数~~ → **已于 2026-08-26 全部完成并验证**。

---

## 执行结果(2026-08-26)

| # | 函数 | 文件 | 原行数 | 处理方式 | 结果 |
|---|---|---|---|---|---|
| 1 | `_pm_p2s1` | src/pipeline/premarket_finalize.py | 126 | 删除重复 docstring + 冗余 unpack 段 | 93 |
| 2 | `run_intraday` | src/pipeline/intraday.py | 204 | 切为 `_id_snapshot_and_crash` / `_id_filter_risky` / `_id_alerts_and_rescore` | 主函数67, 子块≤112 |
| 3 | `train` | src/engine/ml_selector.py | 183 | 三段: `_train_prepare` / `_train_search` / `_train_evaluate`, 经 `self._train_state` 传递 | 主17, 子段≤84 |
| 4 | `_build_factor_rows` | src/engine/reporter.py | 300 | 按因子族四分: `_factor_rows_tech/sentiment/fundamental/event`; `star()` 提为 `_stars` 静态方法; 行序按族重组(add 调用数60不变) | 主35, 子块≤110 |
| 5 | `_init_db` | src/data/db/base.py | 302 | 按表域五分(`_create_recommendations/paper_trade/weight_ic/stock_weight/ml_auction_tables(conn)`), PRAGMA 留主函数; SQL 47条零丢失校验通过 | 15 |
| 6 | `_calc_one_stock` | src/pipeline/premarket_deep.py | 268 | 五块: `_calc_tech/sustain/sentiment_event/policy_deep/composite_block(ctx, cand, history, all_factors)`; all_factors 为唯一累加器原地更新; 因子键集合一致校验通过 | 主36, 子块≤74 |
| 7 | `run_afternoon` | src/pipeline/afternoon.py | 367 | 四段+2辅助: `_am_init/_am_rescore(+_am_rescore_one/_am_ml_assist)/_am_filter_persist/_am_report`, ctx dict 传递 | 主13, 子块≤117 |

## 验证记录

- `python -m ruff check .` 全仓无错
- `python -m pytest tests/ -q` 90 用例全绿
- 7 个模块 import 链验证通过
- DB 冒烟: 临时库建 16 表 + save_recommendation 写读正常
- 遗留顺手修复: `_calc_composite_block` 中 ATR 止损段的裸 `except Exception: pass`
  改为 `logger.warning`(AGENTS.md 红线3)

## 注意事项(下次计划任务触发时观察)

- [ ] 09:25 盘中(intraday)、14:40 午后(afternoon)首次实跑确认输出与拆分前一致
- [ ] 报告 HTML 因子行顺序已按因子族重组(技术→情绪→基本面→事件),
      同族内相对顺序未变; 若观感异常再评估是否恢复原始交错顺序

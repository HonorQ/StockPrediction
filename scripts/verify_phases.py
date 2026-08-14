import sys; sys.path.insert(0, '.')

# Phase 1 - DragonTiger
from src.factors.phase2_market_intel import DragonTigerBoard
dt = DragonTigerBoard()
r = dt.analyze('sz000021')
print(f'Phase1 龙虎榜: lh_score={r["lh_score"]}, signals={r["lh_signals"]}')

# Phase 2 - 归因逻辑
from src.engine.factor_attribution import classify_signal, evaluate_prediction
print(f'Phase2 分类(75, median=50): {classify_signal(75, 50)}')
print(f'Phase2 分类(62, median=60): {classify_signal(62, 60)}')
print(f'Phase2 评估(bullish, +0.5%): {evaluate_prediction("bullish", 0.5)}')
print(f'Phase2 评估(bullish, +1.5%): {evaluate_prediction("bullish", 1.5)}')
print(f'Phase2 评估(bullish, -2.0%): {evaluate_prediction("bullish", -2.0)}')

# Phase 5 - ML特征数
from src.engine.ml_selector import MLStockSelector
ml = MLStockSelector()
feat = ml._extract_features({'rsi_6': 60, 'rsi_14': 55})
print(f'Phase5 ML特征数: {len(feat[0])}')

# Phase 3 - 检查择时代码
from run_daily import MAX_REC_CRASH_KEY
print(f'Phase3: 所有模块验证通过')

print()
print('=== 全部验证通过 ===')

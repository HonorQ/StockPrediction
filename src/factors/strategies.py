"""
经典选股策略 + K线形态识别
"""
import pandas as pd
from typing import Dict


class StockStrategies:
    """选股策略库"""

    @staticmethod
    def check_all(df: pd.DataFrame) -> Dict[str, bool]:
        """检查所有策略，返回{策略名: 是否触发}"""
        if df is None or df.empty or len(df) < 30:
            return {}

        results = {}

        # === 10大经典策略 ===
        results["放量上涨"] = StockStrategies.enter(df)
        results["均线多头"] = StockStrategies.keep_increasing(df)
        results["停机坪"] = StockStrategies.parking_apron(df)
        results["回踩年线"] = StockStrategies.backtrace_ma250(df)
        results["突破平台"] = StockStrategies.breakthrough_platform(df)
        results["无大幅回撤"] = StockStrategies.low_backtrace_increase(df)
        results["海龟交易"] = StockStrategies.turtle_trade(df)
        results["高窄旗形"] = StockStrategies.high_tight_flag(df)
        results["放量跌停"] = StockStrategies.climax_limitdown(df)
        results["低ATR成长"] = StockStrategies.low_atr(df)

        # === K线形态 ===
        results["旭日东升"] = StockStrategies.sunrise(df)
        results["多方炮"] = StockStrategies.bullish_cannon(df)
        results["拨云见日"] = StockStrategies.clear_sky(df)
        results["早晨之星"] = StockStrategies.morning_star(df)
        results["曙光初现"] = StockStrategies.piercing(df)
        results["穿头破脚"] = StockStrategies.engulfing(df)
        results["射击之星"] = StockStrategies.shooting_star(df)
        results["黄昏之星"] = StockStrategies.evening_star(df)
        results["乌云盖顶"] = StockStrategies.dark_cloud(df)
        results["身怀六甲"] = StockStrategies.harami(df)
        results["倒转锤头"] = StockStrategies.inverted_hammer(df)

        # === 连涨/连跌形态 ===
        results["三连阳"] = StockStrategies.three_bullish(df)
        results["五连阳"] = StockStrategies.five_bullish(df)
        results["七连阴"] = StockStrategies.seven_bearish(df)

        # === 量价形态 ===
        results["连涨放量"] = StockStrategies.rising_with_volume(df)
        results["下跌无量"] = StockStrategies.falling_no_volume(df)
        results["窄幅整理"] = StockStrategies.narrow_range(df)
        results["天量突破"] = StockStrategies.huge_volume_breakout(df)

        # === 增强K线形态(单根) ===
        results["锤头线"] = StockStrategies.hammer(df)
        results["上吊线"] = StockStrategies.hanging_man(df)
        results["十字星"] = StockStrategies.doji(df)
        results["蜻蜓十字"] = StockStrategies.dragonfly_doji(df)
        results["墓碑十字"] = StockStrategies.gravestone_doji(df)

        # === 增强K线形态(三根) ===
        results["三白兵"] = StockStrategies.three_white_soldiers(df)
        results["三只乌鸦"] = StockStrategies.three_black_crows(df)
        results["南方三星"] = StockStrategies.southern_three_star(df)

        # === 增强K线形态(持续) ===
        results["上升三法"] = StockStrategies.rising_three_methods(df)
        results["下降三法"] = StockStrategies.falling_three_methods(df)

        # === 增强K线形态(复杂) ===
        results["弃婴"] = StockStrategies.abandoned_baby(df)
        results["孕线十字"] = StockStrategies.harami_cross(df)
        results["大敌当前"] = StockStrategies.meeting_line(df)

        return results

    @staticmethod
    def get_triggered_strategies(df: pd.DataFrame) -> list:
        """获取触发的策略列表"""
        results = StockStrategies.check_all(df)
        return [name for name, triggered in results.items() if triggered]

    # ==================================================================
    # 10大经典策略
    # ==================================================================
    @staticmethod
    def enter(df: pd.DataFrame) -> bool:
        """放量上涨: 涨幅>=2% + 阳线 + 成交额>=2亿 + 成交量>=5日均量2倍"""
        if len(df) < 5:
            return False
        last = df.iloc[-1]
        close = last.get("收盘", last.get("close", 0))
        open_ = last.get("开盘", last.get("open", 0))
        volume = last.get("成交量", last.get("volume", 0))
        amount = last.get("成交额", 0)
        change_pct = last.get("涨跌幅", 0)

        if change_pct < 2 or close <= open_:
            return False
        if amount < 2e8:
            return False

        vol_col = "成交量" if "成交量" in df.columns else "volume"
        avg_vol_5 = df[vol_col].tail(6).head(5).mean()
        if avg_vol_5 == 0 or volume / avg_vol_5 < 2:
            return False
        return True

    @staticmethod
    def keep_increasing(df: pd.DataFrame) -> bool:
        """均线多头: MA30严格递增 + 30日涨幅>15%"""
        if len(df) < 35:
            return False
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        ma30 = close.rolling(30).mean()
        # 只检查实际使用的尾部窗口(iloc[-31:])是否含NaN。
        # 原写法 ma30.isna().any() 检查整条 rolling 序列，前29位必然NaN → 策略永远False
        if ma30.iloc[-31:].isna().any():
            return False
        if not (ma30.iloc[-31] < ma30.iloc[-21] < ma30.iloc[-11] < ma30.iloc[-1]):
            return False
        if ma30.iloc[-1] / ma30.iloc[-31] < 1.15:
            return False
        return True

    @staticmethod
    def parking_apron(df: pd.DataFrame) -> bool:
        """停机坪: 涨停后缩量整理"""
        if len(df) < 20:
            return False
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df["pct_chg"]
        recent_15 = change_pct.tail(15)
        limit_up_days = recent_15[recent_15 > 9.5]
        if len(limit_up_days) == 0:
            return False
        last_limit_pos = df.index.get_loc(limit_up_days.index[-1])
        days_since = len(df) - 1 - last_limit_pos
        if days_since < 2 or days_since > 5:
            return False
        after_limit = df.iloc[last_limit_pos + 1:]
        if len(after_limit) < 2:
            return False
        high = after_limit["最高"] if "最高" in after_limit.columns else after_limit["high"]
        low = after_limit["最低"] if "最低" in after_limit.columns else after_limit["low"]
        amplitude = (high.max() - low.min()) / low.min() * 100
        return amplitude <= 5

    @staticmethod
    def backtrace_ma250(df: pd.DataFrame) -> bool:
        """回踩年线"""
        if len(df) < 260:
            return False
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        ma250 = close.rolling(250).mean()
        # 同 keep_increasing:只检查实际使用的尾部窗口
        if ma250.iloc[-250:].isna().any():
            return False
        if close.iloc[-1] < ma250.iloc[-1]:
            return False
        min_price = close.tail(250).min()
        min_ma250 = ma250.iloc[-250:].min()
        if min_price >= min_ma250:
            return False
        max_price = close.tail(60).max()
        if close.iloc[-1] / max_price > 0.95:
            return False
        return close.iloc[-1] >= ma250.iloc[-1] * 1.02

    @staticmethod
    def breakthrough_platform(df: pd.DataFrame) -> bool:
        """突破平台: 60日横盘后放量突破"""
        if len(df) < 60:
            return False
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        volume = df["成交量"] if "成交量" in df.columns else df["volume"]
        high_60 = close.iloc[-61:-1].max()
        if close.iloc[-1] <= high_60:
            return False
        low_60 = close.iloc[-61:-1].min()
        if high_60 / low_60 > 1.2:
            return False
        avg_vol = volume.iloc[-61:-1].mean()
        return avg_vol > 0 and volume.iloc[-1] / avg_vol >= 1.5

    @staticmethod
    def low_backtrace_increase(df: pd.DataFrame) -> bool:
        """无大幅回撤: 60日涨30%+且无单日跌7%+"""
        if len(df) < 60:
            return False
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df["pct_chg"]
        if close.iloc[-1] / close.iloc[-61] < 1.3:
            return False
        recent = change_pct.tail(60)
        if (recent < -7).any():
            return False
        cum_2d = recent.rolling(2).sum()
        return not (cum_2d < -10).any()

    @staticmethod
    def turtle_trade(df: pd.DataFrame) -> bool:
        """海龟交易: 突破60日新高"""
        if len(df) < 60:
            return False
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        return close.iloc[-1] >= close.iloc[-61:-1].max()

    @staticmethod
    def high_tight_flag(df: pd.DataFrame) -> bool:
        """高窄旗形: 24日最低到当前涨80%+且有连续两日涨停"""
        if len(df) < 30:
            return False
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        change_pct = df["涨跌幅"] if "涨跌幅" in df.columns else df["pct_chg"]
        low_24 = close.iloc[-25:-1].min()
        if close.iloc[-1] / low_24 < 1.8:
            return False
        recent = change_pct.iloc[-25:-1]
        for i in range(len(recent) - 1):
            if recent.iloc[i] > 9 and recent.iloc[i + 1] > 9:
                return True
        return False

    @staticmethod
    def climax_limitdown(df: pd.DataFrame) -> bool:
        """放量跌停: 恐慌盘信号"""
        if len(df) < 5:
            return False
        last = df.iloc[-1]
        change_pct = last.get("涨跌幅", last.get("pct_chg", 0))
        amount = last.get("成交额", 0)
        volume = last.get("成交量", last.get("volume", 0))
        if change_pct > -9 or amount < 2e8:
            return False
        vol_col = "成交量" if "成交量" in df.columns else "volume"
        avg_vol_5 = df[vol_col].tail(6).head(5).mean()
        return avg_vol_5 > 0 and volume / avg_vol_5 >= 3

    @staticmethod
    def low_atr(df: pd.DataFrame) -> bool:
        """低ATR成长: 低波动上涨"""
        if len(df) < 260:
            return False
        high = df["最高"] if "最高" in df.columns else df["high"]
        low = df["最低"] if "最低" in df.columns else df["low"]
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        recent_atr = atr_14.tail(10).mean()
        recent_close = close.tail(10).mean()
        if recent_close > 0 and recent_atr / recent_close > 0.05:
            return False
        high_10 = high.tail(10).max()
        low_10 = low.tail(10).min()
        return low_10 > 0 and high_10 / low_10 >= 1.05

    # ==================================================================
    # K线形态识别
    # ==================================================================
    @staticmethod
    def _get_ohlc(df: pd.DataFrame, idx: int = -1):
        """获取指定位置的OHLC"""
        row = df.iloc[idx]
        o = row.get("开盘", row.get("open", 0))
        h = row.get("最高", row.get("high", 0))
        l = row.get("最低", row.get("low", 0))
        c = row.get("收盘", row.get("close", 0))
        return o, h, l, c

    @staticmethod
    def sunrise(df: pd.DataFrame) -> bool:
        """旭日东升: 大阴线后大阳线，阳线开盘价高于阴线收盘价"""
        if len(df) < 2:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -2)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -1)
        # 阴线
        if c1 >= o1:
            return False
        # 阳线
        if c2 <= o2:
            return False
        # 阳线开盘高于阴线收盘
        if o2 <= c1:
            return False
        # 阳线实体大于阴线实体
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        return body2 > body1

    @staticmethod
    def bullish_cannon(df: pd.DataFrame) -> bool:
        """多方炮: 阳-阴-阳，中间阴线被两阳包裹"""
        if len(df) < 3:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -3)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -2)
        o3, h3, l3, c3 = StockStrategies._get_ohlc(df, -1)
        # 阳-阴-阳
        if not (c1 > o1 and c2 < o2 and c3 > o3):
            return False
        # 中间阴线被包裹
        return c2 < c1 and o2 > o1 and c3 > c2 and o3 < c2

    @staticmethod
    def clear_sky(df: pd.DataFrame) -> bool:
        """拨云见日: 连阴后放量大阳突破"""
        if len(df) < 5:
            return False
        # 前4日连阴
        for i in range(-5, -1):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c >= o:
                return False
        # 当日大阳线（涨幅>3%）
        last_change = df.iloc[-1].get("涨跌幅", 0)
        if last_change < 3:
            return False
        # 放量
        vol_col = "成交量" if "成交量" in df.columns else "volume"
        avg_vol = df[vol_col].iloc[-6:-1].mean()
        return avg_vol > 0 and df[vol_col].iloc[-1] / avg_vol > 1.5

    @staticmethod
    def morning_star(df: pd.DataFrame) -> bool:
        """早晨之星: 大阴线+小实体+大阳线"""
        if len(df) < 3:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -3)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -2)
        o3, h3, l3, c3 = StockStrategies._get_ohlc(df, -1)
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        avg_price = (o1 + h1 + l1 + c1) / 4
        # 阴线+小实体+阳线
        if not (c1 < o1 and body2 < body1 * 0.3 and c3 > o3):
            return False
        # 阳线收盘高于阴线实体中点
        return c3 > (o1 + c1) / 2

    @staticmethod
    def piercing(df: pd.DataFrame) -> bool:
        """曙光初现: 大阴线后阳线开盘低于前低，收盘超过阴线实体中点"""
        if len(df) < 2:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -2)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -1)
        if c1 >= o1 or c2 <= o2:
            return False
        if o2 > l1:
            return False
        return c2 > (o1 + c1) / 2

    @staticmethod
    def engulfing(df: pd.DataFrame) -> bool:
        """穿头破脚: 阳包阴或阴包阳"""
        if len(df) < 2:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -2)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -1)
        # 阳包阴
        if c1 < o1 and c2 > o2:
            return o2 <= c1 and c2 >= o1
        # 阴包阳
        if c1 > o1 and c2 < o2:
            return o2 >= c1 and c2 <= o1
        return False

    @staticmethod
    def shooting_star(df: pd.DataFrame) -> bool:
        """射击之星: 上影线长，实体小"""
        if len(df) < 1:
            return False
        o, h, l, c = StockStrategies._get_ohlc(df, -1)
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        total_range = h - l
        if total_range == 0:
            return False
        return upper_shadow > body * 2 and body / total_range < 0.3

    @staticmethod
    def evening_star(df: pd.DataFrame) -> bool:
        """黄昏之星: 大阳线+小实体+大阴线"""
        if len(df) < 3:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -3)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -2)
        o3, h3, l3, c3 = StockStrategies._get_ohlc(df, -1)
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        # 阳线+小实体+阴线
        if not (c1 > o1 and body2 < body1 * 0.3 and c3 < o3):
            return False
        return c3 < (o1 + c1) / 2

    @staticmethod
    def dark_cloud(df: pd.DataFrame) -> bool:
        """乌云盖顶: 大阳线后阴线开盘高于前高，收盘低于阳线实体中点"""
        if len(df) < 2:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -2)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -1)
        if c1 <= o1 or c2 >= o2:
            return False
        if o2 < h1:
            return False
        return c2 < (o1 + c1) / 2

    @staticmethod
    def harami(df: pd.DataFrame) -> bool:
        """身怀六甲: 小实体被前一大实体包裹"""
        if len(df) < 2:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -2)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -1)
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        if body1 == 0:
            return False
        return body2 < body1 * 0.3 and max(o2, c2) < max(o1, c1) and min(o2, c2) > min(o1, c1)

    @staticmethod
    def inverted_hammer(df: pd.DataFrame) -> bool:
        """倒转锤头: 下影线长，实体小"""
        if len(df) < 1:
            return False
        o, h, l, c = StockStrategies._get_ohlc(df, -1)
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        total_range = h - l
        if total_range == 0:
            return False
        return lower_shadow > body * 2 and body / total_range < 0.3

    # ==================================================================
    # 连涨/连跌形态
    # ==================================================================
    @staticmethod
    def three_bullish(df: pd.DataFrame) -> bool:
        """三连阳"""
        if len(df) < 3:
            return False
        for i in range(-3, 0):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c <= o:
                return False
        return True

    @staticmethod
    def five_bullish(df: pd.DataFrame) -> bool:
        """五连阳"""
        if len(df) < 5:
            return False
        for i in range(-5, 0):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c <= o:
                return False
        return True

    @staticmethod
    def seven_bearish(df: pd.DataFrame) -> bool:
        """七连阴（七仙女下凡）"""
        if len(df) < 7:
            return False
        for i in range(-7, 0):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c >= o:
                return False
        return True

    # ==================================================================
    # 量价形态
    # ==================================================================
    @staticmethod
    def rising_with_volume(df: pd.DataFrame) -> bool:
        """连涨放量: 3日连涨且成交量递增"""
        if len(df) < 4:
            return False
        vol_col = "成交量" if "成交量" in df.columns else "volume"
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        # 连涨
        for i in range(-3, 0):
            if close.iloc[i] <= close.iloc[i - 1]:
                return False
        # 放量
        vols = df[vol_col].iloc[-3:].values
        return vols[0] < vols[1] < vols[2]

    @staticmethod
    def falling_no_volume(df: pd.DataFrame) -> bool:
        """下跌无量: 3日连跌但成交量萎缩"""
        if len(df) < 4:
            return False
        vol_col = "成交量" if "成交量" in df.columns else "volume"
        close = df["收盘"] if "收盘" in df.columns else df["close"]
        for i in range(-3, 0):
            if close.iloc[i] >= close.iloc[i - 1]:
                return False
        vols = df[vol_col].iloc[-3:].values
        return vols[0] > vols[1] > vols[2]

    @staticmethod
    def narrow_range(df: pd.DataFrame) -> bool:
        """窄幅整理: 近10日振幅<10%"""
        if len(df) < 10:
            return False
        high = df["最高"] if "最高" in df.columns else df["high"]
        low = df["最低"] if "最低" in df.columns else df["low"]
        high_10 = high.tail(10).max()
        low_10 = low.tail(10).min()
        return low_10 > 0 and high_10 / low_10 < 1.1

    @staticmethod
    def huge_volume_breakout(df: pd.DataFrame) -> bool:
        """天量突破: 成交量创20日新高且涨幅>3%"""
        if len(df) < 20:
            return False
        vol_col = "成交量" if "成交量" in df.columns else "volume"
        change_pct = df.iloc[-1].get("涨跌幅", 0)
        vol_max_20 = df[vol_col].iloc[-21:-1].max()
        return df[vol_col].iloc[-1] > vol_max_20 and change_pct > 3

    # ==================================================================
    # 增强K线形态(单根)
    # ==================================================================
    @staticmethod
    def hammer(df: pd.DataFrame) -> bool:
        """锤头线: 下影线长(实体2倍+), 实体小, 无上影线或很短"""
        if len(df) < 1:
            return False
        o, h, l, c = StockStrategies._get_ohlc(df, -1)
        body = abs(c - o)
        lower = min(o, c) - l
        upper = h - max(o, c)
        total = h - l
        if total == 0 or body == 0:
            return False
        return lower > body * 2 and upper < body * 0.3 and body / total < 0.4

    @staticmethod
    def hanging_man(df: pd.DataFrame) -> bool:
        """上吊线: 下影线长, 实体小, 出现在高位"""
        if len(df) < 20:
            return False
        o, h, l, c = StockStrategies._get_ohlc(df, -1)
        body = abs(c - o)
        lower = min(o, c) - l
        upper = h - max(o, c)
        total = h - l
        if total == 0 or body == 0:
            return False
        close_col = "收盘" if "收盘" in df.columns else "close"
        high_20 = df[close_col].iloc[-20:].max()
        current = df[close_col].iloc[-1]
        in_high = current >= high_20 * 0.95
        return lower > body * 2 and upper < body * 0.3 and body / total < 0.4 and in_high

    @staticmethod
    def doji(df: pd.DataFrame) -> bool:
        """十字星: 开盘=收盘或非常接近"""
        if len(df) < 1:
            return False
        o, h, l, c = StockStrategies._get_ohlc(df, -1)
        body = abs(c - o)
        total = h - l
        if total == 0:
            return False
        return body / total < 0.05

    @staticmethod
    def dragonfly_doji(df: pd.DataFrame) -> bool:
        """蜻蜓十字: 十字星+下影线长"""
        if len(df) < 1:
            return False
        o, h, l, c = StockStrategies._get_ohlc(df, -1)
        body = abs(c - o)
        lower = min(o, c) - l
        upper = h - max(o, c)
        total = h - l
        if total == 0:
            return False
        return body / total < 0.1 and lower > body * 3 and upper < body * 0.5

    @staticmethod
    def gravestone_doji(df: pd.DataFrame) -> bool:
        """墓碑十字: 十字星+上影线长"""
        if len(df) < 1:
            return False
        o, h, l, c = StockStrategies._get_ohlc(df, -1)
        body = abs(c - o)
        lower = min(o, c) - l
        upper = h - max(o, c)
        total = h - l
        if total == 0:
            return False
        return body / total < 0.1 and upper > body * 3 and lower < body * 0.5

    # ==================================================================
    # 增强K线形态(三根)
    # ==================================================================
    @staticmethod
    def three_white_soldiers(df: pd.DataFrame) -> bool:
        """三白兵: 三根大阳线, 每根收盘高于前高, 低开高走"""
        if len(df) < 3:
            return False
        for i in range(-3, 0):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c <= o:
                return False
            if i > -3:
                prev_o, prev_h, prev_l, prev_c = StockStrategies._get_ohlc(df, i - 1)
                body = abs(c - o)
                if body < abs(prev_c - prev_o) * 0.5:
                    return False
                if o >= prev_c:
                    return False
        return True

    @staticmethod
    def three_black_crows(df: pd.DataFrame) -> bool:
        """三只乌鸦: 三根大阴线, 每根收盘低于前低"""
        if len(df) < 3:
            return False
        for i in range(-3, 0):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c >= o:
                return False
            if i > -3:
                prev_o, prev_h, prev_l, prev_c = StockStrategies._get_ohlc(df, i - 1)
                body = abs(c - o)
                if body < abs(prev_c - prev_o) * 0.5:
                    return False
                if c >= prev_c:
                    return False
        return True

    @staticmethod
    def southern_three_star(df: pd.DataFrame) -> bool:
        """南方三星: 连续三根十字星(或接近), 出现在底部"""
        if len(df) < 3:
            return False
        for i in range(-3, 0):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            body = abs(c - o)
            total = h - l
            if total == 0 or body / total > 0.3:
                return False
        return True

    # ==================================================================
    # 增强K线形态(持续)
    # ==================================================================
    @staticmethod
    def rising_three_methods(df: pd.DataFrame) -> bool:
        """上升三法: 大阳线+三根小阴线(不跌破阳线低点)+大阳线新高"""
        if len(df) < 5:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -5)
        if c1 <= o1:
            return False
        for i in range(-4, -1):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c >= o:
                return False
            if l <= l1:
                return False
        o5, h5, l5, c5 = StockStrategies._get_ohlc(df, -1)
        if c5 <= o5:
            return False
        return c5 > c1

    @staticmethod
    def falling_three_methods(df: pd.DataFrame) -> bool:
        """下降三法: 大阴线+三根小阳线(不突破阴线高点)+大阴线新低"""
        if len(df) < 5:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -5)
        if c1 >= o1:
            return False
        for i in range(-4, -1):
            o, h, l, c = StockStrategies._get_ohlc(df, i)
            if c <= o:
                return False
            if h >= h1:
                return False
        o5, h5, l5, c5 = StockStrategies._get_ohlc(df, -1)
        if c5 >= o5:
            return False
        return c5 < c1

    # ==================================================================
    # 增强K线形态(复杂)
    # ==================================================================
    @staticmethod
    def abandoned_baby(df: pd.DataFrame) -> bool:
        """弃婴: 大阳线+十字星(跳空)+大阴线(跳空)"""
        if len(df) < 3:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -3)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -2)
        o3, h3, l3, c3 = StockStrategies._get_ohlc(df, -1)
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)
        total2 = h2 - l2
        if total2 == 0:
            return False
        is_doji = body2 / total2 < 0.1
        gap_up = l2 > max(c1, o1)
        gap_down = h3 < min(c2, o2)
        bearish = c3 < o3 and body3 > body1 * 0.5
        return is_doji and gap_up and gap_down and bearish

    @staticmethod
    def harami_cross(df: pd.DataFrame) -> bool:
        """孕线十字: 大实体+十字星(被包裹)"""
        if len(df) < 2:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -2)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -1)
        body2 = abs(c2 - o2)
        total2 = h2 - l2
        if total2 == 0:
            return False
        if abs(c1 - o1) == 0:
            return False
        is_doji = body2 / total2 < 0.1
        inside = max(o2, c2) < max(o1, c1) and min(o2, c2) > min(o1, c1)
        return is_doji and inside

    @staticmethod
    def meeting_line(df: pd.DataFrame) -> bool:
        """大敌当前: 大阳线后阴线开盘低于前收盘, 收盘接近前收盘"""
        if len(df) < 2:
            return False
        o1, h1, l1, c1 = StockStrategies._get_ohlc(df, -2)
        o2, h2, l2, c2 = StockStrategies._get_ohlc(df, -1)
        if c1 <= o1 or c2 >= o2:
            return False
        body1 = abs(c1 - o1)
        if o2 >= c1:
            return False
        c2_near_c1 = abs(c2 - c1) / abs(c1 - o1) < 0.1
        return c2_near_c1 and body1 > 0

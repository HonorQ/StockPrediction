"""
全自动新闻→股票匹配系统
三层匹配：
  1. 直接命中：新闻中出现股票名称/代码
  2. 概念映射：预设关键词→概念→股票（concept_map.json）
  3. 行业映射：财报主营构成中的行业关键词→股票
"""
from src.utils.log import get_logger
logger = get_logger()

import os
import json
import re
from typing import List, Dict, Set


class ConceptMap:
    def __init__(self, config_path: str = None, auto_sync: bool = False):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "config", "concept_map.json"
            )
        self.config_path = config_path
        self.sync_cache_path = config_path + ".auto.json"  # 自动同步的增量缓存
        self._data = self._load()
        self._stock_name_index = {}
        self._stock_code_set = set()
        if auto_sync:
            self.maybe_sync_from_akshare()

    def _load(self) -> dict:
        if not os.path.exists(self.config_path):
            logger.info(f'  ⚠️ 概念映射文件不存在: {self.config_path}')
            return {"concepts": {}}
        with open(self.config_path, "r", encoding="utf-8") as f:
            base = json.load(f)
        # 合并自动同步的增量缓存（如果有且未过期）
        if os.path.exists(self.sync_cache_path):
            try:
                mtime = os.path.getmtime(self.sync_cache_path)
                if (os.path.getmtime(self.config_path) - mtime) < 0 or \
                   (__import__("time").time() - mtime) < 7 * 86400:
                    with open(self.sync_cache_path, "r", encoding="utf-8") as f:
                        auto = json.load(f)
                    # 增量合并：只添加原 concepts 中不存在的新概念
                    for k, v in auto.get("concepts", {}).items():
                        if k not in base.get("concepts", {}):
                            base.setdefault("concepts", {})[k] = v
            except Exception:
                pass
        return base

    def maybe_sync_from_akshare(self, force: bool = False) -> bool:
        """从 akshare 同步概念板块成分股，增量更新本地缓存。
        缓存有效期 7 天。返回是否执行了同步。
        """
        import time

        # 检查缓存有效期
        if not force and os.path.exists(self.sync_cache_path):
            try:
                mtime = os.path.getmtime(self.sync_cache_path)
                if (time.time() - mtime) < 7 * 86400:
                    return False  # 缓存未过期
            except Exception:
                pass

        try:
            import akshare as ak
            logger.info('  🔄 同步 akshare 概念板块成分股（首次或过期，需 1-3 分钟）...')
            board_df = ak.stock_board_concept_name_em()
            if board_df is None or board_df.empty:
                return False

            board_names = board_df["板块名称"].tolist()
            auto_concepts = {}
            existing = set(self._data.get("concepts", {}).keys())
            synced_count = 0

            for board_name in board_names:
                # 跳过原有静态映射中已存在的概念（避免覆盖）
                if board_name in existing:
                    continue
                try:
                    cons = ak.stock_board_concept_cons_em(symbol=board_name)
                    if cons is None or cons.empty:
                        continue
                    code_col = "代码" if "代码" in cons.columns else "code"
                    name_col = "名称" if "名称" in cons.columns else "name"
                    codes = cons[code_col].astype(str).tolist()
                    names = cons[name_col].astype(str).tolist() if name_col in cons.columns else []
                    auto_concepts[board_name] = {
                        "stocks": [{"code": c, "name": n} for c, n in zip(codes, names)],
                        "keywords": [board_name],
                        "auto_synced": True,
                    }
                    synced_count += 1
                except Exception:
                    continue

            # 写入缓存
            with open(self.sync_cache_path, "w", encoding="utf-8") as f:
                json.dump({"concepts": auto_concepts, "synced_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                          f, ensure_ascii=False)

            # 重新加载合并后的数据
            self._data = self._load()
            logger.info(f'  ✅ 概念同步完成: 新增 {synced_count} 个概念板块')
            return True
        except Exception as e:
            logger.info(f'  ⚠️ 概念同步失败: {e}')
            return False

    @property
    def concepts(self) -> dict:
        return self._data.get("concepts", {})

    # ----------------------------------------------------------------
    # 构建股票名称索引（从实时行情数据）
    # ----------------------------------------------------------------
    def build_stock_index(self, a_stock_df) -> None:
        """从实时行情DataFrame构建股票名称→代码倒排索引"""
        self._stock_name_index = {}
        self._stock_code_set = set()

        if a_stock_df is None or a_stock_df.empty:
            return

        code_col = "代码" if "代码" in a_stock_df.columns else "股票代码"
        name_col = "名称" if "名称" in a_stock_df.columns else "股票名称"

        for _, row in a_stock_df.iterrows():
            raw_code = str(row.get(code_col, ""))
            name = str(row.get(name_col, ""))
            if not raw_code or not name:
                continue
            # 统一去除前缀
            code = raw_code.lower().replace("sz", "").replace("sh", "").replace("bj", "")
            self._stock_code_set.add(code)
            self._stock_code_set.add(raw_code)  # 保留原始格式
            self._stock_name_index[name] = code
            short_name = name.replace(" ", "")
            if short_name != name:
                self._stock_name_index[short_name] = code

    # ----------------------------------------------------------------
    # 第一层：新闻中直接提取股票名称/代码
    # ----------------------------------------------------------------
    def _extract_direct_hits(self, content: str) -> Set[str]:
        """从新闻正文中直接提取股票代码和名称"""
        hits = set()

        # 匹配6位数字股票代码（带sz/sh前缀或不带）
        codes_in_text = re.findall(r'(?:(?:sz|sh|bj)?\d{6})', content, re.IGNORECASE)
        for c in codes_in_text:
            raw = c.lower().replace("sz", "").replace("sh", "").replace("bj", "")
            if raw in self._stock_code_set:
                hits.add(raw)

        # 匹配纯6位数字（前后没有更多数字）
        pure_codes = re.findall(r'(?<!\d)(\d{6})(?!\d)', content)
        for c in pure_codes:
            if c in self._stock_code_set:
                hits.add(c)

        # 匹配股票名称
        for name, code in self._stock_name_index.items():
            if name in content:
                hits.add(code)

        return hits

    # ----------------------------------------------------------------
    # 第二层：概念关键词匹配
    # ----------------------------------------------------------------
    def _extract_concept_hits(self, content: str) -> Set[str]:
        """通过概念映射表匹配"""
        hits = set()
        content_lower = content.lower()

        for concept_name, concept_data in self.concepts.items():
            keywords = concept_data.get("keywords", [])
            for kw in keywords:
                if kw.lower() in content_lower:
                    for s in concept_data.get("stocks", []):
                        hits.add(s["code"])
                    break

        return hits

    # ----------------------------------------------------------------
    # 对外接口：新闻列表 → 命中股票（区分正负面）
    # ----------------------------------------------------------------
    def match_news_to_stocks(self, news_list: List[Dict], news_factor=None) -> dict:
        """
        输入新闻列表，返回按情感分类的命中股票
        返回: {
            "positive": [...],  # 利好新闻匹配的股票 → 加分
            "negative": [...],  # 利空新闻匹配的股票 → 减分
            "neutral": [...],   # 中性/噪音 → 不处理
        }
        """
        result = {"positive": set(), "negative": set(), "neutral": set()}
        direct_detail = {}
        concept_detail = {}

        # 正负面关键词（综合默认 + news_factor扩展）
        _pos_default = ["利好", "突破", "增长", "超预期", "涨停", "大涨", "新高", "获批", "上市", "交付",
                        "走强", "活跃", "拉升", "爆发", "创新高", "翻红", "连板", "反包", "修复", "高开"]
        _neg_default = ["利空", "下跌", "暴雷", "处罚", "减持", "跌停", "暴跌", "调查", "亏损", "立案",
                        "下挫", "重挫", "闪崩", "炸板", "回落", "出货", "退市", "暂停上市", "连续跌停",
                        # A股风险标记恒为大写"ST"(如 ST股/*ST)，小写"st"永远匹配不到且会误伤英文单词
                        "ST"]
        positive_kw = list(set(_pos_default + getattr(news_factor, 'positive_kw', [])))
        negative_kw = list(set(_neg_default + getattr(news_factor, 'negative_kw', [])))

        def _judge_sentiment(text: str) -> str:
            """判断新闻情感：positive / negative / neutral"""
            pos = sum(1 for kw in positive_kw if kw in text)
            neg = sum(1 for kw in negative_kw if kw in text)
            if pos > neg:
                return "positive"
            if neg > pos:
                return "negative"
            return "neutral"

        for news in news_list:
            content = news.get("content", "")
            if not content:
                continue

            sentiment = _judge_sentiment(content)

            # 层1：直接命中股票名称/代码
            direct = self._extract_direct_hits(content)
            for code in direct:
                direct_detail[code] = direct_detail.get(code, 0) + 1
                result[sentiment].add(code)

            # 层2：概念映射
            concept = self._extract_concept_hits(content)
            for code in concept:
                concept_detail[code] = concept_detail.get(code, 0) + 1
                result[sentiment].add(code)

            # 写回 mentioned_stocks，打通 news_factor.score_stock_news 打分链路
            # （历史缺陷：匹配结果只用于过滤，score_stock_news 只读 mentioned_stocks → 因子恒为 50）
            all_hits = direct | concept
            if all_hits:
                ms = news.get("mentioned_stocks")
                if not isinstance(ms, list):
                    ms = []
                news["mentioned_stocks"] = list(set(ms) | all_hits)

        # 打印日志
        if direct_detail:
            top_direct = sorted(direct_detail.items(), key=lambda x: -x[1])[:5]
            names = []
            for code, count in top_direct:
                code_clean = code.replace("sz", "").replace("sh", "").replace("bj", "")
                stock_info = [s for s in self.get_hot_concept_stocks() if s["code"] == code_clean]
                name_str = stock_info[0]["name"] if stock_info else code_clean
                names.append(f"{name_str}({code_clean}×{count})")
            logger.info(f"  ✅ 新闻直接命中股票: {', '.join(names)}")
            if concept_detail:
                top_conc = sorted(concept_detail.items(), key=lambda x: -x[1])[:5]
            logger.info(f"  ✅ 概念映射命中: {len(concept_detail)}只 (正:{len(result['positive'])} 负:{len(result['negative'])} 中:{len(result['neutral'])})")
        return {
            "positive": list(result["positive"]),
            "negative": list(result["negative"]),
            "neutral": list(result["neutral"]),
        }

    # ----------------------------------------------------------------
    # 辅助接口
    # ----------------------------------------------------------------
    def get_stock_by_code(self) -> Dict[str, dict]:
        result = {}
        for concept_name, concept_data in self.concepts.items():
            for s in concept_data.get("stocks", []):
                code = s["code"]
                if code not in result:
                    result[code] = {
                        "code": code, "name": s["name"],
                        "concepts": [], "reasons": [],
                    }
                result[code]["concepts"].append(concept_name)
                # akshare 同步的概念成分股没有 reason 字段，需降级处理避免 KeyError
                reason = s.get("reason")
                if reason:
                    result[code]["reasons"].append(reason)
        return result

    def get_hot_concept_stocks(self) -> List[dict]:
        stock_map = self.get_stock_by_code()
        result = []
        for code, info in stock_map.items():
            result.append({
                "code": code, "name": info["name"],
                "concepts": info["concepts"],
                "concept_count": len(info["concepts"]),
                "reasons": info["reasons"],
            })
        return result

    def get_stocks_by_concept(self, concept_name: str) -> List[dict]:
        concept = self.concepts.get(concept_name)
        if not concept:
            return []
        return concept.get("stocks", [])

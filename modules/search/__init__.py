"""联网搜索客户端：博查主用，豆包备用（自动回退）。

配置示例：
```yaml
search:
  enabled: true
  trigger_keywords: [...]
  min_interval_seconds: 60
  result_limit: 5
  backends:
    primary: bocha
    bocha:
      api_key: sk-...
      freshness: noLimit
      count: 5
    doubao:
      api_key: ...
      time_range: OneWeek
      count: 5
```
"""
import datetime
import logging
import re
import time

from .base import SearchResult
from .bocha import BochaSearch
from .doubao import DoubaoSearch

logger = logging.getLogger(__name__)

BACKENDS = {"bocha": BochaSearch, "doubao": DoubaoSearch}

# 命中即判定为"时效性问题"：强制用近期时间窗，避免返回几年前的旧数据
# 导致 bot 答错当前版本/天气/新闻。不含"卡池"等历史版本查询词，
# 保证"绝区零1.4卡池"这类明确版本号的历史查询仍能搜到。
TIME_SENSITIVE_SIGNALS = (
    "现在", "今天", "今天", "最新", "最近", "当前", "当下",
    "天气", "温度", "比分", "谁赢了", "赛程", "赛事",
    "新闻", "热搜", "汇率", "油价", "股价", "涨停", "跌了",
    "多少钱", "价格", "开服", "公测", "兑换码", "礼包码",
    "前瞻", "爆料",
)

# 结果里若大量出现"未来上线/前瞻"标记，说明最近没有"当前状态"的新文章，
# 全是下版本预告（如鸣潮3.6 8月20日上线）。这类结果要降权，避免误导。
FUTURE_RESULT_MARKERS = (
    "即将上线", "将于", "月日上线", "才上线", "要等", "才开",
    "前瞻", "预告", "待上线", "X月", "月上线", "上线",
)


class SearchClient:
    """带主备回退的搜索客户端。

    主后端失败或返回空 → 自动尝试备后端。按会话限频，
    避免 bot 对每条消息都发起搜索。
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        primary = config.get("primary", "bocha")
        backends_cfg = config.get("backends", {}) or {}

        self._backends: dict[str, object] = {}
        for name, backend_cls in BACKENDS.items():
            cfg = backends_cfg.get(name, {}) or {}
            if cfg.get("api_key"):
                try:
                    self._backends[name] = backend_cls(cfg.pop("api_key"), **cfg)
                except Exception as exc:
                    logger.error(f"搜索后端 {name} 初始化失败: {exc}")

        self.primary = primary if primary in self._backends else (next(iter(self._backends), ""))
        self._last_search: dict[str, float] = {}  # session_id -> 上次搜索时间
        self.min_interval = max(0.0, float(config.get("min_interval_seconds", 60)))
        self.result_limit = max(1, min(8, int(config.get("result_limit", 5))))

    @property
    def available(self) -> bool:
        return self.enabled and bool(self._backends)

    def can_search(self, session_id: str) -> bool:
        """检查会话是否允许发起新搜索（限频）。"""
        last = self._last_search.get(session_id, 0.0)
        return (time.time() - last) >= self.min_interval

    def _record(self, session_id: str) -> None:
        self._last_search[session_id] = time.time()

    @staticmethod
    def is_time_sensitive(query: str) -> bool:
        """判断搜索词是否为时效性问题（需要近期结果）。"""
        q = (query or "").lower()
        return any(sig in q for sig in TIME_SENSITIVE_SIGNALS)

    async def search(
        self,
        query: str,
        session_id: str = "",
        *,
        force_backend: str = "",
        prefer_recent: bool = False,
    ) -> list[SearchResult]:
        """主后端优先，失败回退备后端。返回格式化好的结果列表。

        时效性问题（天气/新闻/当前版本/比分等）自动用近期时间窗，
        避免博查 noLimit 返回几年前的旧数据导致答错当前状态。

        force_backend：指定优先使用的后端（如补搜当前版本卡池时用豆包，
        它比博查更能命中"当前版本在售卡池"）。
        prefer_recent：强制用近期时间窗（不依赖关键词判断），
        适合"当前版本号+卡池"这类必须新鲜的补搜。
        """
        if not self.available:
            return []
        if session_id and not self.can_search(session_id):
            return []

        time_sensitive = self.is_time_sensitive(query) or prefer_recent
        # 时效性问题自动补当前年月，避免 LLM 凭训练记忆写成错误的年份（如 2025）。
        if time_sensitive:
            query = self._ensure_current_month(query)

        order = [force_backend] if force_backend in self._backends else []
        order += [name for name in self._backends if name not in order]
        for name in order:
            backend = self._backends[name]
            kwargs = {}
            if time_sensitive:
                if name == "bocha":
                    kwargs["freshness"] = "oneMonth"
                elif name == "doubao":
                    kwargs["time_range"] = "OneMonth"
            try:
                results = await backend.search(query, **kwargs)
            except Exception as exc:
                logger.error(f"搜索后端 {name} 异常: {exc}")
                results = []
            if results:
                # 时效性查询：降权"未来上线/前瞻"类结果。搜索引擎按热度排序，
                # 最近最热的常常是"下版本预告"，而用户问"现在"——这些要排到后面。
                if time_sensitive:
                    results = self._deprioritize_future(results)
                if session_id:
                    self._record(session_id)
                logger.info(
                    f"[搜索] 后端={name}, 时效={'是' if time_sensitive else '否'}, "
                    f"query={query[:40]}, 结果={len(results)}"
                )
                return results[: self.result_limit]

        logger.warning(f"[搜索] 所有后端均无结果: query={query[:40]}")
        return []

    @staticmethod
    def _deprioritize_future(results: list[SearchResult]) -> list[SearchResult]:
        """把明显描述"未来上线/前瞻"的结果排到列表末尾。

        依据：标题/摘要含「即将上线/将于/才开/前瞻/预告/上线」等未来标记。
        保留它们（可能含当前版本对比信息），但移到后面，让模型优先看到
        真正描述当前状态的结果。
        """
        future = []
        present = []
        for r in results:
            text = f"{r.title} {r.summary or ''} {r.snippet or ''}"
            if any(m in text for m in FUTURE_RESULT_MARKERS):
                future.append(r)
            else:
                present.append(r)
        if not present:
            # 全是前瞻/预告时保留原序（总比空结果好，模型能据此判断"还没开"）
            return results
        return present + future

    @staticmethod
    def is_future_result(r: SearchResult) -> bool:
        """判断一条结果是否在描述「未来/前瞻」（还没发生的事）。"""
        text = f"{r.title} {r.summary or ''} {r.snippet or ''}"
        return any(m in text for m in FUTURE_RESULT_MARKERS)

    @staticmethod
    def all_future(results: list[SearchResult]) -> bool:
        """结果是否全是前瞻/预告（最近没有"当前状态"的新文章）。"""
        return bool(results) and all(SearchClient.is_future_result(r) for r in results)

    @staticmethod
    def extract_game_name(query: str) -> str:
        """从查询里提取游戏/主题名（去掉"当前/卡池/现在/UP"等修饰词后剩下的词）。

        例：「鸣潮 当前版本 卡池 现在UP」→「鸣潮」；
           「原神现在up角色是谁」→「原神」。
        提取失败返回空串。
        """
        q = query
        # 去掉修饰词与英文缩写
        q = re.sub(r"当前|现在|本期|最新|最近|版本|卡池|角色|是谁|什么|游戏|正在|进行中|的", "", q)
        q = re.sub(r"\bup\b|\bUP\b|谁up", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+", "", q)
        # 取前 2~6 个中文字符作为主题名（游戏名通常 2-4 字）
        match = re.match(r"[一-鿿]{2,6}", q)
        return match.group(0) if match else ""

    @staticmethod
    def infer_current_version(results: list[SearchResult]) -> str:
        """从全是前瞻的结果里推断当前版本号。

        例：结果都说"3.6要8月X号才开" → 当前版本 ≈ 3.5。
        取所有标题里的最大 X.Y 版本号，减 0.1（X.Y → 上一版本）。
        推断失败返回空串。
        """
        versions = []
        for r in results:
            # 不用 \b（中文是 \w 字符，"鸣潮3.6"的 3 前没有词边界）；
            # 用 (?<!\d)(?!\d) 避免匹配日期（如 2026.08）。
            for m in re.finditer(r"(?<!\d)(\d{1,2})\.(\d{1,2})(?!\d)", f"{r.title} {r.summary or ''}"):
                versions.append((int(m.group(1)), int(m.group(2))))
        if not versions:
            return ""
        # 过滤离谱版本号（如把某篇文章里的 100.5 当版本），保留合理的
        versions = [(a, b) for a, b in versions if a <= 20]
        if not versions:
            return ""
        major, minor = max(versions)
        if minor == 0:
            if major <= 1:
                return ""
            major, minor = major - 1, 9
        else:
            minor -= 1
        return f"{major}.{minor}"

    @staticmethod
    def _ensure_current_month(query: str) -> str:
        """保证时效性查询落在当前时间。

        - 模型训练记忆常把年份写成 2025 之类旧年份 → 直接替换为当前年；
        - 没写年份 → 追加「2026年8月」，让搜索引擎优先返回新内容。
        """
        now = datetime.datetime.now()
        # 1) 有 4 位年份但不是当前年 → 替换（用户问"现在"时模型写的旧年份是错的）
        fixed = re.sub(r"(?<!\d)20\d{2}(?!\d)", str(now.year), query)
        if fixed != query:
            return fixed
        # 2) 已含当前年或"x年"字样 → 不动
        if re.search(r"(20)?\d{2}\s*年", query) or str(now.year) in query:
            return query
        # 3) 否则追加当前年月
        return f"{query} {now.year}年{now.month}月"

    def format_results(self, results: list[SearchResult]) -> str:
        """把搜索结果拼成给 LLM 看的文本。

        给「前瞻/预告」性质的结果打标注，避免 LLM 把下版本前瞻当成现状。
        """
        if not results:
            return ""
        lines = ["搜索到的资料："]
        for i, r in enumerate(results, 1):
            lines.append(self._annotate(r, i))
        return "\n".join(lines)

    @staticmethod
    def _annotate(r: SearchResult, index: int) -> str:
        """对可能描述「未来/预告」的内容打标记，提醒 LLM 不要当成现在的情况。

        命中前瞻性关键词（即将上线/前瞻/将于/X月X日上线/下版本等）时，
        在标题前加【前瞻预告】，让 LLM 明确这是还没发生的事。
        """
        text = f"{r.title} {r.summary or ''} {r.snippet or ''}"
        future_markers = (
            "前瞻", "预告", "即将", "将于", "要等", "才开", "才上线", "还未", "待上线",
            "即将上线", "下个版本", "下版本", "爆料", "X月", "月日上线",
        )
        flagged = any(m in text for m in future_markers)
        prefix = "【前瞻预告】" if flagged else ""
        # 复用 to_context 的格式，但可注入前缀
        parts = []
        if r.date:
            parts.append(f"{index}. [{r.date[:10]}] {prefix}{r.title}")
        else:
            parts.append(f"{index}. {prefix}{r.title}")
        if r.site:
            parts.append(f"来源：{r.site}")
        body = r.summary or r.snippet or r.content
        if body:
            parts.append(body[:600])
        if r.url:
            parts.append(r.url[:100])
        return "\n".join(parts)

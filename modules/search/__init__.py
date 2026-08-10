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
import logging
import time

from .base import SearchResult
from .bocha import BochaSearch
from .doubao import DoubaoSearch

logger = logging.getLogger(__name__)

BACKENDS = {"bocha": BochaSearch, "doubao": DoubaoSearch}


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

    async def search(self, query: str, session_id: str = "") -> list[SearchResult]:
        """主后端优先，失败回退备后端。返回格式化好的结果列表。"""
        if not self.available:
            return []
        if session_id and not self.can_search(session_id):
            return []

        order = [self.primary]
        order += [name for name in self._backends if name != self.primary]
        for name in order:
            try:
                results = await self._backends[name].search(query)
            except Exception as exc:
                logger.error(f"搜索后端 {name} 异常: {exc}")
                results = []
            if results:
                if session_id:
                    self._record(session_id)
                logger.info(f"[搜索] 后端={name}, query={query[:40]}, 结果={len(results)}")
                return results[: self.result_limit]

        logger.warning(f"[搜索] 所有后端均无结果: query={query[:40]}")
        return []

    def format_results(self, results: list[SearchResult]) -> str:
        """把搜索结果拼成给 LLM 看的文本。"""
        if not results:
            return ""
        lines = ["搜索到的资料："]
        for i, r in enumerate(results, 1):
            lines.append(r.to_context(i))
        return "\n".join(lines)

"""豆包（火山引擎/feedcoop）Web 搜索后端。

接口：POST https://open.feedcoopapi.com/search_api/web_search
鉴权：Authorization: Bearer {API_KEY}
"""
import logging
from typing import Any

try:
    import aiohttp
except ModuleNotFoundError:
    aiohttp = None

from .base import BaseSearchBackend, SearchResult

logger = logging.getLogger(__name__)


class DoubaoSearch(BaseSearchBackend):
    """豆包搜索（火山引擎 WebSearch API）。支持 web/image 搜索，带正文摘要。"""

    name = "doubao"

    ENDPOINT = "https://open.feedcoopapi.com/search_api/web_search"

    def __init__(self, api_key: str = "", **config):
        super().__init__(api_key, **config)
        self.count = max(1, min(10, int(config.get("count", 5))))
        self.time_range = config.get("time_range", "")  # OneDay/OneWeek/OneMonth/OneYear/日期区间
        self.timeout = max(5.0, float(config.get("timeout", 15)))
        self.need_content = bool(config.get("need_content", False))

    async def search(self, query: str, time_range: str | None = None) -> list[SearchResult]:
        if aiohttp is None:
            logger.warning("aiohttp 未安装，搜索不可用")
            return []
        if not self.api_key:
            logger.warning("豆包搜索未配置 api_key")
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body: dict[str, Any] = {
            "Query": query,
            "SearchType": "web",
            "Count": self.count,
            "Filter": {
                "NeedContent": self.need_content,
                "NeedUrl": True,
            },
        }
        # 时效性问题临时覆盖时间范围（由 SearchClient 传入）；否则用配置值
        if time_range or self.time_range:
            body["TimeRange"] = time_range or self.time_range

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ENDPOINT,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        logger.error(f"豆包搜索失败: HTTP {resp.status} - {text[:200]}")
                        return []
                    payload = await resp.json()
            return self._parse(payload)
        except Exception as exc:
            logger.error(f"豆包搜索异常: {exc}")
            return []

    @staticmethod
    def _parse(payload: dict) -> list[SearchResult]:
        meta = payload.get("ResponseMetadata") or {}
        if meta.get("Error"):
            logger.error("豆包搜索返回错误: %s", meta["Error"])
            return []
        result = payload.get("Result") or {}
        results = []
        for item in result.get("WebResults", []) or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("Title") or "").strip()
            url = str(item.get("Url") or "").strip()
            if not title and not url:
                continue
            results.append(SearchResult(
                title=title[:200],
                url=url,
                snippet=str(item.get("Snippet") or "")[:300],
                summary=str(item.get("Summary") or ""),
                site=str(item.get("SiteName") or "")[:80],
                date=str(item.get("PublishTime") or ""),
                content=str(item.get("Content") or ""),
            ))
        return results

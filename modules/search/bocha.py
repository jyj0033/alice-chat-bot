"""博查（Bocha）Web 搜索后端。

接口：POST https://api.bocha.cn/v1/web-search
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


class BochaSearch(BaseSearchBackend):
    """博查 Web Search API。适合中文场景，返回结构化网页结果。"""

    name = "bocha"

    ENDPOINT = "https://api.bocha.cn/v1/web-search"

    def __init__(self, api_key: str = "", **config):
        super().__init__(api_key, **config)
        self.freshness = config.get("freshness", "noLimit")  # noLimit/oneDay/oneWeek/oneMonth/oneYear
        self.count = max(1, min(10, int(config.get("count", 5))))
        self.timeout = max(5.0, float(config.get("timeout", 15)))

    async def search(self, query: str, freshness: str | None = None) -> list[SearchResult]:
        if aiohttp is None:
            logger.warning("aiohttp 未安装，搜索不可用")
            return []
        if not self.api_key:
            logger.warning("博查搜索未配置 api_key")
            return []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "query": query,
            # 时效性问题（现在/今天/最新等）由 SearchClient 临时覆盖为 oneMonth；
            # 未覆盖时用配置值（默认 noLimit 保留历史版本查询能力）。
            "freshness": freshness or self.freshness,
            "summary": True,  # 让结果带正文相关摘要，更适合 LLM
            "count": self.count,
        }

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
                        logger.error(f"博查搜索失败: HTTP {resp.status} - {text[:200]}")
                        return []
                    payload = await resp.json()
            return self._parse(payload)
        except Exception as exc:
            logger.error(f"博查搜索异常: {exc}")
            return []

    @staticmethod
    def _parse(payload: dict) -> list[SearchResult]:
        data = payload.get("data") or {}
        pages = data.get("webPages") or {}
        results = []
        for item in pages.get("value", []) or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("name") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title and not url:
                continue
            results.append(SearchResult(
                title=title[:200],
                url=url,
                snippet=str(item.get("snippet") or "")[:300],
                summary=str(item.get("summary") or ""),
                site=str(item.get("siteName") or "")[:80],
                date=str(item.get("datePublished") or ""),
            ))
        return results

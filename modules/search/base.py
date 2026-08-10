"""搜索后端基类：统一搜索结果结构。"""
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """一条搜索结果"""
    title: str = ""
    url: str = ""
    snippet: str = ""    # 简短片段（~200字）
    summary: str = ""    # 正文相关摘要（500-1000字，更适合 LLM 场景）
    site: str = ""       # 站点名
    date: str = ""       # 发布时间（ISO）
    content: str = ""    # 完整正文（可能为空）

    def to_context(self, index: int) -> str:
        """格式化成给 LLM 看的一条文本。时间放最前，避免被摘要淹没。"""
        parts = []
        if self.date:
            parts.append(f"{index}. [{self.date[:10]}] {self.title}")
        else:
            parts.append(f"{index}. {self.title}")
        if self.site:
            parts.append(f"来源：{self.site}")
        body = self.summary or self.snippet or self.content
        if body:
            parts.append(body[:600])
        if self.url:
            parts.append(self.url[:100])
        return "\n".join(parts)


class BaseSearchBackend:
    """搜索后端基类。子类实现 async search(query) -> list[SearchResult]。"""

    name = "base"

    def __init__(self, api_key: str = "", **config):
        self.api_key = api_key
        self.config = config

    async def search(self, query: str) -> list[SearchResult]:
        raise NotImplementedError

"""
嵌入 + 重排服务
用于长期记忆的语义检索增强：
- 嵌入：OpenAI 兼容 /v1/embeddings（SiliconFlow 提供 Qwen3-Embedding、bge-m3 等）
- 重排：/v1/rerank（SiliconFlow 提供 bge-reranker 系列）

设计：未配置 api_key 或调用失败时返回 None，调用方自动回退到 TF-IDF。
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
_DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"


class EmbeddingRerankService:
    """嵌入 + 重排服务（SiliconFlow 兼容接口）"""

    def __init__(self, config: dict):
        config = config or {}
        self.api_key = (config.get("api_key") or "").strip()
        self.base_url = (config.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
        self.embed_model = config.get("embed_model") or _DEFAULT_EMBED_MODEL
        self.rerank_model = config.get("rerank_model") or _DEFAULT_RERANK_MODEL
        self.input_type = config.get("input_type") or None  # 可选：query/document/passage
        self.batch_size = int(config.get("batch_size", 32))
        self.enabled = bool(self.api_key)
        self._session = None
        self._embed_dim: Optional[int] = None

    async def _get_session(self):
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ---------- 嵌入 ----------

    async def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        """批量嵌入，返回与输入顺序一致的向量列表；失败返回 None"""
        if not self.enabled or not texts:
            return None
        try:
            import aiohttp
            session = await self._get_session()
            payload = {"model": self.embed_model, "input": texts}
            if self.input_type:
                payload["input_type"] = self.input_type

            async with session.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Embedding API error {resp.status}: {body[:300]}")
                    return None
                data = await resp.json()

            # 按 index 排序保证与输入顺序一致
            emb = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            vecs = [e["embedding"] for e in emb]
            if vecs:
                self._embed_dim = len(vecs[0])
            return vecs
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    async def embed_many(self, texts: list[str]) -> Optional[list[list[float]]]:
        """嵌入大量文本，自动分批并发"""
        if not texts:
            return []
        bs = self.batch_size
        chunks = [texts[i:i + bs] for i in range(0, len(texts), bs)]
        import asyncio
        results = await asyncio.gather(*(self.embed(c) for c in chunks))
        if any(r is None for r in results):
            return None
        out = []
        for r in results:
            out.extend(r)
        return out

    # ---------- 重排 ----------

    async def rerank(self, query: str, documents: list[str], top_n: int = 5) -> Optional[list[int]]:
        """重排文档，返回相关性降序的文档索引；失败返回 None"""
        if not self.enabled or not documents:
            return None
        try:
            import aiohttp
            session = await self._get_session()
            payload = {
                "model": self.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
            async with session.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Rerank API error {resp.status}: {body[:300]}")
                    return None
                data = await resp.json()

            results = data.get("results", [])
            if not results:
                return []
            if "index" in results[0]:
                # 按 relevance_score 降序（接口通常已排好序，这里兜底）
                if "relevance_score" in results[0]:
                    results = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
                return [r["index"] for r in results]
            return [r.get("index", i) for i, r in enumerate(results)]
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return None

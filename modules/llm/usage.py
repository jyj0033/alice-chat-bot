"""LLM token 消耗计量。

回答"钱花在哪了"：每次 LLM 调用都记下花在哪个功能点上，按天汇总，
只保留最近几天（默认 7 天），避免统计表无限膨胀。

用法：给 provider 套一层壳，壳在调用后把 usage 记到指定的功能点上。
同一个底层 provider 可以套多个壳（不同功能点各一个），记账互不干扰。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 功能点 → 中文名。key 与 llm_routing 的角色名对齐。
SITE_LABELS: dict[str, str] = {
    "reply": "回复生成",
    "reply_fallback": "回复备用模型",
    "reply_review": "回复语义复核",
    "meme_review": "表情复核",
    "conversation_judge": "目标判断",
    "search": "检索判断器",
    "search_answer": "带资料生成",
    "vision": "视觉识别",
    "group_analysis": "群日报",
    "digest": "群聊纪要",
    "profile": "用户画像",
    "mbti": "MBTI 分析",
    "slang": "群黑话整理",
    "expression_learning": "表达习惯",
    "memory": "记忆整理",
    "default": "未分类",
}

DEFAULT_RETENTION_DAYS = 7


def site_label(site: str) -> str:
    """功能点的展示名。"""
    key = str(site or "").strip()
    return SITE_LABELS.get(key, key or "未分类")


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class TokenUsageCollector:
    """把每次调用的 token 消耗按天、按功能点记下来。

    记在内存里攒一小批，再落到数据库：一次群消息可能触发好几次模型调用，
    每一次都直接写库会把 SQLite 写锁占满。
    """

    def __init__(
        self,
        storage: Any = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        enabled: bool = True,
        flush_threshold: int = 20,
    ) -> None:
        self.storage = storage
        self.retention_days = max(1, int(retention_days or DEFAULT_RETENTION_DAYS))
        self.enabled = bool(enabled)
        self.flush_threshold = max(1, int(flush_threshold or 20))
        # (日期, 功能点, 模型) → 计数
        self._pending: dict[tuple[str, str, str], dict[str, int]] = {}
        self._last_cleanup_day = ""

    # ---------------------------------------------------------------- 记录

    def record(self, site: str, model: str, usage: dict | None) -> None:
        """记一次调用。usage 形如 {prompt_tokens, completion_tokens, total_tokens}。"""
        if not self.enabled or not usage:
            return
        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        total = _as_int(usage.get("total_tokens")) or (prompt + completion)
        if not (prompt or completion or total):
            return
        key = (
            date.today().isoformat(),
            str(site or "default"),
            str(model or ""),
        )
        row = self._pending.setdefault(
            key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        row["calls"] += 1
        row["prompt_tokens"] += prompt
        row["completion_tokens"] += completion
        row["total_tokens"] += total

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def pending_calls(self) -> int:
        return sum(row["calls"] for row in self._pending.values())

    # ---------------------------------------------------------------- 落库

    async def flush(self) -> int:
        """把攒下的计数写进数据库，返回写入的条数。"""
        if not self.enabled or not self._pending:
            return 0
        storage = self.storage
        if storage is None:
            # 存储还没就绪：计数留在内存里，等下次再落库，不能直接丢掉。
            return 0
        rows = list(self._pending.items())
        self._pending.clear()
        written = 0
        for (day, site, model), counters in rows:
            try:
                await storage.record_llm_token_usage(
                    day=day,
                    site=site,
                    model=model,
                    calls=counters["calls"],
                    prompt_tokens=counters["prompt_tokens"],
                    completion_tokens=counters["completion_tokens"],
                    total_tokens=counters["total_tokens"],
                )
                written += 1
            except Exception as exc:
                logger.debug("token 用量写入失败：%s", exc)
        return written

    async def maybe_flush(self) -> int:
        """攒够一定量就落库。"""
        if self.pending_calls < self.flush_threshold:
            return 0
        return await self.flush()

    async def cleanup(self) -> int:
        """删掉保留期之外的旧记录。"""
        storage = self.storage
        if storage is None:
            return 0
        try:
            removed = await storage.cleanup_llm_token_usage(self.retention_days)
        except Exception as exc:
            logger.debug("token 用量清理失败：%s", exc)
            return 0
        if removed:
            logger.info("已清理 %s 天前的 token 用量记录 %s 条", self.retention_days, removed)
        return removed

    async def maybe_cleanup(self) -> int:
        """每天最多清理一次。"""
        today = date.today().isoformat()
        if today == self._last_cleanup_day:
            return 0
        self._last_cleanup_day = today
        return await self.cleanup()

    def attach_storage(self, storage: Any) -> None:
        """记忆系统初始化完成后把存储接上。"""
        self.storage = storage


class TaggedProvider:
    """给 provider 套一层壳，把这次调用的消耗记到指定功能点。

    壳只暴露 chat()，其余属性透传给底层 provider（调用点会读 model/config 等）。
    """

    def __init__(self, provider: Any, site: str, collector: Optional[TokenUsageCollector]):
        self.raw_provider = provider
        self.site = str(site or "default")
        self.collector = collector

    async def chat(self, request: Any) -> Any:
        response = await self.raw_provider.chat(request)
        if self.collector is not None:
            try:
                self.collector.record(
                    self.site,
                    getattr(response, "model", "") or getattr(self.raw_provider, "model", ""),
                    getattr(response, "usage", None),
                )
            except Exception as exc:
                logger.debug("token 用量记录失败：%s", exc)
        return response

    def __getattr__(self, name: str) -> Any:
        # 只在常规属性查找失败时触发，透传到底层 provider。
        provider = self.__dict__.get("raw_provider")
        if provider is None:
            raise AttributeError(name)
        return getattr(provider, name)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        model = getattr(self.raw_provider, "model", "?")
        return f"<TaggedProvider {self.site} model={model}>"


def unwrap_provider(provider: Any) -> Any:
    """剥掉计量壳，拿到底层 provider。

    同一个底层 provider 可能被多个功能点共用，套壳前先剥一次，
    免得壳套壳导致同一次调用被记两遍。
    """
    if isinstance(provider, TaggedProvider):
        return provider.raw_provider
    return provider


def tag_provider(
    provider: Any,
    site: str,
    collector: Optional[TokenUsageCollector],
) -> Any:
    """给 provider 套上计量壳。已经套过同一个功能点的就直接返回。"""
    if provider is None:
        return None
    raw = unwrap_provider(provider)
    if isinstance(provider, TaggedProvider) and provider.site == str(site or "default"):
        return provider
    if collector is None or not collector.enabled:
        return raw
    return TaggedProvider(raw, site, collector)

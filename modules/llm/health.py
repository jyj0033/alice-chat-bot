"""语言模型 Provider 的健康状态。

只做一件事：记住每个 provider 最近一次成功 / 失败，让「主模型已经废了」
这件事在日志和面板上看得见。

**不改路由、不改回退策略** —— 走不走备用模型由调用点决定，这里只负责
把失败暴露出来。典型场景：主模型的中转账号被封（HTTP 403），回退还能顶住，
但如果不显式记一笔，表面上一切正常，只是回复质量悄悄降级了。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# 同一个 provider 连续失败时，告警日志最多每 60 秒刷一条，避免刷屏。
_ALERT_INTERVAL = 60.0


class ProviderHealth:
    """按 provider 记录调用成功 / 失败。"""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def record_success(self, provider_id: str, model: str = "") -> None:
        state = self._state(provider_id)
        state["successes"] = int(state.get("successes", 0)) + 1
        state["last_success_at"] = time.time()
        if model:
            state["model"] = model
        # 成功一次就算恢复了：面板上不再显示降级。
        state["degraded"] = False
        state.pop("last_error", None)
        state.pop("last_error_at", None)

    def record_failure(self, provider_id: str, error: Any, model: str = "") -> None:
        state = self._state(provider_id)
        state["failures"] = int(state.get("failures", 0)) + 1
        state["last_failure_at"] = time.time()
        if model:
            state["model"] = model
        message = str(error)
        state["last_error"] = message[:500]
        state["degraded"] = True
        now = time.time()
        last_alert = float(state.get("last_alert_at", 0.0))
        if now - last_alert >= _ALERT_INTERVAL:
            state["last_alert_at"] = now
            logger.error(
                "⚠️ 语言模型「%s」调用失败（累计 %d 次），最近一次：%s",
                provider_id,
                state["failures"],
                message[:200],
            )

    def _state(self, provider_id: str) -> dict[str, Any]:
        key = str(provider_id or "default")
        return self._states.setdefault(
            key,
            {"failures": 0, "successes": 0, "degraded": False},
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """给面板用的快照。"""
        result: dict[str, Any] = {}
        for provider_id, state in self._states.items():
            result[provider_id] = {
                "model": state.get("model", ""),
                "failures": int(state.get("failures", 0)),
                "successes": int(state.get("successes", 0)),
                "degraded": bool(state.get("degraded", False)),
                "last_error": state.get("last_error", ""),
                "last_error_at": state.get("last_error_at"),
                "last_success_at": state.get("last_success_at"),
            }
        return result

    def degraded_providers(self) -> list[str]:
        """当前处于降级状态（最近一次调用失败）的 provider。"""
        return [
            provider_id
            for provider_id, state in self._states.items()
            if state.get("degraded")
        ]


class HealthTrackedProvider:
    """给 provider 套一层健康记录壳。

    接口与底层 provider 一致（``chat`` 走底层，其余属性透传），
    调用失败时记一笔再原样抛出，调用方感知不到这层壳。
    """

    def __init__(self, provider: Any, provider_id: str, health: ProviderHealth):
        self.raw_provider = provider
        self.provider_id = str(provider_id)
        self.health = health

    async def chat(self, request: Any) -> Any:
        model = str(getattr(self.raw_provider, "model", "") or "")
        try:
            response = await self.raw_provider.chat(request)
        except Exception as exc:
            # CancelledError 是 BaseException，不会走到这里，任务取消不会被
            # 误记成 provider 失败。
            self.health.record_failure(self.provider_id, exc, model)
            raise
        self.health.record_success(self.provider_id, model)
        return response

    def __getattr__(self, name: str) -> Any:
        provider = self.__dict__.get("raw_provider")
        if provider is None:
            raise AttributeError(name)
        return getattr(provider, name)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        model = getattr(self.raw_provider, "model", "?")
        return f"<HealthTrackedProvider {self.provider_id} model={model}>"

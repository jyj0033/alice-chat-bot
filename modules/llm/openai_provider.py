"""
OpenAI 兼容 Provider
支持 OpenAI、Claude（通过兼容接口）、本地模型等
"""
import asyncio
import logging
from typing import AsyncIterator, Optional
import openai
from openai import AsyncOpenAI

from .base import LLMProvider, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI 兼容 Provider"""

    # 根据 provider_type 自动拼接路径
    BASE_URL_PATHS = {
        "openai": "/v1/chat/completions",
        "siliconflow": "/v1/chat/completions",
        "anthropic": "/v1/messages",  # Claude 兼容格式
        "claude": "/v1/messages",    # Claude 兼容格式
        "minimax": "/v1/messages",   # Claude 兼容格式
        "deepseek": "/v1/chat/completions",
        "nvidia": "/chat/completions",
        "ollama": "/api/chat",
        "compatible": "/v1/chat/completions",
        "openai_compatible": "/v1/chat/completions",
    }

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4o")
        self.timeout = config.get("timeout", 60)
        self.provider_type = config.get("provider_type", "openai").lower()

        # OpenAI SDK 会自动添加 /chat/completions，不需要手动拼接
        # base_url 应该是类似 https://api.openai.com/v1 或 https://api.siliconflow.cn/v1
        pass

        # Claude/minimax 兼容格式需要特殊 headers
        if self.provider_type in ("anthropic", "claude", "minimax"):
            # Claude 兼容格式
            headers = {
                "anthropic-version": "2023-06-01",
                **(config.get("custom_headers", {}) or {}),
            }
        else:
            headers = config.get("custom_headers", {}) or {}

        # 初始化客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
            max_retries=3,
            default_headers=headers if headers else None,
        )

        logger.info(f"Initialized OpenAI provider: {base_url}, model: {self.model}")

    @property
    def provider_name(self) -> str:
        return "openai"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求"""
        try:
            # 构建请求参数
            kwargs = {
                "model": request.model or self.model,
                "messages": self.format_messages(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
            }

            if request.stop:
                kwargs["stop"] = request.stop

            if request.tools:
                kwargs["tools"] = request.tools

            # 发送请求
            response = await self.client.chat.completions.create(**kwargs)

            message = response.choices[0].message
            # Claude 兼容格式响应解析
            if self.provider_type in ("anthropic", "claude", "minimax"):
                # Claude 返回的是 Anthropic 格式，需要转换
                content = message.content if hasattr(message, 'content') else str(message)
            else:
                content = message.content or ""

            # function calling：解析模型请求的工具调用（OpenAI 格式）
            tool_calls = []
            for tc in (getattr(message, "tool_calls", None) or []):
                fn = getattr(tc, "function", None)
                if not fn:
                    continue
                arguments = {}
                try:
                    import json
                    arguments = json.loads(fn.arguments or "{}")
                except (TypeError, ValueError):
                    arguments = {"raw": fn.arguments}
                tool_calls.append({
                    "id": getattr(tc, "id", ""),
                    "name": getattr(fn, "name", ""),
                    "arguments": arguments,
                })

            return ChatResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
                tool_calls=tool_calls,
            )

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}", exc_info=True)
            raise

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """流式聊天请求"""
        try:
            kwargs = {
                "model": request.model or self.model,
                "messages": self.format_messages(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "stream": True,
            }

            if request.stop:
                kwargs["stop"] = request.stop

            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """关闭客户端"""
        await self.client.close()


# Provider 工厂
def create_provider(provider_type: str, config: dict) -> LLMProvider:
    """创建 LLM Provider"""
    provider_type = provider_type.lower()

    # Claude/anthropic 兼容格式使用专门的 Provider（请求 /v1/messages 端点）
    if provider_type in ("claude", "anthropic"):
        from .claude_provider import ClaudeProvider
        return ClaudeProvider(config)

    # OpenAI 兼容格式使用 OpenAIProvider（包括 minimax/siliconflow/nvidia 等）
    if provider_type in ("openai", "compatible", "openai_compatible", "siliconflow", "deepseek", "minimax", "nvidia"):
        return OpenAIProvider(config)

    raise ValueError(f"Unknown provider type: {provider_type}")

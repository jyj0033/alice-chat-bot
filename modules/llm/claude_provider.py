"""
Claude 兼容 Provider (Anthropic API 兼容格式)
用于 Claude 等使用 /v1/messages 端点的服务
"""
import asyncio
import logging
from typing import AsyncIterator
import aiohttp

from .base import LLMProvider, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """Claude 兼容 Provider"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.model = config.get("model", "claude-3-5-sonnet")
        self.timeout = config.get("timeout", 120)

        # 构建完整 URL
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        self.base_url += "v1/messages"

        logger.info(f"Initialized Claude provider: {self.base_url}, model: {self.model}")

    @property
    def provider_name(self) -> str:
        return "claude"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求"""
        try:
            # 构建请求头
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }

            # 构建请求体
            body = {
                "model": request.model or self.model,
                "messages": self.format_claude_messages(request.messages),
                "max_tokens": request.max_tokens or 1024,
            }

            if request.temperature is not None:
                body["temperature"] = request.temperature

            if request.top_p is not None:
                body["top_p"] = request.top_p

            if request.stop:
                body["stop_sequences"] = [request.stop] if isinstance(request.stop, str) else request.stop

            # 发送请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Claude API error: {response.status} - {error_text}")
                        raise Exception(f"API error: {response.status} - {error_text}")

                    result = await response.json()

            # 解析响应
            content = result.get("content", [{}])[0].get("text", "") if result.get("content") else ""

            usage = result.get("usage", {})
            return ChatResponse(
                content=content,
                model=result.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
                finish_reason=result.get("stop_reason", "stop"),
                raw_response=result,
            )

        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat: {e}", exc_info=True)
            raise

    def format_claude_messages(self, messages) -> list:
        """将消息转换为 Claude 格式"""
        result = []
        for msg in messages:
            if msg.role == "system":
                # Claude 不支持 system 消息，转换为 user 消息
                result.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == "user":
                result.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == "assistant":
                result.append({
                    "role": "assistant",
                    "content": msg.content
                })
        return result

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """流式聊天请求"""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }

            body = {
                "model": request.model or self.model,
                "messages": self.format_claude_messages(request.messages),
                "max_tokens": request.max_tokens or 1024,
                "stream": True,
            }

            if request.temperature is not None:
                body["temperature"] = request.temperature

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error: {response.status} - {error_text}")

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            import json
                            try:
                                chunk = json.loads(data)
                                if chunk.get("type") == "content_block_delta":
                                    delta = chunk.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        yield delta.get("text", "")
                            except:
                                pass

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """关闭客户端"""
        pass

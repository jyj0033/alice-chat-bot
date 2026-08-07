"""
LLM Provider 抽象层
支持多种 LLM 服务（OpenAI 兼容接口）
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """聊天消息

    content 支持两种形式：
    - ``str``：纯文本（现有行为不变）
    - ``list``：OpenAI 多模态格式，如
      ``[{"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "..."}}]``
      Claude 兼容 Provider 会把 ``image_url`` 转换为 Anthropic image source。
    """
    role: str  # system / user / assistant
    content: str | list
    name: Optional[str] = None


@dataclass
class ChatRequest:
    """聊天请求"""
    messages: list[ChatMessage] = field(default_factory=list)
    model: str = "gpt-4o"
    temperature: float = 0.8
    max_tokens: int = 500
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[list[str]] = None

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> "ChatRequest":
        """添加消息"""
        self.messages.append(ChatMessage(role=role, content=content, name=name))
        return self

    def add_system(self, content: str) -> "ChatRequest":
        """添加系统消息"""
        return self.add_message("system", content)

    def add_user(self, content: str) -> "ChatRequest":
        """添加用户消息"""
        return self.add_message("user", content)

    def add_assistant(self, content: str) -> "ChatRequest":
        """添加助手消息"""
        return self.add_message("assistant", content)


@dataclass
class ChatResponse:
    """聊天响应"""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    raw_response: Optional[Any] = None


class LLMProvider(ABC):
    """LLM Provider 抽象基类"""

    def __init__(self, config: dict):
        self.config = config
        self.model = config.get("model", "gpt-4o")

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """发送聊天请求"""
        raise NotImplementedError

    @abstractmethod
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """流式聊天请求"""
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供者名称"""
        raise NotImplementedError

    def format_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """格式化消息列表"""
        result = []
        for msg in messages:
            item: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                item["name"] = msg.name
            result.append(item)
        return result

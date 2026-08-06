"""
平台适配器基类
定义统一的接口规范
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .rich_content import MessageSegment


@dataclass
class Message:
    """统一的消息格式"""
    message_id: str
    message_type: str  # private / group
    sender_id: str
    sender_name: str
    group_id: Optional[str] = None
    content: str = ""
    raw_content: str = ""  # 原始消息内容
    mentioned_me: bool = False
    mentioned_others: list[str] = field(default_factory=list)  # 本条消息 @ 的其他 QQ 号（不含 bot）
    reply_to_id: Optional[str] = None
    reply_to_qq: Optional[str] = None  # 被回复消息的发送者 QQ 号（OneBot reply 段扩展字段）
    segments: list[MessageSegment] = field(default_factory=list)
    outer_text: str = ""  # 仅发送者最外层输入，用于触发检测，排除转发/卡片内容
    rich_only: bool = False
    rich_type: str = ""

    @property
    def session_id(self) -> str:
        """会话ID"""
        if self.message_type == "group":
            return f"group_{self.group_id}"
        return f"private_{self.sender_id}"


class PlatformAdapter(ABC):
    """平台适配器基类"""

    def __init__(self, config: dict):
        self.config = config
        self._connected = False
        self._running = False

    @abstractmethod
    async def connect(self) -> None:
        """连接到平台"""
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        raise NotImplementedError

    @abstractmethod
    async def send_message(self, session_id: str, content: str) -> bool:
        """发送消息"""
        raise NotImplementedError

    @abstractmethod
    async def send_group_message(self, group_id: str, content: str) -> bool:
        """发送群消息"""
        raise NotImplementedError

    @abstractmethod
    async def send_private_message(self, user_id: str, content: str) -> bool:
        """发送私聊消息"""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running

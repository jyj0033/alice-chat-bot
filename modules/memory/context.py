"""
上下文窗口管理
管理当前对话的上下文消息
"""
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Deque

logger = logging.getLogger(__name__)


@dataclass
class ContextMessage:
    """上下文消息"""
    sender_id: str
    sender_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    is_bot: bool = False
    message_id: str = ""
    reply_to: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "is_bot": self.is_bot,
            "message_id": self.message_id,
        }


class ContextWindow:
    """滑动上下文窗口"""

    def __init__(
        self,
        max_messages: int = 50,
        max_age: timedelta = timedelta(hours=2)
    ):
        self.messages: Deque[ContextMessage] = deque(maxlen=max_messages)
        self.max_messages = max_messages
        self.max_age = max_age
        self._last_cleanup = datetime.now()

    def add(self, message: ContextMessage) -> None:
        """添加消息"""
        self.messages.append(message)
        self._maybe_cleanup()

    def _maybe_cleanup(self) -> None:
        """定期清理过期消息"""
        now = datetime.now()
        if now - self._last_cleanup > timedelta(minutes=5):
            self._cleanup()
            self._last_cleanup = now

    def _cleanup(self) -> None:
        """清理过期消息"""
        cutoff = datetime.now() - self.max_age
        while self.messages and self.messages[0].timestamp < cutoff:
            self.messages.popleft()

    def get_recent(self, n: int = 20) -> list[ContextMessage]:
        """获取最近N条消息"""
        return list(self.messages)[-n:]

    def get_messages_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> list[ContextMessage]:
        """获取时间范围内的消息"""
        return [
            m for m in self.messages
            if start <= m.timestamp <= end
        ]

    def build_conversation_text(
        self,
        bot_name: str = "Bot",
        include_bot: bool = True,
        max_messages: int = 30
    ) -> str:
        """构建对话文本"""
        lines = []
        recent = self.get_recent(max_messages)

        for msg in recent:
            if not include_bot and msg.is_bot:
                continue

            speaker = bot_name if msg.is_bot else msg.sender_name
            lines.append(f"{speaker}：{msg.content}")

        return "\n".join(lines)

    def count_messages_since(self, since: datetime, sender_id: str = "") -> int:
        """统计自某个时间以来的消息数"""
        count = 0
        for msg in self.messages:
            if msg.timestamp > since:
                if not sender_id or msg.sender_id == sender_id:
                    count += 1
        return count

    def get_activity_level(self, window_minutes: int = 10) -> float:
        """
        计算活动水平 0.0 - 1.0
        """
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        count = self.count_messages_since(cutoff)

        # 归一化：假设每分钟 2 条消息为高活跃度
        expected = window_minutes * 2
        return min(1.0, count / expected)

    def clear(self) -> None:
        """清空上下文"""
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ContextWindow(messages={len(self.messages)}, max={self.max_messages})"


class ContextManager:
    """上下文管理器 - 管理多个会话的上下文"""

    def __init__(self, max_messages: int = 50, max_age_hours: int = 2):
        self._windows: dict[str, ContextWindow] = {}
        self.max_messages = max_messages
        self.max_age = timedelta(hours=max_age_hours)

    def get_window(self, session_id: str) -> ContextWindow:
        """获取会话的上下文窗口"""
        if session_id not in self._windows:
            self._windows[session_id] = ContextWindow(
                max_messages=self.max_messages,
                max_age=self.max_age
            )
        return self._windows[session_id]

    def add_message(
        self,
        session_id: str,
        sender_id: str,
        sender_name: str,
        content: str,
        is_bot: bool = False,
        message_id: str = ""
    ) -> None:
        """添加消息到上下文"""
        window = self.get_window(session_id)
        window.add(ContextMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            is_bot=is_bot,
            message_id=message_id,
        ))

    def build_context_prompt(
        self,
        session_id: str,
        bot_name: str = "Bot",
        persona_prompt: str = "",
        memories: list = None,
        max_messages: int = 30
    ) -> str:
        """构建上下文提示"""
        window = self.get_window(session_id)

        parts = []

        # 1. 人设
        if persona_prompt:
            parts.append(f"[你的设定]\n{persona_prompt}")

        # 2. 相关记忆
        if memories:
            parts.append("[你的记忆]\n" + "\n".join(f"- {m.content}" for m in memories[:5]))

        # 3. 最近对话
        conversation = window.build_conversation_text(
            bot_name=bot_name,
            max_messages=max_messages
        )
        if conversation:
            parts.append(f"[最近对话]\n{conversation}")

        return "\n\n".join(parts)

    def cleanup_inactive(self, max_inactive_minutes: int = 60) -> int:
        """清理不活跃的上下文"""
        # 简单实现：只清理过旧的
        now = datetime.now()
        to_remove = []

        for session_id, window in self._windows.items():
            if len(window.messages) > 0:
                last_msg = window.messages[-1]
                if now - last_msg.timestamp > timedelta(minutes=max_inactive_minutes):
                    to_remove.append(session_id)

        for session_id in to_remove:
            del self._windows[session_id]

        return len(to_remove)

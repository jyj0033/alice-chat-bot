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


def format_message_time(dt: datetime, now: datetime = None) -> str:
    """把时间戳格式化为口语化相对时间，用于对话记录的前缀标记

    - 60秒内   → 刚刚
    - 1小时内  → X分钟前
    - 今天     → HH:MM
    - 昨天     → 昨天 HH:MM
    - 1周内    → X天前
    - 1月内    → X周前
    - 今年     → M月D日
    - 往年     → YYYY年M月D日
    """
    now = now or datetime.now()
    if dt > now:
        dt = now
    delta = now - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "刚刚"
    if secs < 3600:
        return f"{int(secs // 60)}分钟前"
    if dt.date() == now.date():
        return dt.strftime("%H:%M")
    days = (now.date() - dt.date()).days
    if days == 1:
        return "昨天 " + dt.strftime("%H:%M")
    if days < 7:
        return f"{days}天前"
    if days < 30:
        weeks = days // 7
        return f"{weeks}周前"
    if dt.year == now.year:
        return f"{dt.month}月{dt.day}日"
    return f"{dt.year}年{dt.month}月{dt.day}日"


@dataclass
class ContextMessage:
    """上下文消息"""
    sender_id: str
    sender_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    is_bot: bool = False
    message_id: str = ""
    reply_to_id: Optional[str] = None     # 被回复消息的 ID（平台原始字段）
    reply_to_qq: Optional[str] = None     # 被回复消息的发送者 QQ 号
    directed_to_bot: bool = False         # 是否明确对 bot 说

    def to_dict(self) -> dict:
        return {
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "is_bot": self.is_bot,
            "message_id": self.message_id,
            "reply_to_id": self.reply_to_id,
            "reply_to_qq": self.reply_to_qq,
            "directed_to_bot": self.directed_to_bot,
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
        """构建对话文本，每条消息带 [时间] 前缀。

        身份归一：同一 QQ 号改群名片后，窗口里会同时出现"旧名/新名"，
        若原样输出，LLM 会把一个人当成两个人。这里把每个 QQ 号统一到
        "最近一次的昵称"，并在 改名/重名 时附加 (QQ尾号) 绑定身份。
        """
        lines = []
        recent = self.get_recent(max_messages)
        now = datetime.now()

        # id → 窗口内出现过的全部昵称；昵称 → 使用它的全部 id
        id_to_names: dict[str, set] = {}
        name_to_ids: dict[str, set] = {}
        for msg in recent:
            if msg.is_bot:
                continue
            if msg.sender_id:
                id_to_names.setdefault(msg.sender_id, set()).add(msg.sender_name)
            name_to_ids.setdefault(msg.sender_name, set()).add(msg.sender_id)

        # 每个 id 的规范昵称 = 该 id 最近一条消息的昵称（recent 时间正序，覆盖后为最新）
        canonical_name: dict[str, str] = {}
        for msg in recent:
            if not msg.is_bot and msg.sender_id:
                canonical_name[msg.sender_id] = msg.sender_name

        # 需要加 (QQ尾号) 区分的两种情形：
        #  1) 重名：同一昵称被多个 id 使用
        #  2) 改名：同一 id 在窗口内出现多个昵称（群名片变了）→ 用尾号把两个名字绑成同一人
        dup_names = {n for n, ids in name_to_ids.items() if len(ids) > 1}
        renamed_ids = {id_ for id_, names in id_to_names.items() if len(names) > 1}

        def display_name(msg) -> str:
            """消息发言者的显示名：规范昵称，改名/重名时附 (QQ尾号)。"""
            if msg.is_bot:
                return bot_name
            name = canonical_name.get(msg.sender_id, msg.sender_name)
            if not msg.sender_id:
                return name
            if msg.sender_id in renamed_ids or name in dup_names:
                return f"{name}({msg.sender_id[-4:]})"
            return name

        # 建立 QQ号→显示名 映射（回复指向标注也用规范名）
        qq_to_name: dict[str, str] = {}
        for msg in recent:
            if msg.sender_id:
                qq_to_name[msg.sender_id] = display_name(msg)

        # message_id → 发送者，用于 reply 段只有消息 ID、没有发送者 QQ 的情况。
        message_id_to_name: dict[str, str] = {}
        for msg in recent:
            if msg.message_id:
                message_id_to_name[str(msg.message_id)] = display_name(msg)

        for msg in recent:
            if not include_bot and msg.is_bot:
                continue

            speaker = display_name(msg)

            # 标注回复指向：这条消息是"回复谁"的（A→B，或 →@bot）
            pointer = ""
            if msg.reply_to_id or msg.reply_to_qq:
                target = None
                if msg.reply_to_id:
                    target = message_id_to_name.get(str(msg.reply_to_id))
                if not target and msg.reply_to_qq:
                    target = qq_to_name.get(str(msg.reply_to_qq))
                if target:
                    if target == speaker:
                        pointer = f"(回@{target}自己)"
                    else:
                        pointer = f"(回@{target})"
                elif str(msg.reply_to_qq or "") in ("", "0"):
                    pointer = "(回复某条消息)"
                else:
                    pointer = f"(回@尾号{str(msg.reply_to_qq)[-4:]})"

            direction = ""
            if msg.directed_to_bot and not msg.is_bot:
                direction = "(对你说)"

            time_str = format_message_time(msg.timestamp, now)
            lines.append(f"[{time_str}] {speaker}{pointer}{direction}：{msg.content}")

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
        message_id: str = "",
        reply_to_id: Optional[str] = None,
        reply_to_qq: Optional[str] = None,
        directed_to_bot: bool = False,
    ) -> None:
        """添加消息到上下文"""
        window = self.get_window(session_id)
        window.add(ContextMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            is_bot=is_bot,
            message_id=message_id,
            reply_to_id=reply_to_id,
            reply_to_qq=reply_to_qq,
            directed_to_bot=directed_to_bot,
        ))

    def update_message_content(
        self,
        session_id: str,
        message_id: str,
        content: str,
    ) -> bool:
        """富媒体异步解析完成后原位更新上下文，不改变消息先后顺序。"""
        if not message_id or not content:
            return False
        window = self.get_window(session_id)
        for message in reversed(window.messages):
            if str(message.message_id) == str(message_id):
                message.content = content
                return True
        return False

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
        now = datetime.now()

        parts = []

        # 0. 当前时间锚点 - 让 LLM 知道"现在是什么时候"，才能正确理解新旧
        weekday = "周" + "一二三四五六日"[now.weekday()]
        parts.append(f"[当前时间] {now.year}年{now.month}月{now.day}日 {weekday} {now.strftime('%H:%M')}")

        # 1. 人设
        if persona_prompt:
            parts.append(f"[你的设定]\n{persona_prompt}")

        # 2. 相关记忆（带发生时间，明确是旧事）
        if memories:
            # 窗口内每个 id 的最新昵称，用于把记忆里改名前存下的旧昵称改成现在的称呼，
            # 避免"记忆里叫小明、最近对话里叫明哥"，bot 以为是两个人。
            id_name: dict[str, str] = {}
            for m in window.messages:
                if m.sender_id and not m.is_bot:
                    id_name[m.sender_id] = m.sender_name
            mem_lines = []
            for m in memories[:5]:
                content = m.content
                meta = m.metadata or {}
                sid = meta.get("sender_id")
                if sid and sid in id_name:
                    old = meta.get("sender_name")
                    new = id_name[sid]
                    if old and old != new and content.startswith(old + "："):
                        content = new + content[len(old):]
                t = format_message_time(m.created_at, now)
                mem_lines.append(f"- [{t}] {content}")
            parts.append(
                "[你的记忆]（这些是你记得的旧事，[时间]是事情发生的时间，越久远的记忆越模糊）\n"
                + "\n".join(mem_lines)
            )

        # 3. 最近对话
        conversation = window.build_conversation_text(
            bot_name=bot_name,
            max_messages=max_messages
        )
        if conversation:
            parts.append(
                "[最近对话]（[时间]表示距现在多久，如\"昨天 20:15\"是昨晚的事；"
                "标注\"回@某人\"表示回复对象，标注\"对你说\"表示消息明确指向你）\n"
                + conversation
            )

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

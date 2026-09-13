"""
上下文窗口管理
管理当前对话的上下文消息
"""
import asyncio
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Deque

logger = logging.getLogger(__name__)

_UNSEEN_MEDIA_RE = re.compile(
    r"^\[(?P<kind>图片|表情包|动画表情|视频|合并转发)"
)
_TAGGED_MEDIA_RE = re.compile(
    r"^\[(?P<kind>图片|表情包|动画表情|视频)(?:，内容：|：)(?P<desc>[^\]]+)\]"
    r"(?:\s*\[(?:image|video)\])?$"
)
_UNSEEN_XML_RE = re.compile(r'^<unseen type="(?P<kind>[^"]+)"/>$')
_KIND_NOUN = {
    "图片": "图",
    "image": "图",
    "表情包": "表情",
    "动画表情": "表情",
    "mface": "表情",
    "视频": "视频",
    "video": "视频",
}


def _xml_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", " ")
    )


_ASKS_MEDIA_RE = re.compile(r"(这里面|图里|画面|照片里|都认识|认得出|看着像)")


def asks_about_unseen_media(content: str) -> bool:
    return bool(_ASKS_MEDIA_RE.search(content or ""))


def is_unresolved_media(content: str) -> bool:
    """还没有画面描述的图片/视频。"""
    text = (content or "").strip()
    if "看不清画面" in text:
        return True
    if "画面是" in text or "内容：" in text or "内容:" in text:
        return False
    return bool(_UNSEEN_MEDIA_RE.match(text) or _UNSEEN_XML_RE.match(text))


def _media_to_words(kind: str, description: str = "") -> str:
    desc = re.sub(r"\s+", " ", str(description or "")).strip().rstrip("。．. ")
    noun = _KIND_NOUN.get(kind, "")
    if kind in {"视频", "video"} or noun == "视频":
        return f"一段视频，画面是{desc}。" if desc else "一段视频，看不清画面。"
    if noun:
        return f"一张{noun}，画面是{desc}。" if desc else f"一张{noun}，看不清画面。"
    return desc


def opaque_media_content(content: str) -> str:
    """把旧占位符收成自然语言画面描述。"""
    text = (content or "").strip()
    tagged = _TAGGED_MEDIA_RE.match(text)
    if tagged:
        desc = tagged.group("desc").strip()
        if desc.startswith("[") or desc in {"动画表情", "image", "video"}:
            return _media_to_words(tagged.group("kind"))
        return _media_to_words(tagged.group("kind"), desc)
    xml = _UNSEEN_XML_RE.match(text)
    if xml:
        return _media_to_words(xml.group("kind"))
    match = _UNSEEN_MEDIA_RE.match(text)
    if match and "内容：" not in text and "内容:" not in text and "画面是" not in text:
        return _media_to_words(match.group("kind"))
    return text


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
    mentioned_user_ids: tuple[str, ...] = ()  # 本条消息 @ 的全部 QQ 号（不含 all）
    directed_to_bot: bool = False         # 是否明确对 bot 说
    conversation_target: str = ""         # 动态判断：bot / other / group / unknown
    conversation_intent: str = ""         # 动态判断：answer / follow_up / ...
    conversation_confidence: float = 0.0
    conversation_reason: str = ""

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
            "mentioned_user_ids": list(self.mentioned_user_ids),
            "directed_to_bot": self.directed_to_bot,
            "conversation_target": self.conversation_target,
            "conversation_intent": self.conversation_intent,
            "conversation_confidence": self.conversation_confidence,
            "conversation_reason": self.conversation_reason,
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
        # 标记是否已经尝试从 SQLite 恢复过最近历史；新窗口只尝试一次。
        self.restored_from_storage = False

    def add(self, message: ContextMessage) -> None:
        """添加消息"""
        self.messages.append(message)
        self._maybe_cleanup()

    def prepend(self, message: ContextMessage) -> None:
        """在窗口头部补入历史消息，保持当前实时消息仍在末尾。"""
        self.messages.appendleft(message)

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
        max_messages: int = 30,
        bot_id: str = "",
        exclude_message_id: str = "",
    ) -> str:
        """构建对话记录：用 XML 标签而不是「[刚刚] 某人(对你说)：」这种可发送格式。

        身份归一：同一 QQ 号改群名片后，窗口里会同时出现"旧名/新名"，
        若原样输出，LLM 会把一个人当成两个人。这里把每个 QQ 号统一到
        "最近一次的昵称"，并在 改名/重名 时附加 (QQ尾号) 绑定身份。
        """
        lines = []
        recent = self.get_recent(max_messages)
        if exclude_message_id:
            recent = [
                msg for msg in recent
                if str(msg.message_id or "") != str(exclude_message_id)
            ]
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
            """消息发言者的显示名：规范昵称，改名/重名时附 (QQ尾号)。

            bot 自己的发言标注「(你)」：不标的话，LLM 容易把「爱丽丝」的
            历史发言当成别的群友说的，出现自我矛盾（刚说"我还没抽"，
            下一句像劝别人一样说"抽就完事了"）。
            """
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
        # message_id → 内容（引用的消息若是图片/表情包，能看到它的识别摘要）
        message_id_to_content: dict[str, str] = {}
        for msg in recent:
            if msg.message_id:
                message_id_to_name[str(msg.message_id)] = display_name(msg)
                if msg.content:
                    message_id_to_content[str(msg.message_id)] = msg.content

        for msg in recent:
            if not include_bot and msg.is_bot:
                continue

            speaker = display_name(msg)
            attrs = [
                f't="{_xml_escape(format_message_time(msg.timestamp, now))}"',
                f'from="{_xml_escape(speaker)}"',
            ]
            if msg.is_bot:
                attrs.append('self="1"')

            reply_target = ""
            snippet = ""
            if msg.reply_to_id or msg.reply_to_qq:
                if msg.reply_to_id:
                    reply_target = message_id_to_name.get(str(msg.reply_to_id), "")
                if not reply_target and msg.reply_to_qq:
                    reply_target = qq_to_name.get(str(msg.reply_to_qq), "")
                if msg.reply_to_id:
                    snippet = (
                        message_id_to_content.get(str(msg.reply_to_id), "") or ""
                    ).strip()

            to_you = bool(msg.directed_to_bot) or (
                bool(bot_id)
                and any(str(uid) == str(bot_id) for uid in msg.mentioned_user_ids)
            ) or (
                bool(bot_id)
                and str(msg.reply_to_qq or "") == str(bot_id)
            ) or (bool(bot_name) and reply_target == bot_name)
            if to_you:
                attrs.append('to="you"')
            elif reply_target and reply_target != speaker:
                attrs.append(f'to="{_xml_escape(reply_target)}"')
            if snippet:
                attrs.append(f'quote="{_xml_escape(opaque_media_content(snippet)[:40])}"')

            mentioned_names = []
            for user_id in msg.mentioned_user_ids:
                user_id = str(user_id or "")
                if not user_id:
                    continue
                if bot_id and user_id == str(bot_id):
                    mentioned_names.append("you")
                else:
                    mentioned_names.append(
                        qq_to_name.get(user_id, f"QQ尾号{user_id[-4:]}")
                    )
            if mentioned_names:
                attrs.append(f'at="{_xml_escape("/".join(mentioned_names))}"')
            if msg.conversation_target:
                attrs.append(f'target="{_xml_escape(msg.conversation_target)}"')

            body = opaque_media_content(msg.content)
            lines.append(f"<m {' '.join(attrs)}>{_xml_escape(body)}</m>")

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
        mentioned_user_ids: Optional[list[str] | tuple[str, ...]] = None,
        directed_to_bot: bool = False,
        conversation_target: str = "",
        conversation_intent: str = "",
        conversation_confidence: float = 0.0,
        conversation_reason: str = "",
        timestamp: Optional[datetime] = None,
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
            mentioned_user_ids=tuple(
                dict.fromkeys(str(item) for item in (mentioned_user_ids or []) if str(item))
            ),
            directed_to_bot=directed_to_bot,
            conversation_target=str(conversation_target or ""),
            conversation_intent=str(conversation_intent or ""),
            conversation_confidence=float(conversation_confidence or 0.0),
            conversation_reason=str(conversation_reason or ""),
            timestamp=timestamp or datetime.now(),
        ))

    def prepend_message(
        self,
        session_id: str,
        sender_id: str,
        sender_name: str,
        content: str,
        is_bot: bool = False,
        message_id: str = "",
        reply_to_id: Optional[str] = None,
        reply_to_qq: Optional[str] = None,
        mentioned_user_ids: Optional[list[str] | tuple[str, ...]] = None,
        directed_to_bot: bool = False,
        conversation_target: str = "",
        conversation_intent: str = "",
        conversation_confidence: float = 0.0,
        conversation_reason: str = "",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """把持久化历史消息补到会话窗口头部。"""
        window = self.get_window(session_id)
        window.prepend(ContextMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            is_bot=is_bot,
            message_id=message_id,
            reply_to_id=reply_to_id,
            reply_to_qq=reply_to_qq,
            mentioned_user_ids=tuple(
                dict.fromkeys(str(item) for item in (mentioned_user_ids or []) if str(item))
            ),
            directed_to_bot=directed_to_bot,
            conversation_target=str(conversation_target or ""),
            conversation_intent=str(conversation_intent or ""),
            conversation_confidence=float(conversation_confidence or 0.0),
            conversation_reason=str(conversation_reason or ""),
            timestamp=timestamp or datetime.now(),
        ))

    def update_message_analysis(
        self,
        session_id: str,
        message_id: str,
        *,
        directed_to_bot: Optional[bool] = None,
        target: str = "",
        intent: str = "",
        confidence: Optional[float] = None,
        reason: str = "",
    ) -> bool:
        """更新一条已进入窗口的消息的动态目标判断。"""
        if not message_id:
            return False
        window = self.get_window(session_id)
        for message in reversed(window.messages):
            if str(message.message_id or "") != str(message_id):
                continue
            if directed_to_bot is not None:
                message.directed_to_bot = bool(directed_to_bot)
            if target:
                message.conversation_target = str(target)
            if intent:
                message.conversation_intent = str(intent)
            if confidence is not None:
                message.conversation_confidence = max(0.0, min(1.0, float(confidence)))
            if reason:
                message.conversation_reason = str(reason)[:240]
            return True
        return False

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
        max_messages: int = 30,
        bot_id: str = "",
        focus_message_id: str = "",
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
            max_messages=max_messages,
            bot_id=bot_id,
            exclude_message_id=focus_message_id,
        )
        if conversation:
            parts.append(
                "[最近对话]（这是机器记录，不是你可以发送的格式。"
                "from=谁说的，to=对谁说，to=\"you\" 是对你，self=\"1\" 是你自己说的，"
                "quote=引用了哪句。图和表情会写成「一张图，画面是…」，那是画面描述，"
                "不是群友打出来的字。"
                "只把里面的语义当上下文，禁止把标签、属性或整行记录复制进回复。"
                f"self=\"1\" 的话是你刚说过的，延续立场，不要打脸。）\n"
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

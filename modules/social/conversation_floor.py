"""群聊发言权分析。

从最近消息中判断当前是谁在和谁聊天、bot 是否拥有自然接话权，并产出
结构化行为计划。这里故意使用可解释的确定性规则，真实群聊数据只用于后续调参。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from typing import Any


class ActionType(str, Enum):
    """机器人在当前群聊回合可采取的动作。"""

    SILENT = "silent"
    REACT = "react"
    REPLY = "reply"
    ANSWER = "answer"
    FOLLOW_UP = "follow_up"
    INTERRUPT = "interrupt"


@dataclass
class ConversationFloor:
    """当前群聊的发言权快照。"""

    active_speakers: tuple[str, ...] = ()
    likely_target_user: str = ""
    bot_has_floor: bool = False
    two_person_thread: bool = False
    fast_burst: bool = False
    topic_stability: float = 0.5
    interruption_cost: float = 0.5
    observed_message_count: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "active_speakers": list(self.active_speakers),
            "likely_target_user": self.likely_target_user,
            "bot_has_floor": self.bot_has_floor,
            "two_person_thread": self.two_person_thread,
            "fast_burst": self.fast_burst,
            "topic_stability": round(self.topic_stability, 3),
            "interruption_cost": round(self.interruption_cost, 3),
            "reasons": list(self.reasons),
        }


@dataclass
class ActionPlan:
    """一次候选回复的结构化行为计划。"""

    action: ActionType
    target_message_id: str
    target_user_id: str
    confidence: float
    interruption_cost: float
    reason: str
    tone: str
    max_chars: int
    wait_multiplier: float
    directed: bool
    is_question: bool
    target_timestamp: datetime
    topic_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "target_message_id": self.target_message_id,
            "target_user_id": self.target_user_id,
            "confidence": round(self.confidence, 3),
            "interruption_cost": round(self.interruption_cost, 3),
            "reason": self.reason,
            "tone": self.tone,
            "max_chars": self.max_chars,
            "wait_multiplier": self.wait_multiplier,
            "directed": self.directed,
            "is_question": self.is_question,
        }


class ConversationFloorManager:
    """根据短期消息拓扑计算发言权，并在发送前复核计划。"""

    def __init__(
        self,
        active_window_seconds: float = 45.0,
        burst_window_seconds: float = 12.0,
        burst_message_threshold: int = 4,
        topic_shift_threshold: float = 0.12,
    ):
        self.active_window_seconds = active_window_seconds
        self.burst_window_seconds = burst_window_seconds
        self.burst_message_threshold = burst_message_threshold
        self.topic_shift_threshold = topic_shift_threshold

    def analyze(
        self,
        current_message: Any,
        recent_messages: list[Any],
        *,
        bot_id: str = "",
        is_private: bool = False,
        directed_to_bot: bool = False,
        continuing: bool = False,
        mentioned_others: list[str] | None = None,
        topic_relevance: float = 0.5,
        is_question: bool = False,
        rich_message_only: bool = False,
        rich_type: str = "",
    ) -> tuple[ConversationFloor, ActionPlan]:
        """分析当前消息，并返回发言权快照与动作计划。"""
        now = current_message.timestamp
        mentioned_others = mentioned_others or []

        live = [
            m for m in recent_messages
            if not m.is_bot
            and 0 <= (now - m.timestamp).total_seconds() <= self.active_window_seconds
        ]
        active_speakers = self._ordered_unique(m.sender_id for m in live if m.sender_id)
        transitions = sum(
            1 for left, right in zip(live, live[1:])
            if left.sender_id and right.sender_id and left.sender_id != right.sender_id
        )
        two_person_thread = (
            len(live) >= 4
            and len(active_speakers) == 2
            and transitions >= 2
        )

        burst_messages = [
            m for m in recent_messages
            if 0 <= (now - m.timestamp).total_seconds() <= self.burst_window_seconds
        ]
        fast_burst = len(burst_messages) >= self.burst_message_threshold

        previous_text = "".join(m.content for m in live[-4:-1] if m.content)
        current_tokens = self._topic_tokens(current_message.content)
        previous_tokens = self._topic_tokens(previous_text)
        topic_stability = self._overlap(current_tokens, previous_tokens)
        if not previous_tokens:
            topic_stability = 0.5

        replied_user = str(current_message.reply_to_qq or "")
        talking_to_other = bool(mentioned_others) or (
            bool(replied_user)
            and replied_user != str(bot_id or "")
            and not directed_to_bot
        )
        bot_has_floor = is_private or directed_to_bot or continuing

        likely_target = replied_user
        if not likely_target and two_person_thread:
            for msg in reversed(live[:-1]):
                if msg.sender_id != current_message.sender_id:
                    likely_target = msg.sender_id
                    break

        reasons = []
        if bot_has_floor:
            interruption_cost = 0.0
            reasons.append("消息明确对bot或延续对话")
        elif talking_to_other:
            interruption_cost = 0.95
            reasons.append("消息明确回复或提到其他群友")
        else:
            interruption_cost = 0.28
            if two_person_thread:
                interruption_cost += 0.42
                reasons.append("两人连续对聊")
            if fast_burst:
                interruption_cost += 0.16
                reasons.append("群消息正在爆发")
            if previous_tokens and topic_stability < self.topic_shift_threshold:
                interruption_cost += 0.08
                reasons.append("话题正在切换")
            if topic_relevance >= 0.75:
                interruption_cost -= 0.12
                reasons.append("话题与人格兴趣高度相关")
            interruption_cost = max(0.0, min(1.0, interruption_cost))

        floor = ConversationFloor(
            active_speakers=tuple(active_speakers),
            likely_target_user=likely_target,
            bot_has_floor=bot_has_floor,
            two_person_thread=two_person_thread,
            fast_burst=fast_burst,
            topic_stability=topic_stability,
            interruption_cost=interruption_cost,
            observed_message_count=len(recent_messages),
            reasons=reasons,
        )
        plan = self._build_plan(
            current_message=current_message,
            floor=floor,
            is_private=is_private,
            continuing=continuing,
            talking_to_other=talking_to_other,
            topic_relevance=topic_relevance,
            is_question=is_question,
            topic_tokens=current_tokens,
            rich_message_only=rich_message_only,
            rich_type=rich_type,
        )
        return floor, plan

    def should_cancel(
        self,
        plan: ActionPlan,
        recent_messages: list[Any],
        *,
        bot_id: str = "",
    ) -> tuple[bool, str]:
        """思考后/发送前复核：群聊已经向前发展时放弃过期插话。"""
        if plan.directed:
            return False, "明确对bot的消息保留回复"

        newer = [
            m for m in recent_messages
            if not m.is_bot and m.timestamp > plan.target_timestamp
        ]
        if not newer:
            return False, "没有更新的群消息"

        if plan.action == ActionType.REACT:
            return True, "简短反应已错过时机"

        if plan.target_message_id and any(
            str(m.reply_to_id or "") == str(plan.target_message_id)
            and str(m.sender_id) != str(bot_id or "")
            for m in newer
        ):
            return True, "已有群友回复目标消息"

        if plan.is_question and self._looks_like_answer(newer[0].content):
            answer_tokens = self._topic_tokens(newer[0].content)
            if self._overlap(set(plan.topic_tokens), answer_tokens) >= 0.08:
                return True, "群友已经先回答问题"

        if len(newer) >= 3:
            latest_tokens = self._topic_tokens("".join(m.content for m in newer[-3:]))
            if self._overlap(set(plan.topic_tokens), latest_tokens) < self.topic_shift_threshold:
                return True, "群聊已经切换话题"

        if len(newer) >= self.burst_message_threshold + 1:
            return True, "消息爆发，插话窗口已关闭"

        return False, "回复仍然适合当前群聊"

    def _build_plan(
        self,
        *,
        current_message: Any,
        floor: ConversationFloor,
        is_private: bool,
        continuing: bool,
        talking_to_other: bool,
        topic_relevance: float,
        is_question: bool,
        topic_tokens: set[str],
        rich_message_only: bool,
        rich_type: str,
    ) -> ActionPlan:
        expressive = self._is_expressive(current_message.content)

        if floor.bot_has_floor:
            if is_question:
                action = ActionType.ANSWER
                tone, max_chars = "直接、自然地回答", 30
            elif continuing:
                action = ActionType.FOLLOW_UP
                tone, max_chars = "像正在对聊一样自然延续", 28
            else:
                action = ActionType.REPLY
                tone, max_chars = "自然回应，不要客服腔", 30
            confidence = 0.98 if is_private else 0.94
            reason = "bot拥有明确发言权"
            wait_multiplier = 0.85
        elif talking_to_other or floor.interruption_cost >= 0.82:
            action = ActionType.SILENT
            tone, max_chars = "保持旁观", 0
            confidence = 0.94
            reason = "插话会打断正在进行的交流"
            wait_multiplier = 1.0
        elif floor.two_person_thread and floor.interruption_cost >= 0.65:
            action = ActionType.SILENT
            tone, max_chars = "保持旁观", 0
            confidence = 0.88
            reason = "两位群友正在连续对聊"
            wait_multiplier = 1.0
        elif rich_message_only and rich_type in ("image", "mface", "face", "video"):
            action = ActionType.REACT
            tone, max_chars = "只在确实有自然反应时回几个字；不知道内容就沉默", 8
            confidence = 0.38
            reason = "纯媒体消息只适合偶尔短反应"
            wait_multiplier = 0.8
        elif rich_message_only:
            action = ActionType.SILENT
            tone, max_chars = "像普通群友一样略过无人提问的分享", 0
            confidence = 0.9
            reason = "无人提问的链接、卡片或转发不主动点评"
            wait_multiplier = 1.0
        elif expressive and len(current_message.content.strip()) <= 14:
            action = ActionType.REACT
            tone, max_chars = "只做很短的群友式反应", 10
            confidence = 0.62
            reason = "适合短反应而不是完整回答"
            wait_multiplier = 0.75
        elif is_question and topic_relevance >= 0.65:
            action = ActionType.ANSWER
            tone, max_chars = "简短提供有用答案，不抢主导权", 30
            confidence = 0.68
            reason = "面向群里的问题且话题相关"
            wait_multiplier = 1.05
        else:
            action = ActionType.REPLY
            tone, max_chars = "像普通群友一样随意接一句", 26
            confidence = max(0.35, 1.0 - floor.interruption_cost)
            reason = "存在自然接话机会，由概率系统最终决定"
            wait_multiplier = 1.15 if floor.fast_burst else 1.0

        return ActionPlan(
            action=action,
            target_message_id=str(current_message.message_id or ""),
            target_user_id=str(current_message.sender_id or ""),
            confidence=confidence,
            interruption_cost=floor.interruption_cost,
            reason=reason,
            tone=tone,
            max_chars=max_chars,
            wait_multiplier=wait_multiplier,
            directed=floor.bot_has_floor,
            is_question=is_question,
            target_timestamp=current_message.timestamp,
            topic_tokens=tuple(sorted(topic_tokens)),
        )

    @staticmethod
    def _ordered_unique(items) -> list[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @staticmethod
    def _topic_tokens(text: str) -> set[str]:
        tokens = set()
        for chunk in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", (text or "").lower()):
            if re.fullmatch(r"[a-z0-9_]+", chunk):
                if len(chunk) >= 2:
                    tokens.add(chunk)
                continue
            if len(chunk) == 1:
                tokens.add(chunk)
            else:
                tokens.update(chunk[i:i + 2] for i in range(len(chunk) - 1))
        return tokens

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _is_expressive(text: str) -> bool:
        text = (text or "").strip().lower()
        markers = ("哈哈", "笑死", "绝了", "确实", "草", "啊这", "离谱", "牛逼", "牛啊")
        return (
            any(marker in text for marker in markers)
            or bool(re.fullmatch(r"[6６]+[!！~]*", text))
            or text.endswith(("！", "!", "~"))
        )

    @staticmethod
    def _looks_like_answer(text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        answer_markers = (
            "因为", "应该", "可以", "就是", "试试", "需要", "建议",
            "选", "大概", "可能", "直接", "先",
        )
        return len(text) >= 4 and (
            any(marker in text for marker in answer_markers)
            or bool(re.match(r"^(是|不是|对|不对|能|不能|要|不要)", text))
        )

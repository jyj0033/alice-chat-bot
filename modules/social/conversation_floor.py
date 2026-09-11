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
    same_sender_continuation: bool = False
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
            "same_sender_continuation": self.same_sender_continuation,
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
        settle_window_seconds: float = 0.7,
        settle_max_seconds: float = 2.4,
        other_target_context_seconds: float = 900.0,
    ):
        self.active_window_seconds = active_window_seconds
        self.burst_window_seconds = burst_window_seconds
        self.burst_message_threshold = burst_message_threshold
        self.topic_shift_threshold = topic_shift_threshold
        # 普通插话在生成前需要一个很短的收尾窗口：窗口内有新消息就继续等，
        # 但设置上限，避免 bot 因为热闹的群聊永久等不到“最后一条”。
        self.settle_window_seconds = max(0.2, float(settle_window_seconds))
        self.settle_max_seconds = max(
            self.settle_window_seconds,
            float(settle_max_seconds),
        )
        # 仅用于判断“你确定/你觉得”这类无显式 @ 的话是不是在问上一位
        # 群友；窗口过大容易把旧话题误当成当前对话对象，所以默认 15 分钟。
        self.other_target_context_seconds = max(
            0.0, float(other_target_context_seconds)
        )

    @staticmethod
    def _previous_message(current_message: Any, recent_messages: list[Any]):
        """找当前消息之前的最后一条消息，保留 bot 消息作为对话边界。"""
        current_id = str(getattr(current_message, "message_id", "") or "")
        current_time = current_message.timestamp
        previous = None
        for message in recent_messages:
            if message is current_message:
                continue
            message_id = str(getattr(message, "message_id", "") or "")
            if current_id and message_id and message_id == current_id:
                continue
            if message.timestamp >= current_time:
                continue
            if previous is None or message.timestamp > previous.timestamp:
                previous = message
        return previous

    @staticmethod
    def _looks_like_other_user_question(text: str) -> bool:
        """识别明显的二人称询问，避免把所有带问号的群消息都当成公开提问。"""
        text = (text or "").strip()
        if "你" not in text:
            return False
        question_markers = (
            "确定", "觉得", "知道", "用的是", "是不是", "能不能", "怎么",
            "为什么", "什么", "啥", "吗", "呢", "？", "?",
        )
        return any(marker in text for marker in question_markers)

    def _is_same_sender_continuation(
        self, current_message: Any, previous_message: Any
    ) -> bool:
        """当前消息是否是同一用户在短时间内拆开的下一段话。"""
        if previous_message is None or previous_message.is_bot:
            return False
        if not current_message.sender_id or (
            str(previous_message.sender_id) != str(current_message.sender_id)
        ):
            return False
        elapsed = (current_message.timestamp - previous_message.timestamp).total_seconds()
        return 0 <= elapsed <= self.burst_window_seconds

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
        ignore_other_target_signal: bool = False,
        dynamic_target_user_id: str = "",
        allow_dynamic_interjection: bool = False,
        topic_relevance: float = 0.5,
        is_question: bool = False,
        rich_message_only: bool = False,
        rich_type: str = "",
    ) -> tuple[ConversationFloor, ActionPlan]:
        """分析当前消息，并返回发言权快照与动作计划。"""
        now = current_message.timestamp
        mentioned_others = mentioned_others or []
        previous_message = self._previous_message(current_message, recent_messages)
        same_sender_continuation = self._is_same_sender_continuation(
            current_message, previous_message
        )

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
        inferred_other_target = (
            not is_private
            and not directed_to_bot
            and not continuing
            and previous_message is not None
            and not previous_message.is_bot
            and str(previous_message.sender_id) != str(current_message.sender_id)
            and self._looks_like_other_user_question(current_message.content)
            and 0 <= (now - previous_message.timestamp).total_seconds()
            <= self.other_target_context_seconds
        )
        talking_to_other = (
            not ignore_other_target_signal
            and (
                bool(mentioned_others)
                or (
                    bool(replied_user)
                    and replied_user != str(bot_id or "")
                    and not directed_to_bot
                )
                or inferred_other_target
            )
        )
        bot_has_floor = is_private or directed_to_bot or continuing

        likely_target = str(dynamic_target_user_id or replied_user or "")
        if inferred_other_target:
            likely_target = str(previous_message.sender_id or "")
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
            reasons.append(
                "消息明确回复或提到其他群友"
                if not inferred_other_target
                else "二人称问法更像是在问上一位群友"
            )
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
            same_sender_continuation=same_sender_continuation,
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
            same_sender_continuation=same_sender_continuation,
            topic_tokens=current_tokens,
            rich_message_only=rich_message_only,
            rich_type=rich_type,
            allow_dynamic_interjection=allow_dynamic_interjection,
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
            # 简短反应：目标发送者自己已经续了话（如发图后又补一句文字），
            # 反应就过时了——再回图只会接错话茬。其他群友零星聊天则仍可补一句。
            if any(
                not m.is_bot
                and str(m.sender_id) == str(plan.target_user_id or "")
                for m in newer
            ):
                return True, "目标发送者已续话，简短反应过期"
            recent_active = [
                m for m in newer
                if (datetime.now() - m.timestamp).total_seconds() <= self.burst_window_seconds
            ]
            if len(recent_active) >= 2:
                return True, "简短反应已错过时机"
            return False, "后续聊天不密集，仍可补一句"

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
        same_sender_continuation: bool,
        topic_tokens: set[str],
        rich_message_only: bool,
        rich_type: str,
        allow_dynamic_interjection: bool,
    ) -> ActionPlan:
        expressive = self._is_expressive(current_message.content)

        if floor.bot_has_floor:
            if is_question:
                action = ActionType.ANSWER
                tone, max_chars = "直接、自然地回答", 20
            elif continuing:
                action = ActionType.FOLLOW_UP
                tone, max_chars = "像正在对聊一样自然延续", 18
            else:
                action = ActionType.REPLY
                tone, max_chars = "自然回应，不要客服腔", 20
            confidence = 0.98 if is_private else 0.94
            reason = "bot拥有明确发言权"
            wait_multiplier = 0.85
        elif talking_to_other:
            action = ActionType.SILENT
            tone, max_chars = "保持旁观", 0
            confidence = 0.94
            reason = (
                "二人称问法更像是在问上一位群友"
                if "二人称问法更像是在问上一位群友" in floor.reasons
                else "插话会打断正在进行的交流"
            )
            wait_multiplier = 1.0
        elif same_sender_continuation and is_question and not allow_dynamic_interjection:
            # 同一用户短时间拆成多条发送时，最后一条问句通常仍是上一条
            # 的补充。先等这一轮表达结束，避免把问给群友的问题抢答。
            action = ActionType.SILENT
            tone, max_chars = "等待对方把话说完", 0
            confidence = 0.9
            reason = "同一群友连续补充，等待完整表达"
            wait_multiplier = 1.0
        elif floor.two_person_thread and floor.interruption_cost >= 0.65:
            if topic_relevance >= 0.7 and not floor.fast_burst:
                # 两人连续对聊但话题与人格高度相关且非消息爆发：允许偶尔自然
                # 加入讨论，不硬性沉默。真正开不开口由决策概率 + 发送复核把关。
                action = ActionType.REPLY
                tone, max_chars = "像加入讨论一样自然地接一句，不抢主导权", 18
                confidence = 0.45
                reason = "两人连续对聊，话题相关可自然融入"
                wait_multiplier = 1.2
            else:
                # 对聊或消息爆发时不抢完整话题，但保留极低概率的附和机会。
                # 最终是否开口仍由插话成本、群聊热度和随机决策共同压低。
                action = ActionType.REACT
                tone, max_chars = "只在确实适合时短短附和，不展开", 10
                confidence = 0.24
                reason = "两人连续对聊，偶尔附和"
                wait_multiplier = 1.25
        elif floor.interruption_cost >= 0.82:
            # 多人热聊/消息爆发也不是绝对禁言：只允许短反应，避免 Bot
            # 抢走对话主导权；显式回复他人的消息已在上面单独静默。
            action = ActionType.REACT
            tone, max_chars = "只在确实适合时短短附和，不展开", 10
            confidence = 0.2
            reason = "多人聊天节奏较快，偶尔短反应"
            wait_multiplier = 1.3
        elif rich_message_only and rich_type in ("image", "mface", "face", "video"):
            action = ActionType.REACT
            tone, max_chars = "只在确实有自然反应时回一句很短的；不知道内容就沉默", 14
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
            tone, max_chars = "简短提供有用答案，不抢主导权", 20
            confidence = 0.68
            reason = "面向群里的问题且话题相关"
            wait_multiplier = 1.05
        else:
            action = ActionType.REPLY
            tone, max_chars = "像普通群友一样随意接一句", 18
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

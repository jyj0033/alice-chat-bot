"""
发言决策器 - 增强版
集成注意力、情绪、疲劳等系统的综合发言决策
"""
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .awareness import SocialContext, SocialAwarenessManager
from .attention import AttentionManager, AttentionKeywordsDetector
from .conversation_floor import ActionType
from .fatigue import FatigueManager

logger = logging.getLogger(__name__)


@dataclass
class SpeakingDecision:
    """发言决策结果"""
    should_speak: bool
    probability: float
    reason: str
    modifiers: dict


class EnhancedSpeakingDecider:
    """
    增强版发言决策器 - AstrBot风格

    综合考虑：
    1. 基础发言概率（初始2%，有人回复后80%）
    2. 注意力状态
    3. 情绪状态
    4. 疲劳状态
    5. 触发关键词
    6. 冷却状态
    """

    def __init__(
        self,
        # 基础概率
        base_probability: float = 0.02,
        after_reply_probability: float = 0.8,
        probability_duration: float = 120,
        # 人格
        extraversion: float = 0.5,
        neuroticism: float = 0.3,
        # 注意力
        attention_manager: AttentionManager = None,
        attention_keywords_detector: AttentionKeywordsDetector = None,
        # 疲劳
        fatigue_manager: FatigueManager = None,
        # 触发词
        trigger_keywords: list = None,
        command_prefixes: list = None,
    ):
        self.base_probability = base_probability
        self.after_reply_probability = after_reply_probability
        self.probability_duration = probability_duration

        self.extraversion = extraversion
        self.neuroticism = neuroticism

        # 子系统
        self.attention_manager = attention_manager or AttentionManager()
        self.attention_keywords_detector = attention_keywords_detector or AttentionKeywordsDetector()
        self.fatigue_manager = fatigue_manager or FatigueManager()

        # 触发词
        self.trigger_keywords = trigger_keywords or []
        self.command_prefixes = command_prefixes or ["/", "!", "#"]

        # 追踪回复状态
        self._recent_replies: dict = {}  # session_id -> last_reply_time

    def should_speak(
        self,
        context: SocialContext,
        emotional_bonus: float = 0.0
    ) -> Tuple[bool, str, float]:
        """
        决定是否发言

        Args:
            context: 社交上下文
            emotional_bonus: 情感加成

        Returns:
            Tuple[bool, str, float]: (是否发言, 原因, 最终概率)
        """
        decision = self.decide(context, emotional_bonus)
        return decision.should_speak, decision.reason, decision.probability

    def decide(
        self,
        context: SocialContext,
        emotional_bonus: float = 0.0
    ) -> SpeakingDecision:
        """
        完整决策流程

        Returns:
            SpeakingDecision: 决策结果
        """
        session_id = context.session_id
        modifiers = {}

        # === 1. 检查冷却（@ / 引用 / 当前正延续对话时跳过）===
        # 冷却用于限制“低概率随机插话”刷屏；明确对我说或 bot 刚回复过对方、
        # 对方正自然延续对话时，不应被冷却挡住，否则会错过真人之间 30-60s 的接话。
        if self.fatigue_manager.is_in_cooldown(session_id) and not (
            context.mentioned_me
            or context.reply_to_me
            or self.is_conversation_with(session_id, context.sender_id)
        ):
            remaining = self.fatigue_manager.get_cooldown_remaining(session_id)
            return SpeakingDecision(
                should_speak=False,
                probability=0.0,
                reason=f"冷却中（{remaining:.0f}秒）",
                modifiers={}
            )

        # === 2. 检查命令 ===
        decision_text = context.extra.get("outer_text", context.message_content)
        if self._is_command(decision_text):
            return SpeakingDecision(
                should_speak=False,
                probability=0.0,
                reason="命令消息，跳过",
                modifiers={}
            )

        # 私聊中的每条普通消息天然都是对 bot 说的，不走群聊随机插话概率。
        if context.extra.get("is_private", False):
            return SpeakingDecision(
                should_speak=True,
                probability=0.98,
                reason="私聊消息",
                modifiers={"private": True},
            )

        action_plan = context.extra.get("action_plan")
        if (
            context.extra.get("rich_message_only", False)
            and context.extra.get("rich_type") not in ("image", "mface", "face", "video")
            and not context.mentioned_me
            and not context.reply_to_me
            and not context.is_emergency
        ):
            return SpeakingDecision(
                should_speak=False,
                probability=0.0,
                reason="无人提问的链接、卡片或转发不主动点评",
                modifiers={"rich_message_only": True},
            )
        if (
            action_plan
            and action_plan.action == ActionType.SILENT
            and not context.mentioned_me
            and not context.reply_to_me
            and not context.is_emergency
        ):
            return SpeakingDecision(
                should_speak=False,
                probability=0.0,
                reason=action_plan.reason,
                modifiers={
                    "action": action_plan.action.value,
                    "interruption_cost": action_plan.interruption_cost,
                },
            )

        # === 3. 强制触发检查 ===
        trigger_result = context.extra.get("trigger", {})
        if trigger_result.get("forced_trigger", False):
            reasons = trigger_result.get("reasons", ["强制触发"])
            return SpeakingDecision(
                should_speak=True,
                probability=0.95,
                reason=" + ".join(reasons),
                modifiers={"forced": True}
            )

        # === 3.5 明确对我说（引用bot / 名字+提问）→ 高概率必答 ===
        # @ 已被上面的 forced_trigger（0.95）接管；这里覆盖"引用bot回复"和"叫名字提问"，
        # 它们不像 @ 那么"刺耳"，但仍是明确对 bot 说的，应该答。
        reasons = trigger_result.get("reasons", [])
        has_nickname = any(r.startswith("昵称") for r in reasons)
        has_question = "直接提问" in reasons
        if context.reply_to_me or (has_nickname and has_question):
            return SpeakingDecision(
                should_speak=True,
                probability=0.9,
                reason=" + ".join(reasons or ["对我说"]),
                modifiers={"directed": True}
            )

        # === 4. 计算发言概率 ===
        probability = self._calculate_probability(context, emotional_bonus, modifiers)

        # === 5. 添加随机波动 ===
        noise = random.uniform(-0.1, 0.1)
        final_probability = max(0.0, min(1.0, probability + noise))

        # === 6. 决策 ===
        should_speak = random.random() < final_probability

        # === 7. 生成原因 ===
        reason = self._explain_decision(context, probability, modifiers)

        return SpeakingDecision(
            should_speak=should_speak,
            probability=final_probability,
            reason=reason,
            modifiers=modifiers
        )

    def _calculate_probability(
        self,
        context: SocialContext,
        emotional_bonus: float,
        modifiers: dict
    ) -> float:
        """计算发言概率"""
        session_id = context.session_id
        configured_group_probability = context.extra.get("group_base_probability")
        try:
            base_prob = (
                float(configured_group_probability)
                if configured_group_probability is not None
                else self.base_probability
            )
        except (TypeError, ValueError):
            base_prob = self.base_probability
        base_prob = max(0.0, min(1.0, base_prob))

        # === A. 回复后概率提升 ===
        # 仅当"消息明确对bot说"或"是同一个人在延续和bot的对话"时才给加成，
        # 否则普通群友互聊也给 75% 加成会导致 bot 刷屏。
        trigger = context.extra.get("trigger", {})
        reasons = trigger.get("reasons", [])
        is_directed = (
            context.mentioned_me
            or context.reply_to_me
            or any(r.startswith(("昵称", "关键词", "被@", "被回复")) for r in reasons)
        )
        after_reply_bonus = self._get_after_reply_bonus(session_id, context.sender_id, is_directed)
        if after_reply_bonus > 0:
            modifiers["回复提升"] = after_reply_bonus
            base_prob = after_reply_bonus

        # === B. 触发优先级 ===
        trigger = context.extra.get("trigger", {})
        priority = trigger.get("priority", 0.0)
        if priority > 0:
            modifiers["触发优先级"] = priority * 0.4
            base_prob = max(base_prob, priority * 0.4)

        # === C. 注意力影响 ===
        attention = self.attention_manager.get_effective_attention(
            context.group_id, context.sender_id
        )
        if attention > 0.5:
            modifiers["注意力"] = (attention - 0.5) * 0.3

        # === D. 注意力关键词 ===
        if self.attention_manager.enabled:
            attention_change, attention_reason = self.attention_keywords_detector.detect_attention_keywords(
                context.extra.get("outer_text", context.message_content), context.sender_id
            )
        else:
            attention_change, attention_reason = 0.0, ""
        if attention_change != 0:
            modifiers["关键词"] = attention_change

        # === E. 情感加成 ===
        modifiers["情绪"] = emotional_bonus

        # === E.5 话题兴趣与群聊节奏 ===
        # relevance 以 0.5 为中点：感兴趣的话题更愿意接话，厌烦话题更愿意潜水。
        topic_bonus = (context.topic_relevance - 0.5) * 0.3
        if abs(topic_bonus) > 1e-6:
            modifiers["话题兴趣"] = topic_bonus

        # 群越热闹越谨慎插嘴；冷清时只给很小的加成，避免为了暖场而刷屏。
        if context.group_activity > 0.8:
            modifiers["群聊节奏"] = -0.08
        elif context.group_activity < 0.15:
            modifiers["群聊节奏"] = 0.03

        # === E.6 发言权成本 ===
        action_plan = context.extra.get("action_plan")
        if action_plan and not action_plan.directed:
            modifiers["插话成本"] = -action_plan.interruption_cost * 0.28
            if action_plan.action == ActionType.ANSWER:
                modifiers["群问题可回答"] = 0.08
            elif action_plan.action == ActionType.REACT:
                modifiers["仅适合短反应"] = -0.02

        # === F. 疲劳惩罚 ===
        fatigue_penalty = self.fatigue_manager.get_probability_penalty(session_id)
        modifiers["疲劳"] = fatigue_penalty

        # === G. 人格外向性 ===
        extraversion_bonus = (self.extraversion - 0.5) * 0.2
        modifiers["外向性格"] = extraversion_bonus

        # === H. 神经质惩罚 ===
        neuroticism_penalty = (self.neuroticism - 0.3) * 0.1
        modifiers["情绪稳定"] = -neuroticism_penalty

        # === 综合计算 ===
        total = base_prob
        for name, value in modifiers.items():
            if isinstance(value, (int, float)):
                total += value

        # 标准 logistic 映射（单调递增：总修正越高，概率越高）
        # total=0.05 → ~3%，total=0.5 → 50%，total=0.9 → ~96%
        # 原公式 1/(1+(total-0.5)*6) 是错的：total>0.5 时概率反而骤降，且可为负
        probability = 1 / (1 + math.exp(-(total - 0.5) * 8))
        probability = max(0.01, min(0.99, probability))

        return probability

    def _get_after_reply_bonus(
        self,
        session_id: str,
        sender_id: str = "",
        is_directed: bool = False
    ) -> float:
        """获取回复后概率提升

        仅当消息明确对bot说（@/回复/昵称/关键词）或来自"bot刚回复过的那位用户"
        时生效，避免 bot 回复一次后对整个群的每条消息都高概率接话。
        """
        entry = self._recent_replies.get(session_id)
        if not entry:
            return 0.0

        elapsed = time.time() - entry["time"]
        if elapsed >= self.probability_duration:
            return 0.0

        if not (is_directed or (sender_id and sender_id == entry.get("user_id", ""))):
            return 0.0

        # 线性衰减
        factor = 1 - (elapsed / self.probability_duration)
        return self.after_reply_probability * factor

    def _record_reply(self, session_id: str, user_id: str = "") -> None:
        """记录Bot回复（含被回复的用户，用于区分"对bot说"还是普通群聊）"""
        self._recent_replies[session_id] = {"time": time.time(), "user_id": user_id}

    def is_conversation_with(self, session_id: str, sender_id: str) -> bool:
        """判断是否在与该用户延续对话（bot 刚回复过 TA，且在窗口期内）

        用于：同一人没 @、没引用 bot 就接着对 bot 说话时，
        应视为"对我说"（真人对聊不需要每句都点名）。
        """
        if not sender_id:
            return False
        entry = self._recent_replies.get(session_id)
        if not entry:
            return False
        if entry.get("user_id", "") != sender_id:
            return False
        return time.time() - entry["time"] < self.probability_duration

    def _is_command(self, message: str) -> bool:
        """检查是否命令"""
        message = message.strip()
        for prefix in self.command_prefixes:
            if message.startswith(prefix):
                return True
        return False

    def _explain_decision(
        self,
        context: SocialContext,
        probability: float,
        modifiers: dict
    ) -> str:
        """解释决策原因"""
        reasons = []

        # 主要原因
        trigger = context.extra.get("trigger", {})
        if trigger.get("priority", 0) > 0.3:
            reasons.extend(trigger.get("reasons", []))

        # 关键词
        attention_reason = ""
        if self.attention_manager.enabled:
            attention_reason = self.attention_keywords_detector.detect_attention_keywords(
                context.extra.get("outer_text", context.message_content), context.sender_id
            )[1]
        if attention_reason:
            reasons.append(attention_reason)

        # 回复后状态
        trigger_reasons = context.extra.get("trigger", {}).get("reasons", [])
        is_directed = (
            context.mentioned_me
            or context.reply_to_me
            or any(r.startswith(("昵称", "关键词", "被@", "被回复")) for r in trigger_reasons)
        )
        if self._get_after_reply_bonus(context.session_id, context.sender_id, is_directed) > 0.3:
            reasons.append("回复后窗口期")

        if not reasons:
            action_plan = context.extra.get("action_plan")
            if action_plan:
                reasons.append(action_plan.reason)
            else:
                reasons.append("概率触发")

        return " + ".join(reasons)

    # === 状态更新方法 ===

    def on_message(
        self,
        session_id: str,
        group_id: str,
        user_id: str,
        mentioned_bot: bool = False,
        is_reply_to_bot: bool = False,
        is_directed_to_bot: bool = True,
    ) -> None:
        """消息到达时的状态更新"""
        # 疲劳只跟踪 bot 实际参与的对话。无关群聊只影响注意力，不累计轮次。
        if is_directed_to_bot:
            self.fatigue_manager.on_message(session_id, is_bot_message=False)

        # 更新注意力
        self.attention_manager.on_message_received(
            group_id, user_id, mentioned_bot, is_reply_to_bot
        )

    def on_bot_reply(
        self,
        session_id: str,
        group_id: str,
        probability: float = 0.0,
        user_id: str = "",
    ) -> None:
        """Bot回复后的状态更新"""
        # 只有平台实际发送成功后，才把会话标记为已回复。
        self._record_reply(session_id, user_id)

        # 更新疲劳
        self.fatigue_manager.on_message(session_id, is_bot_message=True)

        # 更新注意力
        self.attention_manager.on_bot_reply(group_id)

        # 启动冷却：用本次真实发言概率。
        # 被@/引用回复等强制触发时概率高(>=阈值)，不进入冷却，对话可自然延续；
        # 只有低概率随机插话才冷却，避免连续刷屏。
        self.fatigue_manager.start_cooldown(session_id, probability)

    def cleanup(self) -> None:
        """清理过期状态"""
        self.fatigue_manager.cleanup()

        # 清理过期回复记录
        current_time = time.time()
        to_remove = [
            sid for sid, entry in self._recent_replies.items()
            if current_time - entry["time"] > self.probability_duration * 2
        ]
        for sid in to_remove:
            del self._recent_replies[sid]

"""
发言决策器 - 增强版
集成注意力、情绪、疲劳等系统的综合发言决策
"""
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .awareness import SocialContext, SocialAwarenessManager
from .attention import AttentionManager, AttentionKeywordsDetector
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

        # === 1. 检查冷却（@ 机器人时跳过）===
        if self.fatigue_manager.is_in_cooldown(session_id) and not context.mentioned_me:
            remaining = self.fatigue_manager.get_cooldown_remaining(session_id)
            return SpeakingDecision(
                should_speak=False,
                probability=0.0,
                reason=f"冷却中（{remaining:.0f}秒）",
                modifiers={}
            )

        # === 2. 检查命令 ===
        if self._is_command(context.message_content):
            return SpeakingDecision(
                should_speak=False,
                probability=0.0,
                reason="命令消息，跳过",
                modifiers={}
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

        # === 4. 计算发言概率 ===
        probability = self._calculate_probability(context, emotional_bonus, modifiers)

        # === 5. 添加随机波动 ===
        noise = random.uniform(-0.1, 0.1)
        final_probability = max(0.0, min(1.0, probability + noise))

        # === 6. 决策 ===
        should_speak = random.random() < final_probability

        # === 7. 记录状态更新 ===
        if should_speak:
            self._record_reply(session_id)

        # === 8. 生成原因 ===
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
        base_prob = self.base_probability

        # === A. 回复后概率提升 ===
        after_reply_bonus = self._get_after_reply_bonus(session_id)
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
        attention_change, attention_reason = self.attention_keywords_detector.detect_attention_keywords(
            context.message_content, context.sender_id
        )
        if attention_change != 0:
            modifiers["关键词"] = attention_change

        # === E. 情感加成 ===
        modifiers["情绪"] = emotional_bonus

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

        # Sigmoid 归一化到 0-1
        probability = 1 / (1 + 1e-10 + (total - 0.5) * 6)
        probability = max(0.01, min(0.99, probability))

        return probability

    def _get_after_reply_bonus(self, session_id: str) -> float:
        """获取回复后概率提升"""
        if session_id not in self._recent_replies:
            return 0.0

        last_reply = self._recent_replies[session_id]
        elapsed = time.time() - last_reply

        if elapsed < self.probability_duration:
            # 线性衰减
            factor = 1 - (elapsed / self.probability_duration)
            return self.after_reply_probability * factor

        return 0.0

    def _record_reply(self, session_id: str) -> None:
        """记录Bot回复"""
        self._recent_replies[session_id] = time.time()

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
        attention_reason = self.attention_keywords_detector.detect_attention_keywords(
            context.message_content, context.sender_id
        )[1]
        if attention_reason:
            reasons.append(attention_reason)

        # 回复后状态
        if self._get_after_reply_bonus(context.session_id) > 0.3:
            reasons.append("回复后窗口期")

        if not reasons:
            reasons.append("概率触发")

        return " + ".join(reasons)

    # === 状态更新方法 ===

    def on_message(
        self,
        session_id: str,
        group_id: str,
        user_id: str,
        mentioned_bot: bool = False,
        is_reply_to_bot: bool = False
    ) -> None:
        """消息到达时的状态更新"""
        # 更新疲劳
        self.fatigue_manager.on_message(session_id, is_bot_message=False)

        # 更新注意力
        self.attention_manager.on_message_received(
            group_id, user_id, mentioned_bot, is_reply_to_bot
        )

    def on_bot_reply(self, session_id: str, group_id: str) -> None:
        """Bot回复后的状态更新"""
        # 更新疲劳
        self.fatigue_manager.on_message(session_id, is_bot_message=True)

        # 更新注意力
        self.attention_manager.on_bot_reply(group_id)

        # 启动冷却
        probability = self.base_probability  # 使用基础概率估算
        self.fatigue_manager.start_cooldown(session_id, probability)

    def cleanup(self) -> None:
        """清理过期状态"""
        self.fatigue_manager.cleanup()

        # 清理过期回复记录
        current_time = time.time()
        to_remove = [
            sid for sid, last_time in self._recent_replies.items()
            if current_time - last_time > self.probability_duration * 2
        ]
        for sid in to_remove:
            del self._recent_replies[sid]

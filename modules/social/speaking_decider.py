"""
发言决策器
综合各种因素决定是否发言
"""
import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

from .awareness import SocialContext, SocialAwarenessManager

logger = logging.getLogger(__name__)


class SpeakingDecider:
    """发言决策器"""

    def __init__(
        self,
        personality: dict = None,
        extraversion: float = 0.5,
        neuroticism: float = 0.3
    ):
        """
        初始化

        Args:
            personality: 人格参数字典
            extraversion: 外向性 0-1
            neuroticism: 神经质 0-1
        """
        self.personality = personality or {}
        self.extraversion = extraversion
        self.neuroticism = neuroticism

        # 发言倾向
        self.base_speaking_probability = 0.4  # 基础发言概率

    def should_speak(self, context: SocialContext, emotional_bonus: float = 0.0) -> tuple[bool, str, float]:
        """
        决定是否发言

        Args:
            context: 社交上下文
            emotional_bonus: 情感加成 (-0.3 到 +0.3)

        Returns:
            tuple[bool, str, float]: (是否发言, 原因, 发言概率)
        """
        # 1. 检查强制触发
        trigger = context.extra.get("trigger", {})
        if trigger.get("forced_trigger", False):
            reasons = trigger.get("reasons", ["强制触发"])
            return True, " + ".join(reasons), 0.95

        # 2. 计算发言概率
        probability = self._calculate_probability(context, emotional_bonus)

        # 3. 添加随机性（模拟"看心情"）
        final_probability = probability + random.uniform(-0.15, 0.15)
        final_probability = max(0.0, min(1.0, final_probability))

        # 4. 决策
        should = random.random() < final_probability

        if should:
            reason = self._explain_decision(context, probability)
        else:
            reason = "选择沉默"

        return should, reason, final_probability

    def _calculate_probability(
        self,
        context: SocialContext,
        emotional_bonus: float
    ) -> float:
        """计算发言概率"""
        factors = []

        # === 基础因素 ===

        # 1. 触发优先级
        trigger = context.extra.get("trigger", {})
        priority = trigger.get("priority", 0.0)
        factors.append(("触发优先级", priority * 0.4))

        # 2. 话题相关性
        factors.append(("话题相关", context.topic_relevance * 0.25))

        # 3. 话题熟悉度（越熟悉越愿发言）
        factors.append(("话题熟悉", context.topic_familiarity * 0.2))

        # 4. 情感加成
        factors.append(("情感状态", emotional_bonus))

        # === 环境因素 ===

        # 5. 群活跃度影响
        ambience = context.extra.get("ambience", {})
        activity = ambience.get("activity", 0.5)
        activity_modifier = self._get_activity_modifier(activity)
        factors.append(("群活跃度", (activity_modifier - 1.0) * 0.2))

        # 6. 紧张度（群里有矛盾时减少发言）
        tension = ambience.get("tension", 0.0)
        if tension > 0.5:
            factors.append(("紧张氛围", -0.2))

        # === 人格因素 ===

        # 7. 外向性影响
        extraversion_bonus = (self.extraversion - 0.5) * 0.3
        factors.append(("外向性格", extraversion_bonus))

        # 8. 神经质影响（情绪不稳定时减少发言）
        neuroticism_penalty = (self.neuroticism - 0.3) * 0.1
        factors.append(("情绪稳定", -neuroticism_penalty))

        # === 综合计算 ===
        total = self.base_speaking_probability
        for name, value in factors:
            total += value

        # Sigmoid 归一化
        probability = 1 / (1 + math.exp(-(total - 0.5) * 6))

        return probability

    def _get_activity_modifier(self, activity: float) -> float:
        """获取活跃度修正"""
        if activity < 0.2:
            return 1.3  # 冷群 → 更愿说话
        elif activity > 0.8:
            return 0.7  # 热群 → 倾向潜水
        return 1.0

    def _explain_decision(self, context: SocialContext, probability: float) -> str:
        """解释决策原因"""
        reasons = []

        trigger = context.extra.get("trigger", {})
        if trigger.get("priority", 0) > 0.3:
            reasons.extend(trigger.get("reasons", []))

        if context.topic_relevance > 0.6:
            reasons.append("话题感兴趣")

        if context.topic_familiarity > 0.7:
            reasons.append("话题熟悉")

        if not reasons:
            reasons.append("随机触发")

        return " + ".join(reasons)


class SpeakingDeciderWithSocial:
    """集成社交感知的发言决策器"""

    def __init__(self, social_manager: SocialAwarenessManager, decider: SpeakingDecider):
        self.social_manager = social_manager
        self.decider = decider

    def decide(
        self,
        context: SocialContext,
        emotional_bonus: float = 0.0
    ) -> tuple[bool, str, float]:
        """
        完整决策流程

        Returns:
            tuple[bool, str, float]: (是否发言, 原因, 概率)
        """
        # 1. 社交感知分析
        context = self.social_manager.analyze(context)

        # 2. 发言决策
        return self.decider.should_speak(context, emotional_bonus)

    def quick_decide(self, context: SocialContext) -> bool:
        """
        快速决策（只返回是否发言）
        用于性能敏感场景
        """
        should, _, _ = self.decide(context)
        return should

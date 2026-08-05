"""
注意力系统 - AstrBot风格
管理Bot对群聊中各用户的注意力状态
"""
import logging
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserAttention:
    """用户注意力状态"""
    user_id: str
    attention: float = 0.5  # 注意力值 0-1
    last_update: float = field(default_factory=time.time)
    is_active: bool = True

    def decay(self, elapsed_seconds: float, halflife: float) -> None:
        """注意力衰减"""
        if halflife <= 0:
            return
        decay_rate = math.log(2) / halflife
        self.attention *= math.exp(-decay_rate * elapsed_seconds)
        self.attention = max(0.0, min(1.0, self.attention))

    def boost(self, step: float) -> None:
        """提升注意力"""
        self.attention = min(1.0, self.attention + step)
        self.last_update = time.time()


@dataclass
class BotAttentionState:
    """Bot对群聊的整体注意力状态"""
    group_id: str
    base_attention: float = 0.5  # 基础注意力
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0  # 消息计数（用于疲劳）

    # 用户注意力追踪
    user_attentions: Dict[str, UserAttention] = field(default_factory=dict)

    # 溢出效果
    spillover_attention: float = 0.0
    spillover_last_update: float = field(default_factory=time.time)


class AttentionManager:
    """
    注意力管理器 - AstrBot风格

    核心特性：
    1. 追踪Bot对各用户的注意力
    2. 被@或回复时注意力提升
    3. 长时间无互动注意力衰减
    4. 注意力溢出效应
    """

    def __init__(
        self,
        initial_attention: float = 0.5,
        decay_halflife: float = 300,  # 5分钟半衰期
        boost_step: float = 0.4,
        decrease_step: float = 0.1,
        decrease_threshold: float = 0.3,
        max_tracked_users: int = 10,
        # 溢出配置
        enable_spillover: bool = True,
        spillover_ratio: float = 0.35,
        spillover_halflife: float = 90,
        spillover_min_trigger: float = 0.4,
    ):
        self.initial_attention = initial_attention
        self.decay_halflife = decay_halflife
        self.boost_step = boost_step
        self.decrease_step = decrease_step
        self.decrease_threshold = decrease_threshold
        self.max_tracked_users = max_tracked_users

        # 溢出配置
        self.enable_spillover = enable_spillover
        self.spillover_ratio = spillover_ratio
        self.spillover_halflife = spillover_halflife
        self.spillover_min_trigger = spillover_min_trigger

        # 群组注意力状态
        self._group_states: Dict[str, BotAttentionState] = defaultdict(
            lambda: BotAttentionState(group_id="")
        )

    def get_group_state(self, group_id: str) -> BotAttentionState:
        """获取群组注意力状态"""
        if group_id not in self._group_states:
            self._group_states[group_id] = BotAttentionState(group_id=group_id)
        return self._group_states[group_id]

    def get_user_attention(self, group_id: str, user_id: str) -> float:
        """获取用户注意力"""
        state = self.get_group_state(group_id)

        if user_id not in state.user_attentions:
            state.user_attentions[user_id] = UserAttention(user_id=user_id)

        # 应用衰减
        self._apply_decay(state, user_id)

        return state.user_attentions[user_id].attention

    def on_message_received(
        self,
        group_id: str,
        user_id: str,
        mentioned_bot: bool = False,
        is_reply_to_bot: bool = False
    ) -> None:
        """
        收到消息时的注意力更新

        Args:
            group_id: 群ID
            user_id: 用户ID
            mentioned_bot: 是否@了Bot
            is_reply_to_bot: 是否回复了Bot
        """
        state = self.get_group_state(group_id)
        current_time = time.time()

        # 更新基础活跃状态
        state.last_activity = current_time
        state.message_count += 1

        # 清理过多追踪用户
        self._cleanup_user_tracking(state)

        # 获取或创建用户注意力
        if user_id not in state.user_attentions:
            state.user_attentions[user_id] = UserAttention(user_id=user_id)

        user_attention = state.user_attentions[user_id]

        # 根据消息类型调整注意力
        if mentioned_bot:
            # 被@ - 大幅提升
            user_attention.boost(self.boost_step)
            logger.debug(f"User {user_id} mentioned bot, attention boosted to {user_attention.attention:.2f}")

        elif is_reply_to_bot:
            # 回复Bot - 中等提升
            user_attention.boost(self.boost_step * 0.7)
            logger.debug(f"User {user_id} replied to bot, attention boosted to {user_attention.attention:.2f}")

        # 触发溢出效果
        if self.enable_spillover and user_attention.attention > self.spillover_min_trigger:
            self._apply_spillover(state, user_id, user_attention.attention)

    def on_bot_reply(self, group_id: str) -> None:
        """Bot回复后降低自己注意力"""
        state = self.get_group_state(group_id)

        # Bot回复后，所有用户注意力略微下降
        for user_id, attention in state.user_attentions.items():
            attention.attention *= 0.95

    def on_no_reply(self, group_id: str, user_id: str) -> None:
        """Bot未回复某用户消息"""
        state = self.get_group_state(group_id)

        if user_id in state.user_attentions:
            attention = state.user_attentions[user_id]

            # 如果注意力已经很低，再降
            if attention.attention < self.decrease_threshold:
                attention.attention = max(0.0, attention.attention - self.decrease_step)
                logger.debug(f"User {user_id} attention decreased to {attention.attention:.2f}")

    def get_effective_attention(self, group_id: str, user_id: str = None) -> float:
        """
        获取有效注意力（考虑溢出效果）

        Args:
            group_id: 群ID
            user_id: 用户ID（可选，不指定则返回Bot整体注意力）
        """
        state = self.get_group_state(group_id)
        current_time = time.time()

        # 应用衰减
        elapsed = current_time - state.last_activity
        decay_factor = math.exp(-math.log(2) * elapsed / self.decay_halflife)

        # 基础注意力
        base = state.base_attention * decay_factor

        # 溢出效果衰减
        spillover_elapsed = current_time - state.spillover_last_update
        spillover_decay = math.exp(-math.log(2) * spillover_elapsed / self.spillover_halflife)
        spillover = state.spillover_attention * spillover_decay

        # 用户特定注意力
        if user_id:
            user_attention = self.get_user_attention(group_id, user_id)
            return min(1.0, base + user_attention * 0.3 + spillover * self.spillover_ratio)
        else:
            # Bot整体注意力
            max_user_attention = 0.0
            for ua in state.user_attentions.values():
                max_user_attention = max(max_user_attention, ua.attention)

            return min(1.0, base + max_user_attention * 0.3 + spillover * self.spillover_ratio)

    def _apply_decay(self, state: BotAttentionState, user_id: str) -> None:
        """应用注意力衰减"""
        if user_id not in state.user_attentions:
            return

        current_time = time.time()
        elapsed = current_time - state.user_attentions[user_id].last_update
        state.user_attentions[user_id].decay(elapsed, self.decay_halflife)
        state.user_attentions[user_id].last_update = current_time

    def _apply_spillover(self, state: BotAttentionState, source_user_id: str, source_attention: float) -> None:
        """应用注意力溢出效应"""
        # 高注意力用户的存在会影响Bot对整体的注意力
        if source_attention > self.spillover_min_trigger:
            overflow = (source_attention - self.spillover_min_trigger) * self.spillover_ratio
            state.spillover_attention = min(0.5, state.spillover_attention + overflow)
            state.spillover_last_update = time.time()

    def _cleanup_user_tracking(self, state: BotAttentionState) -> None:
        """清理过多的用户追踪"""
        if len(state.user_attentions) <= self.max_tracked_users:
            return

        # 找出注意力最低的用户移除
        sorted_users = sorted(
            state.user_attentions.items(),
            key=lambda x: x[1].attention
        )

        # 保留高注意力的用户
        to_remove = len(state.user_attentions) - self.max_tracked_users
        for user_id, _ in sorted_users[:to_remove]:
            del state.user_attentions[user_id]

    def reset_group(self, group_id: str) -> None:
        """重置群组注意力状态"""
        if group_id in self._group_states:
            del self._group_states[group_id]

    def cleanup_inactive_groups(self, active_groups: set, inactivity_threshold: float = 3600) -> int:
        """清理不活跃群组"""
        current_time = time.time()
        to_remove = []

        for group_id, state in self._group_states.items():
            elapsed = current_time - state.last_activity
            if elapsed > inactivity_threshold:
                to_remove.append(group_id)

        for group_id in to_remove:
            del self._group_states[group_id]

        return len(to_remove)


class AttentionKeywordsDetector:
    """注意力关键词检测器"""

    def __init__(
        self,
        positive_keywords: list = None,
        negative_keywords: list = None,
        negation_words: list = None
    ):
        self.positive_keywords = positive_keywords or []
        self.negative_keywords = negative_keywords or []
        self.negation_words = negation_words or ["不", "没", "别", "非", "无", "未", "勿", "莫"]

    def detect_attention_keywords(self, message: str, sender_id: str) -> tuple[float, str]:
        """
        检测消息中的注意力关键词

        Returns:
            tuple[float, str]: (注意力变化值, 原因)
        """
        message_lower = message.lower()

        # 检查正面关键词
        for keyword in self.positive_keywords:
            if keyword.lower() in message_lower:
                # 检查是否被否定
                if self._is_negated(message, keyword):
                    return -0.1, f"否定词+{keyword}"
                return 0.15, f"正面关键词「{keyword}」"

        # 检查负面关键词
        for keyword in self.negative_keywords:
            if keyword.lower() in message_lower:
                if self._is_negated(message, keyword):
                    return 0.05, f"否定词+{keyword}"
                return -0.1, f"负面关键词「{keyword}」"

        return 0.0, ""

    def _is_negated(self, message: str, keyword: str) -> bool:
        """检查关键词是否被否定"""
        keyword_pos = message.find(keyword)
        if keyword_pos < 0:
            return False

        # 检查关键词前5个字是否有否定词
        start = max(0, keyword_pos - 5)
        prefix = message[start:keyword_pos]

        for negation in self.negation_words:
            if negation in prefix:
                return True

        return False

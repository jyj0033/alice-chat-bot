"""
疲劳系统 - AstrBot风格
管理Bot的对话疲劳状态
"""
import logging
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class FatigueState:
    """疲劳状态"""
    conversation_rounds: int = 0  # 对话轮次
    consecutive_bot_replies: int = 0  # Bot连续回复次数
    last_bot_speak_time: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)
    last_reset_time: float = field(default_factory=time.time)
    fatigue_level: float = 0.0  # 疲劳等级 0-1

    def reset(self) -> None:
        """重置疲劳状态"""
        self.conversation_rounds = 0
        self.consecutive_bot_replies = 0
        self.fatigue_level = 0.0
        self.last_reset_time = time.time()


class FatigueManager:
    """
    疲劳管理器 - AstrBot风格

    核心特性：
    1. 追踪对话轮次
    2. 长时间无活动自动重置
    3. 疲劳等级影响发言概率
    4. 疲劳时可能主动结束对话
    """

    def __init__(
        self,
        enabled: bool = True,
        # 重置配置
        reset_threshold: float = 300,  # 5分钟无活动重置
        # 疲劳阈值
        threshold_light: int = 3,
        threshold_medium: int = 5,
        threshold_heavy: int = 8,
        # 疲劳影响
        decrease_light: float = 0.1,
        decrease_medium: float = 0.2,
        decrease_heavy: float = 0.35,
        # 主动结束概率
        closing_probability: float = 0.3,
        # 冷却配置
        cooldown_enabled: bool = True,
        cooldown_max_duration: float = 600,  # 10分钟
        cooldown_trigger_threshold: float = 0.3,
        cooldown_attention_decrease: float = 0.2,
    ):
        self.enabled = enabled
        self.reset_threshold = reset_threshold

        # 阈值
        self.threshold_light = threshold_light
        self.threshold_medium = threshold_medium
        self.threshold_heavy = threshold_heavy

        # 疲劳影响
        self.decrease_light = decrease_light
        self.decrease_medium = decrease_medium
        self.decrease_heavy = decrease_heavy

        # 结束概率
        self.closing_probability = closing_probability

        # 冷却
        self.cooldown_enabled = cooldown_enabled
        self.cooldown_max_duration = cooldown_max_duration
        self.cooldown_trigger_threshold = cooldown_trigger_threshold
        self.cooldown_attention_decrease = cooldown_attention_decrease

        # 会话疲劳状态
        self._session_states: Dict[str, FatigueState] = {}
        self._cooldowns: Dict[str, float] = {}  # 会话冷却时间

    def get_state(self, session_id: str) -> FatigueState:
        """获取会话疲劳状态"""
        if session_id not in self._session_states:
            self._session_states[session_id] = FatigueState()
        return self._session_states[session_id]

    def on_message(self, session_id: str, is_bot_message: bool = False) -> None:
        """
        收到消息时的疲劳更新

        Args:
            session_id: 会话ID
            is_bot_message: 是否是Bot自己的消息
        """
        if not self.enabled:
            return
        state = self.get_state(session_id)
        current_time = time.time()

        # 检查是否需要重置
        if current_time - state.last_activity_time > self.reset_threshold:
            state.reset()
            logger.debug(f"Session {session_id} fatigue reset due to inactivity")
            return

        # 更新活动时间
        state.last_activity_time = current_time

        if is_bot_message:
            # 一轮按 bot 实际发出一次回复计算，而不是按群里所有人的消息计算。
            state.conversation_rounds += 1
            state.consecutive_bot_replies += 1
            state.last_bot_speak_time = current_time
        else:
            state.consecutive_bot_replies = 0  # 别人说话则重置连续计数

        # 更新疲劳等级
        self._update_fatigue_level(state)

    def _update_fatigue_level(self, state: FatigueState) -> None:
        """更新疲劳等级"""
        rounds = state.conversation_rounds

        if rounds >= self.threshold_heavy:
            state.fatigue_level = 1.0
        elif rounds >= self.threshold_medium:
            state.fatigue_level = 0.5 + 0.5 * (rounds - self.threshold_medium) / (self.threshold_heavy - self.threshold_medium)
        elif rounds >= self.threshold_light:
            state.fatigue_level = 0.2 + 0.3 * (rounds - self.threshold_light) / (self.threshold_medium - self.threshold_light)
        else:
            state.fatigue_level = 0.0

    def get_probability_penalty(self, session_id: str) -> float:
        """
        获取疲劳导致的概率惩罚

        Returns:
            float: 概率惩罚值（负数）
        """
        if not self.enabled:
            return 0.0
        state = self.get_state(session_id)

        if state.fatigue_level < 0.3:
            return -self.decrease_light * state.fatigue_level / 0.3
        elif state.fatigue_level < 0.6:
            base = -self.decrease_light
            extra = -self.decrease_medium * (state.fatigue_level - 0.3) / 0.3
            return base + extra
        else:
            base = -self.decrease_light - self.decrease_medium
            extra = -self.decrease_heavy * (state.fatigue_level - 0.6) / 0.4
            return base + extra

    def should_close_conversation(self, session_id: str) -> bool:
        """
        决定是否应该主动结束对话

        Returns:
            bool: 是否应该结束
        """
        if not self.enabled:
            return False
        state = self.get_state(session_id)

        # 只有在高疲劳时才可能结束
        if state.fatigue_level < 0.6:
            return False

        # 根据疲劳等级和概率决定
        closing_chance = self.closing_probability * (state.fatigue_level - 0.5)

        import random
        return random.random() < closing_chance

    def get_closing_message(self) -> str:
        """获取结束对话的消息"""
        import random
        messages = [
            "啊我先溜了，有点累",
            "先这样吧，下次再聊",
            "困了，先下线了",
            "聊不动了，你们继续",
            "我先去摸鱼了",
        ]
        return random.choice(messages)

    # === 冷却系统 ===

    def is_in_cooldown(self, session_id: str) -> bool:
        """检查是否在冷却中"""
        if not self.enabled or not self.cooldown_enabled:
            return False

        if session_id not in self._cooldowns:
            return False

        cooldown_end = self._cooldowns[session_id]
        return time.time() < cooldown_end

    def start_cooldown(self, session_id: str, probability: float) -> None:
        """
        开始冷却

        Args:
            session_id: 会话ID
            probability: 当前发言概率
        """
        if not self.enabled or not self.cooldown_enabled:
            return

        if probability >= self.cooldown_trigger_threshold:
            return

        # 冷却时间与概率相关
        deficit = self.cooldown_trigger_threshold - probability
        duration = self.cooldown_max_duration * deficit / self.cooldown_trigger_threshold
        duration = min(self.cooldown_max_duration, max(10, duration))  # 最少10秒

        self._cooldowns[session_id] = time.time() + duration
        logger.debug(f"Session {session_id} entered cooldown for {duration:.0f}s")

    def get_cooldown_remaining(self, session_id: str) -> float:
        """获取冷却剩余时间"""
        if session_id not in self._cooldowns:
            return 0.0

        remaining = self._cooldowns[session_id] - time.time()
        return max(0.0, remaining)

    def cleanup(self) -> int:
        """清理过期冷却"""
        current_time = time.time()
        to_remove = [
            sid for sid, end_time in self._cooldowns.items()
            if current_time > end_time
        ]

        for sid in to_remove:
            del self._cooldowns[sid]

        # 清理过期会话
        to_remove_sessions = []
        for sid, state in self._session_states.items():
            if current_time - state.last_activity_time > self.reset_threshold * 2:
                to_remove_sessions.append(sid)

        for sid in to_remove_sessions:
            del self._session_states[sid]

        return len(to_remove) + len(to_remove_sessions)

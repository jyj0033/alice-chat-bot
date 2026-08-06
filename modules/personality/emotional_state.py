"""
情感状态系统
管理 Bot 的情感状态（精力值、投入度）
"""
import logging
import time
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Emotion(Enum):
    """情感状态"""
    HAPPY = "happy"
    NEUTRAL = "neutral"
    BORED = "bored"
    EXCITED = "excited"
    TIRED = "tired"
    ANXIOUS = "anxious"
    CALM = "calm"


@dataclass
class EmotionalState:
    """情感状态"""

    current_emotion: Emotion = Emotion.NEUTRAL
    energy: float = 0.7  # 精力值 0-1
    engagement: float = 0.5  # 投入度 0-1

    # 衰减速率（每小时）
    energy_decay: float = 0.1
    engagement_decay: float = 0.2

    # 情感修饰
    mood_modifier: float = 1.0  # 心情修正因子

    _last_update: float = None

    def __post_init__(self):
        self._last_update = time.time()

    def update(self, trigger: str, intensity: float = 0.1) -> None:
        """
        根据事件更新情感

        Args:
            trigger: 触发事件类型
            intensity: 影响强度
        """
        current_time = time.time()
        elapsed = (current_time - self._last_update) / 3600  # 转换为小时

        # 情感衰减
        self._apply_decay(elapsed)

        # 事件影响
        trigger_handlers = {
            # 积极事件
            "interesting_topic": lambda: self._update_emotion(Emotion.INTERESTED if hasattr(Emotion, 'INTERESTED') else Emotion.EXCITED, intensity * 0.3),
            "praised": lambda: self._boost(energy=0.1, engagement=intensity * 0.3),
            "funny_moment": lambda: self._update_emotion(Emotion.HAPPY, intensity * 0.2),
            "achievement": lambda: self._update_emotion(Emotion.EXCITED, intensity * 0.4),

            # 消极事件
            "ignored": lambda: self._update_emotion(Emotion.ANXIOUS if hasattr(Emotion, 'ANXIOUS') else Emotion.NEUTRAL, -intensity * 0.2),
            "long_silence": lambda: self._update_emotion(Emotion.BORED, -intensity * 0.1),
            "boring_topic": lambda: self._update_emotion(Emotion.BORED, -intensity * 0.15),
            "tired": lambda: self._update_emotion(Emotion.TIRED, -intensity * 0.2),

            # 中性事件
            "mentioned": lambda: self._boost(engagement=intensity * 0.2),
            "active_discussion": lambda: self._boost(engagement=intensity * 0.1, energy=-intensity * 0.05),
            "normal_message": lambda: None,
        }

        handler = trigger_handlers.get(trigger)
        if handler:
            handler()

        # 更新心情修正因子
        self._update_mood_modifier()

        self._last_update = current_time

    def _apply_decay(self, elapsed_hours: float) -> None:
        """应用时间衰减"""
        self.energy = max(0.1, self.energy - self.energy_decay * elapsed_hours)
        self.engagement = max(0.1, self.engagement - self.engagement_decay * elapsed_hours)

    def _update_emotion(self, emotion: Emotion, delta: float) -> None:
        """更新情感"""
        if delta > 0:
            self.current_emotion = emotion
        else:
            # 消极变化时，只有当前情绪与目标情绪不同时才切换
            if self.current_emotion == emotion:
                self.current_emotion = Emotion.NEUTRAL

    def _boost(self, energy: float = 0, engagement: float = 0) -> None:
        """提升情感值"""
        self.energy = min(1.0, max(0.1, self.energy + energy))
        self.engagement = min(1.0, max(0.1, self.engagement + engagement))

    def _update_mood_modifier(self) -> None:
        """更新心情修正因子"""
        # 基于当前情感状态计算
        emotion_weights = {
            Emotion.HAPPY: 1.2,
            Emotion.EXCITED: 1.3,
            Emotion.NEUTRAL: 1.0,
            Emotion.CALM: 1.0,
            Emotion.ANXIOUS: 0.8,
            Emotion.BORED: 0.7,
            Emotion.TIRED: 0.6,
        }

        base = emotion_weights.get(self.current_emotion, 1.0)
        self.mood_modifier = base * (0.5 + self.energy * 0.5)

    def get_speaking_bonus(self) -> float:
        """
        获取心情加成（影响发言概率）
        返回 -0.3 到 +0.3
        """
        # 基于精力和投入度
        base = (self.energy + self.engagement) / 2

        # 应用心情修正
        bonus = (base - 0.5) * 0.4 * self.mood_modifier

        return max(-0.3, min(0.3, bonus))

    def get_thinking_delay_multiplier(self) -> float:
        """
        获取思考延迟倍数
        精力低时思考更慢
        """
        return 1.0 + (1.0 - self.energy) * 0.5

    def should_reply(self) -> tuple[bool, float]:
        """
        判断是否应该回复
        返回 (是否回复, 概率)
        """
        probability = (self.energy + self.engagement) / 2

        # 加入随机性
        probability += random.uniform(-0.1, 0.1)
        probability = max(0.0, min(1.0, probability))

        return random.random() < probability, probability

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "emotion": self.current_emotion.value,
            "energy": self.energy,
            "engagement": self.engagement,
            "mood_modifier": self.mood_modifier,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        """从字典加载"""
        state = cls()
        state.current_emotion = Emotion(data.get("emotion", "neutral"))
        state.energy = data.get("energy", 0.7)
        state.engagement = data.get("engagement", 0.5)
        state.mood_modifier = data.get("mood_modifier", 1.0)
        return state


class EmotionalManager:
    """情感管理器"""

    def __init__(
        self,
        decay_halflife: float = 600,
        positive_keywords: list = None,
        negative_keywords: list = None,
    ):
        self._states: dict[str, EmotionalState] = {}
        self._default_state = EmotionalState()
        self.decay_halflife = decay_halflife
        self.positive_keywords = positive_keywords or []
        self.negative_keywords = negative_keywords or []

    def get_state(self, session_id: str) -> EmotionalState:
        """获取会话的情感状态"""
        if session_id not in self._states:
            self._states[session_id] = EmotionalState()
        return self._states[session_id]

    def update(self, session_id: str, trigger: str, intensity: float = 0.1) -> None:
        """更新情感"""
        state = self.get_state(session_id)
        state.update(trigger, intensity)

    def trigger_event(self, session_id: str, event_type: str) -> None:
        """触发情感事件"""
        event_mapping = {
            "mentioned": ("mentioned", 0.2),
            "interesting": ("interesting_topic", 0.3),
            "boring": ("boring_topic", 0.2),
            "silence": ("long_silence", 0.15),
            "funny": ("funny_moment", 0.25),
            "normal_message": ("normal_message", 0.0),
        }

        if event_type in event_mapping:
            trigger, intensity = event_mapping[event_type]
            self.update(session_id, trigger, intensity)

    def detect_emotion_keywords(self, message: str) -> tuple[float, str]:
        """
        检测消息中的情绪关键词

        Returns:
            tuple[float, str]: (情绪变化, 描述)
        """
        message_lower = message.lower()

        for keyword in self.positive_keywords:
            if keyword.lower() in message_lower:
                return 0.1, f"正面词「{keyword}」"

        for keyword in self.negative_keywords:
            if keyword.lower() in message_lower:
                return -0.15, f"负面词「{keyword}」"

        return 0.0, ""

    def detect_and_apply_keywords(self, session_id: str, message: str) -> str:
        """检测消息情绪关键词并应用到会话情感状态，返回描述（空串=无变化）"""
        change, desc = self.detect_emotion_keywords(message)
        if change == 0:
            return ""

        state = self.get_state(session_id)
        if change > 0:
            # 被夸奖/感谢 → 心情变好，精力投入提升
            state._boost(energy=0.05, engagement=0.08)
            state.current_emotion = Emotion.HAPPY
        else:
            # 被骂/负面 → 心情变差，兴趣下降
            state._boost(energy=-0.03, engagement=-0.05)
            state.current_emotion = Emotion.BORED
        state._update_mood_modifier()
        return desc

    def reset(self, session_id: str) -> None:
        """重置情感状态"""
        if session_id in self._states:
            self._states[session_id] = EmotionalState()

    def cleanup_inactive(self, active_sessions: set[str]) -> int:
        """清理不活跃的会话状态"""
        to_remove = [s for s in self._states if s not in active_sessions]
        for s in to_remove:
            del self._states[s]
        return len(to_remove)

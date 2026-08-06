"""
社交感知模块
包括触发检测、注意力、疲劳、发言决策等功能
"""
from .awareness import (
    SocialContext,
    SocialAwarenessManager,
    TriggerDetector,
    AmbienceAnalyzer,
    TopicAnalyzer,
)
from .attention import (
    UserAttention,
    BotAttentionState,
    AttentionManager,
    AttentionKeywordsDetector,
)
from .fatigue import (
    FatigueState,
    FatigueManager,
)
from .enhanced_decider import (
    SpeakingDecision,
    EnhancedSpeakingDecider,
)
from .conversation_floor import (
    ActionType,
    ActionPlan,
    ConversationFloor,
    ConversationFloorManager,
)

__all__ = [
    # awareness
    "SocialContext",
    "SocialAwarenessManager",
    "TriggerDetector",
    "AmbienceAnalyzer",
    "TopicAnalyzer",
    # attention
    "UserAttention",
    "BotAttentionState",
    "AttentionManager",
    "AttentionKeywordsDetector",
    # fatigue
    "FatigueState",
    "FatigueManager",
    # enhanced_decider
    "SpeakingDecision",
    "EnhancedSpeakingDecider",
    # conversation_floor
    "ActionType",
    "ActionPlan",
    "ConversationFloor",
    "ConversationFloorManager",
]

"""
人格模块
包括人格设定、情感状态、说话风格、错字生成等功能
"""
from .personality import Personality
from .emotional_state import EmotionalState, EmotionalManager, Emotion
from .speaking_style import SpeakingStyle, SpeakingStyleManager, create_default_style
from .typo import TypoGenerator, TypoPatterns, apply_typo, get_default_generator

__all__ = [
    "Personality",
    "EmotionalState",
    "EmotionalManager",
    "Emotion",
    "SpeakingStyle",
    "SpeakingStyleManager",
    "create_default_style",
    "TypoGenerator",
    "TypoPatterns",
    "apply_typo",
    "get_default_generator",
]

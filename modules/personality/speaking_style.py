"""
说话风格系统
参数化控制 Bot 的说话方式
"""
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeakingStyle:
    """说话风格参数"""

    # 词汇层面
    common_words: list[str] = field(default_factory=list)  # 常用词/口头禅
    banned_words: list[str] = field(default_factory=list)  # 避免的词
    filler_words: list[str] = field(default_factory=lambda: ["呃", "嗯", "那个", "这个"])

    # 句式层面
    min_sentence_length: int = 5   # 最短句子
    max_sentence_length: int = 50  # 最长句子
    avg_sentence_length_range: tuple[int, int] = (10, 30)  # 平均句长范围

    # 标点与格式
    use_ellipsis: bool = True      # 是否使用省略号
    use_emoji: bool = True         # 是否使用emoji
    emoji_frequency: float = 0.2   # emoji使用频率 (0-1)
    use_question_marks: bool = True  # 结尾是否加"？"表示好奇

    # 语气层面
    formality: float = 0.3        # 正式程度 0-1 (0=随意, 1=正式)
    enthusiasm: float = 0.6       # 热情程度 0-1
    humor: float = 0.5            # 幽默程度 0-1

    # 标点习惯
    use_exclamation: bool = True  # 是否使用感叹号
    use_period: bool = True       # 是否使用句号

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakingStyle":
        """从字典加载"""
        return cls(**data)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "common_words": self.common_words,
            "banned_words": self.banned_words,
            "filler_words": self.filler_words,
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "avg_sentence_length_range": self.avg_sentence_length_range,
            "use_ellipsis": self.use_ellipsis,
            "use_emoji": self.use_emoji,
            "emoji_frequency": self.emoji_frequency,
            "use_question_marks": self.use_question_marks,
            "formality": self.formality,
            "enthusiasm": self.enthusiasm,
            "humor": self.humor,
            "use_exclamation": self.use_exclamation,
            "use_period": self.use_period,
        }


class SpeakingStyleManager:
    """说话风格管理器"""

    def __init__(self, style: SpeakingStyle, emoji_set: list[str] = None):
        self.style = style
        self.emoji_set = emoji_set or ["😅", "🤔", "😂", "👍", "🙄", "😏", "🤷", "👀"]

    def get_style_guide(self) -> str:
        """获取风格指南字符串"""
        parts = []
        if self.style.common_words:
            parts.append(f"常用口头禅：{', '.join(self.style.common_words[:3])}")
        if self.style.filler_words:
            parts.append(f"语气词：{', '.join(self.style.filler_words[:3])}")
        if self.style.formality:
            parts.append(f"正式程度：{self.style.formality}")
        if self.style.enthusiasm:
            parts.append(f"热情程度：{self.style.enthusiasm}")
        return "；".join(parts) if parts else ""

    def apply_style(self, text: str) -> str:
        """
        对生成的文本应用说话风格

        Args:
            text: 原始文本

        Returns:
            应用风格后的文本
        """
        if not text:
            return text

        result = text

        # 1. 替换/添加口头禅
        result = self._apply_catchphrases(result)

        # 2. 添加语气词
        result = self._apply_fillers(result)

        # 3. 处理标点
        result = self._apply_punctuation(result)

        # 4. 添加 emoji
        result = self._apply_emoji(result)

        # 5. 长度调整
        result = self._adjust_length(result)

        # 6. 移除禁用词
        result = self._remove_banned_words(result)

        return result

    def _apply_catchphrases(self, text: str) -> str:
        """应用口头禅"""
        if not self.style.common_words:
            return text

        # 30% 概率使用口头禅
        if random.random() < 0.3:
            phrase = random.choice(self.style.common_words)
            # 在合适位置插入
            words = text.split()
            if len(words) > 2:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, phrase)
                return "".join(words)

        return text

    def _apply_fillers(self, text: str) -> str:
        """添加语气词"""
        if not self.style.filler_words or random.random() > 0.2:
            return text

        # 在句首添加语气词
        fillers = self.style.filler_words
        filler = random.choice(fillers)

        if text and text[0].isalpha():
            return filler + text
        elif text.startswith("「") or text.startswith("\""):
            return filler + text[1] + filler + text[1:]

        return text

    def _apply_punctuation(self, text: str) -> str:
        """处理标点符号"""
        # 省略号处理
        if self.style.use_ellipsis and random.random() < 0.15:
            if not text.endswith("..."):
                text = text.rstrip(".。") + "..."

        # 感叹号处理
        if self.style.use_exclamation and self.style.enthusiasm > 0.6:
            if random.random() < 0.2:
                text = text.rstrip(".。!！") + random.choice(["!", "!!", "！"])

        return text

    def _apply_emoji(self, text: str) -> str:
        """添加 emoji"""
        if not self.style.use_emoji:
            return text

        if random.random() < self.style.emoji_frequency:
            emoji = random.choice(self.emoji_set)
            # 在句尾或合适位置添加
            if text.endswith(("。", ".", "！", "!")):
                return text[:-1] + emoji + text[-1]
            else:
                return text + emoji

        return text

    def _adjust_length(self, text: str) -> str:
        """调整文本长度"""
        # 简单实现：不做极端截断
        return text

    def _remove_banned_words(self, text: str) -> str:
        """移除禁用词"""
        for word in self.style.banned_words:
            text = text.replace(word, "*" * len(word))
        return text

    def get_random_delay(self, base: float = 2.0) -> float:
        """
        获取随机延迟时间（模拟思考）

        Returns:
            延迟秒数
        """
        min_delay = base * 0.7
        max_delay = base * 1.5

        # 热情度高时延迟短
        if self.style.enthusiasm > 0.7:
            min_delay *= 0.8
            max_delay *= 0.9
        elif self.style.enthusiasm < 0.4:
            min_delay *= 1.2
            max_delay *= 1.3

        return random.uniform(min_delay, max_delay)

    def should_respond(self) -> bool:
        """
        判断是否应该回复（基于说话风格）

        正式程度高、热情程度低时更倾向不回复
        """
        probability = 0.5

        # 正式程度影响
        probability -= (self.style.formality - 0.5) * 0.3

        # 热情程度影响
        probability += (self.style.enthusiasm - 0.5) * 0.4

        return random.random() < max(0.1, min(0.9, probability))


def create_default_style() -> SpeakingStyle:
    """创建默认说话风格"""
    return SpeakingStyle(
        common_words=["话说", "其实", "感觉", "好像", "有点"],
        filler_words=["呃", "嗯", "那个"],
        min_sentence_length=5,
        max_sentence_length=50,
        avg_sentence_length_range=(10, 30),
        use_ellipsis=True,
        use_emoji=True,
        emoji_frequency=0.2,
        formality=0.3,
        enthusiasm=0.6,
        humor=0.5,
    )

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

_EMOJI_RE = re.compile(
    r'[\U0001F000-\U0001FAFF☀-➿️‍⭐❤❣☮☯〰©®㊗㊙]'
)


@dataclass
class SpeakingStyle:
    """说话风格参数"""

    # 词汇层面
    # 以下几项是旧版配置兼容字段，已不再用于提示词或后处理，避免把回复
    # 固定成一组可识别的口头禅/模板词。
    common_words: list[str] = field(default_factory=list, repr=False)
    banned_words: list[str] = field(default_factory=list)  # 避免的词
    filler_words: list[str] = field(default_factory=list, repr=False)

    # 句式层面
    min_sentence_length: int = 5   # 最短句子
    max_sentence_length: int = 50  # 最长句子
    avg_sentence_length_range: tuple[int, int] = (10, 30)  # 平均句长范围

    # 回复长度
    max_reply_length: int = 20     # 单条回复最大字符数（超长会被智能截断）
    direct_max_reply_length: int | None = None  # 明确问 bot 时的最大字符数

    # 旧版语气词后处理参数，仅保留以兼容旧配置。
    filler_frequency: float = field(default=0.0, repr=False)

    # 标点与格式
    use_ellipsis: bool = True      # 是否使用省略号
    # 省略号后处理频率：给高了会拼出「真人！...」「汗流浃背哈哈...」这类
    # 机械痕迹（原实现是 15% 盲目追加）。
    ellipsis_frequency: float = 0.06
    # 旧版 Emoji 参数，仅保留以兼容旧配置；文字 Emoji 统一禁用。
    use_emoji: bool = field(default=False, repr=False)
    emoji_frequency: float = field(default=0.0, repr=False)
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
        """从字典加载，忽略未知字段（避免配置中的多余键报错）"""
        removed = {
            "common_words",
            "filler_words",
            "filler_frequency",
            "use_emoji",
            "emoji_frequency",
        }
        known = {
            k: v
            for k, v in data.items()
            if k in cls.__dataclass_fields__ and k not in removed
        }
        if "direct_max_reply_length" not in data:
            # 新配置默认给明确问答更大空间；直接构造旧版 SpeakingStyle 时
            # 则保留旧的 max_reply_length 语义，由上层按缺省值回退。
            known["direct_max_reply_length"] = 80
        return cls(**known)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "banned_words": self.banned_words,
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "avg_sentence_length_range": self.avg_sentence_length_range,
            "max_reply_length": self.max_reply_length,
            "direct_max_reply_length": self.direct_max_reply_length,
            "use_ellipsis": self.use_ellipsis,
            "ellipsis_frequency": self.ellipsis_frequency,
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
        # 保留旧参数签名，避免第三方初始化代码报错；文字 Emoji 已统一关闭。
        self.emoji_set = []

    def get_style_guide(self) -> str:
        """获取风格指南字符串（发给 LLM 的说话风格约束）"""
        parts = []
        if self.style.formality >= 0.7:
            parts.append("用词相对稳重，但不要写成公文")
        elif self.style.formality <= 0.3:
            parts.append("用熟人聊天的随意口吻，不要客服腔")
        else:
            parts.append("语气自然，正式和随意之间保持平衡")
        if self.style.enthusiasm >= 0.7:
            parts.append("有兴致时语气活一点，但别每句都大惊小怪")
        elif self.style.enthusiasm <= 0.3:
            parts.append("情绪表达克制，不需要强行热闹")
        else:
            parts.append("语气有温度，但不刻意卖热情")
        return "；".join(parts) if parts else ""

    def apply_style(self, text: str, max_reply_length: int = None) -> str:
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

        # 1. 处理标点
        result = self._apply_punctuation(result)

        # 2. 长度调整
        result = self._adjust_length(result, max_reply_length=max_reply_length)

        # 3. 移除禁用词
        result = self._remove_banned_words(result)

        # 文字 Emoji 不属于固定人格特征：既不自动添加，也清理模型偶尔生成的
        # Emoji。表情包图片仍由 meme_manager 独立处理，不受这里影响。
        result = _EMOJI_RE.sub("", result)

        return result

    # 句尾已经自带语气的情况：再追加省略号会拼出「真人！...」「哈哈...」
    # 这类没人会写的组合。
    _NO_ELLIPSIS_TAILS = ("？", "?", "！", "!", "~", "…", "...", "哈", "呵", "嘿")

    def _apply_punctuation(self, text: str) -> str:
        """处理标点符号"""
        # 省略号处理：低频，且只加在"平铺直叙、句尾没有语气标记"的句子后面
        stripped = text.rstrip()
        if (
            self.style.use_ellipsis
            and len(stripped) >= 6
            and not stripped.endswith(self._NO_ELLIPSIS_TAILS)
            and random.random() < max(0.0, min(1.0, self.style.ellipsis_frequency))
        ):
            text = stripped.rstrip(".。") + "..."

        # 感叹号处理
        if self.style.use_exclamation and self.style.enthusiasm > 0.6:
            if random.random() < 0.2:
                text = text.rstrip(".。!！") + random.choice(["!", "!!", "！"])

        return text


    def _adjust_length(self, text: str, max_reply_length: int = None) -> str:
        """调整文本长度 - 超长时按句子边界智能截断（兜底）"""
        max_len = (
            max_reply_length
            if max_reply_length is not None
            else self.style.max_reply_length
        )
        if not text or len(text) <= max_len:
            return text

        # 按句子边界切分（中文/英文标点）
        sentences = re.split(r'(?<=[。！？!?~…])', text)
        result = ""
        for sent in sentences:
            if len(result) + len(sent) > max_len:
                break
            result += sent

        # 单句就超长时硬截断
        if not result.strip():
            result = text[:max_len]

        result = result.strip()
        # 有内容被截掉时补省略号，读起来像"话说到一半"，更像真人
        if result != text and not result.endswith(("…", "...")):
            result = result.rstrip("。，,.!！ ") + "…"

        return result

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
        min_sentence_length=5,
        max_sentence_length=50,
        max_reply_length=20,
        direct_max_reply_length=80,
        avg_sentence_length_range=(10, 30),
        use_ellipsis=True,
        formality=0.3,
        enthusiasm=0.6,
        humor=0.5,
    )

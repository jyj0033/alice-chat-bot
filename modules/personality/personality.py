"""
人格系统
定义 Bot 的性格特质和背景设定
"""
import logging
from dataclasses import dataclass, field
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Personality:
    """Bot 人格设定"""

    # 基础信息
    name: str = "Bot"
    nickname: str = ""  # 群友称呼
    age_range: str = "20-25"  # 年龄范围
    avatar_description: str = ""  # 头像描述

    # Big Five 人格特质 (0.0 - 1.0)
    traits: dict[str, float] = field(default_factory=lambda: {
        "openness": 0.6,           # 开放性：好奇心、新想法接受度
        "conscientiousness": 0.5,  # 尽责性：计划性、责任感
        "extraversion": 0.5,       # 外向性：社交活跃度
        "agreeableness": 0.7,      # 宜人性：友好、合作程度
        "neuroticism": 0.3,        # 神经质：情绪稳定性（低=稳定）
    })

    # 背景设定
    background: str = ""  # 职业、兴趣、背景故事

    # 偏好设定
    interested_topics: list[str] = field(default_factory=list)  # 感兴趣的话题
    bored_topics: list[str] = field(default_factory=list)  # 不感兴趣的话题
    humor_style: str = "dry"  # dry / slapstick / self-deprecating / sarcastic

    # 禁忌话题
    taboo_topics: list[str] = field(default_factory=list)

    # 常用口头禅
    catchphrases: list[str] = field(default_factory=list)

    # emoji 列表
    emoji_set: list[str] = field(default_factory=lambda: ["😅", "🤔", "😂", "👍", "🙄"])

    @classmethod
    def from_yaml(cls, path: str) -> "Personality":
        """从 YAML 文件加载"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except Exception as e:
            logger.error(f"Failed to load personality from {path}: {e}")
            return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "Personality":
        """从字典加载"""
        # 只保留已知的字段
        valid_fields = {
            'name', 'nickname', 'age_range', 'avatar_description',
            'traits', 'background', 'interested_topics', 'bored_topics',
            'humor_style', 'taboo_topics', 'catchphrases', 'emoji_set'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "nickname": self.nickname,
            "age_range": self.age_range,
            "avatar_description": self.avatar_description,
            "traits": self.traits,
            "background": self.background,
            "interested_topics": self.interested_topics,
            "bored_topics": self.bored_topics,
            "humor_style": self.humor_style,
            "taboo_topics": self.taboo_topics,
            "catchphrases": self.catchphrases,
            "emoji_set": self.emoji_set,
        }

    def build_persona_prompt(self) -> str:
        """构建人格提示词"""
        parts = [
            f"你是「{self.name}」，{'也可以叫我' + self.nickname if self.nickname else '这是你的名字'}。",
            f"你是一个{self.age_range}岁左右的人。",
            "",
            "=== 你的性格特点 ===",
            self._describe_traits(),
            "",
        ]

        if self.background:
            parts.extend([
                "=== 你的背景 ===",
                self.background,
                "",
            ])

        if self.interested_topics:
            parts.extend([
                "=== 你感兴趣的话题 ===",
                "、".join(self.interested_topics),
                "（遇到这些话题你更愿意参与讨论）",
                "",
            ])

        if self.bored_topics:
            parts.extend([
                "=== 你不感兴趣的话题 ===",
                "、".join(self.bored_topics),
                "（遇到这些话题你可能会潜水或敷衍）",
                "",
            ])

        parts.extend([
            "=== 说话风格 ===",
            self._describe_speaking_style(),
            "",
            "=== 行为准则 ===",
            "- 你不是AI助手，你是真实的群聊成员",
            "- 不是每条消息都需要回复，选择性参与",
            "- 可以有自己的观点和情绪",
            "- 有时会犯迷糊、说错话",
            "- 说话像真实的人，不要太正式",
        ])

        if self.taboo_topics:
            parts.extend([
                "",
                "=== 禁忌话题 ===",
                f"避免谈论：{'、'.join(self.taboo_topics)}",
            ])

        return "\n".join(parts)

    def _describe_traits(self) -> str:
        """描述人格特质"""
        descriptions = {
            "openness": {
                (0.0, 0.4): "比较传统，不太接受新事物",
                (0.4, 0.6): "对新鲜事物保持开放态度",
                (0.6, 1.0): "非常好奇，喜欢探索新事物",
            },
            "extraversion": {
                (0.0, 0.4): "偏内向，话不多，但朋友间会很健谈",
                (0.4, 0.6): "社交适度，既能独处也能融入群体",
                (0.6, 1.0): "很活跃，喜欢在群里聊天",
            },
            "agreeableness": {
                (0.0, 0.4): "有时候会直接表达不同意见",
                (0.4, 0.6): "愿意合作，但也有自己的底线",
                (0.6, 1.0): "很友善，总是尽量让大家都舒服",
            },
            "conscientiousness": {
                (0.0, 0.4): "比较随性，不喜欢被束缚",
                (0.4, 0.6): "做事有分寸，知道什么时候该认真",
                (0.6, 1.0): "很靠谱，做事有计划有责任感",
            },
            "neuroticism": {
                (0.0, 0.4): "情绪稳定，心态平和",
                (0.4, 0.6): "偶尔会有情绪波动",
                (0.6, 1.0): "比较敏感，情绪起伏较大",
            },
        }

        lines = []
        for name, value in self.traits.items():
            desc_dict = descriptions.get(name, {})
            desc = ""
            for (low, high), text in desc_dict.items():
                if low <= value < high:
                    desc = text
                    break
            if not desc:
                desc = f"特质{name}值为{value}"
            lines.append(f"- {name}: {desc} (值: {value:.1f})")
        return "\n".join(lines)

    def _describe_speaking_style(self) -> str:
        """描述说话风格"""
        parts = []

        # 句长
        extraversion = self.traits.get("extraversion", 0.5)
        if extraversion > 0.6:
            parts.append("- 说话比较活泼，句子可以稍长")
        elif extraversion < 0.4:
            parts.append("- 话不多，经常用短句")
        else:
            parts.append("- 句子长度适中")

        # emoji
        if self.emoji_set:
            parts.append(f"- 偶尔会使用 emoji：{' '.join(self.emoji_set[:5])}")

        # 口头禅
        if self.catchphrases:
            parts.append(f"- 偶尔会说口头禅：{'、'.join(self.catchphrases[:3])}")

        # 幽默风格
        humor_descriptions = {
            "dry": "冷幽默，喜欢一本正经地说笑话",
            "slapstick": "比较搞笑，喜欢夸张的表达",
            "self-deprecating": "自嘲式幽默",
            "sarcastic": "有点讽刺，喜欢阴阳怪气",
        }
        if self.humor_style in humor_descriptions:
            parts.append(f"- 幽默风格：{humor_descriptions[self.humor_style]}")

        return "\n".join(parts)

    def is_topic_interesting(self, topic: str) -> float:
        """
        判断话题感兴趣程度
        返回 0.0 - 1.0
        """
        topic_lower = topic.lower()

        # 检查禁忌话题
        for taboo in self.taboo_topics:
            if taboo.lower() in topic_lower:
                return 0.0

        # 检查感兴趣话题
        for interested in self.interested_topics:
            if interested.lower() in topic_lower:
                return 0.8 + 0.2 * self.traits.get("openness", 0.5)

        # 检查不感兴趣话题
        for bored in self.bored_topics:
            if bored.lower() in topic_lower:
                return 0.1

        return 0.5  # 中等兴趣

    def get_speaking_enthusiasm(self) -> float:
        """获取说话热情度"""
        base = 0.5
        base += (self.traits.get("extraversion", 0.5) - 0.5) * 0.4
        return max(0.1, min(1.0, base))

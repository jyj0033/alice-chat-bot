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

    # 旧版配置兼容字段：不再注入人格提示词，也不参与回复生成。
    # 保留属性是为了让旧代码读取时不报错；from_dict 会主动忽略旧配置值。
    catchphrases: list[str] = field(default_factory=list, repr=False)
    emoji_set: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Personality":
        """从 YAML 文件加载"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            # 配置文件通常还会包含 speaking_style 等人格旁支字段，不能直接
            # `cls(**data)`，否则一个未知字段就让整个人格回退成默认值。
            return cls.from_dict(data if isinstance(data, dict) else {})
        except Exception as e:
            logger.error(f"从 {path} 加载人格配置失败：{e}")
            return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "Personality":
        """从字典加载"""
        # 只保留已知的字段
        valid_fields = {
            'name', 'nickname', 'age_range', 'avatar_description',
            'traits', 'background', 'interested_topics', 'bored_topics',
            'humor_style', 'taboo_topics'
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
        }

    # 性格的呈现顺序：先社交表现、再做事方式、最后情绪，读起来像在介绍一个人，
    # 而不是罗列五个心理学维度。
    _TRAIT_ORDER = (
        "extraversion", "openness", "agreeableness",
        "conscientiousness", "neuroticism",
    )

    def build_persona_prompt(self) -> str:
        """构建人格提示词。

        刻意避开「=== 标题 ===」+ 字段名 + 数值（如 `agreeableness: … (值: 0.8)`）
        的规格书写法：那种格式本身就在告诉模型"我是一份被配置出来的设定"，
        会把语气推回助手腔。这里写成一段自然的人物介绍——性格用行为描述，
        不出现心理学术语和数值，约束也用平常话说。
        """
        who = f"你叫{self.name}"
        if self.nickname:
            who += f"，熟人喊你{self.nickname}"
        who += f"，{self.age_range}岁。"

        parts = [who]

        if self.background:
            parts.append(self.background.strip())

        if self.avatar_description:
            parts.append(
                f"外在给人的感觉是：{self.avatar_description.strip()}。"
                "这是角色背景，除非别人问到，不要主动介绍自己的外貌。"
            )

        traits = self._describe_traits()
        if traits:
            parts.append(f"性子上，{traits}这些是你本来的样子，不用刻意演出来。")

        preference = []
        if self.interested_topics:
            preference.append(
                f"聊到{self._readable_topics(self.interested_topics)}这些你会来劲，"
                "觉得有合适内容时自然接一句"
            )
        if self.bored_topics:
            preference.append(
                f"碰上{self._readable_topics(self.bored_topics)}就提不起兴趣，"
                "一般划过去不搭理"
            )
        if preference:
            parts.append("；".join(preference) + "。")

        parts.append(self._describe_speaking_style())

        # 人设只描述这个人平时是什么样，具体本轮怎么回复交给对话上下文决定。
        habits = [
            "你像群里普通成员一样聊天。有人拿你是不是机器人开玩笑，就顺着损回去或装傻，不要一本正经承认自己是机器人、AI 或程序。",
            "聊天看当下的兴致：有自然想接的话就说，没什么想补充时安静待着也很正常。",
            "你有自己的想法和脾气，不用总迎合别人；不确定或记不清的事就坦白说。",
            "说话时不用替别人解说群聊。",
        ]
        if self.taboo_topics:
            habits.append(f"{'、'.join(self.taboo_topics)}这些不聊。")
        parts.append("\n".join(habits))

        return "\n\n".join(p for p in parts if p and p.strip())

    @staticmethod
    def _readable_topics(topics: list[str], cap: int = 9) -> str:
        """把话题配置渲染成人话。

        配置里的话题为了让关键词匹配吃到更多词，常写成
        「游戏（什么类型都聊，手游端游主机都OK）」这种带括号和斜杠的描述串。
        原样塞进人格介绍里，一眼就是从配置文件粘过来的。这里剥掉补充说明、
        拆开并列项，只留干净的词。
        """
        import re

        primary: list[str] = []
        extra: list[str] = []
        for topic in topics or []:
            text = re.sub(r"[（(][^）)]*[）)]", "", str(topic or ""))
            pieces = [p.strip() for p in re.split(r"[/、,，]+", text) if p.strip()]
            for i, piece in enumerate(pieces):
                bucket = primary if i == 0 else extra
                if piece not in primary and piece not in extra:
                    bucket.append(piece)
        # 每个配置项先各出一个主词，保证没有话题被截断丢掉，再用并列项补满
        return "、".join((primary + extra)[:cap])

    def _describe_traits(self) -> str:
        """把性格特质描述成一串行为倾向（不出现字段名和数值）。"""
        descriptions = {
            "openness": {
                (0.0, 0.4): "对新东西比较无感，习惯待在熟悉的圈子里",
                (0.4, 0.6): "对新鲜事物保持开放态度",
                (0.6, 1.0): "好奇心重，什么新东西都想试试",
            },
            "extraversion": {
                (0.0, 0.4): "偏内向，话不多，但跟熟人会放得开",
                (0.4, 0.6): "不算话痨，但群里热闹起来也会掺一脚",
                (0.6, 1.0): "挺爱说话，群里有话题就想凑过去",
            },
            "agreeableness": {
                (0.0, 0.4): "不太顾及别人面子，有意见直接说",
                (0.4, 0.6): "好相处，但也有自己的底线",
                (0.6, 1.0): "好说话，不爱跟人抬杠",
            },
            "conscientiousness": {
                (0.0, 0.4): "比较随性，不喜欢被安排",
                (0.4, 0.6): "有分寸，知道什么时候该认真",
                (0.6, 1.0): "做事靠谱，答应的事会记着",
            },
            "neuroticism": {
                (0.0, 0.4): "心态稳，不太容易炸毛",
                (0.4, 0.6): "偶尔会有点小情绪",
                (0.6, 1.0): "比较敏感，情绪起伏大",
            },
        }

        ordered = [n for n in self._TRAIT_ORDER if n in self.traits]
        ordered += [n for n in self.traits if n not in self._TRAIT_ORDER]

        phrases = []
        for name in ordered:
            buckets = descriptions.get(name)
            if not buckets:
                continue
            try:
                # 上界是开区间，1.0 会落空，这里收一下避免描述缺失
                value = min(max(float(self.traits.get(name, 0.5)), 0.0), 0.999)
            except (TypeError, ValueError):
                continue
            for (low, high), text in buckets.items():
                if low <= value < high:
                    phrases.append(text)
                    break

        return "；".join(phrases) + "。" if phrases else ""

    def _describe_speaking_style(self) -> str:
        """把表达习惯写成自然倾向，不把性格误写成固定句长。"""
        parts = []

        extraversion = self.traits.get("extraversion", 0.5)
        if extraversion > 0.6:
            parts.append("语气轻快，愿意接话，碰上感兴趣的话题会多聊两句")
        elif extraversion < 0.4:
            parts.append("偏安静，平时话不多，和熟人聊开了会放松些")
        else:
            parts.append("说话随意自然，长短看话题和当下心情")

        humor_descriptions = {
            "dry": "偏冷幽默，偶尔一本正经地说点怪话",
            "slapstick": "有时会把玩笑开得夸张一点",
            "self-deprecating": "更爱自嘲，不太拿别人开涮",
            "sarcastic": "偶尔有点阴阳怪气，但不刻薄",
        }
        if self.humor_style in humor_descriptions:
            parts.append(humor_descriptions[self.humor_style])

        return "说话上，" + "；".join(parts) + "。"

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

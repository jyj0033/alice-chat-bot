"""
社交感知系统
包括触发检测、发言欲望计算、氛围分析
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SocialContext:
    """社交上下文"""

    # 消息信息
    message_content: str = ""
    sender_id: str = ""
    sender_name: str = ""
    group_id: str = ""
    session_id: str = ""

    # 触发因素
    mentioned_me: bool = False      # @了我
    reply_to_me: bool = False      # 回复了我
    is_direct_question: bool = False  # 是直接提问
    contains_keywords: bool = False  # 包含关键词

    # 群聊氛围
    group_activity: float = 0.5    # 群活跃度 0-1
    recent_message_count: int = 0   # 最近消息数

    # 话题相关
    topic_relevance: float = 0.5   # 话题相关性 0-1
    topic_familiarity: float = 0.5  # 话题熟悉度 0-1

    # 情感
    sender_sentiment: float = 0.5   # 发送者情感 -1到1

    # 其他
    is_emergency: bool = False      # 是否紧急
    extra: dict = field(default_factory=dict)


class TriggerDetector:
    """触发检测器"""

    def __init__(
        self,
        bot_nickname: str = "",
        nicknames: list[str] = None,
        trigger_keywords: list[str] = None
    ):
        self.bot_nickname = bot_nickname
        self.nicknames = [n for n in (nicknames or []) if n]
        self.trigger_keywords = trigger_keywords or []

    def detect(self, context: SocialContext) -> dict:
        """
        检测触发因素

        Returns:
            dict: {
                "forced_trigger": bool,  # 强制触发（必须回复）
                "priority": float,       # 优先级 0-1
                "reasons": list[str]     # 触发原因
            }
        """
        reasons = []
        priority = 0.0

        # 1. @了我 - 高优先级
        if context.mentioned_me:
            reasons.append("被@")
            priority += 0.7

        # 2. 回复了我 - 高优先级
        if context.reply_to_me:
            reasons.append("被回复")
            priority += 0.6

        # 3. 直接提问 - 中高优先级
        question_patterns = [
            r"[？?]$",  # 句尾问号
            r"是不是",
            r"能不能",
            r"怎么",
            r"为什么",
            r"你觉得",
            r"你说",
            r"什么",  # 中文口语常省略问号："吃什么""干啥"等
            r"啥",
            r"哪[个儿]",
        ]
        for pattern in question_patterns:
            if re.search(pattern, context.message_content):
                reasons.append("直接提问")
                priority += 0.3
                context.is_direct_question = True
                break

        # 4. 触发关键词
        for keyword in self.trigger_keywords:
            if keyword.lower() in context.message_content.lower():
                reasons.append(f"关键词「{keyword}」")
                priority += 0.4
                context.contains_keywords = True
                break

        # 5. 昵称检测（主名 + 简称，如"爱丽丝"、"小爱"）
        matched_names = []
        if self.bot_nickname and self.bot_nickname in context.message_content:
            matched_names.append(self.bot_nickname)
        for nick in self.nicknames:
            if nick and nick in context.message_content:
                matched_names.append(nick)
        if matched_names:
            reasons.append(f"昵称「{matched_names[0]}」")
            priority += 0.3

        # 6. 紧急标记
        # 「急」「救命」在群聊里多为玩梗（"他急了""笑死救命"），不能一律当紧急：
        # 单字「急」不再触发；「救命」伴随笑点词时视为玩梗，不算紧急。
        emergency_markers = ["紧急", "SOS", "帮帮忙"]
        laugh_markers = ("笑", "哈", "草", "乐", "xs", "hhh")
        is_emergency = any(
            marker in context.message_content for marker in emergency_markers
        )
        if not is_emergency and "救命" in context.message_content:
            lowered = context.message_content.lower()
            if not any(l in lowered for l in laugh_markers):
                is_emergency = True
        if is_emergency:
            reasons.append("紧急")
            priority += 0.5
            context.is_emergency = True

        # 归一化优先级
        priority = min(1.0, priority)

        return {
            "forced_trigger": priority >= 0.7,
            "priority": priority,
            "reasons": reasons
        }


class AmbienceAnalyzer:
    """氛围分析器"""

    def __init__(self):
        self._recent_messages = []
        self._max_history = 100

    def analyze(self, context: SocialContext) -> dict:
        """
        分析群聊氛围

        Returns:
            dict: {
                "activity": float,      # 活跃度 0-1
                "mood": str,            # 氛围描述
                "tension": float,      # 紧张度 0-1
            }
        """
        # 简化实现：基于传入的参数
        activity = context.group_activity

        # 判断氛围
        mood = "normal"
        if activity > 0.7:
            mood = "active"
        elif activity < 0.2:
            mood = "quiet"

        # 紧张度（简化实现）
        tension = 0.0
        tension_markers = ["吵", "骂", "争", "吵", "滚", "傻"]
        for marker in tension_markers:
            if marker in context.message_content:
                tension += 0.2

        return {
            "activity": activity,
            "mood": mood,
            "tension": min(1.0, tension)
        }

    def get_activity_modifier(self, activity: float) -> float:
        """
        根据活跃度获取修正因子

        活跃度高时降低发言欲望（避免抢话）
        活跃度低时提升发言欲望（活跃气氛）
        """
        if activity < 0.2:
            # 群很冷清，更愿意说话
            return 1.3
        elif activity > 0.8:
            # 群很热闹，倾向潜水
            return 0.7
        return 1.0


class TopicAnalyzer:
    """话题分析器

    配置里的话题常写成描述句（如「游戏（什么类型都聊，手游端游主机都OK）」），
    整串子串匹配永远命不中群消息。这里在启动时把每个话题拆成关键词集合
    （分隔符切分 + jieba 分词），匹配时按关键词命中计算相关度。
    """

    _TOPIC_SPLIT_RE = re.compile(r"[／/、，,。．.！!？?：:（）()\[\]【】\s]+")
    # 描述句里的修饰词/口水词，不能当成话题关键词（否则"无脑""能量"会乱命中）
    _TOPIC_STOPWORDS = {
        "什么", "类型", "都聊", "都可以", "一些", "有点", "相关", "之类",
        "东西", "话题", "内容", "偶尔", "各种", "喜欢", "不要", "无脑",
        "连续", "能量", "ok", "都ok",
    }

    def __init__(self, interested_topics: list[str] = None, bored_topics: list[str] = None):
        self.interested_topics = interested_topics or []
        self.bored_topics = bored_topics or []
        self._interested_keywords = self._extract_keywords(self.interested_topics)
        self._bored_keywords = self._extract_keywords(self.bored_topics)

    @classmethod
    def _extract_keywords(cls, topics: list[str]) -> set[str]:
        """把话题描述串拆成可匹配的关键词集合。"""
        keywords: set[str] = set()
        for topic in topics or []:
            text = str(topic or "").strip().lower()
            if not text:
                continue
            for frag in cls._TOPIC_SPLIT_RE.split(text):
                frag = frag.strip()
                if not frag:
                    continue
                # 短片段整体保留（「抽卡」「微商」本身就是关键词）
                if 2 <= len(frag) <= 6:
                    keywords.add(frag)
                for token in cls._tokenize(frag):
                    if len(token) >= 2:
                        keywords.add(token)
        return {k for k in keywords if k not in cls._TOPIC_STOPWORDS}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        try:
            import jieba
            return [w.strip() for w in jieba.cut(text) if w.strip()]
        except Exception:
            return [text]

    def analyze_relevance(self, message: str) -> float:
        """分析话题相关性（0.5 为中性；命中兴趣词升、厌倦词降）"""
        message_lower = (message or "").lower()
        relevance = 0.5

        interested_hits = sum(
            1 for kw in self._interested_keywords if kw in message_lower
        )
        bored_hits = sum(1 for kw in self._bored_keywords if kw in message_lower)

        # 首个命中给主要权重，多词命中小幅叠加（封顶，避免一句话多关键词爆表）
        if interested_hits:
            relevance += min(0.4, 0.3 + 0.05 * (interested_hits - 1))
        if bored_hits:
            relevance -= min(0.4, 0.3 + 0.05 * (bored_hits - 1))

        return max(0.0, min(1.0, relevance))

    def analyze_familiarity(self, message: str) -> float:
        """
        分析话题熟悉度
        简化实现：基于关键词匹配
        """
        # 常见话题
        common_topics = [
            "游戏", "电影", "音乐", "编程", "学习",
            "工作", "生活", "美食", "旅游", "运动",
            "技术", "AI", "科技", "新闻", "八卦"
        ]

        familiarity = 0.5
        for topic in common_topics:
            if topic in message:
                familiarity += 0.1

        # 专业术语降低熟悉度
        technical_terms = [
            "算法", "架构", "分布式", "微服务", "神经网络",
            "机器学习", "深度学习", "编译原理"
        ]
        for term in technical_terms:
            if term in message:
                familiarity -= 0.1

        return max(0.1, min(1.0, familiarity))


class SocialAwarenessManager:
    """社交感知管理器"""

    def __init__(
        self,
        bot_nickname: str = "",
        interested_topics: list[str] = None,
        bored_topics: list[str] = None
    ):
        # 注意：不创建内部的 trigger_detector，避免循环调用
        self._bot_nickname = bot_nickname
        self.ambience_analyzer = AmbienceAnalyzer()
        self.topic_analyzer = TopicAnalyzer(interested_topics, bored_topics)

    def analyze(self, context: SocialContext) -> SocialContext:
        """完整分析社交上下文"""

        # 触发检测已经在外部完成，结果在 context.extra["trigger"] 中
        # 这里只做氛围和话题分析

        # 氛围分析
        semantic_content = context.message_content
        # 氛围只依据发送者外层发言；转发内容中的争吵不代表当前群正在争吵。
        outer_text = context.extra.get("outer_text")
        if outer_text is not None:
            context.message_content = outer_text
        ambience = self.ambience_analyzer.analyze(context)
        context.message_content = semantic_content
        context.extra["ambience"] = ambience

        # 话题分析
        context.topic_relevance = self.topic_analyzer.analyze_relevance(context.message_content)
        context.topic_familiarity = self.topic_analyzer.analyze_familiarity(context.message_content)

        return context

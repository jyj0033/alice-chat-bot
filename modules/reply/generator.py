"""
回复生成器
调用 LLM 生成回复，并应用说话风格
"""
import asyncio
import logging
import random
from typing import Optional, Dict, Any

from modules.llm.base import ChatRequest, ChatResponse
from modules.personality.speaking_style import SpeakingStyleManager, create_default_style
from modules.personality.emotional_state import EmotionalState

logger = logging.getLogger(__name__)


class ResponseFilter:
    """回复过滤器 - 检测和过滤不合适的回复"""

    def __init__(self):
        # 敏感词列表（示例，实际使用时应配置化）
        self.sensitive_words = [
            "政治敏感词1", "政治敏感词2",  # 请根据实际情况添加
        ]

        # 最小/最大回复长度
        self.min_length = 1
        self.max_length = 500

    def filter(self, text: str) -> tuple[bool, str]:
        """
        过滤回复

        Returns:
            (是否通过, 错误信息或过滤后文本)
        """
        if not text:
            return False, "空回复"

        # 检查长度
        if len(text) < self.min_length:
            return False, "回复过短"

        if len(text) > self.max_length:
            text = text[:self.max_length]

        # 检查敏感词
        for word in self.sensitive_words:
            if word in text:
                logger.warning(f"Reply contains sensitive word: {word}")
                return False, f"包含敏感词: {word}"

        return True, text

    def needs_review(self, text: str) -> bool:
        """检查是否需要人工审核"""
        # 某些关键词触发审核
        review_keywords = ["钱", "转账", "密码", "账号"]
        return any(kw in text for kw in review_keywords)


class ReplyGenerator:
    """回复生成器 - 人格驱动的智能回复"""

    def __init__(
        self,
        llm_provider,
        personality_prompt: str = "",
        speaking_style_manager: SpeakingStyleManager = None,
        thinking_delay: float = 2.0
    ):
        """
        初始化

        Args:
            llm_provider: LLM 提供者
            personality_prompt: 人格提示词
            speaking_style_manager: 说话风格管理器
            thinking_delay: 基础思考延迟（秒）
        """
        self.llm = llm_provider
        self.personality_prompt = personality_prompt
        self.style_manager = speaking_style_manager or SpeakingStyleManager(create_default_style())
        self.base_thinking_delay = thinking_delay
        self.response_filter = ResponseFilter()

        # 统计
        self.replies_generated = 0
        self.replies_filtered = 0

    async def generate(
        self,
        context_prompt: str,
        current_message: str,
        emotional_state: EmotionalState = None,
        temperature: float = 0.8,
        session_context: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        生成回复

        Args:
            context_prompt: 上下文提示
            current_message: 当前消息
            emotional_state: 情感状态
            temperature: 生成温度
            session_context: 会话上下文（群ID、用户ID等）

        Returns:
            str: 生成的回复，如果被过滤则返回 None
        """
        # 1. 构建请求
        request = self._build_request(
            context_prompt,
            current_message,
            emotional_state,
            session_context
        )

        # 2. 调用 LLM
        try:
            response = await self.llm.chat(request)
            reply = response.content.strip()
        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            reply = self._get_fallback_reply()

        # 3. 过滤回复
        passed, result = self.response_filter.filter(reply)
        if not passed:
            logger.info(f"Reply filtered: {result}")
            self.replies_filtered += 1
            return None

        self.replies_generated += 1

        # 4. 应用说话风格
        reply = self.style_manager.apply_style(result)

        # 5. 添加随机延迟（模拟打字）
        delay = self._calculate_delay(emotional_state)
        await asyncio.sleep(delay)

        return reply

    def _build_request(
        self,
        context_prompt: str,
        current_message: str,
        emotional_state: EmotionalState = None,
        session_context: Dict[str, Any] = None
    ) -> ChatRequest:
        """构建 LLM 请求"""

        # 根据情感状态调整温度
        temp = 0.8
        if emotional_state:
            # 兴奋时更有创意
            if emotional_state.energy > 0.8:
                temp = 0.9
            # 疲惫时更保守
            elif emotional_state.energy < 0.3:
                temp = 0.6

        request = ChatRequest(
            temperature=temp,
            max_tokens=500,
            top_p=0.9,
        )

        # 系统提示 - 人格设定
        if self.personality_prompt:
            request.add_system(self.personality_prompt)

        # 添加说话风格指导
        style_guide = self.style_manager.get_style_guide()
        if style_guide:
            request.add_system(f"说话风格指导：{style_guide}")

        # 添加情感状态指导
        if emotional_state:
            emotion_guide = self._get_emotion_guide(emotional_state)
            if emotion_guide:
                request.add_system(emotion_guide)

        # 添加会话上下文
        if session_context:
            context_info = self._format_session_context(session_context)
            request.add_system(f"当前情境：{context_info}")

        # 对话内容
        if context_prompt:
            request.add_user(
                f"【对话历史】\n{context_prompt}\n\n"
                f"【当前消息】{current_message}\n\n"
                f"请以爱丽丝的身份自然回复。"
            )
        else:
            request.add_user(f"{current_message}")

        return request

    def _format_session_context(self, context: Dict[str, Any]) -> str:
        """格式化会话上下文"""
        parts = []

        if group_id := context.get("group_id"):
            parts.append(f"群号：{group_id}")

        if user_name := context.get("user_name"):
            parts.append(f"发送者：{user_name}")

        return "，".join(parts) if parts else "普通会话"

    def _get_emotion_guide(self, state: EmotionalState) -> str:
        """根据情感状态生成指导"""
        guides = []

        if state.energy > 0.8:
            guides.append("你精力充沛，回复可以更积极热情")
        elif state.energy < 0.3:
            guides.append("你有点累了，回复可以简短一些")

        if state.engagement > 0.7:
            guides.append("你对当前话题很感兴趣，可以多说几句")
        elif state.engagement < 0.3:
            guides.append("你对这个话题兴趣一般，保持简短")

        # 情绪影响
        if hasattr(state, 'mood'):
            if state.mood > 0.5:
                guides.append("你心情不错，语气可以更轻松愉快")
            elif state.mood < -0.5:
                guides.append("你心情不太好，回避沉重话题")

        return "，".join(guides) if guides else ""

    def _calculate_delay(self, emotional_state: EmotionalState = None) -> float:
        """计算思考延迟（模拟真人打字前的思考）"""
        delay = self.style_manager.get_random_delay(self.base_thinking_delay)

        # 情感状态影响延迟
        if emotional_state:
            delay *= emotional_state.get_thinking_delay_multiplier()

        return delay

    def _get_fallback_reply(self) -> str:
        """获取备用回复（LLM调用失败时）"""
        fallbacks = [
            "啊？刚才没听清再说一遍？",
            "这我还真不知道诶",
            "有点困，刚才说的啥",
            "抱歉走神了，你再说一遍？",
            "等等让我想想...",
            "emmm...",
            "好像有点道理？",
        ]
        return random.choice(fallbacks)

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "generated": self.replies_generated,
            "filtered": self.replies_filtered,
            "pass_rate": (
                (self.replies_generated - self.replies_filtered) / self.replies_generated
                if self.replies_generated > 0 else 0
            )
        }


class ThinkingDelay:
    """思考延迟管理器 - 模拟真人打字前的思考时间"""

    def __init__(
        self,
        base_delay: float = 2.0,
        min_delay: float = 0.5,
        max_delay: float = 10.0
    ):
        self.base_delay = base_delay
        self.min_delay = min_delay
        self.max_delay = max_delay

    async def wait(
        self,
        message_length: int = 0,
        topic_familiarity: float = 0.5,
        emotional_modifier: float = 1.0
    ) -> float:
        """
        等待思考延迟

        Args:
            message_length: 消息长度（影响思考时间）
            topic_familiarity: 话题熟悉度（越不熟悉越长）
            emotional_modifier: 情感修正

        Returns:
            float: 实际等待时间
        """
        # 基础延迟
        delay = self.base_delay

        # 消息长度影响（长消息需要更长思考）
        if message_length > 100:
            delay += (message_length - 100) * 0.01
        elif message_length > 50:
            delay += (message_length - 50) * 0.005

        # 话题熟悉度影响
        # 不熟悉的话题需要更长的思考时间
        unfamiliarity = 1 - topic_familiarity
        delay += unfamiliarity * 2.0

        # 应用情感修正
        delay *= emotional_modifier

        # 添加随机抖动
        import random
        jitter = random.uniform(-0.5, 0.5)
        delay = max(self.min_delay, min(self.max_delay, delay + jitter))

        # 实际等待
        await asyncio.sleep(delay)

        return delay

    def estimate_delay(
        self,
        message_length: int = 0,
        topic_familiarity: float = 0.5,
        emotional_modifier: float = 1.0
    ) -> float:
        """估算延迟时间（不实际等待）"""
        delay = self.base_delay

        if message_length > 100:
            delay += (message_length - 100) * 0.01
        elif message_length > 50:
            delay += (message_length - 50) * 0.005

        unfamiliarity = 1 - topic_familiarity
        delay += unfamiliarity * 2.0
        delay *= emotional_modifier

        return max(self.min_delay, min(self.max_delay, delay))

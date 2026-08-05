"""
爱丽丝 (Alice) - 群聊AI伙伴
基于 AstrBot 设计的增强版主入口
"""
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.event_bus import EventBus, Event, EventType
from core.adapter.qq_adapter import QQAdapter
from core.adapter.base import Message

from modules.llm.openai_provider import create_provider, LLMProvider
from modules.memory.storage import MemoryStorage, AsyncMemoryStorage, Memory
from modules.memory.context import ContextManager

from modules.personality.personality import Personality
from modules.personality.emotional_state import EmotionalManager
from modules.personality.speaking_style import SpeakingStyleManager, SpeakingStyle
from modules.personality.typo import TypoGenerator

from modules.social.awareness import SocialAwarenessManager, SocialContext, TriggerDetector
from modules.social.attention import AttentionManager, AttentionKeywordsDetector
from modules.social.fatigue import FatigueManager
from modules.social.enhanced_decider import EnhancedSpeakingDecider

from modules.reply.generator import ReplyGenerator, ThinkingDelay, ResponseFilter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GroupChatBot:
    """
    爱丽丝 (Alice) - 群聊AI伙伴

    核心理念：模拟真实人类发言行为
    1. 发言概率系统（初始2%，有人回复后80%）
    2. 注意力机制（追踪用户关注度）
    3. 情绪系统（检测正负面关键词）
    4. 疲劳系统（长时间对话后概率下降）
    5. 错字生成器（模拟真人打字错误）
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = None
        self.personality = None

        # 核心组件
        self.event_bus: Optional[EventBus] = None
        self.llm_providers: dict[str, LLMProvider] = {}  # 多 provider 支持
        self.active_provider_id: str = "primary"  # 当前激活的 provider
        self.qq_adapter: Optional[QQAdapter] = None

        # 记忆系统
        self.memory_storage: Optional[AsyncMemoryStorage] = None
        self.context_manager: Optional[ContextManager] = None

        # 人格系统
        self.emotional_manager: Optional[EmotionalManager] = None
        self.speaking_style_manager: Optional[SpeakingStyleManager] = None
        self.typo_generator: Optional[TypoGenerator] = None

        # 社交系统
        self.social_awareness: Optional[SocialAwarenessManager] = None
        self.trigger_detector: Optional[TriggerDetector] = None
        self.attention_manager: Optional[AttentionManager] = None
        self.attention_keywords_detector: Optional[AttentionKeywordsDetector] = None
        self.fatigue_manager: Optional[FatigueManager] = None
        self.speaking_decider: Optional[EnhancedSpeakingDecider] = None

        # 回复生成
        self.reply_generator: Optional[ReplyGenerator] = None
        self.thinking_delay: Optional[ThinkingDelay] = None
        self.response_filter: Optional[ResponseFilter] = None

        # 状态
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._start_time: float = 0

    async def initialize(self) -> None:
        """初始化 Bot"""
        logger.info("Initializing Alice (爱丽丝)...")
        self._start_time = time.time()

        # 1. 加载配置
        self._load_config()
        logger.info("✓ Config loaded")

        # 2. 初始化事件总线
        self.event_bus = EventBus()
        self._tasks.append(asyncio.create_task(self.event_bus.start()))
        logger.info("✓ EventBus initialized")

        # 3. 初始化 LLM
        self._init_llm()
        logger.info("✓ LLM provider initialized")

        # 4. 初始化记忆系统
        self._init_memory()
        logger.info("✓ Memory system initialized")

        # 5. 初始化人格系统
        self._init_personality()
        logger.info("✓ Personality system initialized")

        # 6. 初始化社交感知
        self._init_social()
        logger.info("✓ Social awareness initialized")

        # 7. 初始化回复生成
        self._init_reply_generator()
        logger.info("✓ Reply generator initialized")

        # 8. 初始化 QQ 适配器
        self._init_qq_adapter()
        logger.info("✓ QQ adapter initialized")

        elapsed = time.time() - self._start_time
        logger.info(f"GroupChatBot initialized successfully in {elapsed:.2f}s!")

    def _load_config(self) -> None:
        """加载配置，如果没有则创建默认配置"""
        config_file = Path(self.config_path)

        # 确保配置目录存在
        config_file.parent.mkdir(parents=True, exist_ok=True)

        if not config_file.exists():
            # 创建默认配置文件
            logger.info("No config file found, creating default configuration...")
            default_config = self._get_default_config()
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)
            self.config = default_config
            logger.info(f"✓ Default config created at {config_file}")
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}

        # 加载人格配置
        personality_config = self.config.get("personality", {})
        self.personality = Personality.from_dict(personality_config)

        logger.info(f"Bot name: {self.personality.name}, nickname: {self.personality.nickname}")

    def _init_llm(self) -> None:
        """初始化 LLM providers - 支持 llm 和 providers 两个配置结构"""
        llm_config = self.config.get("llm", {})
        providers_config = self.config.get("providers", {})
        self.llm_providers = {}

        # 合并两个配置源，providers 优先级更高
        all_providers = {**llm_config, **providers_config}

        # 初始化所有 providers
        for name, provider_config in all_providers.items():
            if isinstance(provider_config, dict) and provider_config.get("enabled", True):
                try:
                    provider = create_provider(
                        provider_config.get("provider_type", "openai"),
                        {
                            "api_key": provider_config.get("api_key", ""),
                            "base_url": provider_config.get("base_url", "https://api.openai.com/v1"),
                            "model": provider_config.get("model", "gpt-4o"),
                            "timeout": provider_config.get("timeout", 120),
                            "temperature": provider_config.get("temperature", 0.8),
                            "max_tokens": provider_config.get("max_tokens", 2000),
                        }
                    )
                    self.llm_providers[name] = provider
                    logger.info(f"✓ LLM provider '{name}': {provider_config.get('base_url')}, model: {provider_config.get('model')}")
                except Exception as e:
                    logger.error(f"✗ Failed to init provider '{name}': {e}")

        # 设置激活的 provider（优先级：primary > claude > siliconflow > 其他）
        priority_order = ["primary", "minimax", "siliconflow", "nvidia", "ark"]
        self.active_provider_id = None
        for priority_name in priority_order:
            if priority_name in self.llm_providers:
                self.active_provider_id = priority_name
                break

        if not self.active_provider_id and self.llm_providers:
            self.active_provider_id = next(iter(self.llm_providers))

        if not self.llm_providers:
            logger.error("No LLM providers available!")

    def get_active_provider(self) -> Optional[LLMProvider]:
        """获取当前激活的 provider"""
        return self.llm_providers.get(self.active_provider_id)

    def _init_memory(self) -> None:
        """初始化记忆系统"""
        memory_config = self.config.get("memory", {})

        self.memory_storage = AsyncMemoryStorage(
            MemoryStorage(memory_config.get("db_path", "data/memory.db"))
        )

        # 嵌入+重排服务（无 key 时自动回退 TF-IDF）
        from modules.memory.embedding import EmbeddingRerankService
        embed_config = memory_config.get("embedding", {}) or {}
        embed_service = EmbeddingRerankService(embed_config)
        self.memory_storage._storage.set_embedding_service(embed_service)
        if embed_service.enabled:
            logger.info(f"✓ 嵌入服务已启用: {embed_service.embed_model} + rerank {embed_service.rerank_model}")
        else:
            logger.info("嵌入服务未配置（无 api_key），记忆检索使用 TF-IDF 回退")

        self.context_manager = ContextManager(
            max_messages=memory_config.get("context_window_size", 30),
            max_age_hours=memory_config.get("context_max_age_hours", 2)
        )

        # 记忆检索参数（向量检索 + 时间衰减）
        self.memory_search_top_k = memory_config.get("retrieval_top_k", 5)
        self.memory_half_life_days = memory_config.get("half_life_days", 30)
        self.memory_similarity_weight = memory_config.get("similarity_weight", 0.85)
        # 衰减清理任务的上次执行时间（初始为"从未"）
        self._last_decay_run: float = 0.0

    def _init_personality(self) -> None:
        """初始化人格系统"""
        speaking_config = self.config.get("speaking", {})

        # 情感管理器
        emotion_config = self.config.get("emotion", {})
        self.emotional_manager = EmotionalManager(
            decay_halflife=emotion_config.get("decay_halflife", 600),
            positive_keywords=emotion_config.get("positive_keywords", []),
            negative_keywords=emotion_config.get("negative_keywords", []),
        )

        # 说话风格
        style_config = self.config.get("personality", {}).get("speaking_style", {})
        speaking_style = SpeakingStyle.from_dict(style_config) if style_config else SpeakingStyle()

        self.speaking_style_manager = SpeakingStyleManager(
            speaking_style,
            emoji_set=self.personality.emoji_set
        )

        # 错字生成器
        typing_config = self.config.get("typing_style", {})
        self.typo_generator = TypoGenerator(
            typo_error_rate=typing_config.get("typo_error_rate", 0.04),
            homophones=typing_config.get("homophones", {}),
            min_chinese_chars=typing_config.get("min_chinese_chars", 3),
            min_message_length=typing_config.get("min_message_length", 10),
        )

    def _init_social(self) -> None:
        """初始化社交感知"""
        speaking_config = self.config.get("speaking", {})
        attention_config = self.config.get("attention", {})
        emotion_config = self.config.get("emotion", {})
        fatigue_config = self.config.get("fatigue", {})
        typing_config = self.config.get("typing_style", {})

        bot_nickname = self.personality.name
        trigger_keywords = speaking_config.get("trigger_keywords", [])

        # 触发检测器
        self.trigger_detector = TriggerDetector(
            bot_nickname=bot_nickname,
            trigger_keywords=trigger_keywords
        )

        # 注意力管理器
        self.attention_manager = AttentionManager(
            initial_attention=attention_config.get("initial_attention", 0.5),
            decay_halflife=attention_config.get("attention_decay_halflife", 300),
            boost_step=attention_config.get("attention_boost_step", 0.4),
            decrease_step=attention_config.get("attention_decrease_step", 0.1),
            decrease_threshold=attention_config.get("attention_decrease_threshold", 0.3),
            max_tracked_users=attention_config.get("max_tracked_users", 10),
            enable_spillover=attention_config.get("enable_spillover", True),
            spillover_ratio=attention_config.get("spillover_ratio", 0.35),
            spillover_halflife=attention_config.get("spillover_decay_halflife", 90),
            spillover_min_trigger=attention_config.get("attention_spillover_min_trigger", 0.4),
        )

        # 注意力关键词检测器
        self.attention_keywords_detector = AttentionKeywordsDetector(
            positive_keywords=attention_config.get("attention_keywords", []),
            negative_keywords=emotion_config.get("negative_keywords", []),
        )

        # 疲劳管理器
        self.fatigue_manager = FatigueManager(
            reset_threshold=fatigue_config.get("reset_threshold", 300),
            threshold_light=fatigue_config.get("threshold_light", 3),
            threshold_medium=fatigue_config.get("threshold_medium", 5),
            threshold_heavy=fatigue_config.get("threshold_heavy", 8),
            decrease_light=fatigue_config.get("decrease_light", 0.1),
            decrease_medium=fatigue_config.get("decrease_medium", 0.2),
            decrease_heavy=fatigue_config.get("decrease_heavy", 0.35),
            closing_probability=fatigue_config.get("closing_probability", 0.3),
        )

        # 社交感知管理器
        self.social_awareness = SocialAwarenessManager(
            bot_nickname=bot_nickname,
            interested_topics=self.personality.interested_topics,
            bored_topics=self.personality.bored_topics
        )

        # 发言决策器
        self.speaking_decider = EnhancedSpeakingDecider(
            base_probability=speaking_config.get("base_probability", 0.02),
            after_reply_probability=speaking_config.get("after_reply_probability", 0.8),
            probability_duration=speaking_config.get("probability_duration", 120),
            extraversion=self.personality.traits.get("extraversion", 0.5),
            neuroticism=self.personality.traits.get("neuroticism", 0.3),
            attention_manager=self.attention_manager,
            attention_keywords_detector=self.attention_keywords_detector,
            fatigue_manager=self.fatigue_manager,
            trigger_keywords=trigger_keywords,
            command_prefixes=speaking_config.get("command_prefixes", ["/", "!", "#"]),
        )

    def _init_reply_generator(self) -> None:
        """初始化回复生成器"""
        thinking_config = self.config.get("thinking", {})

        self.thinking_delay = ThinkingDelay(
            base_delay=thinking_config.get("base_delay", 2.0),
            min_delay=thinking_config.get("random_delay", {}).get("min", 0.5),
            max_delay=thinking_config.get("random_delay", {}).get("max", 10.0)
        )

        self.response_filter = ResponseFilter()

        self.reply_generator = ReplyGenerator(
            llm_provider=self.get_active_provider(),
            personality_prompt=self.personality.build_persona_prompt(),
            speaking_style_manager=self.speaking_style_manager
        )

    def _init_qq_adapter(self) -> None:
        """初始化 QQ 适配器"""
        qq_config = self.config.get("qq", {})
        self.qq_adapter = QQAdapter(
            config=qq_config,
            on_message=self._handle_message
        )

    async def _handle_message(self, message: Message) -> None:
        """处理接收到的消息"""
        logger.info(f"[{message.group_id or '私聊'}] {message.sender_name}: {message.content[:50]}...")

        # 创建事件
        event = Event(
            type=EventType.GROUP_MESSAGE if message.message_type == "group" else EventType.PRIVATE_MESSAGE,
            data={
                "message": message,
                "group_id": message.group_id,
                "session_id": message.session_id,
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
            }
        )

        # 发布事件
        await self.event_bus.publish(event)

        # 处理消息
        await self._process_message(message)

    async def _process_message(self, message: Message) -> None:
        """处理消息主流程"""
        session_id = message.session_id
        group_id = message.group_id or ""

        # === 1. 状态更新 ===
        is_reply_to_bot = self._is_reply_to_bot(message)
        self.speaking_decider.on_message(
            session_id=session_id,
            group_id=group_id,
            user_id=message.sender_id,
            mentioned_bot=message.mentioned_me,
            is_reply_to_bot=is_reply_to_bot,
        )

        # === 2. 更新情感状态 ===
        emotional_trigger = "mentioned" if message.mentioned_me else "normal_message"
        self.emotional_manager.trigger_event(session_id, emotional_trigger)

        emotional_state = self.emotional_manager.get_state(session_id)

        # === 3. 构建社交上下文 ===
        context = SocialContext(
            message_content=message.content,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            group_id=group_id,
            session_id=session_id,
            mentioned_me=message.mentioned_me,
            reply_to_me=is_reply_to_bot,
        )

        # === 4. 社交感知分析 ===
        trigger_result = self.trigger_detector.detect(context)
        context.extra["trigger"] = trigger_result
        context = self.social_awareness.analyze(context)

        # === 5. 添加消息到上下文 ===
        self.context_manager.add_message(
            session_id=session_id,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            content=message.content,
            is_bot=False,
            message_id=message.message_id,
            reply_to_qq=message.reply_to_qq,
        )

        # === 5.5 写入长期记忆（有记忆价值的消息） ===
        await self._store_long_term_memory(message, session_id)

        # === 6. 发言决策 ===
        should_speak, reason, probability = self.speaking_decider.should_speak(
            context,
            emotional_bonus=emotional_state.get_speaking_bonus()
        )

        logger.info(f"[决策] 发言={should_speak}, 概率={probability:.2f}, 原因={reason}")

        if not should_speak:
            return

        # === 6.5 判断消息指向（决定参与方式） ===
        # 明确提到bot的信号：@、回复bot、提到昵称/触发词/紧急
        # 只有这些才当"对我说"，否则归为群友互聊/对大家/自言自语，由LLM判断是否插嘴
        trigger_reasons = context.extra.get("trigger", {}).get("reasons", [])
        has_direct_signal = (
            message.mentioned_me
            or is_reply_to_bot
            or any(r not in ("直接提问",) for r in trigger_reasons)
        )
        direction = "to_bot" if has_direct_signal else "group"
        logger.debug(f"[指向] {direction} (触发: {trigger_reasons})")

        # === 7. 构建上下文提示（含长期记忆检索） ===
        memories = await self._retrieve_memories(message.content, session_id)

        context_prompt = self.context_manager.build_context_prompt(
            session_id=session_id,
            bot_name=self.personality.name,
            persona_prompt=self.personality.build_persona_prompt(),
            memories=memories
        )

        # === 8. 思考延迟 ===
        await self.thinking_delay.wait(
            message_length=len(message.content),
            topic_familiarity=context.topic_familiarity,
            emotional_modifier=emotional_state.get_thinking_delay_multiplier()
        )

        # === 9. 生成回复（direction 控制是否可沉默） ===
        try:
            reply = await self.reply_generator.generate(
                context_prompt=context_prompt,
                current_message=message.content,
                emotional_state=emotional_state,
                direction=direction,
            )
        except Exception as e:
            logger.error(f"Error generating reply: {e}", exc_info=True)
            return

        # LLM 选择沉默（群友互聊/自言自语时的正常行为）
        if reply is None:
            logger.info(f"[沉默] direction={direction}，不参与该条消息")
            return

        # === 10. 过滤回复 ===
        passed, result = self.response_filter.filter(reply)
        if not passed:
            logger.info(f"回复被过滤: {result}")
            return
        reply = result

        # === 11. 应用错字生成 ===
        typing_config = self.config.get("typing_style", {})
        if typing_config.get("enable_typo_generator", True):
            reply = self.typo_generator.apply_typo(reply)

        # === 12. 发送回复（拆成多条，模拟真人分段发送） ===
        import random
        from modules.reply.generator import split_reply_into_messages

        segments = split_reply_into_messages(reply)
        sent_any = False
        for i, seg in enumerate(segments):
            success = await self.qq_adapter.send_message(session_id, seg)
            if success:
                sent_any = True
                logger.info(f"[回复段{i+1}/{len(segments)}] {self.personality.name}: {seg[:50]}")
                # 段间延迟，模拟真人打字停顿
                if i < len(segments) - 1:
                    await asyncio.sleep(random.uniform(0.6, 2.0))

        if sent_any:
            logger.info(f"[回复] {self.personality.name}: {reply[:50]}...")

            # === 13. Bot回复后状态更新 ===
            self.speaking_decider.on_bot_reply(session_id, group_id)

            # === 14. 添加回复到上下文 ===
            self.context_manager.add_message(
                session_id=session_id,
                sender_id=self.config.get("qq", {}).get("self_id", ""),
                sender_name=self.personality.name,
                content=reply,
                is_bot=True,
                message_id="",
                reply_to_qq=message.sender_id,  # bot 回复的是当前这条消息
            )
        else:
            logger.error("Failed to send reply")

        # === 15. 疲劳时可能结束对话 ===
        if self.fatigue_manager.should_close_conversation(session_id):
            closing_msg = self.fatigue_manager.get_closing_message()
            await self.qq_adapter.send_message(session_id, closing_msg)
            logger.info(f"[疲劳] {closing_msg}")

    def _is_reply_to_bot(self, message: Message) -> bool:
        """检查是否回复了Bot：
        1. 被回复消息的发送者是 bot 的 QQ 号（reply 段的 qq 字段）
        2. 内容里提到 bot 昵称
        """
        if message.reply_to_qq:
            self_id = str(self.config.get("qq", {}).get("self_id", ""))
            if self_id and str(message.reply_to_qq) == self_id:
                return True
        bot_name = self.personality.name
        return bot_name in message.content

    # 个人信息关键词 - 出现时消息有较高记忆价值
    _PERSONAL_KEYWORDS = [
        "我叫", "我是", "我喜欢", "我爱", "我的生日", "我在", "我住", "我养",
        "我家", "我的工作", "我今年", "我朋友", "我对象", "我男票", "我女票",
        "我老婆", "我老公", "我同事", "我同学", "记得我", "我叫什么",
    ]

    def _store_long_term_memory(self, message: Message, session_id: str) -> None:
        """把有记忆价值的消息写入 SQLite 情景记忆（异步后台执行）

        打分规则：
        - 提到 bot → +0.2
        - 消息 > 20 字（分享/吐槽）→ +0.2
        - 含个人信息关键词（我叫/我喜欢/我的生日…）→ +0.3
        - 超过阈值 0.5 才值得长期记住
        """
        content = message.content.strip()
        if not content or len(content) < 4:
            return

        importance = 0.3
        if message.mentioned_me:
            importance += 0.2
        if len(content) > 20:
            importance += 0.2
        if any(kw in content for kw in self._PERSONAL_KEYWORDS):
            importance += 0.3

        if importance < 0.5:
            return

        memory = Memory(
            content=f"{message.sender_name}：{content[:200]}",
            memory_type="episodic",
            importance=min(1.0, importance),
            source_session=session_id,
            metadata={
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "mentioned_me": message.mentioned_me,
            },
        )

        async def _save():
            try:
                await self.memory_storage.store(memory)
                logger.debug(f"Saved long-term memory: importance={memory.importance:.2f}")
            except Exception as e:
                logger.error(f"Failed to save memory: {e}")

        # 不阻塞消息处理主流程
        asyncio.create_task(_save())

    async def _retrieve_memories(self, query: str, session_id: str, limit: int = None) -> list:
        """检索相关长期记忆：
        优先向量语义检索（TF-IDF + 余弦），失败或空时退回该会话最近记忆。
        """
        limit = limit or self.memory_search_top_k
        try:
            memories = await self.memory_storage.semantic_search(
                query=query,
                session=session_id,
                limit=limit,
                half_life_days=self.memory_half_life_days,
                similarity_weight=self.memory_similarity_weight,
            )
            if memories:
                return memories
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")

        # 兜底：该会话最近的高价值记忆
        try:
            return await self.memory_storage.retrieve_session_recent(session_id, limit=limit)
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    async def run(self) -> None:
        """运行 Bot"""
        await self.initialize()

        self._running = True
        logger.info("=" * 50)
        logger.info(f"GroupChatBot is running as {self.personality.name}!")
        logger.info("=" * 50)

        # 连接 QQ (可选 - 如果连接失败则继续运行用于测试)
        try:
            await self.qq_adapter.connect()
        except Exception as e:
            logger.warning(f"QQ连接失败 (可继续运行用于测试): {e}")
            logger.warning("提示: 请确保 NapCat QQ机器人 已启动")
            # 不停止 - 让bot以测试模式运行

        # 启动清理任务
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

        # 保持运行
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def _cleanup_loop(self) -> None:
        """定期清理过期状态"""
        while self._running:
            await asyncio.sleep(300)  # 5分钟
            self.speaking_decider.cleanup()
            self.fatigue_manager.cleanup()
            self.context_manager.cleanup_inactive(max_inactive_minutes=60)
            await self._maybe_decay_memories()
            logger.debug("Cleaned up expired states")

    async def _maybe_decay_memories(self) -> None:
        """定期对长期记忆应用时间衰减（每6小时一次，避免频繁写库）"""
        import time
        now = time.time()
        # 距上次执行不足6小时则跳过；进程内首次运行时直接执行一次
        if self._last_decay_run and (now - self._last_decay_run) < 6 * 3600:
            return
        try:
            self._last_decay_run = now
            result = await self.memory_storage.apply_time_decay(
                half_life_days=self.memory_half_life_days,
                min_importance=0.1,
                max_age_days=180,
            )
            if result["decayed"] or result["deleted"]:
                logger.info(f"[记忆衰减] {result}")
        except Exception as e:
            logger.error(f"Memory decay failed: {e}")

    async def stop(self) -> None:
        """停止 Bot"""
        logger.info("Stopping GroupChatBot...")
        self._running = False

        # 取消所有任务
        for task in self._tasks:
            task.cancel()

        # 停止事件总线
        if self.event_bus:
            self.event_bus.stop()

        # 断开 QQ 连接
        if self.qq_adapter:
            await self.qq_adapter.disconnect()

        # 关闭嵌入服务连接
        try:
            svc = self.memory_storage._storage._embedding_service
            if svc:
                await svc.close()
        except Exception:
            pass

        elapsed = time.time() - self._start_time
        logger.info(f"GroupChatBot stopped after {elapsed:.0f}s")

    def _get_default_config(self) -> dict:
        """获取默认配置 - 全部通过Web页面配置"""
        return {
            "bot": {
                "name": "爱丽丝",
                "nickname": "小艾"
            },
            "personality": {
                "name": "爱丽丝",
                "nickname": "小艾",
                "background": "一个活泼可爱、喜欢聊天的女孩，喜欢分享有趣的事情和倾听朋友的故事。",
                "traits": {
                    "openness": 0.7,
                    "conscientiousness": 0.5,
                    "extraversion": 0.8,
                    "agreeableness": 0.85,
                    "neuroticism": 0.3
                },
                "interested_topics": ["美食", "旅行", "音乐", "电影", "八卦", "日常闲聊"],
                "bored_topics": ["广告推销", "政治敏感话题", "重复的无聊话题"]
            },
            "llm": {
                "provider_type": "openai",
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
                "temperature": 0.8,
                "max_tokens": 500,
                "top_p": 0.9,
                "timeout": 120
            },
            "qq": {
                "ws_url": "ws://127.0.0.1:3001",
                "access_token": "",
                "self_id": ""
            },
            "speaking": {
                "base_probability": 0.4,
                "trigger_keywords": ["爱丽丝", "小艾", "bot", "机器人"],
                "thinking_delay": {
                    "base": 2.0,
                    "min": 0.5,
                    "max": 10.0
                }
            },
            "memory": {
                "context_window_size": 50,
                "context_max_age_hours": 2.0,
                "enable_long_term_memory": True,
                "db_path": "data/memory.db"
            },
            "groups": {}
        }


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="爱丽丝 (Alice) - 群聊AI伙伴")
    parser.add_argument("-c", "--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--dashboard", action="store_true", help="Enable web dashboard")
    parser.add_argument("--dashboard-host", default="0.0.0.0", help="Dashboard host")
    parser.add_argument("--dashboard-port", type=int, default=30080, help="Dashboard port")
    args = parser.parse_args()

    # 信号处理
    loop = asyncio.get_event_loop()
    bot_instance = None

    def signal_handler():
        if bot_instance:
            asyncio.create_task(bot_instance.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # 如果启用 Dashboard
    if args.dashboard:
        from dashboard import run_dashboard
        bot = GroupChatBot(config_path=args.config)
        bot_instance = bot

        # 初始化 Bot
        await bot.initialize()
        bot._running = True

        # 在线程中运行 Dashboard
        import threading

        def start_dashboard():
            run_dashboard(bot, host=args.dashboard_host, port=args.dashboard_port)

        dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
        dashboard_thread.start()
        logger.info(f"Dashboard running at http://{args.dashboard_host}:{args.dashboard_port}")

        # 在线程中连接 QQ 适配器
        async def connect_qq():
            try:
                await bot.qq_adapter.connect()
            except Exception as e:
                logger.warning(f"QQ连接失败: {e}")

        qq_thread = threading.Thread(target=lambda: asyncio.run(connect_qq()), daemon=True)
        qq_thread.start()
        logger.info("QQ adapter starting in background...")

        # 保持运行直到被中断
        try:
            while bot._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await bot.stop()
    else:
        # 仅运行 Bot
        bot = GroupChatBot(config_path=args.config)
        bot_instance = bot
        await bot.run()


if __name__ == "__main__":
    asyncio.run(main())

"""
爱丽丝 (Alice) - 群聊AI伙伴
基于 AstrBot 设计的增强版主入口
"""
import asyncio
import logging
import random
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
from modules.social.conversation_floor import ConversationFloorManager

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
        self.conversation_floor_manager: Optional[ConversationFloorManager] = None

        # 回复生成
        self.reply_generator: Optional[ReplyGenerator] = None
        self.thinking_delay: Optional[ThinkingDelay] = None
        self.response_filter: Optional[ResponseFilter] = None

        # 状态
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._start_time: float = 0
        # 每会话正在进行的回复生成任务（同一会话同时只生成一条回复）
        self._reply_tasks: dict[str, asyncio.Task] = {}
        self._rich_media_tasks: set[asyncio.Task] = set()
        # 群聊纪要（滚动总结）：每 N 条消息把新聊天压缩成一条长期记忆，隔天也不会忘
        self._digest_config: dict = {}
        self._last_digest_at: dict[str, float] = {}  # session -> 上次纪要覆盖到的消息时间戳
        self._digest_tasks: dict[str, asyncio.Task] = {}  # 进行中的纪要任务

        # 媒体轨迹：记录最近消息(时间/sender/是否纯图片/图片url)，用于判断"连图"。
        self._media_trail: dict[str, deque] = {}
        self._media_trail_limit = 30

    @property
    def _image_group_config(self) -> dict:
        return self.config.get("image", {}) or {} if self.config else {}

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

        # 兼容旧版 llm.api_key/model 扁平结构，统一视为 primary Provider。
        legacy_keys = {"api_key", "base_url", "model", "provider_type"}
        legacy_llm = (
            isinstance(llm_config, dict)
            and legacy_keys.intersection(llm_config)
            and not any(isinstance(value, dict) for value in llm_config.values())
        )
        if legacy_llm:
            llm_config = {"primary": llm_config}

        # 合并两个配置源，providers 优先级更高
        all_providers = {**llm_config, **providers_config}

        # 初始化所有 providers
        for name, provider_config in all_providers.items():
            if isinstance(provider_config, dict) and provider_config.get("enabled", True):
                try:
                    provider = create_provider(
                        provider_config.get("provider_type", "openai"),
                        {
                            "provider_type": provider_config.get("provider_type", "openai"),
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

        # 群聊纪要配置：把固定条数的群聊滚动总结成长期记忆
        digest_cfg = memory_config.get("digest", {}) or {}
        self._digest_config = {
            "enabled": digest_cfg.get("enabled", True),
            "interval_messages": digest_cfg.get("interval_messages", 20),
            "min_messages": digest_cfg.get("min_messages", 10),
            "max_tokens": digest_cfg.get("max_tokens", 200),
        }
        logger.info(f"✓ 群聊纪要: enabled={self._digest_config['enabled']}, "
                    f"每{self._digest_config['interval_messages']}条消息总结一次")

    def _init_personality(self) -> None:
        """初始化人格系统"""
        speaking_config = self.config.get("speaking", {})

        # 情感管理器
        emotion_config = self.config.get("emotion", {})
        self.emotional_manager = EmotionalManager(
            enabled=emotion_config.get("enabled", True),
            decay_halflife=emotion_config.get("decay_halflife", 600),
            positive_keywords=emotion_config.get("positive_keywords", []),
            negative_keywords=emotion_config.get("negative_keywords", []),
            positive_boost=emotion_config.get("positive_boost", 0.1),
            negative_decrease=emotion_config.get("negative_decrease", 0.15),
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
        floor_config = self.config.get("conversation_floor", {})
        typing_config = self.config.get("typing_style", {})
        cooldown_config = self.config.get("cooldown", {})

        bot_nickname = self.personality.name
        trigger_keywords = speaking_config.get("trigger_keywords", [])

        # 触发检测器
        self.trigger_detector = TriggerDetector(
            bot_nickname=bot_nickname,
            nicknames=[self.personality.nickname],
            trigger_keywords=trigger_keywords
        )

        # 注意力管理器
        self.attention_manager = AttentionManager(
            enabled=attention_config.get("enabled", True),
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
            enabled=fatigue_config.get("enabled", True),
            reset_threshold=fatigue_config.get("reset_threshold", 300),
            threshold_light=fatigue_config.get("threshold_light", 3),
            threshold_medium=fatigue_config.get("threshold_medium", 5),
            threshold_heavy=fatigue_config.get("threshold_heavy", 8),
            decrease_light=fatigue_config.get("decrease_light", 0.1),
            decrease_medium=fatigue_config.get("decrease_medium", 0.2),
            decrease_heavy=fatigue_config.get("decrease_heavy", 0.35),
            closing_probability=fatigue_config.get("closing_probability", 0.3),
            cooldown_enabled=cooldown_config.get("enabled", True),
            cooldown_max_duration=cooldown_config.get("max_duration", 60),
            cooldown_trigger_threshold=cooldown_config.get("trigger_threshold", 0.3),
            cooldown_attention_decrease=cooldown_config.get("attention_decrease", 0.2),
        )

        # 社交感知管理器
        self.social_awareness = SocialAwarenessManager(
            bot_nickname=bot_nickname,
            interested_topics=self.personality.interested_topics,
            bored_topics=self.personality.bored_topics
        )

        # 群聊发言权：判断谁在和谁说话、当前插嘴成本以及候选行为。
        self.conversation_floor_manager = ConversationFloorManager(
            active_window_seconds=floor_config.get("active_window_seconds", 45),
            burst_window_seconds=floor_config.get("burst_window_seconds", 12),
            burst_message_threshold=floor_config.get("burst_message_threshold", 4),
            topic_shift_threshold=floor_config.get("topic_shift_threshold", 0.12),
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

        # 联网搜索（LLM 判断是否需要搜索）。普通闲聊仍走主 LLM，零回归。
        search_client, tool_llm = self._init_search()

        self.reply_generator = ReplyGenerator(
            llm_provider=self.get_active_provider(),
            personality_prompt=self.personality.build_persona_prompt(),
            speaking_style_manager=self.speaking_style_manager,
            tool_llm_provider=tool_llm,
            search_client=search_client,
        )

    def _init_search(self):
        """初始化联网搜索：SearchClient + 支持 function calling 的 LLM。

        工具 LLM 默认复用主 LLM 的密钥/模型，但走 OpenAI 兼容端点
        （MiniMax 的 function calling 在 OpenAI 端点上可用，Anthropic 端点上不支持）。
        """
        from modules.llm.openai_provider import OpenAIProvider
        from modules.search import SearchClient

        search_config = self.config.get("search", {}) or {}
        search_client = SearchClient(search_config)

        tool_llm = None
        if search_client.available and search_config.get("enabled", False):
            llm_cfg = search_config.get("llm", {}) or {}
            primary = self.get_active_provider()
            api_key = llm_cfg.get("api_key") or getattr(primary, "api_key", "")
            model = llm_cfg.get("model") or getattr(primary, "model", "")
            try:
                tool_llm = OpenAIProvider({
                    "provider_type": "openai_compatible",
                    "api_key": api_key,
                    "base_url": llm_cfg.get(
                        "base_url", "https://api.minimaxi.com/v1"
                    ),
                    "model": model or "MiniMax-M3",
                    "timeout": float(llm_cfg.get("timeout", 60)),
                })
                logger.info(f"✓ 联网搜索 LLM: {tool_llm.model}")
            except Exception as e:
                logger.error(f"✗ 联网搜索 LLM 初始化失败: {e}")
                tool_llm = None

        if search_client.available:
            logger.info(
                f"✓ 联网搜索已启用: primary={search_client.primary}, "
                f"后端={list(search_client._backends.keys())}"
            )
        return search_client, tool_llm

    def _init_vision_provider(self) -> Optional[Any]:
        """初始化视觉模型 Provider（图片→文字描述）。未启用/失败返回 None，不阻断启动。"""
        image_config = self.config.get("image", {}) or {}
        vision_config = image_config.get("vision", {}) or {}
        if not vision_config.get("enabled", False):
            return None
        try:
            provider = create_provider(
                vision_config.get("provider_type", "openai"),
                {
                    "provider_type": vision_config.get("provider_type", "openai"),
                    "api_key": vision_config.get("api_key", ""),
                    "base_url": vision_config.get(
                        "base_url", "https://api.openai.com/v1"
                    ),
                    "model": vision_config.get("model", "gpt-4o-mini"),
                    "timeout": vision_config.get("timeout", 60),
                    "temperature": vision_config.get("temperature", 0.4),
                    "max_tokens": 300,
                },
            )
            logger.info(
                f"✓ Vision provider: {vision_config.get('base_url')}, model: {vision_config.get('model')}"
            )
            return provider
        except Exception as e:
            logger.error(f"✗ Failed to init vision provider: {e}")
            return None

    def _init_qq_adapter(self) -> None:
        """初始化 QQ 适配器"""
        qq_config = self.config.get("qq", {})
        # 顶层 image 段（to_text_scope/prompt/vision 等）合并进 rich_media 配置。
        rich_media_config = dict(self.config.get("rich_media", qq_config.get("rich_media", {})))
        image_config = self.config.get("image", {}) or {}
        if image_config:
            merged_image = {**rich_media_config.get("image", {}), **image_config}
            rich_media_config["image"] = merged_image
        adapter_config = {
            **qq_config,
            "rich_media": rich_media_config,
        }
        self.qq_adapter = QQAdapter(
            config=adapter_config,
            on_message=self._handle_message,
            vision_provider=self._init_vision_provider(),
        )

    async def _handle_message(self, message: Message) -> None:
        """处理接收到的消息

        拆成两条路径，避免"思考期间看不到新消息"的失真：
        - 快速路径（立即执行）：状态更新 + 消息写入上下文 + 发言决策，不阻塞接收循环
        - 慢路径（后台任务）：思考延迟 + LLM 生成 + 发送，期间新消息仍会进入上下文
        """
        session_id = message.session_id
        group_config = (
            self.config.get("groups", {}).get(str(message.group_id), {})
            if message.message_type == "group"
            else {}
        )
        if message.message_type == "group" and group_config.get("enabled") is False:
            logger.debug("群 %s 已在 Web 配置中停用", message.group_id)
            return

        # 富媒体增强只依据最外层文字和结构化指向判断。转发/卡片内部即使含有
        # bot 昵称、@ 或问题，也不能把一条普通分享误判为“对 bot 说”。
        is_reply_to_bot = self._is_reply_to_bot(message)
        continuing = self._is_continuing_conversation(message)
        rich_probe = SocialContext(
            message_content=(
                message.outer_text if message.segments else message.content
            ),
            mentioned_me=message.mentioned_me,
            reply_to_me=is_reply_to_bot,
        )
        rich_trigger = self.trigger_detector.detect(rich_probe)
        rich_reasons = rich_trigger.get("reasons", [])
        rich_directed = (
            message.message_type == "private"
            or message.mentioned_me
            or is_reply_to_bot
            or continuing
            or rich_trigger.get("forced_trigger", False)
            or "紧急" in rich_reasons
            or (
                any(reason.startswith("昵称") for reason in rich_reasons)
                and rich_probe.is_direct_question
            )
        )
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

        # 发布事件（入队即可，不等待处理）
        await self.event_bus.publish(event)

        # === 1. 快速路径：状态更新 + 上下文记录（立即执行，bot 实时"看到"消息） ===
        # 私聊天然是对 bot 说；群聊则结合 @、引用/称呼和最近实际回复判断。
        directed_to_bot = (
            message.message_type == "private"
            or message.mentioned_me
            or is_reply_to_bot
            or continuing
        )
        try:
            # 疲劳/注意力更新。注意：注意力按 group_id 键控，私聊没有群号，
            # 用 session_id（private_xxx）作键，避免所有私聊共用同一个注意力桶。
            self.speaking_decider.on_message(
                session_id=session_id,
                group_id=message.group_id or message.session_id,
                user_id=message.sender_id,
                mentioned_bot=message.mentioned_me,
                is_reply_to_bot=is_reply_to_bot,
                is_directed_to_bot=directed_to_bot,
            )

            # 情感更新（@ 提升参与度）
            emotional_trigger = "mentioned" if message.mentioned_me else "normal_message"
            self.emotional_manager.trigger_event(session_id, emotional_trigger)
            # 只有明确对 bot 说的话才按夸奖/冒犯归因，避免把群友互聊误认成针对自己。
            if directed_to_bot:
                emotion_desc = self.emotional_manager.detect_and_apply_keywords(
                    session_id, message.content
                )
                if emotion_desc:
                    logger.debug(f"[情绪] {emotion_desc}")

            # 消息入上下文（关键：立即记录，慢路径构建提示词时能看到这条）
            self.context_manager.add_message(
                session_id=session_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                content=message.content,
                is_bot=False,
                message_id=message.message_id,
                reply_to_id=message.reply_to_id,
                reply_to_qq=message.reply_to_qq,
                directed_to_bot=directed_to_bot,
            )

            # 长期记忆（内部已异步后台执行）
            self._store_long_term_memory(message, session_id)

            # 群聊纪要：距上次总结的新消息达到固定条数 → 后台压缩成纪要存长期记忆
            self._maybe_schedule_digest(session_id)

            # 记录媒体轨迹，供"连图整体识别"判断连续纯图片
            self._record_media_trail(message)
        except Exception as e:
            logger.error(f"Fast path error: {e}", exc_info=True)

        # 富媒体增强可能涉及 NapCat API 或安全网页预览。它在后台执行，决策仍走
        # 快速路径；真正生成回复前会等待本条消息的增强结果。
        enrichment_task = None
        if message.rich_type:
            enrichment_task = asyncio.create_task(
                self._enrich_context_message(message, rich_directed)
            )
            self._rich_media_tasks.add(enrichment_task)
            enrichment_task.add_done_callback(self._rich_media_tasks.discard)

        # === 2. 发言决策（快，无 LLM 调用） ===
        decision = self._decide_reply(message, is_reply_to_bot, continuing)
        if not decision:
            return
        decision["enrichment_task"] = enrichment_task

        # === 3. 调度回复生成（后台任务，避免阻塞接收循环） ===
        current = self._reply_tasks.get(session_id)
        if current and not current.done():
            if decision["direction"] == "to_bot":
                # 正在生成时又来了更强的"对我说"信号 → 取消旧任务改回新消息
                current.cancel()
                self._reply_tasks[session_id] = asyncio.create_task(
                    self._compose_and_send(message, decision)
                )
            else:
                # 旧任务生成时会把新消息纳入上下文，无需另起一条回复
                return
        else:
            self._reply_tasks[session_id] = asyncio.create_task(
                self._compose_and_send(message, decision)
            )

    def _decide_reply(
        self,
        message: Message,
        is_reply_to_bot: bool,
        continuing: bool = False,
    ) -> Optional[dict]:
        """发言决策（同步、快速）：返回回复参数，不发言则返回 None"""
        session_id = message.session_id
        group_id = message.group_id or ""
        group_config = self.config.get("groups", {}).get(str(group_id), {})
        is_private = message.message_type == "private"
        self_id = str(self.config.get("qq", {}).get("self_id", ""))
        quoted_bot = bool(message.reply_to_qq) and bool(self_id) and str(message.reply_to_qq) == self_id

        # 引用/@了别人（非bot）→ 明显在跟别人说话，不当作对bot延续
        talking_to_others = (
            bool(message.mentioned_others)
            or (bool(message.reply_to_qq) and not quoted_bot)
        )

        # 延续对话：bot 最近确实回复成功过该用户，TA 没@没引用就接着对 bot 说。
        continuing = continuing or (
            not is_private
            and not talking_to_others
            and not message.mentioned_me
            and self.speaking_decider.is_conversation_with(session_id, message.sender_id)
        )

        # 构建社交上下文
        context = SocialContext(
            # 触发检测只能看最外层文字，不能被转发记录/卡片标题中的昵称或问题影响。
            message_content=(message.outer_text if message.segments else message.content),
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            group_id=group_id,
            session_id=session_id,
            mentioned_me=message.mentioned_me,
            reply_to_me=is_reply_to_bot,
        )
        # 群活跃度（近10分钟消息量，真实反映冷热，避免恒为默认值）
        context.group_activity = self.context_manager.get_window(session_id).get_activity_level(
            window_minutes=10
        )

        # 触发检测 + 社交感知分析
        trigger_result = self.trigger_detector.detect(context)
        context.extra["trigger"] = trigger_result
        context.extra["is_private"] = is_private
        context.extra["outer_text"] = (
            message.outer_text if message.segments else message.content
        )
        context.extra["rich_message_only"] = message.rich_only
        context.extra["rich_type"] = message.rich_type
        context.extra["group_base_probability"] = group_config.get(
            "speaking_probability"
        )
        # 话题兴趣可以参考安全渲染后的富媒体摘要，但强制触发结果已经在上面锁定。
        context.message_content = message.content
        context = self.social_awareness.analyze(context)

        # 发言权和行为计划。当前消息已在快速路径进入上下文，因此可直接分析消息拓扑。
        window = self.context_manager.get_window(session_id)
        recent_context_messages = window.get_recent(12)
        action_plan = None
        current_is_latest = bool(recent_context_messages) and (
            (
                bool(message.message_id)
                and recent_context_messages[-1].message_id == message.message_id
            )
            or (
                not message.message_id
                and recent_context_messages[-1].sender_id == message.sender_id
                and recent_context_messages[-1].content == message.content
            )
        )
        if current_is_latest:
            current_context_message = recent_context_messages[-1]
            floor_has_name = any(
                reason.startswith("昵称")
                for reason in trigger_result.get("reasons", [])
            )
            directed_for_floor = (
                is_private
                or message.mentioned_me
                or quoted_bot
                or is_reply_to_bot
                or continuing
                or trigger_result.get("forced_trigger", False)
                or context.is_emergency
                or (floor_has_name and context.is_direct_question)
            )
            if directed_for_floor:
                current_context_message.directed_to_bot = True
            floor, action_plan = self.conversation_floor_manager.analyze(
                current_context_message,
                recent_context_messages,
                bot_id=self_id,
                is_private=is_private,
                directed_to_bot=directed_for_floor,
                continuing=continuing,
                mentioned_others=message.mentioned_others,
                topic_relevance=context.topic_relevance,
                is_question=context.is_direct_question,
                rich_message_only=message.rich_only,
                rich_type=message.rich_type,
            )
            context.extra["floor"] = floor
            context.extra["action_plan"] = action_plan
            logger.info(
                f"[发言权] 动作={action_plan.action.value}, "
                f"插话成本={floor.interruption_cost:.2f}, 原因={action_plan.reason}"
            )

        # 发言决策
        emotional_state = (
            self.emotional_manager.get_state(session_id)
            if self.emotional_manager.enabled
            else None
        )
        should_speak, reason, probability = self.speaking_decider.should_speak(
            context,
            emotional_bonus=(
                emotional_state.get_speaking_bonus() if emotional_state else 0.0
            )
        )
        logger.info(f"[决策] 发言={should_speak}, 概率={probability:.2f}, 原因={reason}")

        if not should_speak:
            # 被明确点名却保持沉默 → 降低对发话人的关注（真人也会忙/没看见）
            if message.mentioned_me or is_reply_to_bot:
                # 私聊没有群号，用 session_id（private_xxx）作注意力键，避免所有私聊共用一个桶
                self.attention_manager.on_no_reply(
                    message.group_id or session_id, message.sender_id
                )
            return None

        # 判断消息指向：
        # - 必须回：@、引用bot、强制触发、紧急
        # - 视为对我说：必须回，或（提到名字/昵称 且 带提问）
        # - 同一人延续对话：bot 刚回复过 TA，TA 没@没引用就接着对 bot 说 → 也算对我说
        # - 其余（纯群聊/自言自语/只是顺口提到名字）→ group，由 LLM 决定插嘴还是潜水
        trigger_reasons = trigger_result.get("reasons", [])
        forced = trigger_result.get("forced_trigger", False)

        has_name = any(r.startswith("昵称") for r in trigger_reasons)
        has_question = any(r == "直接提问" for r in trigger_reasons)
        has_emergency = "紧急" in trigger_reasons

        must_reply = is_private or message.mentioned_me or quoted_bot or forced or has_emergency
        is_addressed = must_reply or (has_name and has_question) or continuing
        direction = "to_bot" if is_addressed else "group"
        logger.debug(f"[指向] {direction} (触发: {trigger_reasons}, 延续对话={continuing})")

        return {
            "direction": direction,
            "probability": probability,
            "emotional_state": emotional_state,
            "context": context,
            "action_plan": action_plan,
        }

    async def _compose_and_send(self, message: Message, decision: dict) -> None:
        """慢路径：思考延迟 → 用最新上下文生成回复 → 发送"""
        session_id = message.session_id
        # 私聊没有群号，用 session_id（private_xxx）作注意力键，与 _handle_message 一致。
        group_id = message.group_id or session_id
        direction = decision["direction"]
        probability = decision["probability"]
        emotional_state = decision["emotional_state"]
        context = decision["context"]
        action_plan = decision.get("action_plan")

        try:
            enrichment_task = decision.get("enrichment_task")
            if enrichment_task:
                try:
                    await enrichment_task
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("富媒体增强失败，使用占位符继续：%s", exc)

            # === 检索长期记忆 ===
            memories = await self._retrieve_memories(message.content, session_id)

            # === 思考延迟（期间新消息会进入上下文，等对方把话说完） ===
            await self.thinking_delay.wait(
                message_length=len(message.content),
                topic_familiarity=context.topic_familiarity,
                emotional_modifier=(
                    (emotional_state.get_thinking_delay_multiplier() if emotional_state else 1.0)
                    * (action_plan.wait_multiplier if action_plan else 1.0)
                )
            )

            # 思考期间群聊可能已经向前发展；非定向插话过期时直接放弃。
            if action_plan:
                cancel, cancel_reason = self.conversation_floor_manager.should_cancel(
                    action_plan,
                    self.context_manager.get_window(session_id).get_recent(30),
                    bot_id=str(self.config.get("qq", {}).get("self_id", "")),
                )
                if cancel:
                    logger.info(f"[发送复核] 放弃回复：{cancel_reason}")
                    return

            # === 构建提示词（此刻的上下文 = 思考期间的最新消息，不会回旧话题） ===
            context_prompt = self.context_manager.build_context_prompt(
                session_id=session_id,
                bot_name=self.personality.name,
                # 人格已经作为 system message 注入 ReplyGenerator，避免重复两遍。
                persona_prompt="",
                memories=memories
            )

            # === 生成回复（direction 控制是否可沉默） ===
            try:
                reply = await self.reply_generator.generate(
                    context_prompt=context_prompt,
                    current_message=message.content,
                    emotional_state=emotional_state,
                    direction=direction,
                    action_plan=action_plan.to_dict() if action_plan else None,
                    session_id=session_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error generating reply: {e}", exc_info=True)
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return

            # LLM 选择沉默（群友互聊/自言自语时的正常行为）
            if reply is None:
                logger.info(f"[沉默] direction={direction}，不参与该条消息")
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return

            # LLM 调用和模拟打字也会耗时，发送前再复核一次群聊局势。
            if action_plan:
                cancel, cancel_reason = self.conversation_floor_manager.should_cancel(
                    action_plan,
                    self.context_manager.get_window(session_id).get_recent(30),
                    bot_id=str(self.config.get("qq", {}).get("self_id", "")),
                )
                if cancel:
                    logger.info(f"[发送复核] 生成后放弃回复：{cancel_reason}")
                    return

            # === 过滤回复 ===
            passed, result = self.response_filter.filter(reply)
            if not passed:
                logger.info(f"回复被过滤: {result}")
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return
            reply = result

            # === 应用错字生成 ===
            typing_config = self.config.get("typing_style", {})
            if typing_config.get("enable_typo_generator", True):
                reply = self.typo_generator.apply_typo(reply)

            # === 发送回复（拆成多条，模拟真人分段发送） ===
            from modules.reply.generator import split_reply_into_messages

            segments = split_reply_into_messages(reply)
            # 插话进入群聊讨论时，引用触发消息，让群友明确知道在回应哪一条；
            # 明确对我说（directed）的消息本来就指向清楚，不用再引用。
            quote_id = (
                action_plan.target_message_id
                if action_plan and not action_plan.directed and action_plan.target_message_id
                else ""
            )
            sent_segments = []
            for i, seg in enumerate(segments):
                if i > 0 and action_plan:
                    cancel, cancel_reason = self.conversation_floor_manager.should_cancel(
                        action_plan,
                        self.context_manager.get_window(session_id).get_recent(30),
                        bot_id=str(self.config.get("qq", {}).get("self_id", "")),
                    )
                    if cancel:
                        logger.info(f"[分段复核] 停止剩余消息：{cancel_reason}")
                        break
                success = await self.qq_adapter.send_message(
                    session_id, seg, reply_to_id=(quote_id if i == 0 else "")
                )
                if success:
                    sent_segments.append(seg)
                    logger.info(f"[回复段{i+1}/{len(segments)}] {self.personality.name}: {seg[:50]}")
                    # 段间延迟，模拟真人打字停顿
                    if i < len(segments) - 1:
                        await asyncio.sleep(random.uniform(0.6, 2.0))
                else:
                    # 保持分段顺序；前一段失败后继续发后一段会显得语义残缺。
                    break

            if sent_segments:
                sent_reply = "".join(sent_segments)
                all_segments_sent = len(sent_segments) == len(segments)
                logger.info(f"[回复] {self.personality.name}: {sent_reply[:50]}...")

                # Bot回复后状态更新（真实概率：高概率的@/回复不触发冷却，对话可延续）
                self.speaking_decider.on_bot_reply(
                    session_id,
                    group_id,
                    probability=probability,
                    user_id=message.sender_id,
                )

                # 添加回复到上下文
                self.context_manager.add_message(
                    session_id=session_id,
                    sender_id=self.config.get("qq", {}).get("self_id", ""),
                    sender_name=self.personality.name,
                    content=sent_reply,
                    is_bot=True,
                    message_id="",
                    reply_to_id=message.message_id,
                    reply_to_qq=message.sender_id,  # bot 回复的是当前这条消息
                )

                # bot 的回复也写入长期记忆，让会话历史两侧完整
                self._store_bot_memory(session_id, sent_reply)

                # 疲劳收尾只在主回复成功后发送；成功后重置，避免连续多轮重复说“先走了”。
                if (
                    all_segments_sent
                    and self.fatigue_manager.should_close_conversation(session_id)
                ):
                    closing_msg = self.fatigue_manager.get_closing_message()
                    closing_sent = await self.qq_adapter.send_message(
                        session_id, closing_msg
                    )
                    if closing_sent:
                        self.context_manager.add_message(
                            session_id=session_id,
                            sender_id=self.config.get("qq", {}).get("self_id", ""),
                            sender_name=self.personality.name,
                            content=closing_msg,
                            is_bot=True,
                        )
                        self.fatigue_manager.get_state(session_id).reset()
                        logger.info(f"[疲劳] {closing_msg}")
                        # 疲劳收尾消息同样入历史
                        self._store_bot_memory(session_id, closing_msg)
            else:
                logger.error("Failed to send reply")
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)

        except asyncio.CancelledError:
            # 被更新的"对我说"消息取代，静默退出
            logger.debug(f"Reply task cancelled: {session_id}")
            raise
        except Exception as e:
            logger.error(f"Compose/send error: {e}", exc_info=True)
        finally:
            # 释放会话锁：只有自己仍是当前登记的任务才移除，避免误删被更新的任务
            if self._reply_tasks.get(session_id) is asyncio.current_task():
                self._reply_tasks.pop(session_id, None)

    async def _enrich_context_message(
        self,
        message: Message,
        directed: bool,
    ) -> None:
        """后台增强富媒体，并原位更新已经进入滑动窗口的那条消息。"""
        original_content = message.content
        conversation_context = self._build_image_conversation_context(message)
        group_image_urls = self._collect_group_image_urls(message)
        await self.qq_adapter.enrich_message(
            message,
            directed=directed,
            conversation_context=conversation_context,
            group_image_urls=group_image_urls,
        )
        if message.content != original_content:
            self.context_manager.update_message_content(
                message.session_id,
                message.message_id,
                message.content,
            )
            logger.debug("[富媒体] %s", message.content[:120])

    def _record_media_trail(self, message: Message) -> None:
        """记录消息轨迹（时间/sender/是否纯图片/图片url），用于判断连续连图。"""
        try:
            sid = message.session_id
            is_pure_image = message.rich_only and message.rich_type in ("image", "mface")
            url = ""
            if is_pure_image:
                for seg in message.segments:
                    if seg.type in ("image", "mface") and seg.url:
                        url = seg.url
                        break
            trail = self._media_trail.setdefault(sid, deque(maxlen=self._media_trail_limit))
            trail.append({
                "ts": time.time(),
                "sender_id": message.sender_id,
                "is_image": is_pure_image,
                "url": url,
                "message_id": message.message_id,
            })
        except Exception as e:
            logger.debug("Record media trail failed: %s", e)

    def _collect_group_image_urls(self, message: Message) -> list[str]:
        """取当前图片消息之前、同一人连续发的纯图片 URL（相邻间隔≤60s）。

        从轨迹中往回扫：同 sender 的纯图片且与前一条间隔在窗口内 → 收进组；
        遇到不同 sender / 非图片 / 超窗 → 停止（严格连续）。返回时间正序。
        """
        if not (self.config.get("image", {}).get("group_enabled", True)):
            return []
        trail = self._media_trail.get(message.session_id)
        if not trail:
            return []
        interval = float(self.config.get("image", {}).get("group_interval_seconds", 60))
        cap = int(self.config.get("image", {}).get("group_max_images", 4)) - 1  # 除当前图
        now = time.time()
        sender = message.sender_id
        current_msg_id = getattr(message, "message_id", "") or ""
        collected: list[str] = []
        for entry in reversed(list(trail)):
            # 跳过当前这条消息本身（快路径已把它写入轨迹末尾）
            if entry.get("message_id") and current_msg_id and entry["message_id"] == current_msg_id:
                continue
            if entry["sender_id"] != sender:
                break
            if not entry["is_image"] or not entry["url"]:
                break
            # 相邻间隔：该条目与其后一条（或当前消息）的时间差超过窗口则视为断组
            if now - entry["ts"] > interval:
                break
            collected.append(entry["url"])
            if len(collected) >= cap:
                break
            now = entry["ts"]  # 向前推移，判断下一条与前一条的间隔
        collected.reverse()  # 时间正序
        if collected:
            logger.debug("[连图] 当前图前收集 %d 张组图: %s", len(collected), collected)
        return collected

    def _build_image_conversation_context(self, message: Message) -> str:
        """取当前图片消息之前的若干条对话，供视觉模型判断图片/表情包的意图。"""
        try:
            window = self.context_manager.get_window(message.session_id)
            recent = window.get_recent(8)
            lines = []
            current_id = getattr(message, "message_id", "") or ""
            for m in reversed(recent):
                # 排除当前这条（后台增强时它已在窗口内）
                if current_id and m.message_id and m.message_id == current_id:
                    continue
                if not current_id and m.sender_id == message.sender_id and m.content == message.content:
                    continue
                speaker = "爱丽丝" if m.is_bot else (m.sender_name or m.sender_id)
                lines.append(f"[{m.timestamp.strftime('%H:%M')}] {speaker}：{m.content[:80]}")
                if len(lines) >= 6:
                    break
            return "\n".join(reversed(lines))
        except Exception as e:
            logger.debug("Build image conversation context failed: %s", e)
            return ""

    def _is_reply_to_bot(self, message: Message) -> bool:
        """检查消息是否引用了 Bot；名字/昵称提及由触发检测器单独处理。"""
        if message.reply_to_qq:
            self_id = str(self.config.get("qq", {}).get("self_id", ""))
            if self_id and str(message.reply_to_qq) == self_id:
                return True
        return False

    def _is_continuing_conversation(self, message: Message) -> bool:
        """判断消息是否自然延续了 bot 与同一用户的最近一次真实对话。"""
        if message.message_type == "private":
            return True

        self_id = str(self.config.get("qq", {}).get("self_id", ""))
        quoted_bot = (
            bool(message.reply_to_qq)
            and bool(self_id)
            and str(message.reply_to_qq) == self_id
        )
        talking_to_others = (
            bool(message.mentioned_others)
            or (bool(message.reply_to_qq) and not quoted_bot)
        )
        return (
            not talking_to_others
            and not message.mentioned_me
            and self.speaking_decider.is_conversation_with(
                message.session_id, message.sender_id
            )
        )

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
        # 卡片/转发是外部引用材料，不把其内容误记成发送者自己的事实。
        # 如果外层另有文字，只记外层文字和附件类型。
        if message.rich_only:
            return
        content = (message.outer_text or message.content).strip()
        if message.rich_type:
            content = f"{content} [{message.rich_type}]".strip()
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

    def _store_bot_memory(self, session_id: str, content: str) -> None:
        """把 bot 自己发出的消息写入 SQLite 情景记忆（异步后台执行）。

        用户消息走 _store_long_term_memory（按重要性筛选）；bot 回复全部保存，
        保证会话历史两侧完整（不依赖内存窗口，重启后仍在）。重要性略低，
        语义召回时以群友的消息为主，不至于被 bot 闲聊刷屏。
        """
        content = content.strip()
        if not content:
            return
        self_id = str(self.config.get("qq", {}).get("self_id", ""))
        memory = Memory(
            content=f"{self.personality.name}：{content[:200]}",
            memory_type="episodic",
            importance=0.45,
            source_session=session_id,
            metadata={
                "sender_id": self_id,
                "sender_name": self.personality.name,
                "is_bot": True,
            },
        )

        async def _save():
            try:
                await self.memory_storage.store(memory)
            except Exception as e:
                logger.error(f"Failed to save bot memory: {e}")

        # 不阻塞消息处理主流程
        asyncio.create_task(_save())

    def _maybe_schedule_digest(self, session_id: str) -> None:
        """检查是否该生成群聊纪要：距上次纪要的新消息达到固定条数就调度后台总结"""
        if not self._digest_config.get("enabled", True):
            return
        if session_id in self._digest_tasks and not self._digest_tasks[session_id].done():
            return  # 已有纪要任务在跑，等它结束后重新计数

        since = self._last_digest_at.get(session_id, 0.0)
        window = self.context_manager.get_window(session_id)
        new_msgs = [m for m in window.messages if m.timestamp.timestamp() > since]
        if len(new_msgs) < self._digest_config.get("interval_messages", 20):
            return

        self._digest_tasks[session_id] = asyncio.create_task(
            self._generate_digest(session_id, new_msgs)
        )

    async def _generate_digest(self, session_id: str, messages: list) -> None:
        """把一批群聊消息压缩成纪要，存入长期记忆（后台执行，不阻塞消息流）"""
        try:
            if len(messages) < self._digest_config.get("min_messages", 10):
                return

            # 组织消息文本（带时间和发送者，供 LLM 总结）
            lines = []
            for m in messages:
                speaker = self.personality.name if m.is_bot else m.sender_name
                t = m.timestamp.strftime("%H:%M")
                lines.append(f"[{t}] {speaker}：{m.content[:80]}")
            chat_text = "\n".join(lines)

            provider = self.get_active_provider()
            if not provider:
                return

            from modules.llm.base import ChatRequest
            req = ChatRequest(
                temperature=0.4,
                max_tokens=self._digest_config.get("max_tokens", 200),
                top_p=0.9,
            )
            req.add_system(
                "你是群聊纪要助手。把下面的群聊记录总结成2-4句简短的纪要："
                "①主要聊了什么话题；②谁提到的（用昵称）；③有没有值得记住的信息（约定、喜好、八卦）。"
                "口语化、像群友转述，不要寒暄、不要列点、不要复述原话。"
            )
            req.add_user(f"【群聊记录】\n{chat_text}")

            resp = await provider.chat(req)
            summary = (resp.content or "").strip()
            if not summary:
                return

            now = datetime.now()
            memory = Memory(
                content=f"【群聊纪要 {now.month}月{now.day}日 {now.strftime('%H:%M')}】{summary}",
                memory_type="session_summary",
                importance=0.75,
                source_session=session_id,
                tags=["纪要"],
                metadata={"kind": "session_summary"},
            )
            await self.memory_storage.store(memory)
            logger.info(f"[纪要] {session_id}: {summary[:60]}...")

            # 纪要覆盖到的最后一条消息时间；之后新到的消息下次再总结
            self._last_digest_at[session_id] = max(m.timestamp.timestamp() for m in messages)
        except Exception as e:
            logger.error(f"Digest generation failed: {e}", exc_info=True)
        finally:
            self._digest_tasks.pop(session_id, None)

    async def _retrieve_memories(self, query: str, session_id: str, limit: int = None) -> list:
        """检索相关长期记忆：
        1. 该会话最近的群聊纪要（始终带上，bot 记得"最近群里聊过什么"，不会隔天失忆）
        2. 向量语义检索（TF-IDF + 余弦），失败或空时退回该会话最近记忆
        """
        limit = limit or self.memory_search_top_k

        memories: list = []
        try:
            memories = await self.memory_storage.semantic_search(
                query=query,
                session=session_id,
                limit=limit,
                half_life_days=self.memory_half_life_days,
                similarity_weight=self.memory_similarity_weight,
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")

        if not memories:
            # 兜底：该会话最近的高价值记忆
            try:
                memories = await self.memory_storage.retrieve_session_recent(session_id, limit=limit)
            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")
                memories = []

        # 最近纪要作为"近况"放最前，细节记忆在后（去重避免重复出现）
        try:
            digests = await self.memory_storage.retrieve_session_recent(
                session_id, limit=2, memory_type="session_summary"
            )
        except Exception as e:
            logger.debug(f"Digest retrieval skipped: {e}")
            digests = []
        if digests:
            digest_ids = {d.id for d in digests}
            memories = digests + [m for m in memories if m.id not in digest_ids]

        return memories

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
        background_tasks = (
            list(self._reply_tasks.values())
            + list(self._rich_media_tasks)
            + list(self._digest_tasks.values())
        )
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        self._reply_tasks.clear()
        self._rich_media_tasks.clear()
        self._digest_tasks.clear()

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
                "age_range": "20-25",
                "avatar_description": "",
                "background": "一个活泼可爱、喜欢聊天的女孩，喜欢分享有趣的事情和倾听朋友的故事。",
                "traits": {
                    "openness": 0.7,
                    "conscientiousness": 0.5,
                    "extraversion": 0.8,
                    "agreeableness": 0.85,
                    "neuroticism": 0.3
                },
                "interested_topics": ["美食", "旅行", "音乐", "电影", "八卦", "日常闲聊"],
                "bored_topics": ["广告推销", "政治敏感话题", "重复的无聊话题"],
                "humor_style": "dry",
                "taboo_topics": [],
                "catchphrases": [],
                "emoji_set": ["😅", "🤔", "😂", "👍", "🙄"],
                "speaking_style": {
                    "common_words": [],
                    "banned_words": [],
                    "filler_words": ["呃", "嗯", "那个", "这个"],
                    "max_reply_length": 60,
                    "use_ellipsis": True,
                    "use_emoji": True,
                    "emoji_frequency": 0.2,
                    "formality": 0.3,
                    "enthusiasm": 0.6,
                    "use_exclamation": True
                }
            },
            "llm": {
                "primary": {
                    "provider_type": "openai_compatible",
                    "api_key": "",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "temperature": 0.8,
                    "max_tokens": 500,
                    "timeout": 120,
                    "enabled": True
                }
            },
            "qq": {
                "ws_host": "0.0.0.0",
                "ws_port": 3001,
                "access_token": "",
                "self_id": ""
            },
            "speaking": {
                "base_probability": 0.02,
                "after_reply_probability": 0.8,
                "probability_duration": 120,
                "command_prefixes": ["/", "!", "#"],
                "trigger_keywords": ["爱丽丝", "小艾", "bot", "机器人"],
            },
            "thinking": {
                "base_delay": 2.0,
                "random_delay": {
                    "min": 0.5,
                    "max": 10.0
                }
            },
            "typing_style": {
                "enable_typo_generator": True,
                "typo_error_rate": 0.04,
                "min_chinese_chars": 3,
                "min_message_length": 10,
                "homophones": {}
            },
            "emotion": {
                "enabled": True,
                "decay_halflife": 600,
                "positive_boost": 0.1,
                "negative_decrease": 0.15,
                "positive_keywords": [],
                "negative_keywords": []
            },
            "attention": {
                "enabled": True,
                "initial_attention": 0.5,
                "attention_decay_halflife": 300,
                "attention_boost_step": 0.4,
                "attention_decrease_step": 0.1,
                "attention_decrease_threshold": 0.3,
                "max_tracked_users": 10,
                "enable_spillover": True,
                "spillover_ratio": 0.35,
                "spillover_decay_halflife": 90,
                "attention_spillover_min_trigger": 0.4
            },
            "fatigue": {
                "enabled": True,
                "reset_threshold": 300,
                "threshold_light": 3,
                "threshold_medium": 5,
                "threshold_heavy": 8,
                "decrease_light": 0.1,
                "decrease_medium": 0.2,
                "decrease_heavy": 0.35,
                "closing_probability": 0.3
            },
            "cooldown": {
                "enabled": True,
                "max_duration": 60,
                "trigger_threshold": 0.3
            },
            "conversation_floor": {
                "active_window_seconds": 45,
                "burst_window_seconds": 12,
                "burst_message_threshold": 4,
                "topic_shift_threshold": 0.12
            },
            "rich_media": {
                "enabled": True,
                "forward": {
                    "enabled": True,
                    "expand_when_undirected": True,
                    "max_nodes": 12,
                    "max_chars": 600,
                    "timeout": 5.0
                },
                "links": {
                    "enabled": True,
                    "directed_only": True,
                    "timeout": 3.0,
                    "max_bytes": 262144,
                    "max_redirects": 3,
                    "cache_ttl": 1800
                },
                "image": {
                    "ocr_enabled": False,
                    "ocr_action": "ocr_image",
                    "ocr_timeout": 5.0,
                    "to_text_scope": "all",
                    "to_text_prompt": "用一两句话（50字以内）描述图片内容，如果是表情包/梗图，重点说明它想表达的情绪和意图（如调侃、无语、赞同、嘲讽、自嘲），并结合前文对话判断含义。",
                    "to_text_context": True,
                    "context_window": 6,
                    "to_text_timeout": 60,
                    "max_download_bytes": 5242880,
                    "cache_ttl": 600,
                    "group_enabled": True,
                    "group_interval_seconds": 60,
                    "group_max_images": 4,
                    "vision": {
                        "enabled": False,
                        "provider_type": "openai",
                        "api_key": "",
                        "base_url": "https://api.openai.com/v1",
                        "model": "gpt-4o-mini",
                        "timeout": 60
                    }
                }
            },
            "memory": {
                "context_window_size": 50,
                "context_max_age_hours": 2.0,
                "enable_long_term_memory": True,
                "db_path": "data/memory.db",
                "retrieval_top_k": 5,
                "half_life_days": 30,
                "similarity_weight": 0.85,
                "digest": {
                    "enabled": True,
                    "interval_messages": 20,
                    "min_messages": 10,
                    "max_tokens": 200
                }
            },
            "search": {
                "enabled": False,
                "primary": "doubao",
                "min_interval_seconds": 60,
                "result_limit": 5,
                "trigger_keywords": [
                    "新闻", "热搜", "最新", "今天", "现在", "天气", "温度",
                    "价格", "多少钱", "比分", "谁赢了", "汇率", "油价",
                    "涨幅", "跌了", "涨停", "什么情况", "怎么回事", "发生了",
                    "结果", "发布", "官宣", "宣布", "百科", "是什么", "为什么",
                    # 游戏类时效性意图（游戏名无法枚举，用通用意图词覆盖）
                    "卡池", "抽卡", "UP", "复刻", "兑换码", "礼包码",
                    "角色", "是谁", "谁up",
                    "前瞻", "爆料", "开服", "公测", "内测", "强度",
                    "评测", "节奏榜", "T0", "版本", "赛程", "赛事", "联动", "攻略", "阵容",
                ],
                "llm": {
                    "api_key": "",
                    "base_url": "https://api.minimaxi.com/v1",
                    "model": "",
                    "timeout": 60
                },
                "backends": {
                    "bocha": {
                        "api_key": "",
                        "freshness": "noLimit",
                        "count": 5
                    },
                    "doubao": {
                        "api_key": "",
                        "time_range": "",
                        "count": 5
                    }
                }
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

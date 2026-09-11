"""
爱丽丝 (Alice) - 群聊AI伙伴
基于 AstrBot 设计的增强版主入口
"""
import asyncio
from dataclasses import replace
import logging
import random
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from core.event_bus import EventBus, Event, EventType
from core.adapter.qq_adapter import QQAdapter
from core.adapter.base import Message
from core.adapter.rich_media import RichMediaEnricher
from core.config_store import load_config as load_config_file, save_config as save_config_file

from modules.llm.openai_provider import create_provider, LLMProvider
from modules.memory.storage import MemoryStorage, AsyncMemoryStorage, Memory
from modules.memory.context import ContextManager
from modules.group_analysis import GroupDailyAnalysis
from modules.meme_manager import MemeManager

from modules.personality.personality import Personality
from modules.personality.emotional_state import EmotionalManager
from modules.personality.speaking_style import SpeakingStyleManager, SpeakingStyle
from modules.personality.typo import TypoGenerator

from modules.social.awareness import SocialAwarenessManager, SocialContext, TriggerDetector
from modules.social.attention import AttentionManager, AttentionKeywordsDetector
from modules.social.fatigue import FatigueManager
from modules.social.enhanced_decider import EnhancedSpeakingDecider
from modules.social.conversation_floor import ActionType, ConversationFloorManager
from modules.social.conversation_judge import ConversationJudge, ConversationJudgeResult

from modules.reply.generator import ReplyGenerator, ThinkingDelay, ResponseFilter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# 第三方组件的信息日志通常是英文且重复；保留警告和错误，避免污染 Bot 中文日志。
for _quiet_logger_name in (
    "asyncio",
    "jieba",
    "websockets",
    "websockets.server",
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
):
    logging.getLogger(_quiet_logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 第一人称代词：只认「我」会漏掉习惯说「俺」「咱」的人——他们的自述永远
# 匹配不上，画像里就只剩零碎日常，提炼不出东西。
FIRST_PERSON_PRONOUNS = ("我", "俺", "咱")
_SELF_PREDICATES = (
    "叫", "是", "喜欢", "爱", "的生日", "在", "住", "养", "家",
    "的工作", "今年", "对象", "男票", "女票", "老婆", "老公",
    # 家庭成员：家庭构成是关于这个人生活的稳定事实
    "爸", "妈", "爸妈", "父母",
)
PERSONAL_KEYWORDS = [
    pronoun + predicate
    for pronoun in FIRST_PERSON_PRONOUNS
    for predicate in _SELF_PREDICATES
] + ["记得我", "我叫什么"]


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
        self.conversation_judge: Optional[ConversationJudge] = None

        # 回复生成
        self.reply_generator: Optional[ReplyGenerator] = None
        self.thinking_delay: Optional[ThinkingDelay] = None
        self.response_filter: Optional[ResponseFilter] = None
        self.meme_manager: Optional[MemeManager] = None

        # 状态
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._start_time: float = 0
        # 每会话正在进行的回复生成任务（同一会话同时只生成一条回复）
        self._reply_tasks: dict[str, asyncio.Task] = {}
        self._reply_task_decisions: dict[str, dict] = {}
        # 只保护同一会话的快速入库/状态更新；LLM 目标判断在锁外执行，
        # 避免一次 8 秒的判断阻塞后续群消息进入上下文。
        self._message_ingest_locks: dict[str, asyncio.Lock] = {}
        # 目标判断按到达顺序完成，避免多个 LLM 调用乱序返回后旧消息反过来
        # 抢占新消息的回复任务；它不影响前面的快速入库。
        self._conversation_judge_locks: dict[str, asyncio.Lock] = {}
        # 同一会话的入站版本号。目标判断可能排队数秒，版本号用来在真正
        # 调模型前丢弃已经被后续群消息覆盖的普通候选，减少过期判断和费用。
        self._session_message_versions: dict[str, int] = {}
        self._rich_media_tasks: set[asyncio.Task] = set()
        self._meme_collect_tasks: set[asyncio.Task] = set()
        # 用户消息、Bot 回复等轻量记忆写入任务。单独跟踪，停止时等待/取消，
        # 避免 create_task 后进程退出导致最后几条记忆丢失。
        self._memory_tasks: set[asyncio.Task] = set()
        # 群聊纪要（滚动总结）：每 N 条消息把新聊天压缩成一条长期记忆，隔天也不会忘
        self._digest_config: dict = {}
        self._last_digest_at: dict[str, float] = {}  # session -> 上次纪要覆盖到的消息时间戳
        self._digest_tasks: dict[str, asyncio.Task] = {}  # 进行中的纪要任务
        # 群聊日报：独立保存完整群消息，避免长期记忆筛选导致日报只看到少数消息。
        self._group_analysis_config: dict = {}
        self._group_analysis_tasks: dict[str, asyncio.Task] = {}
        self._group_analysis_write_tasks: set[asyncio.Task] = set()
        self._last_group_analysis_cleanup: float = 0.0

        # 媒体轨迹：记录最近消息(时间/sender/是否纯图片/图片url)，用于判断"连图"。
        self._media_trail: dict[str, deque] = {}
        self._media_trail_limit = 30

        # 提炼不出稳定特征的人 → 记下当时的素材规模，素材没变就不再重复问 LLM
        self._profile_failed_material: dict[tuple, tuple] = {}
        # Dashboard 和 QQ 可能在不同事件循环；用线程锁防止手动/定时提炼并发覆盖画像。
        self._profile_distill_guard = threading.Lock()
        # 黑话自动清理的上次执行时间（进程内首次启动时立即检查）
        self._last_slang_cleanup: float = 0.0

    @property
    def _image_group_config(self) -> dict:
        return self.config.get("image", {}) or {} if self.config else {}

    async def initialize(self) -> None:
        """初始化 Bot"""
        logger.info("正在初始化 Alice（爱丽丝）……")
        self._start_time = time.time()

        # 1. 加载配置
        self._load_config()
        logger.info("✓ 配置已加载")

        # 2. 初始化事件总线
        self.event_bus = EventBus()
        self._tasks.append(asyncio.create_task(self.event_bus.start()))
        logger.info("✓ 事件总线已初始化")

        # 3. 初始化 LLM
        self._init_llm()
        logger.info("✓ 语言模型提供商已初始化")

        # 4. 初始化记忆系统
        self._init_memory()
        logger.info("✓ 记忆系统已初始化")

        # 4.5 初始化本地表情库；素材不进入普通记忆和 LLM 召回。
        self._init_meme_manager()
        logger.info("✓ 表情库已初始化")

        # 5. 初始化人格系统
        self._init_personality()
        logger.info("✓ 人格系统已初始化")

        # 6. 初始化社交感知
        self._init_social()
        logger.info("✓ 社交感知系统已初始化")

        # 7. 初始化回复生成
        self._init_reply_generator()
        logger.info("✓ 回复生成器已初始化")

        # 8. 初始化 QQ 适配器
        self._init_qq_adapter()
        logger.info("✓ QQ 适配器已初始化")

        elapsed = time.time() - self._start_time
        logger.info(f"爱丽丝初始化完成，耗时 {elapsed:.2f} 秒")

    def _load_config(self) -> None:
        """加载配置，如果没有则创建默认配置"""
        config_file = Path(self.config_path)

        # 确保配置目录存在
        config_file.parent.mkdir(parents=True, exist_ok=True)

        if not config_file.exists():
            # 创建默认配置文件
            logger.info("未找到配置文件，正在创建默认配置...")
            default_config = self._get_default_config()
            save_config_file(default_config, config_file)
            self.config = default_config
            logger.info(f"✓ 默认配置已创建：{config_file}")
        else:
            self.config = load_config_file(config_file)

        # 加载人格配置
        personality_config = self.config.get("personality", {})
        self.personality = Personality.from_dict(personality_config)

        logger.info(
            f"机器人名称：{self.personality.name}，昵称：{self.personality.nickname}"
        )

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
                            "top_p": provider_config.get("top_p", 0.9),
                        }
                    )
                    self.llm_providers[name] = provider
                    logger.info(
                        f"✓ 语言模型提供商「{name}」：地址={provider_config.get('base_url')}，"
                        f"模型={provider_config.get('model')}"
                    )
                except Exception as e:
                    logger.error(f"✗ 初始化语言模型提供商「{name}」失败：{e}")

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
            logger.error("没有可用的语言模型提供商！")

    def get_active_provider(self) -> Optional[LLMProvider]:
        """获取当前激活的 provider"""
        return self.llm_providers.get(self.active_provider_id)

    def _init_memory(self) -> None:
        """初始化记忆系统"""
        memory_config = self.config.get("memory", {})
        self.long_term_memory_enabled = bool(
            memory_config.get("enable_long_term_memory", True)
        )
        self.memory_share_across_sessions = bool(
            memory_config.get("share_across_sessions", False)
        )

        self.memory_storage = AsyncMemoryStorage(
            MemoryStorage(
                memory_config.get("db_path", "data/memory.db"),
                share_across_sessions=self.memory_share_across_sessions,
            )
        )

        logger.info(
            "✓ 长期记忆：启用=%s，跨会话共享=%s",
            self.long_term_memory_enabled,
            self.memory_share_across_sessions,
        )

        # 嵌入+重排服务（无 key 时自动回退 TF-IDF）
        from modules.memory.embedding import EmbeddingRerankService
        embed_config = memory_config.get("embedding", {}) or {}
        embed_service = EmbeddingRerankService(embed_config)
        self.memory_storage._storage.set_embedding_service(embed_service)
        if embed_service.enabled:
            logger.info(
                f"✓ 嵌入服务已启用：向量模型={embed_service.embed_model}，"
                f"重排模型={embed_service.rerank_model}"
            )
        else:
            logger.info("嵌入服务未配置（无 API 密钥），记忆检索使用 TF-IDF 回退")

        self.context_manager = ContextManager(
            max_messages=memory_config.get("context_window_size", 30),
            max_age_hours=memory_config.get("context_max_age_hours", 2)
        )

        # 记忆检索参数（向量检索 + 时间衰减）
        self.memory_search_top_k = memory_config.get("retrieval_top_k", 5)
        self.memory_half_life_days = memory_config.get("half_life_days", 30)
        self.memory_similarity_weight = memory_config.get("similarity_weight", 0.85)
        self.memory_decay_presets = {
            "semantic": {"half_life_days": 90, "max_age_days": 365},
            "session_summary": {"half_life_days": 45, "max_age_days": 180},
            "episodic": {"half_life_days": 14, "max_age_days": 90},
        }
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
        logger.info(
            f"✓ 群聊纪要：启用={self._digest_config['enabled']}，"
            f"每{self._digest_config['interval_messages']}条消息总结一次"
        )

        # 群聊日报：默认不自动发送，避免升级后突然增加 LLM 调用和群消息；
        # Dashboard 手动触发不受 auto_enabled 影响。
        analysis_cfg = memory_config.get("group_analysis", {}) or {}
        auto_times = analysis_cfg.get("auto_times", analysis_cfg.get("auto_time", "23:50"))
        if isinstance(auto_times, str):
            auto_times = [auto_times]
        if not isinstance(auto_times, list):
            auto_times = ["23:50"]
        self._group_analysis_config = {
            "enabled": bool(analysis_cfg.get("enabled", True)),
            "auto_enabled": bool(analysis_cfg.get("auto_enabled", False)),
            "auto_times": [str(value).strip() for value in auto_times if str(value).strip()][:8]
            or ["23:50"],
            "min_messages": max(3, min(5000, int(analysis_cfg.get("min_messages", 10)))),
            "max_messages": max(20, min(5000, int(analysis_cfg.get("max_messages", 500)))),
            "max_prompt_chars": max(4000, min(60000, int(analysis_cfg.get("max_prompt_chars", 24000)))),
            "max_topics": max(1, min(10, int(analysis_cfg.get("max_topics", 5)))),
            "max_quotes": max(1, min(8, int(analysis_cfg.get("max_quotes", 3)))),
            "max_titles": max(1, min(8, int(analysis_cfg.get("max_titles", 5)))),
            "max_tokens": max(400, min(6000, int(analysis_cfg.get("max_tokens", 2400)))),
            "max_report_chars": max(1200, min(10000, int(analysis_cfg.get("max_report_chars", 6000)))),
            "retention_days": max(3, min(365, int(analysis_cfg.get("retention_days", 30)))),
            "send_report": bool(analysis_cfg.get("send_report", True)),
            "retries": max(1, min(8, int(analysis_cfg.get("retries", 2)))),
            "avatars_enabled": bool(analysis_cfg.get("avatars_enabled", True)),
            "avatar_cache_days": max(1, min(365, int(analysis_cfg.get("avatar_cache_days", 7)))),
            "avatar_max_count": max(1, min(30, int(analysis_cfg.get("avatar_max_count", 12)))),
            "avatar_timeout": max(1.0, min(20.0, float(analysis_cfg.get("avatar_timeout", 6)))),
        }
        logger.info(
            "✓ 群聊日报：启用=%s，自动生成=%s，时间=%s",
            self._group_analysis_config["enabled"],
            self._group_analysis_config["auto_enabled"],
            ",".join(self._group_analysis_config["auto_times"]),
        )

        # 用户画像配置：从情景记忆提炼每个群友的稳定个人特征（semantic 记忆）
        profile_cfg = memory_config.get("profile", {}) or {}
        self._profile_config = {
            "enabled": profile_cfg.get("enabled", True),
            "interval_minutes": int(profile_cfg.get("interval_minutes", 30)),
            "min_facts": max(1, int(profile_cfg.get("min_facts", 2))),
            # 日常发言也参与提炼：有足够多的日常发言时，即使自述少也能画像
            "daily_min_messages": max(3, int(profile_cfg.get("daily_min_messages", 8))),
            "daily_window_days": max(3, int(profile_cfg.get("daily_window_days", 15))),
            # 已有画像后，日常新增消息达到该数量才重炼（避免每次发言都刷）
            "refresh_daily_count": max(1, int(profile_cfg.get("refresh_daily_count", 5))),
            # MBTI 是额外的娱乐向分析；已有结果时积累更多新素材再刷新，降低调用量。
            "mbti_refresh_daily_count": max(
                1, int(profile_cfg.get("mbti_refresh_daily_count", 12))
            ),
            # MBTI 性格倾向分析（与画像共用素材，每次更新画像多一次 LLM 调用）
            "mbti_enabled": bool(profile_cfg.get("mbti_enabled", True)),
            # 对短句/回复句补充同会话上下文，只用于消歧，不作为目标用户证据
            "context_enabled": bool(profile_cfg.get("context_enabled", True)),
            "context_neighbors": max(
                1, min(3, int(profile_cfg.get("context_neighbors", 2)))
            ),
            "context_max_targets": max(
                1, min(20, int(profile_cfg.get("context_max_targets", 12)))
            ),
        }
        self._last_profile_distill: float = 0.0
        logger.info(
            f"✓ 用户画像：启用={self._profile_config['enabled']}，"
            f"每{self._profile_config['interval_minutes']}分钟提炼一次（自述+日常发言）"
        )

        # 群聊黑话：定时从聊天记录提取群内特有的梗/称呼，回复时按需注入
        slang_cfg = memory_config.get("slang", {}) or {}
        cleanup_cfg = slang_cfg.get("cleanup", {}) or {}
        self._slang_config = {
            "enabled": bool(slang_cfg.get("enabled", True)),
            "interval_hours": max(1, int(slang_cfg.get("interval_hours", 24))),
            "lookback_hours": max(6, int(slang_cfg.get("lookback_hours", 48))),
            "max_inject": max(1, int(slang_cfg.get("max_inject", 8))),
            "cleanup": {
                "enabled": bool(cleanup_cfg.get("enabled", True)),
                "interval_hours": max(1, int(cleanup_cfg.get("interval_hours", 24))),
                "lookback_hours": max(24, int(cleanup_cfg.get("lookback_hours", 168))),
                "max_entries": max(1, int(cleanup_cfg.get("max_entries", 40))),
            },
        }
        self._last_slang_extract: float = 0.0
        logger.info(
            f"✓ 群聊黑话：启用={self._slang_config['enabled']}，"
            f"每{self._slang_config['interval_hours']}小时提取一次；"
            f"自动清理={self._slang_config['cleanup']['enabled']}，"
            f"每{self._slang_config['cleanup']['interval_hours']}小时检查一次"
        )

    def _init_meme_manager(self) -> None:
        """初始化本地表情包素材库。"""
        self.meme_manager = MemeManager(
            self.config.get("meme_manager", {}) or {},
            base_dir=Path(__file__).resolve().parent,
        )
        logger.info(
            "✓ 表情库：启用=%s，自动收集=%s，自动发送=%s，总数=%d",
            self.meme_manager.enabled,
            self.meme_manager.auto_collect_enabled,
            self.meme_manager.auto_send_enabled,
            self.meme_manager.stats().get("total", 0),
        )

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
        style_config = self.config.get("personality", {}).get("speaking_style", {}) or {}
        speaking_style = SpeakingStyle.from_dict(style_config)

        self.speaking_style_manager = SpeakingStyleManager(
            speaking_style
        )

        # 错字生成器
        typing_config = self.config.get("typing_style", {})
        self.typo_generator = TypoGenerator(
            typo_error_rate=typing_config.get("typo_error_rate", 0.04),
            homophones=typing_config.get("homophones", {}),
            min_chinese_chars=typing_config.get("min_chinese_chars", 3),
            min_message_length=typing_config.get("min_message_length", 10),
        )

    def _refresh_personality_runtime(self) -> None:
        """热更新人格提示词和说话风格，但保留情绪/注意力等会话状态。"""
        personality_config = self.config.get("personality", {}) or {}
        self.personality = Personality.from_dict(personality_config)

        style_config = personality_config.get("speaking_style", {}) or {}
        speaking_style = SpeakingStyle.from_dict(
            style_config if isinstance(style_config, dict) else {}
        )
        if self.speaking_style_manager is None:
            self.speaking_style_manager = SpeakingStyleManager(speaking_style)
        else:
            self.speaking_style_manager.style = speaking_style
            self.speaking_style_manager.emoji_set = []

        if self.conversation_floor_manager:
            direct_limit = speaking_style.direct_max_reply_length
            if direct_limit is None:
                direct_limit = speaking_style.max_reply_length
            try:
                self.conversation_floor_manager.direct_answer_max_chars = max(
                    1, int(direct_limit)
                )
            except (TypeError, ValueError):
                self.conversation_floor_manager.direct_answer_max_chars = 80

        if self.reply_generator:
            self.reply_generator.personality_prompt = self.personality.build_persona_prompt()
            self.reply_generator.style_manager = self.speaking_style_manager
            self.reply_generator.bot_name = self.personality.name
            self.reply_generator.taboo_topics = [
                str(topic).strip()
                for topic in (self.personality.taboo_topics or [])
                if str(topic).strip()
            ]

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
            bored_topics=self.personality.bored_topics,
            taboo_topics=self.personality.taboo_topics,
        )

        # 群聊发言权：判断谁在和谁说话、当前插嘴成本以及候选行为。
        personality_style_config = (
            self.config.get("personality", {}).get("speaking_style", {}) or {}
        )
        self.conversation_floor_manager = ConversationFloorManager(
            active_window_seconds=floor_config.get("active_window_seconds", 45),
            burst_window_seconds=floor_config.get("burst_window_seconds", 12),
            burst_message_threshold=floor_config.get("burst_message_threshold", 4),
            topic_shift_threshold=floor_config.get("topic_shift_threshold", 0.12),
            settle_window_seconds=floor_config.get("settle_window_seconds", 0.7),
            settle_max_seconds=floor_config.get("settle_max_seconds", 2.4),
            other_target_context_seconds=floor_config.get(
                "other_target_context_seconds", 900
            ),
            direct_answer_max_chars=personality_style_config.get(
                "direct_max_reply_length", 80
            ),
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

        # 每条群消息先做一次轻量的目标/接话意图判断；随机发言概率仍负责
        # 拟人化节奏，但不再独自决定“这句话是不是在对 Bot 说”。
        self._configure_conversation_judge()

    def _configure_conversation_judge(self) -> None:
        """按当前配置创建目标判断器。

        这个组件是无状态的，热更新时直接替换实例即可；正在执行的旧请求仍
        会持有旧实例，不会因为 Web 保存配置而被中途改写。
        """
        judge_config = self.config.get("conversation_judge", {}) or {}
        judge_provider_id = str(
            judge_config.get("provider_id") or self.active_provider_id or ""
        )
        judge_provider = self.llm_providers.get(judge_provider_id)
        if judge_provider is None:
            judge_provider = self.get_active_provider()
        self.conversation_judge = ConversationJudge(
            provider=judge_provider,
            bot_id=str(self.config.get("qq", {}).get("self_id", "") or ""),
            bot_name=self.personality.name,
            enabled=judge_config.get("enabled", True),
            timeout=judge_config.get("timeout", 8.0),
            max_tokens=judge_config.get("max_tokens", 220),
            context_messages=judge_config.get("context_messages", 16),
        )
        logger.info(
            "✓ 群聊目标判断：启用=%s，提供商=%s，超时=%.1f秒",
            self.conversation_judge.enabled,
            judge_provider_id or "active",
            self.conversation_judge.timeout,
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
            bot_name=self.personality.name,
            taboo_topics=self.personality.taboo_topics,
            meme_manager=self.meme_manager,
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
                logger.info(f"✓ 联网搜索语言模型：{tool_llm.model}")
            except Exception as e:
                logger.error(f"✗ 初始化联网搜索语言模型失败：{e}")
                tool_llm = None

        if search_client.available:
            logger.info(
                f"✓ 联网搜索已启用：主后端={search_client.primary}，"
                f"备用后端={list(search_client._backends.keys())}"
            )
        return search_client, tool_llm

    def _init_vision_provider(self) -> Optional[Any]:
        """初始化视觉模型 Provider（图片→文字描述）。未启用/失败返回 None，不阻断启动。"""
        image_config = self.config.get("image", {}) or {}
        rich_media_image = (self.config.get("rich_media", {}) or {}).get("image", {}) or {}
        image_config = {**rich_media_image, **image_config}
        if isinstance(rich_media_image.get("vision"), dict) and isinstance(
            image_config.get("vision"), dict
        ):
            image_config["vision"] = {
                **rich_media_image["vision"],
                **image_config["vision"],
            }
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
                f"✓ 视觉模型提供商：地址={vision_config.get('base_url')}，"
                f"模型={vision_config.get('model')}"
            )
            return provider
        except Exception as e:
            logger.error(f"✗ 初始化视觉模型提供商失败：{e}")
            return None

    def _get_rich_media_config(self) -> dict:
        """合并兼容的 rich_media/image 配置，供启动和热更新共用。"""
        qq_config = self.config.get("qq", {}) or {}
        rich_media_config = dict(
            self.config.get("rich_media", qq_config.get("rich_media", {})) or {}
        )
        image_config = self.config.get("image", {}) or {}
        if image_config:
            merged_image = {**rich_media_config.get("image", {}), **image_config}
            base_vision = rich_media_config.get("image", {}).get("vision", {})
            override_vision = image_config.get("vision", {})
            if isinstance(base_vision, dict) and isinstance(override_vision, dict):
                merged_image["vision"] = {**base_vision, **override_vision}
            rich_media_config["image"] = merged_image
        return rich_media_config

    def _init_qq_adapter(self) -> None:
        """初始化 QQ 适配器"""
        qq_config = self.config.get("qq", {}) or {}
        rich_media_config = self._get_rich_media_config()
        adapter_config = {
            **qq_config,
            "rich_media": rich_media_config,
        }
        self.qq_adapter = QQAdapter(
            config=adapter_config,
            on_message=self._handle_message,
            vision_provider=self._init_vision_provider(),
        )

    def apply_runtime_config(self) -> None:
        """重新读取 Web 配置并热更新可安全替换的运行组件。"""
        self._load_config()
        self._refresh_personality_runtime()
        self._init_llm()

        if self.reply_generator:
            self.reply_generator.llm = self.get_active_provider()
            search_client, tool_llm = self._init_search()
            self.reply_generator.search_client = search_client
            self.reply_generator.tool_llm = tool_llm

        # 目标/接话判断也使用独立 provider 配置；否则 Web 切换 provider、
        # 超时或开关后，当前进程仍会继续使用启动时的旧判断器。
        self._configure_conversation_judge()

        if self.meme_manager:
            self.meme_manager.update_config(self.config.get("meme_manager", {}))

        if self.qq_adapter:
            qq_config = self.config.get("qq", {}) or {}
            rich_media_config = self._get_rich_media_config()
            self.qq_adapter.config = {
                **qq_config,
                "rich_media": rich_media_config,
            }
            self.qq_adapter.self_id = qq_config.get("self_id", "")
            self.qq_adapter.access_token = qq_config.get("access_token", "")
            self.qq_adapter.rich_media_enricher = RichMediaEnricher(
                rich_media_config,
                self.qq_adapter.call_api,
                vision_provider=self._init_vision_provider(),
            )
        logger.info("✓ Web 配置已热更新（正在进行的请求继续使用旧实例）")

    async def _handle_message(self, message: Message) -> None:
        """处理接收到的消息

        拆成两条路径，避免"思考期间看不到新消息"的失真：
        - 快速路径（立即执行）：状态更新 + 消息写入上下文 + 发言决策，不阻塞接收循环
        - 慢路径（后台任务）：思考延迟 + 群聊收尾窗口 + LLM 生成 + 发送，期间新消息仍会进入上下文
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
        # 富媒体增强要知道“可能是对 Bot 说”还是普通分享，但这里的
        # continuing 只是旧规则线索，不能在动态目标判断前把消息定性。
        rich_directed = (
            message.message_type == "private"
            or message.mentioned_me
            or is_reply_to_bot
        )
        logger.info(f"[{message.group_id or '私聊'}] {message.sender_name}：{message.content[:50]}……")

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
        )
        recent_context_for_judgement = []
        ingest_lock = self._message_ingest_locks.setdefault(
            session_id, asyncio.Lock()
        )
        await ingest_lock.acquire()
        message_version = None
        try:
            # 进程重启后，先恢复该会话最近仍在上下文有效期内的 episodic 消息，
            # 再追加当前消息；这样首条消息不会让 Bot 突然失去刚才的对话。
            await self._restore_recent_context(
                session_id, current_message_id=message.message_id
            )

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
                mentioned_user_ids=message.mentioned_user_ids,
                directed_to_bot=directed_to_bot,
            )
            recent_context_for_judgement = self.context_manager.get_window(
                session_id
            ).get_recent(30)
            self._session_message_versions[session_id] = (
                self._session_message_versions.get(session_id, 0) + 1
            )
            message_version = self._session_message_versions[session_id]

            # 长期记忆（内部已异步后台执行）
            self._store_long_term_memory(message, session_id)

            # 群聊纪要：距上次总结的新消息达到固定条数 → 后台压缩成纪要存长期记忆
            self._maybe_schedule_digest(session_id)

            # 群日报单独保存完整群消息，不参与普通长期记忆筛选和召回。
            self._store_group_analysis_message(message, session_id)

            # 记录媒体轨迹，供"连图整体识别"判断连续纯图片
            self._record_media_trail(message)
        except Exception as e:
            logger.error(f"快速消息处理路径出错：{e}", exc_info=True)
        finally:
            # 只锁到快速入库结束；动态判断和回复生成必须在锁外，后续消息
            # 才能继续进入上下文，供收尾窗口观察。
            ingest_lock.release()

        # 目标和接话意图由独立的轻量模型动态判断。它只返回结构化结果，
        # 不生成回复；失败时由 _decide_reply 使用保守的旧逻辑回退。
        judge_lock = self._conversation_judge_locks.setdefault(
            session_id, asyncio.Lock()
        )
        async with judge_lock:
            # 非显式定向的旧消息不值得在队列里继续消耗一次 LLM 判断：后面
            # 的消息才是当前群聊状态。显式 @/引用 Bot 仍保留，避免用户的
            # 定向问题因为旁边新消息到达而被静默。
            explicit_directed = (
                message.message_type == "private"
                or message.mentioned_me
                or is_reply_to_bot
            )
            if (
                message.message_type == "group"
                and not explicit_directed
                and self._session_message_versions.get(session_id, 0)
                != message_version
            ):
                conversation_judgement = ConversationJudgeResult.unavailable(
                    "stale_message"
                )
                logger.info(
                    "[目标判断] 跳过已过期的普通消息：%s",
                    message.message_id or message.sender_id,
                )
            else:
                conversation_judgement = await self._judge_conversation_message(
                    message,
                    continuation_hint=continuing,
                    heuristic_reasons=rich_reasons,
                    recent_messages=recent_context_for_judgement,
                )
        if conversation_judgement.available:
            self.context_manager.update_message_analysis(
                session_id,
                message.message_id,
                directed_to_bot=(conversation_judgement.target == "bot"),
                target=conversation_judgement.target,
                intent=conversation_judgement.intent,
                confidence=conversation_judgement.confidence,
                reason=conversation_judgement.reason,
            )
            rich_directed = (
                rich_directed
                or (
                    conversation_judgement.target == "bot"
                    and conversation_judgement.should_reply
                )
            )

        # 富媒体增强可能涉及 NapCat API 或安全网页预览。它在后台执行，决策仍走
        # 快速路径；真正生成回复前会等待本条消息的增强结果。
        enrichment_task = None
        if message.rich_type:
            enrichment_task = asyncio.create_task(
                self._enrich_context_message(message, rich_directed)
            )
            self._rich_media_tasks.add(enrichment_task)
            enrichment_task.add_done_callback(self._rich_media_tasks.discard)

        # 表情自动收集放在富媒体增强之后：如果视觉识别已经给出简短描述，
        # 描述会一并写进素材元数据；整个过程独立于回复任务，不拖慢发言决策。
        if (
            self.meme_manager
            and self.meme_manager.auto_collect_enabled
            and any(
                getattr(segment, "type", "") in {"image", "mface"}
                for segment in (message.segments or [])
            )
        ):
            collect_task = asyncio.create_task(
                self._collect_meme_after_enrichment(message, enrichment_task)
            )
            self._meme_collect_tasks.add(collect_task)
            collect_task.add_done_callback(self._meme_collect_tasks.discard)

        # === 2. 发言决策（目标判断已完成，下面只做本地策略合并） ===
        decision = self._decide_reply(
            message,
            is_reply_to_bot,
            continuing,
            conversation_judgement=conversation_judgement,
        )
        if not decision:
            return
        decision["enrichment_task"] = enrichment_task
        # 这里必须记录“触发本次候选回复的那条消息”，不能记录判断完成时
        # 的最新消息。目标判断可能耗时数秒；如果期间又来了群消息，记录
        # latest 会让旧任务误以为自己仍然是最新候选，继续回答旧话题。
        decision["context_marker"] = self._context_marker_for_message(
            session_id, message
        )

        # === 3. 调度回复生成（后台任务，避免阻塞接收循环） ===
        current = self._reply_tasks.get(session_id)
        if current and not current.done():
            current_decision = self._reply_task_decisions.get(session_id) or {}
            if (
                decision["direction"].startswith("to_bot")
                or current_decision.get("direction") == "group"
            ):
                # 新消息需要重新成为候选目标：定向消息优先替换旧任务；普通
                # 插话任务也不能继续回答已经过期的旧消息。
                current.cancel()
                task = asyncio.create_task(
                    self._compose_and_send(message, decision)
                )
                self._reply_tasks[session_id] = task
                self._reply_task_decisions[session_id] = decision
            else:
                # 定向回复保留原问题；普通新消息会在上面的分支替换旧插话。
                return
        else:
            task = asyncio.create_task(
                self._compose_and_send(message, decision)
            )
            self._reply_tasks[session_id] = task
            self._reply_task_decisions[session_id] = decision

    async def _collect_meme_after_enrichment(
        self,
        message: Message,
        enrichment_task: asyncio.Task | None = None,
    ) -> None:
        """等待图片增强完成后，把直接图片交给本地表情库。"""
        try:
            if enrichment_task:
                await enrichment_task
            if self.meme_manager and self.qq_adapter:
                await self.meme_manager.collect_message(message, self.qq_adapter)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("表情自动收集任务失败：%s", exc)

    async def send_meme(
        self,
        session_id: str,
        *,
        meme_id: str = "",
        category: str = "",
        reply_to_id: str = "",
        automatic: bool = False,
    ) -> dict[str, Any]:
        """发送一张表情包；Web 手动发送和 LLM 自动发送共用此链路。"""
        manager = self.meme_manager
        if not manager or not manager.enabled or not self.qq_adapter:
            return {"success": False, "error": "表情包功能未启用"}
        if automatic and not manager.auto_send_enabled:
            return {"success": False, "error": "自动发送未启用"}
        session_id = str(session_id or "").strip()
        if not session_id.startswith(("group_", "private_")):
            return {"success": False, "error": "会话格式不正确"}
        if automatic:
            available, remaining = manager.auto_send_available(session_id)
            if not available:
                if not manager.auto_send_enabled:
                    return {"success": False, "error": "自动发送未启用"}
                return {
                    "success": False,
                    "error": f"自动发图冷却中，还需 {remaining:.0f} 秒",
                }

        # 发送前同步磁盘最新清单：自动收集的新素材可能未进运行内存，
        # 避免「列表能看到、发送却说不存在」。
        await asyncio.to_thread(manager.reload)

        if meme_id:
            item = await asyncio.to_thread(manager.resolve, meme_id)
            if not item:
                return {"success": False, "error": "指定的表情包不存在"}
        else:
            item = await asyncio.to_thread(manager.choose, category, session_id)
        if not item:
            return {"success": False, "error": "没有找到可发送的表情包"}
        payload = await asyncio.to_thread(manager.get_bytes, item.get("id", ""))
        if not payload:
            return {"success": False, "error": "表情文件不存在"}
        _, image_bytes = payload
        try:
            # 历史清单可能还留有 GIF/WebP；发送前统一转成 PNG，避免 NapCat
            # 对动图格式的兼容差异，也保证新旧表情包走同一种发送格式。
            image_bytes = await asyncio.to_thread(manager.to_png_bytes, image_bytes)
        except Exception as exc:
            logger.warning("[表情库] 图片转 PNG 失败：%s", exc)
            return {"success": False, "error": "表情图片无法转换为 PNG", "item": item}
        try:
            send_with_id = getattr(self.qq_adapter, "send_image_with_id", None)
            if callable(send_with_id):
                success, outbound_message_id = await send_with_id(
                    session_id,
                    image_bytes,
                    reply_to_id=(reply_to_id or None),
                )
            else:
                success = await self.qq_adapter.send_image(
                    session_id,
                    image_bytes,
                    reply_to_id=(reply_to_id or None),
                )
                outbound_message_id = str(
                    getattr(self.qq_adapter, "last_sent_message_id", "") or ""
                )
        except Exception as exc:
            logger.warning("[表情库] 发送失败：%s", exc)
            return {"success": False, "error": str(exc)}
        if not success:
            return {"success": False, "error": "QQ 适配器发送失败", "item": item}
        if automatic:
            manager.remember_choice(session_id, item.get("id", ""))
            manager.record_auto_send(session_id)
        await asyncio.to_thread(manager.record_use, item.get("id", ""))
        logger.info("[表情库] 发送成功：分类=%s，会话=%s", item.get("category", ""), session_id)
        return {
            "success": True,
            "item": item,
            "message_id": str(outbound_message_id or ""),
        }

    async def _judge_conversation_message(
        self,
        message: Message,
        *,
        continuation_hint: bool = False,
        heuristic_reasons: list[str] | None = None,
        recent_messages: list[Any] | None = None,
    ) -> ConversationJudgeResult:
        """对当前消息做动态目标/接话判断。"""
        if message.message_type == "private":
            return ConversationJudgeResult(
                target="bot",
                intent="answer" if self.trigger_detector else "follow_up",
                should_reply=True,
                confidence=0.99,
                reference_message_id=str(message.message_id or ""),
                target_user_id=str(message.sender_id or ""),
                reason="私聊消息默认进入对话",
                available=True,
                evidence={"source": "private"},
            )

        judge = getattr(self, "conversation_judge", None)
        if not judge:
            return ConversationJudgeResult.unavailable("not_initialized")

        try:
            recent = (
                list(recent_messages)
                if recent_messages is not None
                else self.context_manager.get_window(message.session_id).get_recent(
                    judge.context_messages + 2
                )
            )
            result = await judge.judge(
                message,
                recent,
                heuristic_signals={
                    "mentioned_me": message.mentioned_me,
                    "mentioned_others": message.mentioned_others,
                    "reply_to_me": self._is_reply_to_bot(message),
                    "reply_to_qq": message.reply_to_qq or "",
                    "continuation_hint": continuation_hint,
                    "trigger_reasons": heuristic_reasons or [],
                },
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[目标判断] 准备判断失败，回退旧决策：%s", exc)
            return ConversationJudgeResult.unavailable(str(exc))

    def _decide_reply(
        self,
        message: Message,
        is_reply_to_bot: bool,
        continuing: bool = False,
        conversation_judgement: ConversationJudgeResult | None = None,
    ) -> Optional[dict]:
        """发言决策（同步、快速）：返回回复参数，不发言则返回 None"""
        session_id = message.session_id
        group_id = message.group_id or ""
        group_config = self.config.get("groups", {}).get(str(group_id), {})
        is_private = message.message_type == "private"
        self_id = str(self.config.get("qq", {}).get("self_id", ""))
        quoted_bot = bool(message.reply_to_qq) and bool(self_id) and str(message.reply_to_qq) == self_id

        judge = (
            conversation_judgement.to_dict()
            if isinstance(conversation_judgement, ConversationJudgeResult)
            else {}
        )
        judge_available = bool(judge.get("available"))
        judge_target = str(judge.get("target") or "")
        judge_should_reply = bool(judge.get("should_reply"))
        judge_intent = str(judge.get("intent") or "")

        # 旧规则只作为模型不可用时的回退线索；不再因为“最近回复过同一人”
        # 自动把任意无标记消息定性成连续对话。
        heuristic_talking_to_others = (
            bool(message.mentioned_others)
            or (bool(message.reply_to_qq) and not quoted_bot)
        )
        if judge_available:
            # 动态判断是最终的目标依据；程序提取的 @/回复信息仍会放入模型上下文。
            if not judge_should_reply:
                logger.info(
                    "[目标判断] 本轮不参与：目标=%s，意图=%s，理由=%s",
                    judge_target,
                    judge_intent,
                    judge.get("reason", ""),
                )
                return None
            talking_to_others = judge_target == "other"
            continuing = judge_target == "bot" and judge_intent in {
                "follow_up", "acknowledge", "answer"
            }
        else:
            # 判断服务不可用时保守降级：显式信号仍可回复，但不自动续接。
            talking_to_others = heuristic_talking_to_others
            continuing = False

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
        context.extra["conversation_judgement"] = judge
        context.extra["dynamic_target"] = judge_target if judge_available else ""
        context.extra["group_base_probability"] = group_config.get(
            "speaking_probability"
        )
        # 话题兴趣可以参考安全渲染后的富媒体摘要，但强制触发结果已经在上面锁定。
        context.message_content = message.content
        context = self.social_awareness.analyze(context)

        # 话题兴趣不仅影响“想不想插话”，极端厌烦/禁忌话题还应让情绪和
        # 发言权系统看到；明确问到时仍允许生成一条边界回复。
        if context.topic_relevance >= 0.75:
            self.emotional_manager.trigger_event(session_id, "interesting")
        elif context.topic_relevance <= 0.25:
            self.emotional_manager.trigger_event(session_id, "boring")

        # 发言权和行为计划。当前消息已在快速路径进入上下文，因此可直接分析消息拓扑。
        # 解析失败的占位文本（segment 都识别不出）没有可用语义输入，调 LLM
        # 只会产出复读上文 / 输出自己名字这种垃圾回复；按"沉默"对待，直接
        # 跳过整个 bot 决策链，避免触发后续 stale-plan 链路。
        if (message.content or "").strip() == "[无法识别的消息]":
            logger.info(
                f"[静默] 无法识别的消息，跳过机器人决策：{message.content}"
            )
            return

        window = self.context_manager.get_window(session_id)
        recent_context_messages = window.get_recent(12)
        action_plan = None
        # Bot 自己刚刚发出的回复可能已经追加到窗口末尾，因此不能用
        # `window[-1]` 判断；这里只比较“最新的群友消息”。
        current_is_latest = self._is_latest_user_message(session_id, message)
        current_context_message = self._context_item_for_message(session_id, message)
        if current_is_latest and current_context_message is not None:
            floor_has_name = any(
                reason.startswith("昵称")
                for reason in trigger_result.get("reasons", [])
            )
            if judge_available:
                directed_for_floor = is_private or judge_target == "bot"
            else:
                directed_for_floor = (
                    is_private
                    or message.mentioned_me
                    or quoted_bot
                    or is_reply_to_bot
                    or trigger_result.get("forced_trigger", False)
                    or context.is_emergency
                    or (floor_has_name and context.is_direct_question)
                )
            if directed_for_floor:
                current_context_message.directed_to_bot = True
            if judge_available and judge_intent == "answer":
                context.is_direct_question = True
            floor, action_plan = self.conversation_floor_manager.analyze(
                current_context_message,
                recent_context_messages,
                bot_id=self_id,
                is_private=is_private,
                directed_to_bot=directed_for_floor,
                continuing=continuing,
                mentioned_others=message.mentioned_others,
                ignore_other_target_signal=(
                    judge_available and judge_should_reply
                ),
                allow_dynamic_interjection=(
                    judge_available
                    and judge_should_reply
                    and judge_target != "bot"
                ),
                dynamic_target_user_id=(
                    str(judge.get("target_user_id") or "")
                    if judge_available
                    else ""
                ),
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
            if message.mentioned_me or is_reply_to_bot or (
                judge_available and judge_target == "bot"
            ):
                # 私聊没有群号，用 session_id（private_xxx）作注意力键，避免所有私聊共用一个桶
                self.attention_manager.on_no_reply(
                    message.group_id or session_id, message.sender_id
                )
            return None

        # 判断消息指向：动态判断可用时由模型结果决定；旧触发器仅作回退。
        # - 必须回：@、引用bot、强制触发、紧急
        # - 视为对我说：必须回，或（提到名字/昵称 且 带提问）
        # - 同一人延续对话：bot 刚回复过 TA，TA 没@没引用就接着对 bot 说 → 也算对我说
        # - 其余（纯群聊/自言自语/只是顺口提到名字）→ group，由 LLM 决定插嘴还是潜水
        trigger_reasons = trigger_result.get("reasons", [])
        forced = trigger_result.get("forced_trigger", False)

        has_name = any(r.startswith("昵称") for r in trigger_reasons)
        has_question = any(r == "直接提问" for r in trigger_reasons)
        has_emergency = "紧急" in trigger_reasons

        if judge_available:
            direction = "to_bot" if judge_target == "bot" else "group"
        else:
            must_reply = is_private or message.mentioned_me or quoted_bot or forced or has_emergency
            if must_reply or (has_name and has_question):
                direction = "to_bot"
            else:
                direction = "group"

        # 目标判断排队期间，后面的群友消息可能已经先进入窗口。普通插话
        # 只能针对最新一条群友消息创建候选任务；否则 action_plan 为空时，
        # 收尾/发送复核没有可靠的过期边界，旧消息会在新消息之后抢答。
        # 定向消息（包括模型识别出的隐式续话）仍保留原问题，交给发送前的
        # 定向回复逻辑处理。
        if (
            message.message_type == "group"
            and not current_is_latest
            and direction == "group"
        ):
            logger.info(
                "[发言决策] 消息已过期，跳过普通插话：%s",
                message.message_id or message.sender_id,
            )
            return None
        logger.debug(
            f"[指向] {direction}（触发：{trigger_reasons}，延续对话={continuing}）"
        )

        return {
            "direction": direction,
            "probability": probability,
            "emotional_state": emotional_state,
            "context": context,
            "action_plan": action_plan,
            "conversation_judgement": judge,
        }

    def _refresh_action_plan_after_wait(
        self,
        message: Message,
        direction: str,
        action_plan,
        conversation_judgement: dict | None = None,
    ) -> tuple[object, bool]:
        """思考延迟后用最新群聊重新校验非定向行为计划。

        定向回复由新消息到达时的任务替换机制负责；普通插话则可能在等待期间
        迎来新群消息。此时沿用旧的 REACT/REPLY 计划容易出现"按旧消息组织，
        却对新上下文作答"，所以在未过期时将计划切换到最新群友消息。
        """
        if not action_plan or action_plan.directed or direction != "group":
            return action_plan, False

        recent = self.context_manager.get_window(message.session_id).get_recent(30)
        newer = [
            item for item in recent
            if not item.is_bot and item.timestamp > action_plan.target_timestamp
        ]
        if not newer:
            return action_plan, False

        judgement = conversation_judgement or {}
        judgement_available = bool(judgement.get("available"))
        dynamic_should_reply = bool(judgement.get("should_reply"))
        dynamic_target = str(judgement.get("target") or "")

        # 如果最新消息已经经过动态判断并明确允许参与，不能再用旧 plan
        # 的机械“有人先回答/话题已变”信号把它取消；否则模型判断只在前一
        # 步骤生效，到了发送复核又被旧规则覆盖。
        if not (judgement_available and dynamic_should_reply):
            cancel, cancel_reason = self.conversation_floor_manager.should_cancel(
                action_plan,
                recent,
                bot_id=str(self.config.get("qq", {}).get("self_id", "")),
            )
            if cancel:
                logger.info(f"[发送复核] 放弃回复：{cancel_reason}")
                return None, True

        latest = newer[-1]
        self_id = str(self.config.get("qq", {}).get("self_id", ""))
        if (
            not judgement_available
            and (
                latest.directed_to_bot
                or (
                    latest.reply_to_qq and self_id
                    and str(latest.reply_to_qq) == self_id
                )
            )
        ):
            # 新的定向消息应该由 _handle_message 创建的新任务负责，旧插话不抢答。
            logger.info("[发送复核] 新消息已明确对机器人说，放弃旧插话")
            return None, True

        # 内容无法渲染（segment 都解析不出来）→ 没有可用语义输入，硬插话
        # 只会让 LLM 复读上文或输出自己名字这种垃圾。直接放弃旧插话。
        if latest.content.strip() == "[无法识别的消息]":
            logger.info(
                "[发送复核] 最新消息无法识别（%s），放弃旧插话", latest.content
            )
            return None, True

        try:
            topic_relevance = self.social_awareness.topic_analyzer.analyze_relevance(
                latest.content
            )
        except Exception:
            topic_relevance = 0.5

        latest_probe = SocialContext(message_content=latest.content)
        try:
            self.trigger_detector.detect(latest_probe)
        except Exception:
            pass
        rich_markers = (
            "[链接", "[卡片", "[小程序", "[图片", "[表情包", "[动画表情",
            "[视频", "[合并转发",
        )
        # 渲染失败（无法识别的消息段）的占位文本也算"无可用文字"，与
        # 纯富媒体等同——LLM 没有可用的语义输入，硬调只会产出"复读上
        # 一句"或"输出自己名字"这类垃圾回复。
        unparseable_placeholder = latest.content.strip() == "[无法识别的消息]"
        rich_message_only = latest.content.lstrip().startswith(rich_markers) or unparseable_placeholder
        rich_type = ""
        if latest.content.lstrip().startswith("[图片"):
            rich_type = "image"
        elif latest.content.lstrip().startswith(("[表情包", "[动画表情")):
            rich_type = "mface"
        elif latest.content.lstrip().startswith("[视频"):
            rich_type = "video"
        elif unparseable_placeholder:
            # 没有可用文字、也没有明确富类型，至少别让 plan 当成普通文本。
            rich_type = "unknown"

        # 最新消息本身已经由快速路径写入上下文；这里仅重新计算发言权计划，
        # 不重新抽一次随机概率，避免同一条候选回复被随机数重复改变。
        dynamic_is_directed = judgement_available and dynamic_target == "bot"
        dynamic_continuing = dynamic_is_directed and str(
            judgement.get("intent") or ""
        ) in {"follow_up", "acknowledge", "answer"}
        # 动态判断已经确认要参与时，不能再让旧的 @/回复机械信号或“纯链接
        # 不点评”规则把 action plan 改回 silent；对话模型才是当前消息的
        # 目标裁判，floor 只负责决定以多短的方式接话。
        allow_dynamic_interjection = judgement_available and dynamic_should_reply
        plan_rich_only = rich_message_only
        if allow_dynamic_interjection and rich_type not in (
            "image", "mface", "face", "video"
        ):
            plan_rich_only = False
        _, refreshed = self.conversation_floor_manager.analyze(
            latest,
            recent,
            bot_id=self_id,
            is_private=False,
            directed_to_bot=dynamic_is_directed,
            continuing=dynamic_continuing,
            mentioned_others=[
                str(user_id)
                for user_id in getattr(latest, "mentioned_user_ids", ())
                if str(user_id) != self_id
            ],
            ignore_other_target_signal=allow_dynamic_interjection,
            dynamic_target_user_id=str(judgement.get("target_user_id") or ""),
            allow_dynamic_interjection=allow_dynamic_interjection,
            topic_relevance=topic_relevance,
            is_question=latest_probe.is_direct_question,
            rich_message_only=plan_rich_only,
            rich_type=rich_type,
        )
        if allow_dynamic_interjection and refreshed.action == ActionType.SILENT:
            # 仍保留“短接话”边界，不把动态判断升级成长篇回答；这里只是
            # 将没有机械 floor 分支的消息转成可执行的最小回复计划。
            refreshed = replace(
                refreshed,
                action=ActionType.REACT,
                tone="自然接一句，不复述前文",
                max_chars=14,
                wait_multiplier=1.0,
                directed=False,
                reason="动态判断允许参与，采用短接话",
            )
        if refreshed.action.value == "silent":
            logger.info("[发送复核] 最新上下文不适合插话，放弃旧回复")
            return None, True

        logger.info(
            "[发送复核] 插话计划更新：%s → %s，目标=%s",
            action_plan.action.value,
            refreshed.action.value,
            latest.message_id or latest.sender_id,
        )
        return refreshed, False

    def _latest_user_context_message(self, session_id: str):
        """返回会话中最新的群友消息对象。"""
        recent = self.context_manager.get_window(session_id).get_recent(50)
        for item in reversed(recent):
            if not item.is_bot:
                return item
        return None

    def _context_message_as_platform_message(
        self,
        session_id: str,
        item,
    ) -> Message:
        """把窗口消息转换成动态判断器/生成器可复用的统一消息对象。"""
        self_id = str(self.config.get("qq", {}).get("self_id", "") or "")
        mentioned_ids = [
            str(value) for value in getattr(item, "mentioned_user_ids", ()) if str(value)
        ]
        mentioned_others = [
            value for value in mentioned_ids if value not in {self_id, "all"}
        ]
        content = str(getattr(item, "content", "") or "")
        rich_type = ""
        if content.startswith("[图片"):
            rich_type = "image"
        elif content.startswith(("[表情包", "[动画表情")):
            rich_type = "mface"
        elif content.startswith("[视频"):
            rich_type = "video"
        elif content.startswith("[链接"):
            rich_type = "link"
        return Message(
            message_id=str(getattr(item, "message_id", "") or ""),
            message_type="group",
            sender_id=str(getattr(item, "sender_id", "") or ""),
            sender_name=str(getattr(item, "sender_name", "") or "历史用户"),
            group_id=str(session_id).removeprefix("group_"),
            content=content,
            raw_content=content,
            mentioned_me=self_id in mentioned_ids if self_id else False,
            mentioned_user_ids=mentioned_ids,
            mentioned_others=mentioned_others,
            reply_to_id=getattr(item, "reply_to_id", None),
            reply_to_qq=getattr(item, "reply_to_qq", None),
            outer_text=content,
            rich_only=content.startswith((
                "[链接", "[卡片", "[小程序", "[图片", "[表情包", "[动画表情",
                "[视频", "[合并转发",
            )),
            rich_type=rich_type,
        )

    async def _judge_context_message(
        self,
        session_id: str,
        item,
        *,
        continuation_hint: bool = False,
    ) -> ConversationJudgeResult:
        """等待期间有新消息时，对最新消息再次做目标判断。"""
        judge = getattr(self, "conversation_judge", None)
        if not judge or not item:
            return ConversationJudgeResult.unavailable("not_initialized")
        platform_message = self._context_message_as_platform_message(session_id, item)
        recent = self.context_manager.get_window(session_id).get_recent(
            judge.context_messages + 2
        )
        return await judge.judge(
            platform_message,
            recent,
            heuristic_signals={
                "mentioned_me": platform_message.mentioned_me,
                "mentioned_others": platform_message.mentioned_others,
                "reply_to_me": self._is_reply_to_bot(platform_message),
                "reply_to_qq": platform_message.reply_to_qq or "",
                "continuation_hint": continuation_hint,
            },
        )

    async def _prepare_automatic_meme(
        self,
        session_id: str,
        current_message: Message,
        reply: str,
        direction: str,
        *,
        category: str | None = None,
        meme_id: str = "",
    ) -> tuple[dict[str, Any] | None, str]:
        """为自动发图挑选候选素材，并做发送前语义复核。"""
        manager = self.meme_manager
        if not manager:
            return None, "表情库未初始化"
        if not manager.auto_send_enabled:
            return None, "自动发送未启用"

        available, remaining = manager.auto_send_available(session_id)
        if not available:
            return None, f"自动发图冷却中，还需 {remaining:.0f} 秒"

        await asyncio.to_thread(manager.reload)
        if meme_id:
            candidate = await asyncio.to_thread(manager.resolve, meme_id)
        else:
            candidate = await asyncio.to_thread(
                manager.choose,
                category or "",
                session_id,
                remember=False,
            )
        if not candidate:
            return None, "没有找到匹配的表情素材"

        judge = getattr(self, "conversation_judge", None)
        if not judge:
            return None, "语义复核器不可用"
        recent = self.context_manager.get_window(session_id).get_recent(16)
        try:
            review = await judge.review_meme_send(
                current_message,
                recent,
                candidate,
                reply=reply,
                direction=direction,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return None, f"表情语义复核失败：{exc}"
        if (
            not review.available
            or not review.should_reply
            or not (review.evidence or {}).get("should_send_meme")
        ):
            return None, review.reason or "当前语境不适合发图"
        return candidate, review.reason or "语境匹配"

    @staticmethod
    def _context_marker(item):
        """返回窗口消息的稳定标记；同时兼容平台 Message 和 ContextMessage。"""
        timestamp = getattr(item, "timestamp", None)
        if hasattr(timestamp, "isoformat"):
            timestamp = timestamp.isoformat()
        return (
            str(getattr(item, "message_id", "") or ""),
            str(timestamp or ""),
            str(getattr(item, "sender_id", "") or ""),
            str(getattr(item, "content", "") or ""),
        )

    @staticmethod
    def _context_item_matches_message(item, message: Message) -> bool:
        """判断窗口中的消息是否对应一条刚收到的平台消息。"""
        item_id = str(getattr(item, "message_id", "") or "")
        message_id = str(getattr(message, "message_id", "") or "")
        if item_id and message_id:
            return item_id == message_id
        return (
            str(getattr(item, "sender_id", "") or "")
            == str(getattr(message, "sender_id", "") or "")
            and str(getattr(item, "content", "") or "")
            == str(getattr(message, "content", "") or "")
        )

    def _latest_user_context_marker(self, session_id: str):
        """返回会话里最新群友消息的稳定标记，用于检测收尾窗口是否被打断。"""
        recent = self.context_manager.get_window(session_id).get_recent(30)
        for item in reversed(recent):
            if not item.is_bot:
                return self._context_marker(item)
        return None

    def _context_marker_for_message(self, session_id: str, message: Message):
        """返回某条入站消息在窗口中的标记，而不是调用时刻的 latest 标记。"""
        item = self._context_item_for_message(session_id, message)
        if item is not None:
            return self._context_marker(item)
        # 快速路径异常时仍给出可比较的退化标记；后续会被最新消息检测挡住。
        return self._context_marker(message)

    def _context_item_for_message(self, session_id: str, message: Message):
        """找出窗口中对应的入站消息，忽略之后追加的 Bot 回复。"""
        recent = self.context_manager.get_window(session_id).get_recent(50)
        for item in reversed(recent):
            if not item.is_bot and self._context_item_matches_message(item, message):
                return item
        return None

    def _is_latest_user_message(self, session_id: str, message: Message) -> bool:
        """判断入站消息是否仍是窗口里最新的群友消息。"""
        latest = self._latest_user_context_message(session_id)
        return bool(latest and self._context_item_matches_message(latest, message))

    async def _wait_for_group_settle(self, session_id: str, action_plan) -> None:
        """等待普通群聊短暂安静，再把最新上下文交给 LLM。

        这是插话和定向回复的边界：明确 @/引用 bot 的消息不走这里；普通群聊
        只等待一个很短的 idle 窗口。新消息会重置 idle 计时，但总等待有上限，
        所以热闹群聊最终仍会进入一次最新上下文的判断。
        """
        if not action_plan or action_plan.directed or not self.conversation_floor_manager:
            return

        floor = self.conversation_floor_manager
        idle_seconds = max(0.2, float(getattr(floor, "settle_window_seconds", 0.7)))
        max_seconds = max(
            idle_seconds,
            float(getattr(floor, "settle_max_seconds", 2.4)),
        )
        marker = self._latest_user_context_marker(session_id)
        if marker is None:
            return

        started = time.monotonic()
        while True:
            remaining = max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                logger.debug("[发言收尾] 达到 %.1fs 上限，使用当前群聊上下文", max_seconds)
                return

            await asyncio.sleep(min(idle_seconds, remaining))
            latest = self._latest_user_context_marker(session_id)
            if latest == marker:
                logger.debug("[发言收尾] 群聊已安静 %.1fs，开始分析", idle_seconds)
                return

            marker = latest
            logger.debug("[发言收尾] 检测到新群消息，继续等待 %.1fs", idle_seconds)

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
        conversation_judgement = decision.get("conversation_judgement") or {}
        # 动态判断为 bot 的隐式续话即使在排队期间变成了“非最新用户消息”，
        # 仍应保留原问题的定向回复语义；普通群聊插话则必须在收尾时重新判断。
        initial_plan_directed = bool(
            (action_plan and action_plan.directed) or direction == "to_bot"
        )
        effective_context_item = None
        effective_message_id = str(message.message_id or "")

        try:
            enrichment_task = decision.get("enrichment_task")
            if enrichment_task:
                try:
                    await enrichment_task
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("富媒体增强失败，使用占位符继续：%s", exc)

            # === 思考延迟（期间新消息会进入上下文，等对方把话说完） ===
            await self.thinking_delay.wait(
                message_length=len(message.content),
                topic_familiarity=context.topic_familiarity,
                emotional_modifier=(
                    (emotional_state.get_thinking_delay_multiplier() if emotional_state else 1.0)
                    * (action_plan.wait_multiplier if action_plan else 1.0)
                )
            )

            # 定向回复按原有节奏及时处理；普通插话再补一个很短的 debounce，
            # 避免 LLM 只看到“享年4级”就抢先点评，错过后面紧接着的“猝/翻车”语境。
            await self._wait_for_group_settle(session_id, action_plan)

            # 普通插话等待期间如果出现了新消息，先对最新消息重新做动态
            # 目标判断；定向回复仍保留原始问题，不被旁边的新话题带走。
            if (
                message.message_type == "group"
                and not initial_plan_directed
                and decision.get("context_marker")
                != self._latest_user_context_marker(session_id)
            ):
                latest_item = self._latest_user_context_message(session_id)
                latest_judgement = await self._judge_context_message(
                    session_id,
                    latest_item,
                    continuation_hint=False,
                )
                if latest_judgement.available:
                    latest_id = str(getattr(latest_item, "message_id", "") or "")
                    self.context_manager.update_message_analysis(
                        session_id,
                        latest_id,
                        directed_to_bot=(latest_judgement.target == "bot"),
                        target=latest_judgement.target,
                        intent=latest_judgement.intent,
                        confidence=latest_judgement.confidence,
                        reason=latest_judgement.reason,
                    )
                    if not latest_judgement.should_reply:
                        logger.info("[发送复核] 最新消息经动态判断不应参与")
                        return
                    if latest_judgement.target == "bot":
                        # 新定向消息会由 _handle_message 的任务替换机制负责。
                        logger.info("[发送复核] 最新消息已动态判断为对机器人说，放弃旧插话")
                        return
                    conversation_judgement = latest_judgement.to_dict()
                    effective_context_item = latest_item
                    effective_message_id = str(latest_id or "")
                    decision["context_marker"] = self._latest_user_context_marker(
                        session_id
                    )
                else:
                    # 旧判断只属于原始触发消息，不能在目标已经切换后继续
                    # 作为新消息的生成约束；本轮改用本地 floor 的保守回退。
                    conversation_judgement = {}

            # 思考期间群聊可能已经向前发展；普通插话过期时放弃，仍适合时
            # 把行为计划切换到最新群友消息，避免旧计划套新上下文。
            action_plan, plan_cancelled = self._refresh_action_plan_after_wait(
                message,
                direction,
                action_plan,
                conversation_judgement=conversation_judgement,
            )
            if plan_cancelled:
                return

            # 如果插话计划已经切到等待期间的新消息，搜索判断、挫败检测等仍使用
            # 旧触发消息会造成“计划回 m2、搜索却搜 m1”的错位。保持定向回复的
            # 原始消息语义不变，只对非定向插话同步当前目标内容。
            generation_message = message.content
            if action_plan and not action_plan.directed and action_plan.target_message_id:
                for candidate in reversed(
                    self.context_manager.get_window(session_id).get_recent(30)
                ):
                    if str(candidate.message_id or "") == str(action_plan.target_message_id):
                        generation_message = candidate.content
                        effective_context_item = candidate
                        effective_message_id = str(candidate.message_id or "")
                        break

            if effective_context_item is None and effective_message_id:
                for candidate in reversed(
                    self.context_manager.get_window(session_id).get_recent(50)
                ):
                    if str(candidate.message_id or "") == effective_message_id:
                        effective_context_item = candidate
                        break

            effective_message = message
            if effective_context_item is not None and (
                str(getattr(effective_context_item, "message_id", "") or "")
                != str(message.message_id or "")
            ):
                effective_message = self._context_message_as_platform_message(
                    session_id, effective_context_item
                )

            # 普通插话的目标如果在动态判断后又变了，新的消息处理任务会重新
            # 决定是否发言；旧任务不得把过期草稿发进群。
            if (
                message.message_type == "group"
                and not initial_plan_directed
                and decision.get("context_marker")
                != self._latest_user_context_marker(session_id)
                and effective_message_id == str(message.message_id or "")
            ):
                logger.info("[发送复核] 普通插话目标已过期，放弃旧草稿")
                return

            # 记忆查询按最终候选消息执行，避免等待后换了目标但仍召回旧问题。
            memories = await self._retrieve_memories(
                generation_message,
                session_id,
                exclude_id=effective_message_id or message.message_id,
            )

            # === 构建提示词（此刻的上下文 = 思考期间的最新消息，不会回旧话题） ===
            context_prompt = self.context_manager.build_context_prompt(
                session_id=session_id,
                bot_name=self.personality.name,
                # 人格已经作为 system message 注入 ReplyGenerator，避免重复两遍。
                persona_prompt="",
                memories=memories,
                bot_id=str(self.config.get("qq", {}).get("self_id", "") or ""),
                focus_message_id=effective_message_id,
                max_messages=self.context_manager.max_messages,
            )

            # === 注入最近群日报摘要：让 bot 知道"昨天/今天群里聊过啥，群友标签是啥" ===
            context_prompt = await self._augment_context_with_group_reports(
                session_id, context_prompt
            )

            # === 命中的群黑话：只注入本轮对话里真正出现的词条 ===
            glossary = []
            try:
                if (
                    getattr(self, "long_term_memory_enabled", True)
                    and self._slang_config.get("enabled", True)
                ):
                    glossary = await self.memory_storage.match_slang(
                        f"{context_prompt}\n{generation_message}",
                        session=session_id,
                        limit=self._slang_config.get("max_inject", 8),
                    )
                    if glossary:
                        await self.memory_storage.bump_slang_hits(
                            [g["id"] for g in glossary]
                        )
                        logger.info(
                            "[黑话] 本轮注入 %d 条：%s",
                            len(glossary), "、".join(g["term"] for g in glossary),
                        )
            except Exception as exc:
                logger.debug("黑话匹配失败：%s", exc)

            # === 生成回复（direction 控制是否可沉默） ===
            try:
                gen_result = await self.reply_generator.generate(
                    context_prompt=context_prompt,
                    current_message=generation_message,
                    emotional_state=emotional_state,
                    direction=direction,
                    action_plan=action_plan.to_dict() if action_plan else None,
                    session_id=session_id,
                    glossary=glossary,
                    conversation_judgement=conversation_judgement,
                    current_message_context={
                        "sender_id": effective_message.sender_id,
                        "sender_name": effective_message.sender_name,
                        "message_id": effective_message.message_id,
                        "mentioned_user_ids": effective_message.mentioned_user_ids,
                        "reply_to_id": effective_message.reply_to_id,
                        "reply_to_qq": effective_message.reply_to_qq,
                        "conversation_judgement": conversation_judgement,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"生成回复时出错：{e}", exc_info=True)
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return

            # generator 返回 dict / None；老调用兜底仍兼容字符串返回。
            if gen_result is None:
                reply = None
                tool_meme_category = None
                tool_meme_id = ""
                tool_meme_called = False
            elif isinstance(gen_result, dict):
                reply = gen_result.get("reply")
                tool_meme_category = gen_result.get("meme_category")
                tool_meme_id = (gen_result.get("meme_id") or "").strip()
                tool_meme_called = bool(gen_result.get("meme_called"))
            else:
                # 兼容：旧版本直接返回 str
                reply = gen_result
                tool_meme_category = None
                tool_meme_id = ""
                tool_meme_called = False

            # LLM 选择沉默（群友互聊/自言自语时的正常行为）。
            # 但 generator 可能用 send_meme 工具调用单独发图、文本为空——这是合法
            # 的「只发表情包」路径，让它继续走到 send_meme 通道，不要被吞掉。
            # tool_meme_called 是 LLM 显式调工具的事实旗标（独立于 args 是否为空），
            # 用它做兜底，避免空 category/空 id 的随机抽图路径被这里误判为沉默。
            has_meme_intent = bool(tool_meme_called or tool_meme_category or tool_meme_id)
            if reply is None and not has_meme_intent:
                logger.info(f"[沉默] 回复方向={direction}，不参与该条消息")
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return
            # reply 为 None 但有表情包要发 → 当成空字符串处理，下游分支会跳过
            # 文字发送直接走 send_meme 通道。
            if reply is None:
                reply = ""

            # 语义复读复核：先用低成本词面重叠筛出可疑草稿，再让动态判断器
            # 区分“正常沿用关键词回答”和“只是把用户的话换个说法重述”。
            judge = getattr(self, "conversation_judge", None)
            review_text = reply
            if self.meme_manager:
                review_text = self.meme_manager.strip_directives(review_text)
            recent_context_for_review = self.context_manager.get_window(
                session_id
            ).get_recent(16)
            recent_texts = [
                m.content for m in recent_context_for_review if not m.is_bot
            ]
            if (
                judge
                and review_text.strip()
                and self.reply_generator.looks_like_paraphrase_candidate(
                    review_text, recent_texts
                )
            ):
                review = await judge.review_reply(
                    effective_message,
                    recent_context_for_review,
                    review_text,
                    direction=direction,
                )
                review_evidence = review.evidence or {}
                if (
                    review.available
                    and review_evidence.get("is_paraphrase")
                    and not review_evidence.get("adds_information")
                ):
                    logger.info(
                        "[复读判断] 草稿被判定为语义复读，要求重新接话：%s",
                        review.reason or "未增加信息",
                    )
                    retry_reply = await self.reply_generator.generate(
                        context_prompt=context_prompt,
                        current_message=generation_message,
                        emotional_state=emotional_state,
                        direction=direction,
                        action_plan=action_plan.to_dict() if action_plan else None,
                        session_id=session_id,
                        glossary=glossary,
                        conversation_judgement=conversation_judgement,
                        current_message_context={
                            "sender_id": effective_message.sender_id,
                            "sender_name": effective_message.sender_name,
                            "message_id": effective_message.message_id,
                            "mentioned_user_ids": effective_message.mentioned_user_ids,
                            "reply_to_id": effective_message.reply_to_id,
                            "reply_to_qq": effective_message.reply_to_qq,
                            "conversation_judgement": conversation_judgement,
                        },
                        avoid_paraphrase=True,
                    )
                    # generate() 当前返回包含表情包工具结果的 dict；兼容旧版
                    # 直接返回字符串，避免复读重答分支把 dict 当作文本继续处理。
                    if isinstance(retry_reply, dict):
                        retry_text = retry_reply.get("reply")
                        retry_meme_category = retry_reply.get("meme_category")
                        retry_meme_id = (retry_reply.get("meme_id") or "").strip()
                        retry_meme_called = bool(retry_reply.get("meme_called"))
                    else:
                        retry_text = retry_reply
                        retry_meme_category = None
                        retry_meme_id = ""
                        retry_meme_called = False

                    retry_has_meme_intent = bool(
                        retry_meme_called or retry_meme_category or retry_meme_id
                    )
                    if retry_text is None and not retry_has_meme_intent:
                        logger.info("[复读判断] 重答选择沉默")
                        return
                    # 重答是完整替代结果，不能沿用上一版已经被判定为复读的
                    # 工具调用或 marker；否则可能出现“新文字 + 旧表情包”。
                    reply = retry_text if retry_text is not None else ""
                    tool_meme_category = retry_meme_category
                    tool_meme_id = retry_meme_id
                    tool_meme_called = retry_meme_called

            # 表情包选择标记只在内部流转，不能进入群聊文本、记忆或日报。
            meme_category = None
            meme_id = tool_meme_id  # 工具调用优先；下面兜底仍允许 marker 走老路
            marker_requested = False
            if self.meme_manager:
                # 工具调用已经给出 meme_category / meme_id 时，LLM 偶发残留的
                # [[表情:xxx]] marker 也要从 reply 里剥干净，避免重复发图/重复标签。
                raw_reply = reply or ""
                reply, marker_category = self.meme_manager.extract_directive(raw_reply)
                marker_requested = reply != raw_reply or marker_category is not None
                if isinstance(marker_category, str) and marker_category.startswith("@id:"):
                    candidate_id = marker_category[4:]
                    if not meme_id:
                        meme_id = candidate_id
                else:
                    if not tool_meme_called and marker_category is not None:
                        tool_meme_category = marker_category
                # 兜底：解析失败残留的标记（如 LLM 输出了格式外的变体）不能原样发进
                # 群里，剥成普通文字后再继续，避免”[[表情:xxx:yyy”直接出现在聊天里。
                # 同时把半截标记里的分类名抢救回来当作 meme_category，让”想发表情包”
                # 的意图不会因为 LLM 漏写 `]]` 而彻底丢失。
                residue, recovered_category = self.meme_manager.strip_directives(reply)
                if recovered_category and not meme_id and not tool_meme_category:
                    tool_meme_category = recovered_category
                if residue != reply:
                    marker_requested = True
                    logger.warning(
                        "[表情库] 残留标记未解析成功，已剥离并回收分类 %s: %s",
                        recovered_category or "<空>",
                        reply if len(reply) <= 80 else reply[:80] + "...",
                    )
                    reply = residue
                # 工具调用优先：send_meme 已经告诉系统要发图，marker 路径只用来
                # 兜老 LLM 输出；正常情况直接以 tool_use 为准。
                meme_category = tool_meme_category

            # 只有真实工具调用或被剥离的内部标记才算发图请求。
            # meme_category="" 本身可能只是“未指定分类”，不能单独作为依据。
            meme_requested = bool(tool_meme_called or marker_requested)
            if meme_requested:
                candidate, candidate_reason = await self._prepare_automatic_meme(
                    session_id,
                    effective_message,
                    reply,
                    direction,
                    category=meme_category,
                    meme_id=meme_id,
                )
                if candidate:
                    meme_id = str(candidate.get("id") or "")
                    meme_category = str(candidate.get("category") or "")
                else:
                    logger.info("[表情复核] 取消自动发图：%s", candidate_reason)
                    meme_requested = False
                    tool_meme_category = None
                    tool_meme_id = ""
                    tool_meme_called = False
                    meme_category = None
                    meme_id = ""
            has_meme_intent = meme_requested

            # 复读兜底：把群友的原话原样说一遍不如不说。表情包识别摘要会引用
            # 前文原话，短反应档下 LLM 容易直接抓那句引文当自己的发言。
            recent_texts = [
                m.content
                for m in self.context_manager.get_window(session_id).get_recent(12)
                if not m.is_bot
            ]
            if reply and self.reply_generator.is_parroting(reply, recent_texts) and not has_meme_intent:
                logger.info(f"[复读] 放弃回复（复述了群友原话）：{reply[:40]}")
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return

            # 兜底：极短回复（≤8 字）只是把对方最后一句的关键词原样重复，
            # 没有补充任何新信息（典型："确实真实""就是真实""对"）。
            # 这种复读群里会被直接当成废话，提前挡掉让 LLM 重答。
            # 注意：LLM 调 send_meme 时偶尔会写几个字尾巴（如"发个"+"调工具"），
            # 复读检测会把尾巴和原话前缀误判，但用户真正要的是图，丢掉整条不可接受。
            # 有图要发就跳过复读检查。
            if reply and recent_texts and not has_meme_intent:
                last_msg = recent_texts[-1].strip()
                if last_msg and self.reply_generator.is_short_echo(reply, last_msg):
                    logger.info(
                        f"[复读] 放弃短复读回（{reply!r} 复读自 {last_msg!r}）"
                    )
                    if direction == "to_bot":
                        self.attention_manager.on_no_reply(group_id, message.sender_id)
                    return

            # LLM 调用和模拟打字也会耗时，发送前再复核一次群聊局势。
            if (
                message.message_type == "group"
                and not initial_plan_directed
                and decision.get("context_marker")
                != self._latest_user_context_marker(session_id)
            ):
                logger.info("[发送复核] 生成期间出现新消息，放弃过期草稿")
                return
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
            passed, result = (
                (True, "")
                if not reply and meme_requested
                else self.response_filter.filter(reply)
            )
            if not passed:
                logger.info(f"回复被过滤：{result}")
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return
            reply = result

            # === 应用错字生成 ===
            typing_config = self.config.get("typing_style", {})
            if reply and typing_config.get("enable_typo_generator", True):
                reply = self.typo_generator.apply_typo(reply)

            # === 发送回复（拆成多条，模拟真人分段发送） ===
            from modules.reply.generator import split_reply_into_messages

            segments = split_reply_into_messages(reply)
            # 动态判断给出的引用目标优先使用；定向群聊回复也引用原消息，
            # 避免 Bot 思考期间群里继续聊天后，看不出它到底在回答谁。
            quote_id = ""
            target_id = str(
                conversation_judgement.get("reference_message_id")
                or (action_plan.target_message_id if action_plan else "")
                or effective_message_id
                or ""
            )
            recent_ids = {
                str(m.message_id or "")
                for m in self.context_manager.get_window(session_id).get_recent(50)
                if m.message_id
            }
            if target_id and target_id in recent_ids and (
                direction == "to_bot"
                or (action_plan and not action_plan.directed)
            ):
                quote_id = target_id
            elif action_plan and not action_plan.directed and action_plan.target_message_id:
                target_id = str(action_plan.target_message_id)
                if any(
                    not m.is_bot
                    and str(m.message_id or "") != target_id
                    and m.timestamp > action_plan.target_timestamp
                    for m in self.context_manager.get_window(session_id).get_recent(30)
                ):
                    quote_id = action_plan.target_message_id
            sent_segments = []
            first_sent_message_id = ""
            meme_sent = False
            meme_item = None
            try:
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
                    send_with_id = getattr(
                        self.qq_adapter, "send_message_with_id", None
                    )
                    if callable(send_with_id):
                        success, outbound_message_id = await send_with_id(
                            session_id,
                            seg,
                            reply_to_id=(quote_id if i == 0 else ""),
                        )
                    else:
                        success = await self.qq_adapter.send_message(
                            session_id,
                            seg,
                            reply_to_id=(quote_id if i == 0 else ""),
                        )
                        outbound_message_id = str(
                            getattr(self.qq_adapter, "last_sent_message_id", "") or ""
                        )
                    if success:
                        sent_segments.append(seg)
                        if not first_sent_message_id:
                            first_sent_message_id = str(outbound_message_id or "")
                        logger.info(
                            f"[回复段{i+1}/{len(segments)}] {self.personality.name}：{seg[:50]}"
                        )
                        # 段间延迟，模拟真人打字停顿
                        if i < len(segments) - 1:
                            await asyncio.sleep(random.uniform(0.6, 2.0))
                    else:
                        # 保持分段顺序；前一段失败后继续发后一段会显得语义残缺。
                        break

                # 文本完整发送后再发图，避免一条回复被拆成“半句文字 + 表情”。
                text_complete = len(sent_segments) == len(segments)
                if meme_requested and text_complete:
                    meme_result = await self.send_meme(
                        session_id,
                        meme_id=meme_id,
                        category=meme_category,
                        reply_to_id=(quote_id if not segments else ""),
                        # 这是模型在回复过程中主动选择的自动发图，必须经过
                        # auto_send_enabled 闸门；Web 手动发送仍使用 automatic=False。
                        automatic=True,
                    )
                    meme_sent = bool(meme_result.get("success"))
                    meme_item = meme_result.get("item")
                    if meme_sent and not first_sent_message_id:
                        # 纯表情包回复也必须有真实消息 ID；否则下一条群友
                        # 回复这张图时，目标判断器无法建立“回复了 Bot”的关系。
                        first_sent_message_id = str(
                            meme_result.get("message_id") or ""
                        )
                    if not meme_sent:
                        logger.info("[表情库] 本轮没有可发送的表情：%s", meme_result.get("error", "未知原因"))
            finally:
                # 任务可能在段间被新的"对我说"消息取消；已经发到群里的内容
                # 必须记录状态和上下文（全部是同步操作，取消中也能安全执行），
                # 否则 bot 会忘记自己刚说过的话，下一条回复可能重复或矛盾。
                if sent_segments or meme_sent:
                    sent_reply = "".join(sent_segments)
                    if meme_sent:
                        meme_label = (meme_item or {}).get("category", "表情")
                        sent_reply = f"{sent_reply} [发送表情包：{meme_label}]".strip()
                        logger.info(f"[回复] {self.personality.name}：{sent_reply[:50]}……")

                    # Bot回复后状态更新（真实概率：高概率的@/回复不触发冷却，对话可延续）
                    self.speaking_decider.on_bot_reply(
                        session_id,
                        group_id,
                        probability=probability,
                        user_id=effective_message.sender_id,
                    )

                    # 添加回复到上下文
                    self.context_manager.add_message(
                        session_id=session_id,
                        sender_id=self.config.get("qq", {}).get("self_id", ""),
                        sender_name=self.personality.name,
                        content=sent_reply,
                        is_bot=True,
                        message_id=first_sent_message_id,
                        reply_to_id=(quote_id or effective_message.message_id),
                        reply_to_qq=effective_message.sender_id,
                    )

                    # bot 的回复也写入长期记忆，让会话历史两侧完整
                    self._store_bot_memory(
                        session_id,
                        sent_reply,
                        message_id=first_sent_message_id,
                        reply_to_id=(quote_id or effective_message.message_id),
                        reply_to_qq=effective_message.sender_id,
                    )
                    self._store_group_analysis_bot_message(session_id, sent_reply)

            if not sent_segments and not meme_sent:
                logger.error("回复发送失败")
                if direction == "to_bot":
                    self.attention_manager.on_no_reply(group_id, message.sender_id)
                return

            # 疲劳收尾只在主回复完整发送后考虑；成功后重置，避免连续多轮重复说“先走了”。
            if (
                len(sent_segments) == len(segments)
                and (meme_category is None or meme_sent)
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
                    self._store_group_analysis_bot_message(session_id, closing_msg)

        except asyncio.CancelledError:
            # 被更新的"对我说"消息取代，静默退出
            logger.debug(f"回复任务已取消：{session_id}")
            raise
        except Exception as e:
            logger.error(f"组织或发送回复时出错：{e}", exc_info=True)
        finally:
            # 释放会话锁：只有自己仍是当前登记的任务才移除，避免误删被更新的任务
            if self._reply_tasks.get(session_id) is asyncio.current_task():
                self._reply_tasks.pop(session_id, None)
                self._reply_task_decisions.pop(session_id, None)

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
        """记录消息轨迹（时间/sender/是否纯图片/图片url+file），用于判断连续连图。

        file 用于图片 URL 直连失败时走 NapCat get_image 取本地文件兜底。
        """
        try:
            sid = message.session_id
            is_pure_image = message.rich_only and message.rich_type in ("image", "mface")
            url = ""
            file = ""
            if is_pure_image:
                for seg in message.segments:
                    if seg.type in ("image", "mface"):
                        url = seg.url or url
                        file = seg.file or seg.file_id or file
                        break
            trail = self._media_trail.setdefault(sid, deque(maxlen=self._media_trail_limit))
            trail.append({
                "ts": time.time(),
                "sender_id": message.sender_id,
                "is_image": is_pure_image,
                "url": url,
                "file": file,
                "message_id": message.message_id,
            })
        except Exception as e:
            logger.debug("记录媒体轨迹失败：%s", e)

    def _collect_group_image_urls(self, message: Message) -> list[dict]:
        """取当前图片消息之前、同一人连续发的纯图片（URL+file，相邻间隔≤60s）。

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
        collected: list[dict] = []
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
            collected.append({"url": entry["url"], "file": entry.get("file", "")})
            if len(collected) >= cap:
                break
            now = entry["ts"]  # 向前推移，判断下一条与前一条的间隔
        collected.reverse()  # 时间正序
        if collected:
            logger.debug("[连图] 当前图前收集 %d 张组图：%s", len(collected), [c["url"] for c in collected])
        return collected

    def _build_image_conversation_context(self, message: Message) -> str:
        """取当前图片消息相关的前后对话，供视觉模型判断图片/表情包的意图。

        引用关系是关键：群友引用一张图说"这是好事啊"，视觉模型若不知道
        "谁发的图、谁引用了图说了什么"，会把图的意图错安到引用者头上
        （把引用者的话当成图的内容）。因此这里额外提供：
        1. 发图人（当前图片是谁、什么时候发的；若它引用了别的消息一并说明）；
        2. 窗口里引用/回复了这张图的消息：`X 引用了这张图，说：...`；
        3. 最近对话带引用标记，帮助模型分清每条话是谁说的。
        """
        try:
            window = self.context_manager.get_window(message.session_id)
            recent = window.get_recent(12)
            current_id = getattr(message, "message_id", "") or ""
            self_id = str(self.config.get("qq", {}).get("self_id", ""))

            # 当前图片在窗口里对应消息的时间（Message 本身不带 timestamp）
            now = datetime.now()
            for m in recent:
                if current_id and str(m.message_id or "") == str(current_id):
                    now = m.timestamp
                    break

            # 改名归一：同一 QQ 号在窗口内改群名片时，统一到最近一次昵称，并附尾号绑定身份
            id_to_names: dict[str, set] = {}
            latest_name: dict[str, str] = {}
            for m in recent:
                if not m.sender_id or m.is_bot:
                    continue
                id_to_names.setdefault(m.sender_id, set()).add(m.sender_name)
                latest_name[m.sender_id] = m.sender_name  # recent 时间正序，覆盖后为最新
            renamed_ids = {i for i, ns in id_to_names.items() if len(ns) > 1}

            def display(m) -> str:
                if m.is_bot:
                    return "爱丽丝"
                sid = m.sender_id
                name = latest_name.get(sid, m.sender_name) or sid
                if sid and sid in renamed_ids:
                    return f"{name}({sid[-4:]})"
                return name

            def display_by_id(sid: str) -> str:
                if sid and self_id and str(sid) == self_id:
                    return self.personality.name
                name = latest_name.get(str(sid), "")
                if sid and sid in renamed_ids:
                    return f"{name}({sid[-4:]})"
                return name or str(sid)

            def quoted_target(m) -> str:
                """返回某条消息引用的对象描述：窗口内能找到内容就给内容，否则给昵称。"""
                qid = getattr(m, "reply_to_id", None)
                qqq = getattr(m, "reply_to_qq", None)
                if qid:
                    for other in recent:
                        if other is not m and str(other.message_id or "") == str(qid):
                            return f"{display(other)}：{other.content[:40]}"
                if qqq:
                    name = display_by_id(str(qqq))
                    if name:
                        return name
                return ""

            lines: list[str] = []
            # 1. 最近对话（当前图片之前，时间正序，最多6条），带引用标记
            recent_list = list(recent)
            cur_index = None
            for i, m in enumerate(recent_list):
                if current_id and m.message_id and str(m.message_id) == str(current_id):
                    cur_index = i
                    break
                if not current_id and m.sender_id == message.sender_id and m.content == message.content:
                    cur_index = i
                    break
            if cur_index is None:
                cur_index = len(recent_list)  # 找不到当前图时退化为取全部
            for m in recent_list[max(0, cur_index - 6):cur_index]:
                seg = m.content[:80]
                tgt = quoted_target(m)
                if tgt:
                    seg = f"{seg}（回复：{tgt}）"
                lines.append(f"[{m.timestamp.strftime('%H:%M')}] {display(m)}：{seg}")

            # 2. 当前图片：发图人 + 时间；若它引用了别的消息，一并说明
            sender_name = (message.sender_name or message.sender_id or "未知用户")
            header = f"[{now.strftime('%H:%M')}] {sender_name} 发来这张图"
            tgt = quoted_target(message)
            if tgt:
                header += f"，回复的是：{tgt}"
            lines.append(header)

            # 3. 谁引用了这张图（窗口内 reply_to_id == 当前图 id），时间正序
            for m in recent:
                if not current_id or str(m.reply_to_id or "") != str(current_id):
                    continue
                if str(m.message_id or "") == current_id:
                    continue
                lines.append(
                    f"[{m.timestamp.strftime('%H:%M')}] {display(m)} 引用了这张图，说：{m.content[:60]}"
                )

            return "\n".join(lines)
        except Exception as e:
            logger.debug("构建图片对话上下文失败：%s", e)
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
    # （在模块级构建：类体内的列表推导访问不到同级类变量）
    _FIRST_PERSON = FIRST_PERSON_PRONOUNS
    _PERSONAL_KEYWORDS = PERSONAL_KEYWORDS

    # 群里转发的领取口令/拉人广告：文本是模板，不代表发送者本人，
    # 既没有记忆价值也会稀释画像素材。
    _PROMO_MARKERS = ("复制粘贴", "口令接取", "农友钱", "扫码领取", "点击领取")

    @classmethod
    def _is_promo_text(cls, content: str) -> bool:
        """判断是否是转发的领取口令/推广模板文本。"""
        text = content or ""
        if "复制粘贴" in text and ("口令" in text or "接取" in text):
            return True
        if any(marker in text for marker in cls._PROMO_MARKERS[1:]):
            return True
        return "【" in text and "价值" in text and "领取" in text

    async def _restore_recent_context(
        self, session_id: str, current_message_id: str = ""
    ) -> None:
        """从 SQLite 恢复重启前仍有效的近期对话到工作记忆窗口。

        SQLite 里的 episodic 记录不是完整聊天日志，只恢复其中已经被长期
        记忆筛选留下的消息；窗口仍以当前配置的消息数和时间范围为上限。
        """
        if not getattr(self, "long_term_memory_enabled", True):
            return
        if not self.memory_storage or not self.context_manager:
            return

        window = self.context_manager.get_window(session_id)
        if window.restored_from_storage:
            return
        # 先标记，避免同一会话的并发首条消息重复恢复。
        window.restored_from_storage = True

        restore_limit = max(0, self.context_manager.max_messages - 1)
        if restore_limit <= 0:
            return

        try:
            rows = await self.memory_storage.get_session_messages(
                session_id,
                limit=restore_limit * 4,  # 较大的取样窗口，供下面过滤“仅上下文”与低价值
            )
            cutoff = datetime.now() - window.max_age
            restored_count = 0
            for memory in rows:  # 数据库按新到旧；头插后自然变成时间正序
                if memory.created_at < cutoff:
                    break
                meta = memory.metadata or {}
                # 只恢复「有实义内容」的消息：纯应声/仅上下文的历史不是一个
                # 可衔接的对话（见写入端的 meaningful 标记），跳过它们，
                # 避免重启后窗口被“嗯”“好”“对”刷满而挤掉真人对话。
                if meta.get("profile_context_only") or not meta.get("meaningful"):
                    continue
                message_id = str(meta.get("message_id") or "")
                if current_message_id and message_id == str(current_message_id):
                    continue

                bot_name = getattr(self.personality, "name", "爱丽丝")
                sender_name = str(
                    meta.get("sender_name")
                    or (bot_name if meta.get("is_bot") else "历史用户")
                )
                content = (memory.content or "").strip()
                prefix = f"{sender_name}："
                if content.startswith(prefix):
                    content = content[len(prefix):].strip()
                if not content:
                    continue

                self.context_manager.prepend_message(
                    session_id=session_id,
                    sender_id=str(meta.get("sender_id") or ""),
                    sender_name=sender_name,
                    content=content,
                    is_bot=bool(meta.get("is_bot")),
                    message_id=message_id,
                    reply_to_id=meta.get("reply_to_id"),
                    reply_to_qq=meta.get("reply_to_qq"),
                    mentioned_user_ids=meta.get("mentioned_user_ids") or [],
                    directed_to_bot=bool(meta.get("directed_to_bot")),
                    conversation_target=str(meta.get("conversation_target") or ""),
                    conversation_intent=str(meta.get("conversation_intent") or ""),
                    conversation_confidence=float(
                        meta.get("conversation_confidence") or 0.0
                    ),
                    conversation_reason=str(meta.get("conversation_reason") or ""),
                    timestamp=memory.created_at,
                )
                restored_count += 1
                if restored_count >= restore_limit:
                    break
        except Exception as exc:
            # 恢复失败不能阻塞当前消息；下次新建窗口时仍可再尝试。
            window.restored_from_storage = False
            logger.debug("恢复近期记忆失败：%s", exc)

    def _store_group_analysis_message(self, message: Message, session_id: str) -> None:
        """保存完整群聊流水，供日报使用，不进入普通记忆链路。"""
        config = getattr(self, "_group_analysis_config", {}) or {}
        if not config.get("enabled", True) or message.message_type != "group":
            return
        content = (message.content or message.outer_text or "").strip()
        if message.rich_type:
            content = f"{content} [{message.rich_type}]".strip()
        if not content:
            return
        message_id = str(message.message_id or "").strip()
        if not message_id:
            message_id = f"{message.sender_id}-{time.time_ns()}"
        memory = Memory(
            content=content[:2000],
            memory_type="group_analysis",
            importance=0.05,
            source_session=session_id,
            metadata={
                "message_id": message_id,
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "is_bot": False,
                "reply_to_id": message.reply_to_id,
                "reply_to_qq": message.reply_to_qq,
            },
        )

        async def _save():
            try:
                await self.memory_storage.store_group_analysis_message(memory)
            except Exception as exc:
                logger.debug("群日报消息保存失败：%s", exc)

        task = self._track_memory_task(_save())
        self._group_analysis_write_tasks.add(task)
        task.add_done_callback(self._group_analysis_write_tasks.discard)

    def _store_group_analysis_bot_message(self, session_id: str, content: str) -> None:
        """保存 Bot 群消息到日报流水，但不把 Bot 计入群友统计。"""
        config = getattr(self, "_group_analysis_config", {}) or {}
        if not config.get("enabled", True) or not str(session_id).startswith("group_"):
            return
        content = (content or "").strip()
        if not content:
            return
        self_id = str(self.config.get("qq", {}).get("self_id", ""))
        memory = Memory(
            content=content[:2000],
            memory_type="group_analysis",
            importance=0.05,
            source_session=session_id,
            metadata={
                "message_id": f"bot-{time.time_ns()}",
                "sender_id": self_id,
                "sender_name": self.personality.name,
                "is_bot": True,
            },
        )

        async def _save():
            try:
                await self.memory_storage.store_group_analysis_message(memory)
            except Exception as exc:
                logger.debug("机器人群日报消息保存失败：%s", exc)

        task = self._track_memory_task(_save())
        self._group_analysis_write_tasks.add(task)
        task.add_done_callback(self._group_analysis_write_tasks.discard)

    async def _wait_group_analysis_writes(self) -> None:
        """日报读取前等待已进入队列的群消息写入，避免漏掉最近几条。"""
        tasks = [task for task in self._group_analysis_write_tasks if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def trigger_group_analysis(
        self, session_id: str, days: int = 1, report_date: str = ""
    ) -> dict:
        """供 Dashboard 调用的手动日报入口；群内消息不再触发日报。

        report_date 传 "YYYY-MM-DD" 时，按指定自然日重新分析并补发该日报告；
        不传时固定分析今天。
        """
        session_id = str(session_id or "").strip()
        if not session_id.startswith("group_"):
            return {"success": False, "error": "只能分析群聊会话"}
        # 校验这个群真实存在（有记忆或活跃窗口），避免对不存在的 group 误触发
        # 后向该群发送“素材不足”等群内消息。
        if (
            not self.context_manager
            or session_id not in self.context_manager._windows
        ):
            try:
                exists = await self.memory_storage.get_session_messages(
                    session_id, limit=1
                )
            except Exception:
                exists = []
            if not exists:
                return {"success": False, "error": "该群没有可分析的会话记录"}
        config = getattr(self, "_group_analysis_config", {}) or {}
        if not config.get("enabled", True):
            return {"success": False, "error": "群聊日报功能目前没有开启"}
        current = self._group_analysis_tasks.get(session_id)
        if current and not current.done():
            return {"success": False, "error": "这个群的日报还在整理中"}
        # 日报固定分析自然日，不再允许按滚动 N 天回看。
        days = 1
        report_date = str(report_date or "").strip()
        trigger = Message(
            message_id="",
            message_type="group",
            sender_id="",
            sender_name="",
            group_id=session_id.removeprefix("group_"),
        )
        started = await self._start_group_analysis(
            trigger, days, automatic=False, send_ack=False, report_date=report_date
        )
        if not started:
            return {"success": False, "error": "日报任务未能启动"}
        return {
            "success": True,
            "session": session_id,
            "days": days,
            "report_date": report_date or "today",
        }

    async def _start_group_analysis(
        self,
        message: Message,
        days: int,
        automatic: bool = False,
        send_ack: bool = True,
        report_date: str = "",
    ) -> bool:
        """启动单群日报任务；同一群同时只允许一个分析任务。"""
        session_id = message.session_id
        config = getattr(self, "_group_analysis_config", {}) or {}
        if not config.get("enabled", True):
            if not automatic and send_ack and self.qq_adapter:
                await self.qq_adapter.send_message(session_id, "群聊日报功能目前没有开启")
            return False
        current = self._group_analysis_tasks.get(session_id)
        if current and not current.done():
            if not automatic and send_ack and self.qq_adapter:
                await self.qq_adapter.send_message(session_id, "这群的日报还在整理中，稍等一下")
            return False
        if not automatic and send_ack and self.qq_adapter:
            await self.qq_adapter.send_message(
                session_id, "收到，正在整理今天的群聊，等我一会儿～"
            )

        task = asyncio.create_task(
            self._run_group_analysis(
                session_id, days, automatic=automatic, report_date=report_date
            )
        )
        self._group_analysis_tasks[session_id] = task

        def _cleanup(done_task, session=session_id):
            if self._group_analysis_tasks.get(session) is done_task:
                self._group_analysis_tasks.pop(session, None)

        task.add_done_callback(_cleanup)
        return True

    async def _run_group_analysis(
        self,
        session_id: str,
        days: int,
        automatic: bool = False,
        report_date: str = "",
    ) -> None:
        """读取完整群聊流水、生成日报并保存/发送。"""
        config = getattr(self, "_group_analysis_config", {}) or {}
        try:
            await self._wait_group_analysis_writes()
            now = datetime.now()
            target_date = None
            report_label = "今日"
            try:
                target_date = datetime.strptime(str(report_date or ""), "%Y-%m-%d")
                report_label = f"{target_date.strftime('%m-%d')}"
            except ValueError:
                target_date = None
            if target_date is not None:
                since = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                until = since + timedelta(days=1)
            else:
                since = now.replace(hour=0, minute=0, second=0, microsecond=0)
                until = now + timedelta(seconds=1)
            messages = await self.memory_storage.get_group_analysis_messages(
                session_id,
                since=since,
                until=until,
                limit=config.get("max_messages", 500),
            )
            usable_count = len(GroupDailyAnalysis.human_messages(messages))
            min_messages = config.get("min_messages", 10)
            if usable_count < min_messages:
                if not automatic and self.qq_adapter:
                    await self.qq_adapter.send_message(
                        session_id,
                        f"今天只有 {usable_count} 条可分析的群友发言，至少需要 {min_messages} 条。",
                    )
                logger.info(
                    "[群日报] %s 素材不足：%d/%d",
                    session_id,
                    usable_count,
                    min_messages,
                )
                return

            provider = self.get_active_provider()
            report = await GroupDailyAnalysis.analyze(
                messages,
                provider=provider,
                max_chars=config.get("max_prompt_chars", 24000),
                max_topics=config.get("max_topics", 5),
                max_quotes=config.get("max_quotes", 3),
                max_titles=config.get("max_titles", 5),
                max_tokens=config.get("max_tokens", 2400),
                retries=config.get("retries", 2),
                bot_name=getattr(getattr(self, "personality", None), "name", "爱丽丝"),
                bot_persona=(
                    self.personality.build_persona_prompt()
                    if getattr(self, "personality", None)
                    else ""
                ),
            )
            report_text = GroupDailyAnalysis.render_report(
                report,
                report_label=report_label,
                max_chars=config.get("max_report_chars", 6000),
            )
            if target_date is not None:
                report_date_str = report_date
            else:
                report_date_str = now.strftime("%Y-%m-%d")
            report_memory = Memory(
                content=report_text,
                memory_type="group_report",
                importance=0.7,
                source_session=session_id,
                tags=["群日报"],
                metadata={
                    "kind": "group_daily_analysis",
                    "report_date": report_date_str,
                    "days": 1,
                    "message_count": report["statistics"].get("message_count", 0),
                    "participant_count": report["statistics"].get("participant_count", 0),
                    "analysis_error": report.get("analysis_error", ""),
                },
            )
            await self.memory_storage.store_group_analysis_report(report_memory)
            should_send = (not automatic) or config.get("send_report", True)
            send_mode = "none"
            if should_send and self.qq_adapter:
                avatar_fetcher = getattr(self.qq_adapter, "fetch_user_avatar", None)
                if config.get("avatars_enabled", True) and callable(avatar_fetcher):
                    async def _fetch_avatar(sender_id: str):
                        return await avatar_fetcher(
                            sender_id,
                            timeout=config.get("avatar_timeout", 6),
                        )

                    report["avatars"] = await GroupDailyAnalysis.fetch_avatars(
                        report,
                        _fetch_avatar,
                        Path(__file__).resolve().parent
                        / "data"
                        / "group_analysis"
                        / "avatars",
                        max_count=config.get("avatar_max_count", 12),
                        cache_days=config.get("avatar_cache_days", 7),
                    )
                image_bytes = GroupDailyAnalysis.render_report_image(
                    report, report_label="今日"
                )
                send_image = getattr(self.qq_adapter, "send_image", None)
                sent_as_image = False
                if image_bytes and callable(send_image):
                    try:
                        sent_as_image = bool(await send_image(session_id, image_bytes))
                    except Exception as image_exc:
                        logger.warning("[群日报] 图片发送失败，回退文本：%s", image_exc)
                if sent_as_image:
                    send_mode = "image"
                elif await self.qq_adapter.send_message(session_id, report_text):
                    send_mode = "text_fallback"
            logger.info(
                "[群日报] %s 完成：%d 条消息、%d 人、发送=%s、格式=%s",
                session_id,
                report["statistics"].get("message_count", 0),
                report["statistics"].get("participant_count", 0),
                should_send,
                send_mode,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[群日报] %s 生成失败：%s", session_id, exc, exc_info=True)
            if not automatic and self.qq_adapter:
                await self.qq_adapter.send_message(session_id, "群日报生成失败了，具体错误已记到日志里")

    @staticmethod
    def _analysis_time_matches(now: datetime, value: str) -> bool:
        try:
            hour, minute = (int(part) for part in str(value).strip().split(":", 1))
            target = hour * 60 + minute
            current = now.hour * 60 + now.minute
            return 0 <= current - target <= 6 and 0 <= hour <= 23 and 0 <= minute <= 59
        except (TypeError, ValueError):
            return False

    async def _maybe_group_analysis(self) -> None:
        """在配置时间窗口内为有素材的群自动生成日报。"""
        config = getattr(self, "_group_analysis_config", {}) or {}
        if not config.get("enabled", True) or not config.get("auto_enabled", False):
            return
        now = datetime.now()
        if not any(
            self._analysis_time_matches(now, value)
            for value in config.get("auto_times", ["23:50"])
        ):
            return
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            sessions = await self.memory_storage.get_group_analysis_sessions(since=since)
        except Exception as exc:
            logger.debug("[群日报] 获取自动分析群失败：%s", exc)
            return
        report_date = now.strftime("%Y-%m-%d")
        for item in sessions:
            session_id = str(item.get("session") or "")
            if not session_id.startswith("group_"):
                continue
            if int(item.get("message_count") or 0) < config.get("min_messages", 10):
                continue
            current = self._group_analysis_tasks.get(session_id)
            if current and not current.done():
                continue
            try:
                existing = await self.memory_storage.get_group_analysis_report(
                    session_id, report_date
                )
            except Exception:
                existing = None
            if existing:
                continue
            # 自动任务没有触发消息对象，构造一个最小的会话载体即可复用启动逻辑。
            trigger = Message(
                message_id="",
                message_type="group",
                sender_id="",
                sender_name="",
                group_id=session_id.removeprefix("group_"),
            )
            await self._start_group_analysis(trigger, 1, automatic=True)

    async def _maybe_cleanup_group_analysis(self) -> None:
        """定期清理日报原始流水，控制 SQLite 体积。"""
        config = getattr(self, "_group_analysis_config", {}) or {}
        if not config.get("enabled", True):
            return
        now_ts = time.time()
        if self._last_group_analysis_cleanup and now_ts - self._last_group_analysis_cleanup < 6 * 3600:
            return
        self._last_group_analysis_cleanup = now_ts
        try:
            cutoff = datetime.now() - timedelta(days=config.get("retention_days", 30))
            deleted = await self.memory_storage.delete_group_analysis_before(cutoff)
            if deleted:
                logger.info("[群日报] 清理过期分析数据 %d 条", deleted)
        except Exception as exc:
            logger.debug("[群日报] 清理过期数据失败：%s", exc)

    def _store_long_term_memory(self, message: Message, session_id: str) -> None:
        """把有记忆价值的消息写入 SQLite 情景记忆（异步后台执行）

        打分规则：
        - 提到 bot → +0.2
        - 消息 > 20 字（分享/吐槽）→ +0.2
        - 含个人信息关键词（我叫/我喜欢/我的生日…）→ +0.3
        - 超过阈值 0.5 才值得长期记住
        - 带引用但不满足画像条件的短句以低重要性保存，仅作对话上下文
        """
        if not getattr(self, "long_term_memory_enabled", True):
            return
        # 卡片/转发是外部引用材料，不把其内容误记成发送者自己的事实。
        # 如果外层另有文字，只记外层文字和附件类型。
        if message.rich_only:
            return
        content = (message.outer_text or message.content).strip()
        if message.rich_type:
            content = f"{content} [{message.rich_type}]".strip()
        has_reply = bool(message.reply_to_id or message.reply_to_qq)
        # 明确回复的“我也是”“对”等极短接话也要留下，方便画像提炼时读取前后文；
        # 它们会被标记为仅上下文，不会进入画像事实或普通记忆召回。
        if not content or (len(content) < 4 and not has_reply):
            return
        # 转发的领取口令/推广模板不是这个人说的话，不进长期记忆
        if self._is_promo_text(content):
            return

        self_statement = self._is_self_statement(content)
        # 普通短闲聊不进长期库，避免“说过一句就记住”；明确对 bot 说、
        # 自我描述和较完整的分享/吐槽仍然保留。带引用的短句另存为“仅上下文”，
        # 供画像理解前后文，但不会作为这个人的画像证据。
        context_only = has_reply and not (
            message.mentioned_me or self_statement or len(content) > 20
        )
        if not (message.mentioned_me or self_statement or len(content) > 20 or context_only):
            return

        importance = 0.25 if context_only else 0.3
        if message.mentioned_me:
            importance += 0.2
        if len(content) > 20:
            importance += 0.2
        if any(kw in content for kw in self._PERSONAL_KEYWORDS):
            importance += 0.3 if self_statement else 0.1

        if not context_only and importance < 0.5:
            return

        bot_config = getattr(self, "config", {}) or {}
        memory = Memory(
            content=f"{message.sender_name}：{content[:200]}",
            memory_type="episodic",
            importance=min(1.0, importance),
            source_session=session_id,
            metadata={
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "mentioned_me": message.mentioned_me,
                "message_id": message.message_id,  # 检索时排除当前消息自召回
                "reply_to_id": message.reply_to_id,
                "reply_to_qq": message.reply_to_qq,
                "mentioned_user_ids": message.mentioned_user_ids,
                "directed_to_bot": message.mentioned_me
                or (
                    str(message.reply_to_qq or "")
                    == str(bot_config.get("qq", {}).get("self_id", ""))
                    and bool(message.reply_to_qq)
                ),
                "profile_context_only": context_only,
                # 有实义内容（非纯应声）：重启恢复会话窗口时只取这类消息，
                # 避免“嗯”“好”“对”把真人对话挤出窗口。
                "meaningful": not context_only
                and (message.mentioned_me or self_statement or len(content) > 20),
            },
        )

        async def _save():
            try:
                # 写入去重：同会话同发送者已有近似内容 → 强化旧记忆，不新增重复条目
                dup = await self.memory_storage.find_similar(
                    session_id, message.sender_id, content
                )
                if dup is not None:
                    boost = max(0.05, memory.importance - dup.importance)
                    await self.memory_storage.bump_memories([dup.id], importance_boost=boost)
                    logger.debug(
                        "记忆去重：更新已有 #%d（重要性 %.2f→+%.2f），而非新增",
                        dup.id, dup.importance, boost,
                    )
                    return
                await self.memory_storage.store(memory)
                logger.debug(f"长期记忆已保存：重要性={memory.importance:.2f}")
            except Exception as e:
                logger.error(f"保存长期记忆失败：{e}")

        # 不阻塞消息处理主流程
        self._track_memory_task(_save())

    def _store_bot_memory(
        self,
        session_id: str,
        content: str,
        *,
        message_id: str = "",
        reply_to_id: str | None = None,
        reply_to_qq: str | None = None,
    ) -> None:
        """把 bot 自己发出的消息写入 SQLite 情景记忆（异步后台执行）。

        用户消息走 _store_long_term_memory（按重要性筛选）；bot 的极短应声/表情
        不进长期库，避免 bot 自己的闲聊把会话记忆刷满；有实际内容的回复仍会
        保存，保证重启后可以恢复有意义的对话片段。
        """
        if not getattr(self, "long_term_memory_enabled", True):
            return
        content = content.strip()
        if not content or (
            len(content) < 8
            and not any(mark in content for mark in ("?", "？", "!", "！"))
        ):
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
                "message_id": str(message_id or ""),
                "reply_to_id": reply_to_id,
                "reply_to_qq": reply_to_qq,
                # 有实义内容才能参与重启后的会话窗口恢复
                "meaningful": len(content) >= 8
                or any(mark in content for mark in ("?", "？", "!", "！")),
            },
        )

        async def _save():
            try:
                if self_id:
                    dup = await self.memory_storage.find_similar(
                        session_id, self_id, content
                    )
                    if dup is not None:
                        await self.memory_storage.bump_memories(
                            [dup.id], importance_boost=0.02
                        )
                        return
                await self.memory_storage.store(memory)
            except Exception as e:
                logger.error(f"保存机器人记忆失败：{e}")

        # 不阻塞消息处理主流程
        self._track_memory_task(_save())

    def _track_memory_task(self, coroutine) -> asyncio.Task:
        """创建并跟踪轻量记忆后台任务。"""
        task = asyncio.create_task(coroutine)
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)
        return task

    def _maybe_schedule_digest(self, session_id: str) -> None:
        """检查是否该生成群聊纪要：距上次纪要的新消息达到固定条数就调度后台总结"""
        if (
            not getattr(self, "long_term_memory_enabled", True)
            or not self._digest_config.get("enabled", True)
        ):
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
            logger.info(f"[纪要] {session_id}：{summary[:60]}...")

            # 纪要覆盖到的最后一条消息时间；之后新到的消息下次再总结
            self._last_digest_at[session_id] = max(m.timestamp.timestamp() for m in messages)
        except Exception as e:
            logger.error(f"生成群聊纪要失败：{e}", exc_info=True)
        finally:
            self._digest_tasks.pop(session_id, None)

    def _is_personal_fact(self, memory) -> bool:
        """判断一条 episodic 记忆是否包含"关于发送者自己的事实"。

        判据只有句式（见 _is_self_statement）。曾经还有一条"高重要性且含第一
        人称即算自述"的兜底，但普通消息的 importance 可能因重复写入强化而升高，
        结果是「被保存/命中得多」的普通消息漂到高重要性后冒充自述。
        判不准的留给日常发言，信息不丢。
        """
        return self._is_self_statement(self._strip_speaker(memory.content or ""))

    # 「我是/我在」极易切错：「对我|是吧」「帮我|在群里」「我|是说」都会命中，
    # 但都不是在讲自己的身份或所在地。命中这类词时要看后面接的是不是名词性成分。
    _AMBIGUOUS_SELF_PREDICATES = ("是", "在")
    # 使役/受益动词后面的「我」是宾语（帮我…、让我…），不是在讲自己
    _OBJECT_MARKERS = ("帮", "让", "叫", "替", "陪", "请")
    # 否定/反讽式回应：形式像自述，意思正相反（「我喜欢个🥚」＝一点也不喜欢）
    _DISMISSIVE_MARKERS = (
        "个屁", "个鬼", "个头", "个球", "个毛", "个蛋", "个锤", "个🥚", "才怪",
    )
    # 名词性词性（含数词、时间词，覆盖「我是95年的」这类）
    _NOUNISH_POS = ("n", "j", "eng", "m", "t", "s", "f")
    # 没有 jieba 时用于识别“我是/我在”后面的动词、语气词。这里只拦
    # 明显的转折/宾语结构，保留“我在看电影”“我是学生”这类正常自述。
    _NON_NOUNISH_TAIL_PREFIXES = (
        "说", "想", "要", "帮", "让", "能", "会", "是不是",
        "看我", "看谁", "吗", "吧", "呢", "啊", "呀", "了",
    )

    @classmethod
    def _is_self_statement(cls, content: str) -> bool:
        """判断文本是否真的在描述发送者自己。

        群聊里的单条消息是孤立片段，很可能是在回别人的话，所以不能只做
        子串匹配——「就这么对我是吧」「我喜欢个🥚」形式上都能命中。
        判不准的不算自述，但仍会作为日常发言参与提炼，信息不会丢。
        """
        text = (content or "").strip()
        if not text or any(mark in text for mark in cls._DISMISSIVE_MARKERS):
            return False
        hits = [kw for kw in PERSONAL_KEYWORDS if kw in text]
        if not hits:
            return False
        # 非歧义关键词（我家/我住/我老婆…）命中即可
        if any(kw[1:] not in cls._AMBIGUOUS_SELF_PREDICATES for kw in hits):
            return True
        return any(cls._is_identity_statement(text, kw) for kw in hits)

    @classmethod
    def _is_identity_statement(cls, text: str, keyword: str) -> bool:
        """「我是X」「我在X」中的 X 是否是名词性成分（身份、所在地）。"""
        index = text.find(keyword)
        while index >= 0:
            preceded_by_object_marker = (
                index > 0 and text[index - 1] in cls._OBJECT_MARKERS
            )
            if not preceded_by_object_marker:
                tail = text[index + len(keyword):].strip()
                if tail and cls._tail_is_nounish(tail):
                    return True
            index = text.find(keyword, index + 1)
        return False

    @classmethod
    def _tail_is_nounish(cls, tail: str) -> bool:
        """判断「我是/我在」后面接的是不是名词性成分。"""
        # 「我是…的」是典型判断句（我是玩周瑜的），照样算身份陈述
        normalized = tail.strip()
        if normalized.rstrip("。！!？?～~ ").endswith("的"):
            return True
        if not normalized or normalized.startswith(cls._NON_NOUNISH_TAIL_PREFIXES):
            return False
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            import jieba.posseg as pseg
            for word, flag in pseg.cut(normalized):
                if not word.strip():
                    continue
                return flag.startswith(cls._NOUNISH_POS)
        except Exception:
            # 没有分词器时采用保守回退：至少需要两个有效字符，避免
            # “我是吧/我是说……”等语气或话头被记成个人事实。
            return len(normalized.strip("。！!？?～~ ")) >= 2
        return False

    def _is_meaningful_chat(self, memory) -> bool:
        """判断一条记忆是否是有信息量的日常发言（可参与画像推断）。

        - 太短 / 纯富媒体占位（图片、表情包、转发、链接摘要）没有稳定信号
        - 只留正常聊天内容，供画像从"常聊话题 / 说话风格 / 行为习惯"中推断
        """
        content = self._strip_speaker(memory.content or "")
        if (memory.metadata or {}).get("is_bot"):
            return False
        if len(content) < 8:
            return False
        if content.startswith("[") and content.endswith("]"):
            return False
        # 历史库里已存的口令广告：模板文本，不参与画像提炼
        if self._is_promo_text(content):
            return False
        return True

    @staticmethod
    def _strip_speaker(content: str) -> str:
        """剥掉记忆内容开头的「昵称：」前缀（昵称可能随改群名片变化）。"""
        if "：" in content:
            head, rest = content.split("：", 1)
            if len(head) <= 40 and not head.startswith(("http", "[", "【")):
                return rest.strip()
        return content.strip()

    @staticmethod
    def _dedupe_messages(msgs: list, cap: int) -> list:
        """按内容去重（近似），保留时间较新的，最多 cap 条。"""
        out: list = []
        seen: set = set()
        for m in msgs:  # 调用方已按时间倒序
            key = GroupChatBot._strip_speaker(m.content or "")[:30]
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
            if len(out) >= cap:
                break
        return out

    # 这些消息即使字数不短，也常常只是接话、附和或指代前文；补上下文后再交给
    # 模型判断，避免把别人刚说的兴趣/经历套到目标用户身上。
    _PROFILE_CONTEXT_MARKERS = (
        "我也是", "我也", "俺也", "我呢", "同上", "一样", "你说的",
        "这个", "那个", "这样", "那样", "上面", "楼上", "刚才", "刚刚",
        "确实", "对啊", "是啊", "不是吧", "笑死", "他说", "她说",
        "他们", "她们", "别人", "对方", "那个人",
    )

    @classmethod
    def _needs_profile_context(cls, memory) -> bool:
        """判断一条目标发言是否可能脱离前后文就无法准确理解。"""
        content = cls._strip_speaker(memory.content or "")
        if not content:
            return False
        metadata = memory.metadata or {}
        if metadata.get("reply_to_id") or metadata.get("reply_to_qq"):
            return True
        if len(content) <= 24:
            return True
        return any(marker in content for marker in cls._PROFILE_CONTEXT_MARKERS)

    async def _render_profile_context(
        self,
        targets: list,
        before: int = 2,
        after: int = 2,
        max_targets: int = 12,
        since: Optional[datetime] = None,
    ) -> str:
        """渲染少量消歧上下文；上下文中的他人发言永远不作为画像证据。"""
        reader = getattr(self.memory_storage, "get_profile_context", None)
        if not callable(reader):
            return ""
        selected = sorted(
            [m for m in (targets or []) if getattr(m, "id", None)],
            key=lambda m: m.created_at,
            reverse=True,
        )[:max(1, int(max_targets))]
        blocks = []
        seen_ids = set()
        for target in selected:
            if target.id in seen_ids:
                continue
            seen_ids.add(target.id)
            try:
                rows = await reader(
                    target.id, before=before, after=after, since=since
                )
            except Exception as exc:
                logger.debug("[画像] 读取对话上下文失败（编号%s）：%s", target.id, exc)
                continue
            if len(rows) <= 1:
                continue
            lines = []
            has_target = False
            for row in rows:
                content = self._strip_speaker(row.content or "")[:120]
                if not content:
                    continue
                metadata = row.metadata or {}
                sender = (
                    metadata.get("sender_name")
                    or (
                        getattr(getattr(self, "personality", None), "name", "Bot")
                        if metadata.get("is_bot") else "群友"
                    )
                    or metadata.get("sender_id")
                    or "群友"
                )
                if row.id == target.id:
                    label = "目标发言"
                    has_target = True
                else:
                    label = "上下文"
                lines.append(f"[{label}] {sender}：{content}")
            if has_target and len(lines) > 1:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # 画像里不该出现的用词：prompt 已明令禁止推测，但模型偶尔还是会写。
    # 这里不直接丢弃（画像其余部分可能是对的），而是打标记交给 Web 端展示，
    # 方便人工核对原句后修正。
    _PROFILE_HEDGE_WORDS = (
        "可能", "似乎", "也许", "大概", "或许", "貌似", "估计", "应该是",
    )
    # 对任何人都成立、等于没说的描述
    _PROFILE_VAGUE_PATTERNS = (
        "关注游戏话题", "涉及游戏话题", "喜欢和群友互动", "与群友互动",
        "发言活跃", "群内活跃", "热衷于群聊", "关注度高",
    )

    # 画像正文的最大长度：累加机制下画像会一轮轮变长，不设上限最终会膨胀成
    # 一段小作文，既挤占提示词也没人看得下去。
    _PROFILE_MAX_CHARS = 120
    # 模型偶尔会带上元话语开头（"更新后画像：""综合来看："），那不是画像内容
    _PROFILE_META_PREFIXES = (
        "更新后画像：", "更新后的画像：", "更新画像：", "画像：", "更新：",
        "综合来看：", "综合以上：", "总结：", "以下是画像：", "新画像：",
    )

    @classmethod
    def _normalize_profile_summary(cls, summary: str) -> str:
        """清理画像正文：剥掉元话语前缀，并按句子边界收进长度上限。"""
        text = (summary or "").strip().strip('"\'“”')
        changed = True
        while changed:
            changed = False
            for prefix in cls._PROFILE_META_PREFIXES:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    changed = True
        if len(text) <= cls._PROFILE_MAX_CHARS:
            return text

        # 超长：按句子边界保留完整句，避免从半句切断
        import re as _re
        sentences = _re.split(r"(?<=[。；;！!？?])", text)
        kept = ""
        for sentence in sentences:
            if kept and len(kept) + len(sentence) > cls._PROFILE_MAX_CHARS:
                break
            kept += sentence
        kept = kept.strip()
        if kept:
            return kept[:cls._PROFILE_MAX_CHARS].rstrip("，,、;； ")
        return text[:cls._PROFILE_MAX_CHARS].rstrip("，,、;； ")

    # === MBTI 性格倾向分析（与画像共用素材） ===
    # 说明：MBTI 本身不是严谨心理测量，这里只作为"从群聊行为看性格倾向"的
    # 娱乐向分析展示，不参与回复生成，避免 bot 在群里给人贴标签。
    _MBTI_AXES = (
        ("E", "I", "E/I"),
        ("S", "N", "S/N"),
        ("T", "F", "T/F"),
        ("J", "P", "J/P"),
    )
    # 素材太少时四个维度基本靠猜，不如不给
    _MBTI_MIN_SAMPLES = 8

    @staticmethod
    def _parse_confidence(raw: str) -> float:
        """宽松解析置信度：兼容 0.65 / 65% / 「0.6（中）」等写法，解析不出按 0 处理。"""
        import re as _re

        text = (raw or "").strip()
        match = _re.search(r"\d+(?:\.\d+)?", text)
        if not match:
            return 0.0
        try:
            value = float(match.group(0))
        except ValueError:
            return 0.0
        if "%" in text or value > 1.0:
            value = value / 100 if value > 1.0 else value
        return max(0.0, min(1.0, value))

    @classmethod
    def _parse_mbti_response(cls, content: str) -> Optional[dict]:
        """解析 MBTI 四行输出；任一维度缺失就返回 None（宁可不显示也不猜）。

        期望格式（每行）：`E/I: I | 0.65 | 很少主动开话题，多是被点名才回`
        置信度一栏容错：模型偶尔写成百分比或带注释，解析不出按 0 处理。
        """
        import re as _re

        text = _re.sub(r"<think>.*?</think>", "", content or "", flags=_re.DOTALL)
        dimensions = []
        letters = []
        for first, second, label in cls._MBTI_AXES:
            pattern = (
                rf"{first}\s*/\s*{second}\s*[:：]\s*([{first}{second}])"
                rf"\s*[|｜]\s*([^|｜\n]*)[|｜]\s*(.+)"
            )
            match = _re.search(pattern, text, _re.IGNORECASE)
            if not match:
                return None
            letter = match.group(1).upper()
            dimensions.append({
                "axis": label,
                "letter": letter,
                "confidence": round(cls._parse_confidence(match.group(2)), 2),
                "reason": match.group(3).strip().strip("。").strip()[:60],
            })
            letters.append(letter)
        return {"type": "".join(letters), "dimensions": dimensions}

    async def _analyze_mbti(
        self,
        provider,
        name: str,
        profile_summary: str,
        self_text: str,
        daily_text: str,
    ) -> Optional[dict]:
        """根据群聊行为判断 MBTI 四维倾向。失败返回 None，不影响画像本身。"""
        from modules.llm.base import ChatRequest

        req = ChatRequest(temperature=0.2, max_tokens=400, top_p=0.9)
        req.add_system(
            "你是性格倾向分析助手。根据群友在群聊里的实际行为判断 MBTI 四个维度。\n"
            "维度按群聊行为理解，不是做量表题：\n"
            "E/I 外向-内向：主动开话题、拉人互动、发言频繁 vs 多为被动回应、话少\n"
            "S/N 实感-直觉：聊具体的事、细节、操作和数值 vs 聊想法、联想、类比、玩抽象梗\n"
            "T/F 思考-情感：就事论事、讲道理、直接指出问题 vs 照顾对方感受、情绪表达多\n"
            "J/P 判断-知觉：有计划、守时守规律、说到做到 vs 随性、临时起意、作息不定\n"
            "\n校准（很重要）：\n"
            "- 群聊场景天然显得随意，不要因为「在群里闲聊」就一律判 P。"
            "有固定作息、按计划推进的事、长期坚持的习惯、守时守约，这些都是 J 的证据。\n"
            "- 资料里只有TA说过的话，不要因为「有发言」就默认判 E。"
            "要看是否主动发起话题、拉人互动；多为被动接话、被点名才出现的是 I。\n"
            "- 四个维度要分别独立判断，不要为了凑成常见类型而互相迁就。\n"
            "\n严格输出四行，每行格式为「维度: 字母 | 置信度 | 依据」，不要输出其他内容：\n"
            "E/I: X | 0.65 | 依据\n"
            "S/N: X | 0.6 | 依据\n"
            "T/F: X | 0.55 | 依据\n"
            "J/P: X | 0.7 | 依据\n"
            "\n置信度是 0 到 1 的小数。依据必须是资料里出现过的具体行为，20字以内，"
            "不要复述原话。某个维度资料不足以判断时，仍要给字母，但置信度填 0.3 以下。"
        )
        parts = [f"群友「{name}」的资料："]
        if profile_summary:
            parts.append(f"【已有画像】\n{profile_summary}")
        if self_text:
            parts.append(f"【自我描述】\n{self_text}")
        if daily_text:
            parts.append(f"【日常发言】\n{daily_text}")
        req.add_user("\n".join(parts))

        try:
            resp = await provider.chat(req)
        except Exception as exc:
            logger.warning("[MBTI] %s 分析失败：%s", name, exc)
            return None
        parsed = self._parse_mbti_response(resp.content if resp else "")
        if not parsed:
            logger.info("[MBTI] %s 输出无法解析，跳过", name)
            return None
        parsed["analyzed_at"] = datetime.now().isoformat()
        logger.info(
            "[MBTI] %s：类型=%s（%s）",
            name, parsed["type"],
            "、".join(f"{d['axis']}{d['letter']}{d['confidence']}" for d in parsed["dimensions"]),
        )
        return parsed

    @staticmethod
    def _strip_profile_prefix(content: str) -> str:
        """取画像正文（去掉「【用户画像 名字】」前缀），用于喂回给模型做增量修订。"""
        text = (content or "").strip()
        if text.startswith("【用户画像") and "】" in text:
            text = text.split("】", 1)[1]
        return text.strip()

    @classmethod
    def _profile_quality_warnings(cls, summary: str) -> list:
        """标出画像里的可疑表述（推测、性别不明、空话），供人工复核。"""
        text = summary or ""
        warnings = []
        hedges = [w for w in cls._PROFILE_HEDGE_WORDS if w in text]
        if hedges:
            warnings.append("含推测用词：" + "、".join(hedges))
        if "他/她" in text or "TA" in text:
            warnings.append("性别不明确")
        vague = [p for p in cls._PROFILE_VAGUE_PATTERNS if p in text]
        if vague:
            warnings.append("描述空泛：" + "、".join(vague))
        return warnings

    @staticmethod
    def _is_profile_refusal(summary: str) -> bool:
        """判断画像提炼输出是否是 LLM 的拒绝文本（素材太散时会答"无法提炼"）。

        拒绝文本存成画像会污染检索：召回时 bot 会"记得"一段关于某人的废话。
        拒绝语可能藏在句中（如"……从现有发言中难以提炼出明确特征"），所以扫全文；
        这些短语不会出现在真正的画像里，全文匹配不会误伤。
        """
        markers = (
            "无法提炼", "无法判断", "无法从", "信息不足", "不足以",
            "难以提炼", "难以判断", "没有足够", "缺乏可识别", "无明确语义",
            "提炼不出", "无稳定",
        )
        text = summary or ""
        return any(m in text for m in markers)

    async def _distill_user_profiles(self, force: bool = False) -> dict:
        """从情景记忆提炼用户画像（semantic 记忆，后台执行）。

        每个发送者的画像由两部分信号合成：
        - 【自我描述】TA直接说过的关于自己的话（较可信）
        - 【日常发言】TA的日常聊天，从中推断兴趣/性格/习惯等稳定倾向
        已有画像且没有"新自述 + 足够新日常发言"时跳过；force=True 强制重炼。
        画像存为 semantic 记忆，语义检索时可被召回。
        """
        result = {"distilled": 0, "skipped": 0, "failed": 0, "error": ""}
        if (
            not getattr(self, "long_term_memory_enabled", True)
            or not self._profile_config.get("enabled", True)
        ):
            return result
        guard = getattr(self, "_profile_distill_guard", None)
        if guard is None:
            guard = threading.Lock()
            self._profile_distill_guard = guard
        if not guard.acquire(blocking=False):
            result["error"] = "profile distillation already running"
            result["skipped"] += 1
            return result
        min_facts = self._profile_config.get("min_facts", 2)
        daily_min = self._profile_config.get("daily_min_messages", 8)
        window_days = self._profile_config.get("daily_window_days", 15)
        refresh_daily = self._profile_config.get("refresh_daily_count", 5)
        mbti_refresh_daily = self._profile_config.get("mbti_refresh_daily_count", 12)
        mbti_enabled = self._profile_config.get("mbti_enabled", True)
        context_enabled = self._profile_config.get("context_enabled", True)
        context_neighbors = self._profile_config.get("context_neighbors", 2)
        context_max_targets = self._profile_config.get("context_max_targets", 12)
        try:
            profiles = await self.memory_storage.get_profiles()
            profile_by_scope = {}
            for p in profiles:
                sender_id = (p.metadata or {}).get("sender_id")
                if not sender_id:
                    continue
                scope = p.source_session or (p.metadata or {}).get("profile_session", "")
                key = (str(sender_id), "" if self.memory_share_across_sessions else scope)
                # get_profiles 按最近访问/更新倒序；旧库若存在重复画像，保留最新的一条。
                if key not in profile_by_scope:
                    profile_by_scope[key] = p

            # 先列出有素材的用户/会话，再分别取每个范围自己的最近 2000 条。
            # 不能用全局最近 2000 条，否则活跃群会把冷门用户挤出画像素材。
            scopes = await self.memory_storage.get_profile_scopes(
                share_across_sessions=self.memory_share_across_sessions,
            )
            suppressions = await self.memory_storage.get_profile_suppressions()
            profile_attempts = await self.memory_storage.get_profile_attempts()

            async def remember_profile_attempt(scope_key, material_sig) -> None:
                """持久化无结果/失败的素材签名，避免重启后重复问同一批素材。"""
                self._profile_failed_material[scope_key] = material_sig
                try:
                    await self.memory_storage.set_profile_attempt(
                        scope_key[0],
                        scope_key[1],
                        material_sig[0],
                        material_sig[1],
                    )
                except Exception as exc:
                    # 记录失败不能反过来阻断其它用户画像；内存标记仍然有效。
                    logger.debug("[画像] 保存提炼尝试记录失败：%s", exc)

            by_scope: dict[tuple, dict] = {}
            for sid, source_scope in scopes:
                scope_key = (
                    str(sid),
                    "" if self.memory_share_across_sessions else source_scope,
                )
                try:
                    materials = await self.memory_storage.get_profile_materials(
                        sid,
                        source_session=source_scope,
                        share_across_sessions=self.memory_share_across_sessions,
                        limit=2000,
                        after=suppressions.get(scope_key),
                    )
                except Exception as exc:
                    logger.warning("[画像] %s 素材读取失败，跳过本范围：%s", sid, exc)
                    result["failed"] += 1
                    if not result["error"]:
                        result["error"] = str(exc)
                    continue
                if not materials:
                    continue
                entry = by_scope.setdefault(
                    scope_key, {"self": [], "daily": [], "latest_name": ""}
                )
                for m in materials:
                    meta = m.metadata or {}
                    if meta.get("is_bot") or meta.get("profile_context_only"):
                        continue
                    if meta.get("sender_name"):
                        entry["latest_name"] = meta["sender_name"]
                    if self._is_personal_fact(m):
                        entry["self"].append(m)
                    elif self._is_meaningful_chat(m):
                        entry["daily"].append(m)

            provider = self.get_active_provider()
            if not provider:
                result["error"] = "no active provider"
                return result

            from modules.llm.base import ChatRequest
            from datetime import timedelta

            now = datetime.now()
            cutoff = now - timedelta(days=window_days)

            for scope_key, entry in by_scope.items():
                sid, profile_scope = scope_key
                self_facts = sorted(entry["self"], key=lambda m: m.created_at, reverse=True)
                daily = sorted(entry["daily"], key=lambda m: m.created_at, reverse=True)
                existing = profile_by_scope.get(scope_key)
                latest_name = entry["latest_name"] or "该用户"

                # 素材不足：自述少且日常发言也少 → 无可提炼
                if len(self_facts) < min_facts and len(daily) < daily_min:
                    logger.debug(
                        "[画像] 跳过 %s：素材不足（自述%d < %d 且日常%d < %d）",
                        latest_name, len(self_facts), min_facts, len(daily), daily_min,
                    )
                    result["skipped"] += 1
                    continue

                # 提炼不出稳定特征的人：素材没变化就别每轮都再问一次 LLM。
                # （已有画像的人由下面的"新素材"门槛把关，这条覆盖的是还没有画像的人）
                material_sig = (len(self_facts), len(daily))
                failed_sig = self._profile_failed_material.get(scope_key)
                if failed_sig is None:
                    failed_sig = profile_attempts.get(scope_key)
                if not force and failed_sig == material_sig:
                    logger.debug(
                        "[画像] 跳过 %s：上次提炼不出且素材无变化（自述%d 日常%d）",
                        latest_name, len(self_facts), len(daily),
                    )
                    result["skipped"] += 1
                    continue

                # 已有画像且没有足够新素材 → 跳过（避免每次发言都刷）
                if not force and existing and existing.created_at:
                    newest_self = self_facts[0].created_at if self_facts else None
                    has_new_self = newest_self is not None and newest_self > existing.created_at
                    new_daily = [m for m in daily if m.created_at > existing.created_at]
                    if not has_new_self and len(new_daily) < refresh_daily:
                        result["skipped"] += 1
                        continue

                # 构建样本：自述最多 12 条，日常发言限最近窗口内最多 20 条
                unique_facts = self._dedupe_messages(self_facts, 12)
                daily_window = [m for m in daily if m.created_at > cutoff]
                unique_daily = self._dedupe_messages(daily_window, 20)

                def render_section(msgs: list, cap_len: int = 120) -> str:
                    return "\n".join(
                        self._strip_speaker(m.content or "")[:cap_len] for m in reversed(msgs)
                    )

                def source_lines(msgs: list, cap_len: int = 120) -> list:
                    """留存提炼素材原句，供 Web 端追溯"哪句话导致了这个结论"。"""
                    return [
                        self._strip_speaker(m.content or "")[:cap_len]
                        for m in reversed(msgs)
                        if (m.content or "").strip()
                    ]

                self_text = render_section(unique_facts)
                daily_text = render_section(unique_daily)
                context_text = ""
                if context_enabled:
                    context_targets = [
                        item for item in unique_facts + unique_daily
                        if self._needs_profile_context(item)
                    ]
                    context_text = await self._render_profile_context(
                        context_targets,
                        before=context_neighbors,
                        after=context_neighbors,
                        max_targets=context_max_targets,
                        since=suppressions.get(scope_key),
                    )
                existing_meta = (existing.metadata or {}) if existing else {}
                # 带质量警告的旧画像仅供面板复核，不能当作可信结论回灌给 LLM，
                # 否则一次错误推测会在后续增量更新中反复自我强化。
                previous_summary = "" if existing_meta.get("warnings") else self._strip_profile_prefix(
                    existing.content if existing else ""
                )

                # 累加而非重写：素材窗口只有最近15天，若每次从零重写，
                # 职业、所在地这类稳定事实会随旧消息滚出窗口而被"忘掉"。
                # 把已有画像一并给模型，让它在旧结论上修订。
                req = ChatRequest(temperature=0.3, max_tokens=300, top_p=0.9)
                req.add_system(
                    "你是用户画像提炼助手。根据群友的【自我描述】和【日常发言】，写出这个人"
                    "长期稳定的特征（职业、常玩的游戏、作息、习惯、性格、常聊话题等）。\n"
                    "- 【自我描述】是TA直接说过的关于自己的话，较可信\n"
                    "- 【日常发言】是TA的日常聊天，只从反复出现的模式里归纳\n"
                    "- 两类素材都是从群聊里摘出来的孤立单句，很多是在回别人的话，"
                    "看不到上下文。读不懂或像是在接别人话茬的，直接跳过，不要硬解释。\n"
                    "- 【对话上下文】只用于理解【目标发言】的指代和语气；上下文里的其他人"
                    "说过的兴趣、经历、身份和观点，绝对不能归到TA身上。\n"
                    "- 【已有画像】是之前根据更早的聊天总结的结论（本次资料里可能不再提到）\n"
                    "\n给了【已有画像】时，要在它的基础上更新，而不是重写：\n"
                    "A. 保留仍然成立的稳定事实（职业、所在地、长期爱好），"
                    "即使这次资料里没再提到也要留着——人不会因为最近没说就不是那样的人。\n"
                    "B. 新资料和已有画像冲突时以新资料为准，直接改掉旧结论。\n"
                    "C. 新资料里发现的新的稳定特征，补充进去。\n"
                    "D. 已有画像里明显是一次性事件、或事后看判断错的，删掉。\n"
                    "\n判断标准：\n"
                    "1. 只写反复出现、多次印证的事。一次性事件（某天加班、家里进水、某次消费）"
                    "是经历不是特征，不要写。\n"
                    "2. 群聊里大量是玩笑、玩梗、反讽，不要当真；被认真提过或多次提到的才算。\n"
                    "3. 不确定的就不写，不要用「可能」「似乎」「也许」「大概」这类词。\n"
                    "4. 不要写对谁都成立的空话，例如「关注游戏话题」「喜欢和群友互动」"
                    "「发言活跃」。只写能把TA和别人区分开的信息。\n"
                    "5. 不要猜年龄、性别、收入、家境，资料没明说就不写。\n"
                    "6. 用第三人称，不复述原话，不解释你的推理过程。\n"
                    "7. 只能写这份资料或已有画像里出现过的内容。没出现过的职业、地名、游戏名，"
                    "一个字都不能出现——绝对不要把别人的情况套到TA身上。\n"
                    "\n粒度要求：写具体的东西——做什么工作或在哪读书、常玩什么、什么作息或"
                    "习惯、说话有什么特点，每一项都必须有出处；"
                    "不要写「是个游戏爱好者」「喜欢和大家聊天」这种笼统评价。\n"
                    "\n输出一段完整画像（不是修改说明、不要分点、不要加「更新后画像：」"
                    "这类开头），必须控制在120字以内——这是硬性要求，"
                    "累积的内容变多时要做取舍，只保留最能代表这个人的信息。\n"
                    "只要能找到一条具体且反复出现的信息，就要写出来。"
                    "只有当资料和已有画像里都找不到任何具体信息时，才输出「无法提炼」四个字。"
                )
                parts = [f"群友「{latest_name}」的资料："]
                if previous_summary:
                    parts.append(f"【已有画像】\n{previous_summary}")
                if self_text:
                    parts.append(f"【自我描述】\n{self_text}")
                if daily_text:
                    parts.append(f"【日常发言】\n{daily_text}")
                if context_text:
                    parts.append(
                        "【对话上下文（仅用于消歧，非画像证据）】\n" + context_text
                    )
                req.add_user("\n".join(parts))
                try:
                    resp = await provider.chat(req)
                except Exception as exc:
                    logger.warning("[画像] %s 提炼失败，继续处理其他用户：%s", latest_name, exc)
                    result["failed"] += 1
                    await remember_profile_attempt(scope_key, material_sig)
                    if not result["error"]:
                        result["error"] = str(exc)
                    continue
                summary = (getattr(resp, "content", "") or "").strip().strip('"\'“”')
                if not summary:
                    logger.info("[画像] 跳过 %s：语言模型返回空", latest_name)
                    await remember_profile_attempt(scope_key, material_sig)
                    result["skipped"] += 1
                    continue
                if self._is_profile_refusal(summary):
                    logger.info("[画像] 跳过 %s：素材判断不出稳定特征（%s）",
                                latest_name, summary[:40])
                    await remember_profile_attempt(scope_key, material_sig)
                    result["skipped"] += 1
                    continue
                summary = self._normalize_profile_summary(summary)
                if not summary:
                    logger.info("[画像] 跳过 %s：清理后无有效内容", latest_name)
                    await remember_profile_attempt(scope_key, material_sig)
                    result["skipped"] += 1
                    continue
                self._profile_failed_material.pop(scope_key, None)

                # 样本里最新一条的来源会话作为画像归属会话
                sample = unique_facts + unique_daily
                src_session = (
                    profile_scope
                    if not self.memory_share_across_sessions
                    else (sample[0].source_session if sample else "")
                )
                warnings = self._profile_quality_warnings(summary)
                if warnings:
                    logger.info("[画像] %s 质量提示：%s", latest_name, "；".join(warnings))
                # MBTI：与画像共用素材，只在画像确实更新且素材足够时才分析。
                # 失败或素材不足时沿用上一次的结果，不影响画像写入。
                mbti = existing_meta.get("mbti")
                sample_size = len(unique_facts) + len(unique_daily)
                new_material_count = (
                    sample_size
                    if not existing
                    else sum(
                        1 for item in unique_facts + unique_daily
                        if item.created_at > existing.created_at
                    )
                )
                refresh_mbti = (
                    not mbti
                    or force
                    or new_material_count >= mbti_refresh_daily
                )
                if mbti_enabled and sample_size >= self._MBTI_MIN_SAMPLES and refresh_mbti:
                    fresh = await self._analyze_mbti(
                        provider, latest_name, summary, self_text, daily_text
                    )
                    if fresh:
                        fresh["sample_size"] = sample_size
                        mbti = fresh

                # 面板删除画像与本轮提炼可能并发；删除端会持有同一把锁，
                # 这里再读一次抑制点，兼容其它直接调用 storage.delete 的路径。
                try:
                    latest_suppressions = await self.memory_storage.get_profile_suppressions()
                except Exception as exc:
                    logger.warning("[画像] %s 保存前无法确认删除抑制，跳过本轮：%s", latest_name, exc)
                    result["failed"] += 1
                    if not result["error"]:
                        result["error"] = str(exc)
                    continue
                if latest_suppressions.get(scope_key):
                    logger.info("[画像] %s 已在提炼期间被删除，跳过本轮写回", latest_name)
                    result["skipped"] += 1
                    continue

                new_profile = Memory(
                    content=f"【用户画像 {latest_name}】{summary}",
                    memory_type="semantic",
                    importance=0.8,
                    source_session=src_session,
                    tags=["用户画像"],
                    created_at=now,
                    last_accessed=now,
                    metadata={
                        "profile": True,
                        "sender_id": sid,
                        "sender_name": latest_name,
                        "profile_session": src_session,
                        "fact_count": len(unique_facts),
                        "daily_count": len(unique_daily),
                        # 留存提炼素材：Web 端据此追溯"哪句话导致了这个结论"
                        "source_facts": source_lines(unique_facts),
                        "source_daily": source_lines(unique_daily),
                        "warnings": warnings,
                        # 累积轨迹：画像是在历次结论上迭代出来的，不是本次素材的快照
                        "first_distilled_at": (
                            existing_meta.get("first_distilled_at") or now.isoformat()
                        ),
                        "last_distilled_at": now.isoformat(),
                        "distill_count": int(existing_meta.get("distill_count", 0) or 0) + 1,
                        "previous_summary": previous_summary,
                        "mbti": mbti,
                    },
                )
                try:
                    if existing and existing.id:
                        new_profile.id = existing.id
                        await self.memory_storage.update_memory(new_profile)
                    else:
                        await self.memory_storage.store(new_profile)
                    await self.memory_storage.clear_profile_suppression(sid, profile_scope)
                    await self.memory_storage.clear_profile_attempt(sid, profile_scope)
                except Exception as exc:
                    logger.warning("[画像] %s 保存失败，继续处理其他用户：%s", latest_name, exc)
                    result["failed"] += 1
                    if not result["error"]:
                        result["error"] = str(exc)
                    continue
                result["distilled"] += 1
                logger.info(f"[画像] {latest_name}：{summary[:50]}...")
        except Exception as e:
            logger.error(f"提炼用户画像失败：{e}", exc_info=True)
            result["error"] = str(e)
        finally:
            guard.release()
        return result

    async def _maybe_distill_profiles(self) -> None:
        """定时触发画像提炼（按 interval_minutes 节流，避免频繁调用 LLM）"""
        if (
            not getattr(self, "long_term_memory_enabled", True)
            or not self._profile_config.get("enabled", True)
        ):
            return
        import time
        interval = max(5, self._profile_config.get("interval_minutes", 30))
        now = time.time()
        if self._last_profile_distill and (now - self._last_profile_distill) < interval * 60:
            return
        self._last_profile_distill = now
        await self._distill_user_profiles(force=False)

    # === 群聊黑话提取 ===

    # 提取结果的过滤：这些是通用网络用语，不是这个群特有的，收进词表没意义
    _SLANG_TOO_COMMON = {
        "yyds", "绝了", "笑死", "牛逼", "破防", "emo", "666", "awsl",
        "无语", "离谱", "摆烂", "内卷", "润", "上头", "真香", "社死",
    }

    @classmethod
    def _parse_slang_lines(cls, content: str) -> list[dict]:
        """解析黑话提取结果：每行 `词 | 含义 | 例句`（例句可省略）。"""
        import re as _re

        text = _re.sub(r"<think>.*?</think>", "", content or "", flags=_re.DOTALL)
        items = []
        seen = set()
        for line in text.splitlines():
            line = line.strip().lstrip("-•*0123456789. ").strip()
            if not line or "|" not in line and "｜" not in line:
                continue
            parts = [p.strip() for p in _re.split(r"[|｜]", line)]
            if len(parts) < 2:
                continue
            term, meaning = parts[0].strip("「」\"'`【】 "), parts[1]
            example = parts[2] if len(parts) > 2 else ""
            if not term or not meaning or len(term) > 20 or len(meaning) < 2:
                continue
            if term.lower() in cls._SLANG_TOO_COMMON:
                continue
            # 「词」这一栏偶尔会被写成说明句，明显过长的丢掉
            if len(term) > 12 and len(term) > len(meaning):
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "term": term,
                "meaning": meaning[:120],
                "example": example[:80],
            })
        return items

    @classmethod
    def _parse_slang_cleanup_ids(cls, content: str, valid_ids) -> list[int]:
        """解析黑话审核结果，只接受明确标记为删除的词条 ID。"""
        import json as _json
        import re as _re

        allowed = set()
        for value in valid_ids or []:
            try:
                allowed.add(int(value))
            except (TypeError, ValueError):
                continue
        if not allowed:
            return []

        text = _re.sub(r"<think>.*?</think>", "", content or "", flags=_re.DOTALL)
        deleted = set()

        def add(value) -> None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                return
            if value in allowed:
                deleted.add(value)

        # 兼容模型偶尔返回的 JSON：{"delete_ids": [1, 2]} 或
        # [{"id": 1, "action": "delete"}]。
        def collect_json(value) -> None:
            if isinstance(value, dict):
                for key in ("delete_ids", "remove_ids", "删除", "移除"):
                    values = value.get(key)
                    if isinstance(values, (list, tuple)):
                        for item in values:
                            add(item)
                action = str(
                    value.get("action") or value.get("status")
                    or value.get("decision") or ""
                ).lower()
                if value.get("id") is not None and any(
                    word in action for word in ("delete", "remove", "invalid", "删除", "移除")
                ):
                    add(value.get("id"))
                for nested in value.values():
                    if isinstance(nested, (dict, list, tuple)):
                        collect_json(nested)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (dict, list, tuple)):
                        collect_json(item)

        for block in _re.findall(r"```(?:json)?\s*(.*?)```", text, flags=_re.I | _re.S):
            try:
                collect_json(_json.loads(block.strip()))
            except (TypeError, ValueError, _json.JSONDecodeError):
                pass

        # 约定格式：`删除 12 | 原因`。只解析包含删除动作的行，避免把原因里的
        # 普通数字误当成词条 ID。
        for line in text.splitlines():
            if not _re.search(r"删除|移除|清理|delete|remove|invalid|wrong", line, _re.I):
                continue
            match = _re.search(
                r"(?:删除|移除|清理|delete|remove|invalid|wrong)"
                r"[^0-9]{0,20}(\d+)",
                line,
                flags=_re.I,
            )
            if match:
                add(match.group(1))

        return sorted(deleted)

    @classmethod
    def _is_obviously_invalid_slang(cls, row: dict) -> bool:
        """过滤数据库里历史遗留的明显坏词条，人工词条由调用方先排除。"""
        import re as _re

        term = str(row.get("term") or "").strip()
        meaning = str(row.get("meaning") or "").strip()
        normalized = _re.sub(r"\s+", "", term).casefold()
        common = {
            _re.sub(r"\s+", "", str(item)).casefold()
            for item in cls._SLANG_TOO_COMMON
        }
        return (
            not term
            or not meaning
            or len(term) > 20
            or len(meaning) < 2
            or normalized in common
            or (len(term) > 12 and len(term) > len(meaning))
        )

    async def _review_slang_batch(
        self, session_id: str, entries: list[dict], lines: list[str]
    ) -> int:
        """让 LLM 复核一批自动词条，证据不足时保留。"""
        provider = self.get_active_provider()
        if not provider or not entries or len(lines) < 10:
            return 0

        from modules.llm.base import ChatRequest

        req = ChatRequest(temperature=0.1, max_tokens=900, top_p=0.9)
        req.add_system(
            "你是群聊黑话词表审核员。请根据近期真实群聊记录，审核下面的自动提取词条。\n"
            "只有在以下情况明确成立时才删除：词条在记录中没有依据、含义与实际用法明显不符、"
            "它只是普通词/全网通用网络用语/游戏官方名词，或明显是模型臆造。\n"
            "证据不足时保留，不要因为近期没出现就删除；不要修改含义。人工词条不会出现在本批。\n"
            "只输出需要删除的行，格式严格为：删除 <ID> | 原因。没有需要删除的就输出：无。"
        )
        entry_text = "\n".join(
            f"ID={row['id']} | 词={row.get('term', '')} | 含义={row.get('meaning', '')}"
            f" | 例句={row.get('example') or '无'}"
            for row in entries
        )
        req.add_user(
            f"【会话】{session_id or '全部群'}\n"
            "【近期群聊记录】\n"
            + "\n".join(lines)
            + "\n【待审核自动词条】\n"
            + entry_text
        )

        resp = await provider.chat(req)
        ids = self._parse_slang_cleanup_ids(
            resp.content if resp else "", [row["id"] for row in entries]
        )
        deleted = await self.memory_storage.delete_auto_slang(ids)
        if deleted:
            logger.info(
                "[黑话清理] %s 删除 %d 条明显不符合语境的自动词条：%s",
                session_id or "全部群",
                deleted,
                "、".join(str(item) for item in ids[:8]),
            )
        return deleted

    async def cleanup_slang_all(self, hours: int = None) -> dict:
        """清理明显错误的自动黑话；人工词条永远不参与自动删除。"""
        cleanup_cfg = self._slang_config.get("cleanup", {}) or {}
        lookback_hours = max(
            24, int(hours if hours is not None else cleanup_cfg.get("lookback_hours", 168))
        )
        result = {
            "sessions": 0,
            "checked": 0,
            "deleted": 0,
            "obvious_deleted": 0,
            "skipped": 0,
            "error": "",
        }

        try:
            rows = await self.memory_storage.list_slang()
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("列出黑话供自动清理失败：%s", exc, exc_info=True)
            return result

        auto_rows = [
            row for row in rows
            if row.get("source") == "auto" and row.get("enabled", 1)
        ]
        obvious_ids = [
            row["id"] for row in auto_rows if self._is_obviously_invalid_slang(row)
        ]
        if obvious_ids:
            result["obvious_deleted"] = await self.memory_storage.delete_auto_slang(obvious_ids)
            result["deleted"] += result["obvious_deleted"]
            removed = set(obvious_ids)
            auto_rows = [row for row in auto_rows if row.get("id") not in removed]

        if not auto_rows:
            return result

        provider = self.get_active_provider()
        if not provider:
            result["error"] = "no active provider"
            return result

        from datetime import timedelta

        try:
            episodes = await self.memory_storage.get_all(
                limit=4000, memory_type="episodic"
            )
            since = datetime.now() - timedelta(hours=lookback_hours)
            lines_by_session: dict[str, list[str]] = {}
            all_lines: list[str] = []
            for memory in episodes:
                if not memory.source_session or memory.created_at <= since:
                    continue
                line = (memory.content or "").strip()
                if not line or line.startswith("["):
                    continue
                lines_by_session.setdefault(memory.source_session, []).append(line)
                if str(memory.source_session).startswith("group_"):
                    all_lines.append(line)
            for lines in lines_by_session.values():
                lines.reverse()
            all_lines.reverse()
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("读取黑话清理素材失败：%s", exc, exc_info=True)
            return result

        max_entries = max(1, int(cleanup_cfg.get("max_entries", 40)))
        reviewed_sessions = set()
        for session_id, session_rows in self._group_slang_rows(auto_rows).items():
            lines = lines_by_session.get(session_id, []) if session_id else all_lines
            if len(lines) < 10:
                result["skipped"] += len(session_rows)
                continue
            reviewed_sessions.add(session_id or "__all__")
            for start in range(0, len(session_rows), max_entries):
                batch = session_rows[start:start + max_entries]
                result["checked"] += len(batch)
                try:
                    result["deleted"] += await self._review_slang_batch(
                        session_id, batch, lines[:160]
                    )
                except Exception as exc:
                    logger.error(
                        "黑话自动审核失败（%s）：%s", session_id, exc, exc_info=True
                    )
                    if not result["error"]:
                        result["error"] = str(exc)
        result["sessions"] = len(reviewed_sessions)
        return result

    @staticmethod
    def _group_slang_rows(rows: list[dict]) -> dict[str, list[dict]]:
        """按词条所属会话分组，避免不同群的同名黑话互相影响。"""
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("session") or ""), []).append(row)
        return grouped

    async def _extract_slang(self, session_id: str, hours: int = 48) -> dict:
        """从最近的群聊记录里提取这个群特有的黑话，写入词表。"""
        from datetime import timedelta

        result = {"session": session_id, "found": 0, "saved": 0, "error": ""}
        provider = self.get_active_provider()
        if not provider:
            result["error"] = "no active provider"
            return result
        try:
            since = datetime.now() - timedelta(hours=hours)
            episodes = await self.memory_storage.get_all(
                limit=2000, memory_type="episodic"
            )
            lines = [
                (m.content or "").strip()
                for m in episodes
                if m.source_session == session_id and m.created_at > since
            ]
            # 富媒体占位/识别摘要不是群友说的话，不参与黑话提取
            lines = [l for l in lines if l and not l.startswith("[")][:120]
            if len(lines) < 10:
                result["error"] = f"素材不足（{len(lines)} 条）"
                return result

            existing = await self.memory_storage.list_slang(session=session_id)
            known = "、".join(sorted({row["term"] for row in existing}))[:400]

            from modules.llm.base import ChatRequest
            req = ChatRequest(temperature=0.2, max_tokens=800, top_p=0.9)
            req.add_system(
                "你是群聊黑话整理助手。从聊天记录里找出这个群特有的「黑话」——"
                "外人看了不懂、或者字面意思和实际意思不一样的词。\n"
                "算黑话的：群内绰号和外号（谁被叫什么）、这个群自造的梗和缩写、"
                "反复出现的固定说法、把普通词用成别的意思。\n"
                "不算黑话的：全网通用的网络用语（yyds、绝了、破防、摆烂这些）、"
                "游戏和动漫的官方名词（角色名、装备名、副本名）、普通中文词、"
                "只出现过一次且看不出含义的怪话。\n"
                "\n每行输出一条，格式为「词 | 含义 | 例句」：\n"
                "含义要写清楚它在这个群里实际指什么（20字以内）；"
                "例句从记录里摘一句原话（可省略）。\n"
                "只输出这些行，不要编号、不要解释、不要总结。"
                "找不到任何群特有的黑话就输出「无」。"
            )
            user_parts = [f"【群聊记录】\n" + "\n".join(lines)]
            if known:
                user_parts.append(
                    f"\n【已收录的黑话】\n{known}\n"
                    "这些已经收录了：含义如果和记录里的用法不符可以重新给出，否则不用重复输出。"
                )
            req.add_user("\n".join(user_parts))

            resp = await provider.chat(req)
            items = self._parse_slang_lines(resp.content if resp else "")
            result["found"] = len(items)
            for item in items:
                saved = await self.memory_storage.upsert_slang(
                    term=item["term"],
                    meaning=item["meaning"],
                    session=session_id,
                    example=item["example"],
                    source="auto",
                )
                if saved:
                    result["saved"] += 1
            if items:
                logger.info(
                    "[黑话] %s 提取 %d 条：%s",
                    session_id, len(items),
                    "、".join(i["term"] for i in items[:8]),
                )
        except Exception as exc:
            logger.error("黑话提取失败：%s", exc, exc_info=True)
            result["error"] = str(exc)
        return result

    async def extract_slang_all(self, hours: int = 48) -> dict:
        """对所有活跃会话跑一遍黑话提取（供定时任务和 Web 手动触发）。"""
        summary = {"sessions": 0, "found": 0, "saved": 0}
        sessions = set(self.context_manager._windows.keys())
        try:
            episodes = await self.memory_storage.get_all(limit=2000, memory_type="episodic")
            from datetime import timedelta
            since = datetime.now() - timedelta(hours=hours)
            sessions |= {
                m.source_session for m in episodes
                if m.source_session and m.created_at > since
            }
        except Exception as exc:
            logger.debug("列出活跃会话失败：%s", exc)
        for session_id in sorted(sessions):
            if not str(session_id).startswith("group_"):
                continue  # 黑话是群内共识，私聊没有这个概念
            result = await self._extract_slang(session_id, hours=hours)
            summary["sessions"] += 1
            summary["found"] += result["found"]
            summary["saved"] += result["saved"]
        return summary

    async def _maybe_extract_slang(self) -> None:
        """按配置间隔（默认每天）触发一次黑话提取。"""
        if (
            not getattr(self, "long_term_memory_enabled", True)
            or not self._slang_config.get("enabled", True)
        ):
            return
        interval_hours = max(1, int(self._slang_config.get("interval_hours", 24)))
        now = time.time()
        if self._last_slang_extract and (now - self._last_slang_extract) < interval_hours * 3600:
            return
        self._last_slang_extract = now
        await self.extract_slang_all(
            hours=int(self._slang_config.get("lookback_hours", 48))
        )

    async def _maybe_cleanup_slang(self) -> None:
        """按配置间隔审核并删除明显错误的自动黑话。"""
        if not getattr(self, "long_term_memory_enabled", True):
            return
        cleanup_cfg = self._slang_config.get("cleanup", {}) or {}
        if not cleanup_cfg.get("enabled", True):
            return
        interval_hours = max(1, int(cleanup_cfg.get("interval_hours", 24)))
        now = time.time()
        if self._last_slang_cleanup and (now - self._last_slang_cleanup) < interval_hours * 3600:
            return
        self._last_slang_cleanup = now
        try:
            result = await self.cleanup_slang_all(
                hours=int(cleanup_cfg.get("lookback_hours", 168))
            )
        except Exception as exc:
            logger.error("[黑话清理] 定时任务失败：%s", exc, exc_info=True)
            return
        if result["deleted"] or result["checked"]:
            logger.info(
                "[黑话清理] 检查 %d 个会话、%d 条自动词条，删除 %d 条（明显格式问题 %d 条）",
                result["sessions"],
                result["checked"],
                result["deleted"],
                result["obvious_deleted"],
            )
        if result["error"]:
            logger.warning("[黑话清理] %s", result["error"])

    async def _retrieve_memories(self, query: str, session_id: str, limit: int = None,
                                 exclude_id: str = "") -> list:
        """检索相关长期记忆：
        1. 该会话最近的群聊纪要（始终带上，bot 记得"最近群里聊过什么"，不会隔天失忆）
        2. 向量语义检索（TF-IDF + 余弦），失败或空时退回该会话最近记忆

        exclude_id：当前消息的 message_id。快路径会先异步把当前消息写入 episodic，
        检索发生在写入之后，若不排除，刚发的这条会把自己的内容召回——等于让 bot
        复述自己刚说的话。这里通过 memory_id 映射排除。
        """
        if not getattr(self, "long_term_memory_enabled", True):
            return []

        limit = limit or self.memory_search_top_k
        # 略微提高单次召回数，让"近况纪要"和"群友画像"有更多被覆盖到的概率
        if limit < 8:
            limit = 8

        memories: list = []
        try:
            memories = await self.memory_storage.semantic_search(
                query=query,
                session=session_id,
                limit=limit,
                half_life_days=self.memory_half_life_days,
                similarity_weight=self.memory_similarity_weight,
                decay_presets=self.memory_decay_presets,
            )
        except Exception as e:
            logger.error(f"语义检索失败：{e}")

        if not memories:
            # 兜底只取该会话最近的消息，不按重要性拿一条无关的旧画像/旧事实。
            try:
                memories = await self.memory_storage.retrieve_session_recent(
                    session_id, limit=limit, memory_type="episodic"
                )
            except Exception as e:
                logger.error(f"读取相关记忆失败：{e}")
                memories = []

        # 排除当前消息自己（若已落库且被召回）：通过 metadata.message_id 匹配
        if exclude_id:
            excluded = {m.id for m in memories
                        if str((m.metadata or {}).get("message_id") or "") == str(exclude_id)}
            if excluded:
                memories = [m for m in memories if m.id not in excluded]

        # 近况纪要：今天 + 昨天各 1 条，让 bot 跨日也记得群里聊过啥
        try:
            digests = await self.memory_storage.retrieve_session_recent(
                session_id, limit=2, memory_type="session_summary"
            )
        except Exception as e:
            logger.debug(f"读取群聊纪要失败，跳过：{e}")
            digests = []
        if digests:
            digest_ids = {d.id for d in digests}
            memories = (digests + [m for m in memories if m.id not in digest_ids])[:limit]

        # 群友画像补充：semantic_search 不一定召回到当前活跃的常驻群友，
        # 这里再单独拉本会话最近的画像（最多 3 条），保证至少有"谁是谁"的锚点
        try:
            profile_existing_ids = {m.id for m in memories}
            profiles = await self.memory_storage.retrieve_session_recent(
                session_id, limit=12, memory_type="semantic"
            )
            profile_picks: list = []
            seen_senders: set[str] = set()
            for mem in profiles:
                meta = mem.metadata or {}
                tags = meta.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]
                if "用户画像" not in tags:
                    continue
                sender_key = str(meta.get("sender_id") or "")
                if sender_key and sender_key in seen_senders:
                    continue
                if mem.id in profile_existing_ids:
                    continue
                profile_picks.append(mem)
                if sender_key:
                    seen_senders.add(sender_key)
                if len(profile_picks) >= 3:
                    break
            if profile_picks:
                memories = profile_picks + memories
        except Exception as e:
            logger.debug(f"读取用户画像失败，跳过：{e}")

        # 只刷新真正命中的画像访问时间；不提升 importance，也不把无关召回当成强化。
        profile_ids = [
            m.id for m in memories
            if m.id and (m.metadata or {}).get("profile")
        ]
        if profile_ids:
            try:
                await self.memory_storage.update_access_many(profile_ids)
            except Exception as e:
                logger.debug(f"更新用户画像访问记录失败，跳过：{e}")

        return memories

    async def _augment_context_with_group_reports(
        self, session_id: str, context_prompt: str
    ) -> str:
        """把最近两天的群日报里的"话题"和"群友标签"段拼进 context_prompt。

        bot 跨日不会自动记得群里聊过啥、人是谁，日报里已经有结构化结论，
        直接复用，避免每次都"群友 X 是谁"猜错。
        """
        try:
            reports = await self.memory_storage.get_group_analysis_reports(
                session=session_id, limit=2
            )
        except Exception as exc:
            logger.debug(f"读取群日报失败，跳过：{exc}")
            return context_prompt
        if not reports:
            return context_prompt

        # 取报告里有用的两段：话题 + 群友标签（忽略金句/逆天等闲聊段子）
        blocks: list[str] = ["\n[近期群日报摘要]（这些是昨天/前天群里聊过的主题和群友标签，"
                              "用得上就顺手提一下，不要主动复述）："]
        used = False
        for report in reports:
            meta = report.metadata or {}
            report_date = meta.get("report_date") or "?"
            content = report.content or ""
            profiles_segment = _extract_section(content, "我给几位群友留了个小标签", "我注意到的话题")
            topics_segment = _extract_section(content, "我注意到的话题", "我忍不住记下的几句")
            if not topics_segment and not profiles_segment:
                continue
            blocks.append(f"--- {report_date} ---")
            if topics_segment:
                used = True
                # 截短：每段最多 3 行，避免 prompt 过长
                short_topics = "\n".join(topics_segment.splitlines()[:3])
                blocks.append(f"话题：\n{short_topics}")
            if profiles_segment:
                used = True
                short_profiles = "\n".join(profiles_segment.splitlines()[:5])
                blocks.append(f"群友标签：\n{short_profiles}")
        if not used:
            return context_prompt
        return context_prompt + "\n" + "\n".join(blocks)

    async def run(self) -> None:
        """运行 Bot"""
        await self.initialize()

        self._running = True
        logger.info("=" * 50)
        logger.info(f"爱丽丝已开始运行，当前名称：{self.personality.name}")
        logger.info("=" * 50)

        # 清理任务必须在连接前启动：connect() 成功后会一直阻塞服务 WebSocket，
        # 放在它后面的代码在正常运行时永远执行不到。
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

        # 连接 QQ (可选 - 如果连接失败则继续运行用于测试)
        try:
            await self.qq_adapter.connect()
        except Exception as e:
            logger.warning(f"QQ 连接失败（可继续运行用于测试）：{e}")
            logger.warning("提示：请确保 NapCat QQ 机器人已启动")
            # 不停止 - 让bot以测试模式运行

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
            await self._maybe_distill_profiles()
            await self._maybe_extract_slang()
            await self._maybe_cleanup_slang()
            await self._maybe_group_analysis()
            await self._maybe_cleanup_group_analysis()
            logger.debug("已清理过期状态")

    async def _maybe_decay_memories(self) -> None:
        """定期对长期记忆应用时间衰减（每6小时一次，避免频繁写库）"""
        if not getattr(self, "long_term_memory_enabled", True):
            return
        import time
        now = time.time()
        # 距上次执行不足6小时则跳过；进程内首次运行时直接执行一次
        if self._last_decay_run and (now - self._last_decay_run) < 6 * 3600:
            return
        try:
            self._last_decay_run = now
            # 分级衰减：个人事实(semantic)活得最久、纪要(session_summary)次之、
            # 消息流水(episodic)最短——"该记住的"不随闲聊一起老去。
            result = await self.memory_storage.apply_time_decay(
                half_life_days=self.memory_half_life_days,
                min_importance=0.1,
                max_age_days=180,
                presets=self.memory_decay_presets,
            )
            if result["deleted"]:
                logger.info(f"[记忆衰减] 检查结果：{result}")
        except Exception as e:
            logger.error(f"记忆衰减失败：{e}")

    async def stop(self) -> None:
        """停止 Bot"""
        logger.info("正在停止爱丽丝...")
        self._running = False

        current_loop = asyncio.get_running_loop()

        async def finish_group(tasks: list[asyncio.Task], preserve_memory: bool = False) -> None:
            """在任务所属事件循环中取消/等待任务。"""
            pending = [task for task in tasks if not task.done()]
            if not pending:
                return
            if preserve_memory:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=5.0,
                    )
                    return
                except asyncio.TimeoutError:
                    # 超时后仍要把未完成的写入任务收掉，避免关闭 SQLite 时撞上后台写入。
                    pass
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        async def finish_tasks(tasks: list[asyncio.Task], preserve_memory: bool = False) -> None:
            """兼容普通模式和 Dashboard/QQ 双事件循环模式。"""
            unique = list(dict.fromkeys(task for task in tasks if task is not None))
            by_loop: dict[asyncio.AbstractEventLoop, list[asyncio.Task]] = {}
            for task in unique:
                try:
                    task_loop = task.get_loop()
                except RuntimeError:
                    continue
                by_loop.setdefault(task_loop, []).append(task)

            for task_loop, group in by_loop.items():
                if task_loop is current_loop:
                    await finish_group(group, preserve_memory=preserve_memory)
                    continue
                if task_loop.is_closed():
                    continue

                async def runner(group=group):
                    await finish_group(group, preserve_memory=preserve_memory)

                runner_coro = runner()
                try:
                    future = asyncio.run_coroutine_threadsafe(runner_coro, task_loop)
                except RuntimeError:
                    runner_coro.close()
                    continue
                try:
                    await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=6.0 if preserve_memory else 3.0,
                    )
                except (asyncio.TimeoutError, RuntimeError):
                    future.cancel()
                    # 事件循环仍在运行时，跨线程取消是安全的；若刚好已关闭则忽略。
                    for task in group:
                        if task.done():
                            continue
                        try:
                            task_loop.call_soon_threadsafe(task.cancel)
                        except RuntimeError:
                            pass

        # 先停掉清理/回复/纪要任务，防止它们在退出阶段继续创建记忆写入。
        background_tasks = (
            list(self._reply_tasks.values())
            + list(self._rich_media_tasks)
            + list(self._meme_collect_tasks)
            + list(self._digest_tasks.values())
            + list(self._group_analysis_tasks.values())
        )
        await finish_tasks(list(self._tasks) + background_tasks)

        # 记忆写入是退出前必须尽量保住的数据；先等待，超时才取消。
        memory_tasks = list(self._memory_tasks)
        await finish_tasks(memory_tasks, preserve_memory=True)
        self._tasks.clear()
        self._reply_tasks.clear()
        self._reply_task_decisions.clear()
        self._message_ingest_locks.clear()
        self._conversation_judge_locks.clear()
        self._session_message_versions.clear()
        self._rich_media_tasks.clear()
        self._meme_collect_tasks.clear()
        self._digest_tasks.clear()
        self._group_analysis_tasks.clear()
        self._memory_tasks.clear()

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

        # SQLite 连接也要显式关闭；Dashboard 与 QQ 共用它时尤其重要。
        try:
            if self.memory_storage:
                await self.memory_storage.close()
        except Exception:
            pass

        elapsed = time.time() - self._start_time
        logger.info(f"爱丽丝已停止，运行时长 {elapsed:.0f} 秒")

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
                "speaking_style": {
                    "banned_words": [],
                    "max_reply_length": 20,
                    "direct_max_reply_length": 80,
                    "use_ellipsis": True,
                    "ellipsis_frequency": 0.06,
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
                    "top_p": 0.9,
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
            "meme_manager": {
                "enabled": True,
                "storage_path": "data/memes",
                "auto_collect_enabled": False,
                "auto_send_enabled": False,
                "auto_send_cooldown_seconds": 60,
                "collect_private": False,
                "collect_scope": [],
                "collect_plain_images": False,
                "skip_screenshots": True,
                "min_collect_dimension": 64,
                "max_collect_dimension": 2400,
                "max_collect_pixels": 6000000,
                "default_category": "待整理",
                "max_image_bytes": 8388608,
                "max_images_per_message": 2,
                "daily_collect_limit": 80,
                "collect_cooldown_seconds": 15,
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
                "topic_shift_threshold": 0.12,
                "settle_window_seconds": 0.7,
                "settle_max_seconds": 2.4,
                "other_target_context_seconds": 900
            },
            "conversation_judge": {
                "enabled": True,
                "provider_id": "primary",
                "timeout": 8.0,
                "max_tokens": 220,
                "context_messages": 16,
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
                    "to_text_prompt": "用一两句话（50字以内）客观描述图片中能直接看到的内容：主体、动作或表情、画面文字、明显颜色和构图；不要推测人物关系、前因后果、情绪意图或适用场景。",
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
                "share_across_sessions": False,
                "db_path": "data/memory.db",
                "retrieval_top_k": 5,
                "half_life_days": 30,
                "similarity_weight": 0.85,
                "digest": {
                    "enabled": True,
                    "interval_messages": 20,
                    "min_messages": 10,
                    "max_tokens": 200
                },
                "group_analysis": {
                    "enabled": True,
                    "auto_enabled": False,
                    "auto_times": ["23:50"],
                    "min_messages": 10,
                    "max_messages": 500,
                    "max_prompt_chars": 24000,
                    "max_topics": 5,
                    "max_quotes": 3,
                    "max_titles": 5,
                    "max_tokens": 1800,
                    "max_report_chars": 6000,
                    "retention_days": 30,
                    "send_report": True,
                    "avatars_enabled": True,
                    "avatar_cache_days": 7,
                    "avatar_max_count": 12,
                    "avatar_timeout": 6
                },
                "slang": {
                    "enabled": True,
                    "interval_hours": 24,
                    "lookback_hours": 48,
                    "max_inject": 8,
                    "cleanup": {
                        "enabled": True,
                        "interval_hours": 24,
                        "lookback_hours": 168,
                        "max_entries": 40
                    }
                },
                "profile": {
                    "enabled": True,
                    "interval_minutes": 30,
                    "min_facts": 2,
                    "daily_min_messages": 8,
                    "daily_window_days": 15,
                    "refresh_daily_count": 5,
                    "mbti_refresh_daily_count": 12,
                    "mbti_enabled": True,
                    "context_enabled": True,
                    "context_neighbors": 2,
                    "context_max_targets": 12
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


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """从群日报渲染文本里截两 marker 之间的段落。任一 marker 缺失返回空串。"""
    if not text or not start_marker or not end_marker:
        return ""
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        return text[start:].strip()
    return text[start:end].strip()


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
        logger.info(
            f"管理面板已启动：http://{args.dashboard_host}:{args.dashboard_port}"
        )

        # 在线程中连接 QQ 适配器
        async def connect_qq():
            # 清理循环（记忆衰减/画像提炼/状态清理）必须与消息处理跑在同一个
            # 事件循环上（即这个 QQ 线程），否则会跨线程访问同一批状态。
            # 注意 connect() 成功后会一直阻塞，清理任务要在它之前创建。
            cleanup_task = asyncio.create_task(bot._cleanup_loop())
            bot._tasks.append(cleanup_task)
            try:
                await bot.qq_adapter.connect()
            except Exception as e:
                logger.warning(f"QQ 连接失败：{e}")
                # 连接失败也保持本线程的清理循环运行（测试模式）
                await asyncio.gather(cleanup_task, return_exceptions=True)

        qq_thread = threading.Thread(target=lambda: asyncio.run(connect_qq()), daemon=True)
        qq_thread.start()
        logger.info("QQ 适配器正在后台启动...")

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

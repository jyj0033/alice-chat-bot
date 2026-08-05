"""
Pipeline 调度器 - 多阶段消息处理
参考 AstrBot 的 Pipeline 架构
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Pipeline 处理上下文"""
    # 原始消息数据
    raw_message: str = ""
    sender_id: str = ""
    sender_name: str = ""
    group_id: str = ""
    session_id: str = ""
    is_private: bool = False

    # 解析后的数据
    mentioned_me: bool = False  # 是否@了Bot
    reply_to_me: bool = False   # 是否回复了Bot的消息
    message_type: str = "text"  # 消息类型

    # 决策数据
    should_speak: bool = False
    speak_reason: str = ""
    speaking_probability: float = 0.0

    # 生成数据
    context_prompt: str = ""    # 构建的上下文提示
    generated_reply: str = ""   # 生成的回复
    final_reply: str = ""       # 处理后的最终回复

    # 额外数据
    extra: dict = field(default_factory=dict)

    # 错误
    error: Optional[str] = None


class PipelineStage(ABC):
    """Pipeline 阶段基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext | bool:
        """
        处理上下文
        返回 PipelineContext 继续处理
        返回 True 继续处理
        返回 False 中断 Pipeline
        """
        raise NotImplementedError


class Pipeline:
    """消息处理 Pipeline"""

    def __init__(self, event_bus=None):
        self.stages: list[PipelineStage] = []
        self.event_bus = event_bus

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """添加阶段"""
        self.stages.append(stage)
        return self

    def add_stages(self, *stages: PipelineStage) -> "Pipeline":
        """批量添加阶段"""
        self.stages.extend(stages)
        return self

    async def execute(self, context: PipelineContext) -> bool:
        """
        执行 Pipeline
        返回是否成功完成
        """
        logger.debug(f"Pipeline starting with {len(self.stages)} stages")

        for i, stage in enumerate(self.stages):
            try:
                logger.debug(f"Executing stage: {stage.name}")
                result = await stage.process(context)

                # 处理不同返回类型
                if result is False:
                    logger.debug(f"Pipeline aborted at stage: {stage.name}")
                    return False

                # 更新上下文
                if isinstance(result, PipelineContext):
                    context = result

            except Exception as e:
                logger.error(f"Stage {stage.name} error: {e}", exc_info=True)
                context.error = str(e)
                return False

        logger.debug("Pipeline completed successfully")
        return True


# ============== 具体阶段实现 ==============

class ParseStage(PipelineStage):
    """消息解析阶段"""

    def __init__(self, bot_nickname: str = ""):
        super().__init__("Parse")
        self.bot_nickname = bot_nickname

    async def process(self, context: PipelineContext) -> PipelineContext:
        """解析消息"""
        message = context.raw_message

        # 检测是否@了Bot
        if self.bot_nickname and self.bot_nickname in message:
            context.mentioned_me = True

        # 检测是否回复了Bot（简单实现）
        if message.strip().startswith("//") or "[CQ:reply" in message:
            context.reply_to_me = True

        logger.debug(f"Parsed message: mentioned={context.mentioned_me}, reply={context.reply_to_me}")
        return context


class SkipStage(PipelineStage):
    """跳过阶段 - 用于测试或调试"""

    def __init__(self, skip: bool = False):
        super().__init__("Skip")
        self.skip = skip

    async def process(self, context: PipelineContext) -> bool:
        return not self.skip


# Pipeline 工厂函数
def create_default_pipeline() -> Pipeline:
    """创建默认 Pipeline"""
    return Pipeline()

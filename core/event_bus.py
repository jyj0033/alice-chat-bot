"""
事件总线 - 发布订阅模式
用于异步消息分发和组件解耦
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Awaitable, Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型"""
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"
    BOT_MENTIONED = "bot_mentioned"
    GROUP_MESSAGE = "group_message"
    PRIVATE_MESSAGE = "private_message"
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    ERROR = "error"


@dataclass
class Event:
    """事件对象"""
    type: EventType
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # 来源平台/群组
    session_id: str = ""  # 会话ID

    def __post_init__(self):
        if not self.source and "group_id" in self.data:
            self.source = f"group:{self.data['group_id']}"
        if not self.session_id and "session_id" in self.data:
            self.session_id = self.data["session_id"]


EventHandler = Callable[[Event], Awaitable[Any]]


class EventBus:
    """事件总线 - 发布订阅模式"""

    def __init__(self):
        self._subscribers: dict[EventType, list[EventHandler]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._pending_tasks: set[asyncio.Task] = set()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """订阅事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed handler to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """取消订阅"""
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type.value}")

    async def publish(self, event: Event) -> None:
        """发布事件到队列"""
        await self._queue.put(event)
        logger.debug(f"Published event: {event.type.value}")

    async def start(self) -> None:
        """启动事件循环"""
        self._running = True
        logger.info("EventBus started")

        while self._running:
            try:
                event = await self._queue.get()

                # 分发到对应处理器
                handlers = self._subscribers.get(event.type, [])
                for handler in handlers:
                    task = asyncio.create_task(self._safe_handle(handler, event))
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBus error: {e}", exc_info=True)

        logger.info("EventBus stopped")

    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        """安全执行处理器"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Event handler error for {event.type.value}: {e}", exc_info=True)

    def stop(self) -> None:
        """停止事件循环"""
        self._running = False
        logger.info("EventBus stopping...")

    @property
    def queue_size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running

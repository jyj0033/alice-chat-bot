"""
QQ 适配器 - 使用 NapCat/OneBot v11 协议
作为 WebSocket 服务端接收 NapCat 的连接
"""
import asyncio
import contextlib
import json
import logging
import uuid
import websockets
from typing import Any, Callable, Awaitable, Optional, Set

from .base import PlatformAdapter, Message
from .rich_content import (
    is_rich_only,
    parse_message_segments,
    primary_rich_type,
    render_outer_text,
    render_segments,
)
from .rich_media import RichMediaEnricher

logger = logging.getLogger(__name__)


class QQAdapter(PlatformAdapter):
    """QQ 平台适配器 - NapCat/OneBot v11 (WebSocket 服务端)"""

    def __init__(
        self,
        config: dict,
        on_message: Optional[Callable[[Message], Awaitable[None]]] = None,
        vision_provider: Any | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.on_message_callback = on_message
        self.self_id = config.get("self_id", "")
        self.access_token = config.get("access_token", "")

        # WebSocket 服务端配置
        self.ws_host = config.get("ws_host", "0.0.0.0")
        self.ws_port = config.get("ws_port", 3001)

        self._server = None
        self._running = False
        self._connected = False
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._pending_api: dict[str, asyncio.Future] = {}
        self._message_tasks: set[asyncio.Task] = set()

        # 消息 ID 计数器
        self._message_id = 0

        self.messages_sent = 0
        self.messages_received = 0
        # 用最近消息 ID 补全 reply 段缺失的发送者信息，帮助判断谁在回复谁。
        self._message_senders: dict[str, str] = {}
        self._message_sender_cache_size = 2000
        self.rich_media_enricher = RichMediaEnricher(
            config.get("rich_media", {}) or {},
            self.call_api,
            vision_provider=vision_provider,
        )

    async def connect(self) -> None:
        """启动 WebSocket 服务端接收 NapCat 连接"""
        self._running = True
        self._connected = False

        logger.info(f"Starting WebSocket server at {self.ws_host}:{self.ws_port}...")

        try:
            self._server = await websockets.serve(
                self._handle_client,
                self.ws_host,
                self.ws_port,
                ping_interval=30,
                ping_timeout=10,
            )
            self._connected = True
            logger.info(f"WebSocket server started successfully! Waiting for NapCat connection...")

            # 保持运行
            async with self._server:
                await asyncio.Future()

        except Exception as e:
            logger.error(f"Server error: {e}")
            self._connected = False

    async def _handle_client(self, websocket) -> None:
        """处理 NapCat 客户端连接"""
        logger.info(f"NapCat connected from {websocket.remote_address}")
        self._clients.add(websocket)

        try:
            async for message in websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("NapCat disconnected")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            self._clients.discard(websocket)

    async def _handle_message(self, raw_message: str) -> None:
        """处理接收到的消息"""
        try:
            data = json.loads(raw_message)
            post_type = data.get("post_type", "")
            action = data.get("action", "")
            echo = data.get("echo")

            # API 响应：唤醒发起 call_api 的协程。消息回调在独立任务中运行，
            # 接收循环可继续处理 echo，避免 get_forward_msg 等调用死锁。
            if echo is not None and ("status" in data or "retcode" in data):
                self._resolve_api_response(str(echo), data)
                return

            # API 调用
            if action:
                await self._handle_api_call(action, data.get("params", {}), data.get("echo"))
                return

            # 消息
            if post_type == "message":
                self.messages_received += 1
                message_type = data.get("message_type", "private")
                message = self._parse_message(data)
                logger.info(f"[{'群聊' if message_type == 'group' else '私聊'}] {message.sender_name}: {message.content[:50]}...")
                task = asyncio.create_task(self._dispatch_message(message))
                self._message_tasks.add(task)
                task.add_done_callback(self._message_tasks.discard)

            # 元事件
            elif post_type == "meta_event":
                meta_type = data.get("meta_event_type", "")
                if meta_type == "lifecycle":
                    logger.info("NapCat connected!")
                elif meta_type == "heartbeat":
                    logger.debug("Heartbeat received")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _dispatch_message(self, message: Message) -> None:
        """独立处理消息，让接收循环能继续接收 API 回执和后续群消息。"""
        try:
            if self.on_message_callback:
                await self.on_message_callback(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Message callback failed: %s", exc, exc_info=True)

    def _resolve_api_response(self, echo: str, response: dict) -> None:
        future = self._pending_api.get(echo)
        if not future or future.done():
            return
        status = str(response.get("status", "ok")).lower()
        retcode = int(response.get("retcode", 0) or 0)
        if status in ("ok", "async") and retcode == 0:
            future.set_result(response.get("data"))
        else:
            message = response.get("message") or response.get("wording") or "NapCat API 调用失败"
            future.set_exception(RuntimeError(f"{message} (retcode={retcode})"))

    async def call_api(
        self,
        action: str,
        params: dict,
        timeout: float | None = 10.0,
    ):
        """向 NapCat 调用 API 并等待匹配 echo 的响应。"""
        if not self._clients:
            raise ConnectionError("No NapCat connected")
        echo = f"alice-{uuid.uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self._pending_api[echo] = future
        try:
            await self._broadcast(json.dumps({
                "action": action,
                "params": params,
                "echo": echo,
            }, ensure_ascii=False))
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout=max(0.1, float(timeout)))
        finally:
            self._pending_api.pop(echo, None)

    async def enrich_message(
        self,
        message: Message,
        *,
        directed: bool = False,
        conversation_context: str = "",
        group_image_urls: list[dict] | None = None,
    ) -> Message:
        """按配置展开转发、链接标题和图片识别/意图解读。"""
        return await self.rich_media_enricher.enrich(
            message,
            directed=directed,
            conversation_context=conversation_context,
            group_image_urls=group_image_urls,
        )

    async def _handle_api_call(self, action: str, params: dict, echo: str = None) -> None:
        """处理 API 调用"""
        logger.debug(f"API call: {action}")

        response = {"status": "ok", "retcode": 0, "data": {}}
        if echo:
            response["echo"] = echo

        # 处理常用 API
        if action == "get_login_info":
            response["data"] = {"user_id": int(self.self_id) if self.self_id else 0, "nickname": "爱丽丝"}
        elif action == "get_group_list":
            response["data"] = []
        elif action == "get_group_member_info":
            response["data"] = {
                "group_id": params.get("group_id"),
                "user_id": params.get("user_id"),
                "nickname": "Unknown",
                "role": "member"
            }
        elif action == "send_group_msg":
            # 实际发送消息
            group_id = params.get("group_id")
            message = params.get("message", "")
            response["data"] = {"message_id": self._gen_message_id()}
            logger.info(f"Sending to group {group_id}: {message[:50]}...")
        elif action == "send_private_msg":
            user_id = params.get("user_id")
            message = params.get("message", "")
            response["data"] = {"message_id": self._gen_message_id()}
            logger.info(f"Sending to user {user_id}: {message[:50]}...")
        else:
            logger.debug(f"API call: {action}")

        # 发送到所有客户端
        await self._broadcast(json.dumps(response))

    def _gen_message_id(self) -> int:
        """生成消息 ID"""
        self._message_id += 1
        return self._message_id

    async def _broadcast(self, message: str) -> None:
        """广播消息到所有客户端"""
        if self._clients:
            await asyncio.gather(
                *[client.send(message) for client in self._clients],
                return_exceptions=True
            )

    def _parse_message(self, data: dict) -> Message:
        """解析消息"""
        raw_content = data.get("message", "")
        segments = parse_message_segments(raw_content)
        outer_message_id = str(data.get("message_id", "") or "")
        for segment in segments:
            if segment.type == "forward" and not (
                segment.data.get("id") or segment.file_id
            ):
                # 部分 NapCat 版本只给外层消息 ID，也可用于 get_forward_msg。
                segment.file_id = outer_message_id
        content = render_segments(segments) or "[无法识别的消息]"
        outer_text = render_outer_text(segments)

        # 解析被回复的对象（[CQ:reply] 消息段），两种格式都支持
        reply_to_id, reply_to_qq = None, None
        for segment in segments:
            if segment.type == "reply":
                reply_to_id = str(segment.data.get("id", "")) or None
                qq = segment.data.get("qq")
                reply_to_qq = str(qq) if qq is not None else None
                break

        if reply_to_id and not reply_to_qq:
            reply_to_qq = self._message_senders.get(str(reply_to_id))

        # 检查是否 @ 了 bot，并收集 @ 的其他 QQ 号（区分"对bot说"和"对别人说"）
        mentioned_me = False
        mentioned_others: list[str] = []
        for segment in segments:
            if segment.type != "at":
                continue
            qq = segment.data.get("qq")
            if str(qq) == str(self.self_id):
                mentioned_me = True
            elif qq is not None and str(qq) not in (str(self.self_id), "all"):
                mentioned_others.append(str(qq))

        # 回复了 bot 的消息，等同于 @（也算"提到我"）
        if not mentioned_me and reply_to_qq and str(reply_to_qq) == str(self.self_id):
            mentioned_me = True

        sender = data.get("sender", {}) or {}
        message = Message(
            message_id=str(data.get("message_id", "")),
            message_type=data.get("message_type", "private"),
            sender_id=str(data.get("user_id", "")),
            # 群名片比 QQ 昵称更符合群友实际看到的称呼。
            sender_name=(
                sender.get("card")
                or sender.get("nickname")
                or f"User_{data.get('user_id', '')}"
            ),
            group_id=str(data.get("group_id", "")) if data.get("message_type") == "group" else None,
            content=content,
            raw_content=(
                raw_content
                if isinstance(raw_content, str)
                else json.dumps(raw_content, ensure_ascii=False, separators=(",", ":"))
            ),
            mentioned_me=mentioned_me,
            mentioned_others=mentioned_others,
            reply_to_id=reply_to_id,
            reply_to_qq=reply_to_qq,
            segments=segments,
            outer_text=outer_text,
            rich_only=is_rich_only(segments),
            rich_type=primary_rich_type(segments),
        )

        if message.message_id:
            self._message_senders[message.message_id] = message.sender_id
            while len(self._message_senders) > self._message_sender_cache_size:
                self._message_senders.pop(next(iter(self._message_senders)))

        return message

    def _extract_text(self, content) -> str:
        """兼容旧调用：富媒体现在统一走结构化解析器。"""
        return render_segments(parse_message_segments(content))

    async def send_message(self, session_id: str, content: str, reply_to_id: str | None = None) -> bool:
        """发送消息（可选引用一条消息，供插话时指明回应对象）"""
        if not self._clients:
            logger.error("No NapCat connected")
            return False

        try:
            # OneBot v11 需要消息数组格式；带引用时在首段前加 reply 段。
            message_array = []
            if reply_to_id:
                message_array.append({"type": "reply", "data": {"id": reply_to_id}})
            message_array.append({"type": "text", "data": {"text": content}})

            if session_id.startswith("group_"):
                group_id = int(session_id.replace("group_", ""))
                message_data = {
                    "action": "send_group_msg",
                    "params": {"group_id": group_id, "message": message_array},
                }
            else:
                user_id = int(session_id.replace("private_", ""))
                message_data = {
                    "action": "send_private_msg",
                    "params": {"user_id": user_id, "message": message_array},
                }

            await self._broadcast(json.dumps(message_data))
            self.messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send: {e}")
            return False

    async def send_group_message(self, group_id: str, content: str, reply_to_id: str | None = None) -> bool:
        return await self.send_message(f"group_{group_id}", content, reply_to_id=reply_to_id)

    async def send_private_message(self, user_id: str, content: str) -> bool:
        return await self.send_message(f"private_{user_id}", content)

    async def disconnect(self) -> None:
        self._running = False
        self._connected = False
        if self._message_tasks:
            tasks = list(self._message_tasks)
            for task in tasks:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks)
            self._message_tasks.clear()
        for future in self._pending_api.values():
            if not future.done():
                future.set_exception(ConnectionError("NapCat disconnected"))
        self._pending_api.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and len(self._clients) > 0

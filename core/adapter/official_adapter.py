"""QQ 开放平台官方机器人适配器（API v2）。

和 NapCat/OneBot 通道的根本差别：

* **不登录任何 QQ 号**。只用 AppID + AppSecret 换 Access Token 直连开放平台，
  所以不存在「账号被风控掐断」这回事，掉线只剩网络/服务端两类原因。
* **用户标识不是 QQ 号**，而是 openid：群是 ``group_openid``，群成员是
  ``member_openid``，单聊是 ``user_openid``。上层看到的会话标识形如
  ``group_<group_openid>`` / ``private_<user_openid>``。

对外接口与 ``QQAdapter`` 保持一致，``main.py`` 只需按配置二选一，
决策、记忆、词条、日报这些上层业务一行都不用改。

文档：https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote

import aiohttp
import websockets

from .base import Message, PlatformAdapter
from .rate_limit import RateLimitedError, SendRateLimiter
from .rich_content import (
    is_rich_only,
    parse_message_segments,
    primary_rich_type,
    render_outer_text,
    render_segments,
)
from .rich_media import RichMediaEnricher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 官方平台常量（取值全部来自官方文档，勿凭记忆改）
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "https://api.bot.qq.com"
TOKEN_PATH = "/app/getAppAccessToken"
GATEWAY_PATH = "/gateway"

# 事件订阅位：GROUP_AND_C2C_EVENT = 1 << 25
INTENT_GROUP_AND_C2C = 1 << 25
INTENT_INTERACTION = 1 << 26
INTENT_PUBLIC_GUILD_MESSAGES = 1 << 30

# 网关 opcode
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_RESUME = 6
OP_RECONNECT = 7
OP_INVALID_SESSION = 9
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

GROUP_EVENTS = frozenset({"GROUP_AT_MESSAGE_CREATE", "GROUP_MESSAGE_CREATE"})
C2C_EVENTS = frozenset({"C2C_MESSAGE_CREATE"})

# 官方回「被限频 / 频率超限」的错误码（见官方文档的错误码表）：
#   1100100 安全打击:消息被限频
#   504000~504999 消息频率相关错误
#   40034100 / 40034105 / 40054013 主动消息不可用或被拒绝
#   1100304 @次数鉴权失败
RATE_LIMIT_ERR_CODES = frozenset(
    {1100100, 1100304, 40034100, 40034105, 40054013}
)


def is_rate_limit_code(err_code: Any) -> bool:
    """这个错误码是不是「被限频了」。"""
    try:
        code = int(err_code)
    except (TypeError, ValueError):
        return False
    if code in RATE_LIMIT_ERR_CODES:
        return True
    return 504000 <= code <= 504999


class OfficialAPIError(RuntimeError):
    """官方接口返回了错误。带上 err_code，方便上层判断是不是被限频。"""

    def __init__(self, message: str, err_code: Any = None):
        super().__init__(message)
        self.err_code = err_code

    @property
    def is_rate_limit(self) -> bool:
        return is_rate_limit_code(self.err_code)

# 被动回复窗口：群聊 5 分钟 5 次，单聊 60 分钟 4 次
PASSIVE_WINDOW_GROUP = 300.0
PASSIVE_WINDOW_C2C = 3600.0
PASSIVE_MAX_REPLY_GROUP = 5
PASSIVE_MAX_REPLY_C2C = 4

# 富媒体 file_type
FILE_TYPE_IMAGE = 1
FILE_TYPE_VIDEO = 2
FILE_TYPE_VOICE = 3
FILE_TYPE_FILE = 4

# 二进制直传上限 5MB，超过要走分片上传
DIRECT_UPLOAD_LIMIT = 5 * 1024 * 1024
# md5_10m：文件前 10002432 字节的 MD5
MD5_10M_SIZE = 10002432

# 官方群里 @ 某人写成 <@member_openid>，@全体成员写成 <@all>
AT_PATTERN = re.compile(r"<@([^>]+)>")


class OfficialQQAdapter(PlatformAdapter):
    """QQ 开放平台官方机器人适配器。"""

    # 官方开放平台没有「戳一戳」接口。上层据此跳过主动戳 —— 否则群里每条
    # 消息都会白掷一次骰子、白起一个后台任务、白打一条日志。
    supports_poke = False

    def __init__(
        self,
        config: dict,
        on_message: Optional[Callable[[Message], Awaitable[None]]] = None,
        on_notice: Optional[Callable[[dict], Awaitable[None]]] = None,
        vision_provider: Any | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.on_message_callback = on_message
        self.on_notice_callback = on_notice
        self.self_id = str(config.get("self_id") or "")
        self.rich_media_enricher = RichMediaEnricher(
            config.get("rich_media", {}) or {},
            self.call_api,
            vision_provider=vision_provider,
        )

        self._access_token = ""
        self._token_expires_at = 0.0
        self._http: aiohttp.ClientSession | None = None
        self._ws: Any = None
        self._gateway_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._stopped = asyncio.Event()

        self._connected = False
        self._gateway_session_id = ""
        self._latest_s: int | None = None
        self._last_seq: int | None = None

        # 心跳 ACK 看门狗。官方通道最阴险的掉线是「半开连接」：写得进去、
        # 收不到包。不主动断开就永远发现不了 —— 面板一直显示「已连接」，
        # 但平台已经不再投递事件，群里 @ 机器人只会得到「服务不可用」。
        # 官方网关每个心跳周期回一次 op 11 ACK，连续三个周期不回即判定已死。
        self._last_ack_at = 0.0
        self._ack_seen = False
        self._ack_watchdog_logged = False
        self._last_event_at = 0.0
        self._connected_at = 0.0
        self._resumed_connection = False

        self.messages_sent = 0
        self.messages_received = 0
        self.last_sent_message_id = ""

        # 官方通道没有 OneBot 的 echo 机制，但面板要读这个属性，保持形状一致。
        self._pending_api: dict[str, asyncio.Future] = {}
        # 出站消息 ID -> 发送者，用来判断「谁回复了谁」
        self._message_senders: dict[str, str] = {}

        # 同一条消息可能重复推送（官方明确说要用 msg_seq 去重）
        self._seen_message_ids: set[str] = set()
        self._seen_message_order = deque(maxlen=4000)
        # 已经打过招呼的群：首次收到某群消息时记一条 INFO，方便拿到 group_openid
        self._announced_groups: set[str] = set()

        # message_id ↔ REFIDX 双向索引：官方通道的引用回复只认 REFIDX
        self._ref_idx_by_msg_id: dict[str, str] = {}
        self._msg_id_by_ref_idx: dict[str, str] = {}
        # 同一 msg_id 的被动回复序号（群 5 次 / 单聊 4 次）
        self._msg_seq_used: dict[str, int] = {}
        # 最近收到的入站事件：msg_id -> (session_id, 到达时间)
        self._recent_events: dict[str, tuple[str, float]] = {}
        self._recent_event_order: deque = deque(maxlen=2000)
        # 机器人自己在各场景下的 id 集合（AppID / 全局 id / 群内 member_openid）
        self._bot_ids: set[str] = set()
        # 已经提醒过的、官方通道不支持的 OneBot 动作
        self._warned_actions: set[str] = set()
        # 戳一戳不支持，只提醒一次，别刷屏
        self._poke_warned = False

        # 群成员 openid -> 用户名。官方的成员接口是白名单能力（11253），
        # 但消息事件里本来就带 author.username —— 先从收到的消息里攒一份，
        # 查不到成员资料时用它兜底，省得为个昵称去申请白名单。
        self._member_names: dict[str, str] = {}
        self._member_name_order: deque = deque(maxlen=4000)

        # 发送端频控：官方对主动消息有 qpm 和每日条数限制，触顶直接拒收，
        # 拒了那条消息就没了。阈值和说明见 core/adapter/rate_limit.py。
        official_cfg = self._official_config()
        self.rate_limiter = SendRateLimiter(
            bot_per_minute=int(official_cfg.get("rate_limit_bot_per_minute") or 25),
            session_per_minute=int(official_cfg.get("rate_limit_session_per_minute") or 15),
            daily_per_session=int(official_cfg.get("rate_limit_daily_per_session") or 900),
            wait_seconds=float(official_cfg.get("rate_limit_wait_seconds") or 5.0),
        )

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------

    def _official_config(self) -> dict:
        raw = self.config.get("official") if isinstance(self.config, dict) else None
        return raw if isinstance(raw, dict) else {}

    @property
    def _app_id(self) -> str:
        return str(self._official_config().get("app_id") or "").strip()

    @property
    def _app_secret(self) -> str:
        return str(self._official_config().get("app_secret") or "").strip()

    @property
    def _api_base(self) -> str:
        base = str(self._official_config().get("api_base") or DEFAULT_API_BASE).strip()
        return base.rstrip("/") or DEFAULT_API_BASE

    @property
    def _intents(self) -> int:
        raw = self._official_config().get("intents")
        if raw is None:
            return INTENT_GROUP_AND_C2C
        try:
            return int(raw)
        except (TypeError, ValueError):
            return INTENT_GROUP_AND_C2C

    def _canonical_bot_id(self) -> str:
        """机器人对外的规范 id。

        上层（决策、上下文重建）拿 ``qq.self_id`` 和消息里的 id 比对，官方
        通道下这个 id 只能是 AppID —— 群成员 id（member_openid）和它不是一个
        命名空间，混用会导致「回复了机器人」永远判定不出来。
        """
        return str(self.self_id or self._app_id or "").strip()

    def _bot_id_candidates(self) -> set[str]:
        ids = {value for value in self._bot_ids if value}
        canonical = self._canonical_bot_id()
        if canonical:
            ids.add(canonical)
        return ids

    @property
    def access_token(self) -> str:
        return self._access_token

    @access_token.setter
    def access_token(self, value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            # main.py 热更新配置时会无条件把 access_token 写回来。官方通道的
            # Access Token 由适配器自己刷新，空值不能把已获取的凭证清掉。
            return
        self._access_token = text
        self._token_expires_at = time.time() + 7200

    # ------------------------------------------------------------------
    # HTTP / 鉴权
    # ------------------------------------------------------------------

    async def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    async def _ensure_access_token(self, force: bool = False) -> str:
        """取 Access Token；有效期 7200 秒，提前 60 秒自动续期。"""
        if not force and self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        app_id = self._app_id
        app_secret = self._app_secret
        if not app_id or not app_secret:
            raise RuntimeError(
                "官方通道缺少凭证：请在 config/config.yaml 的 qq.official 下配置 "
                "app_id 与 app_secret（app_secret 会自动落到 secrets.yaml）"
            )
        data = await self._request_json(
            "POST",
            TOKEN_PATH,
            json_body={"appId": app_id, "clientSecret": app_secret},
            auth=False,
            timeout=20.0,
        )
        token = str(data.get("access_token") or "").strip()
        if not token:
            code = data.get("code")
            message = data.get("message")
            raise RuntimeError(f"获取 access_token 失败：{code} {message}")
        try:
            expires_in = int(data.get("expires_in") or 7200)
        except (TypeError, ValueError):
            expires_in = 7200
        self._access_token = token
        self._token_expires_at = time.time() + expires_in
        logger.info("已获取官方 access_token（有效期 %s 秒）", expires_in)
        return token

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict | None = None,
        auth: bool = True,
        timeout: float = 20.0,
        retry_auth: bool = True,
    ) -> dict:
        """调用官方 OpenAPI。

        官方接口的错误体是 ``{"err_code": ..., "message": ...}``，HTTP 状态码
        可能是 200；反过来取 access_token 的接口用 ``code`` 报业务错误。
        两种都在这里统一转成异常。
        """
        session = await self._ensure_http()
        url = path if path.startswith("http") else f"{self._api_base}{path}"
        headers: dict[str, str] = {}
        if auth:
            token = await self._ensure_access_token()
            headers["Authorization"] = f"QQBot {token}"
        if json_body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        try:
            async with session.request(
                method,
                url,
                json=json_body,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=max(1.0, float(timeout))),
            ) as response:
                status = response.status
                text = await response.text()
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"官方接口超时：{method} {path}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"官方接口请求失败：{method} {path}：{exc}") from exc

        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {"data": data}

        if status == 401 and auth and retry_auth:
            # 凭证过期：强制换一次再重试。
            logger.info("官方接口返回 401，刷新 access_token 后重试一次")
            await self._ensure_access_token(force=True)
            return await self._request_json(
                method,
                path,
                json_body=json_body,
                params=params,
                auth=auth,
                timeout=timeout,
                retry_auth=False,
            )
        if status >= 400:
            err = data.get("err_code") or data.get("code") or status
            raise OfficialAPIError(
                f"官方接口 {method} {path} 失败：{err} {str(data.get('message') or '')}".strip(),
                err,
            )
        err = data.get("err_code")
        if err not in (None, 0, "0"):
            raise OfficialAPIError(
                f"官方接口 {method} {path} 失败：{err} {str(data.get('message') or '')}".strip(),
                err,
            )
        return data

    async def _put_bytes(self, url: str, blob: bytes, timeout: float = 120.0) -> None:
        """把分片 PUT 到预签名地址（预签名 URL 自带鉴权，不加 QQBot 头）。"""
        session = await self._ensure_http()
        try:
            async with session.put(
                url,
                data=blob,
                timeout=aiohttp.ClientTimeout(total=max(1.0, float(timeout))),
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise RuntimeError(f"分片上传失败：{response.status} {text[:200]}")
        except asyncio.TimeoutError as exc:
            raise RuntimeError("分片上传超时") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"分片上传请求失败：{exc}") from exc

    # ------------------------------------------------------------------
    # 网关（WebSocket）
    # ------------------------------------------------------------------

    async def _fetch_gateway_url(self) -> str:
        data = await self._request_json("GET", GATEWAY_PATH, timeout=15.0)
        url = str(data.get("url") or "").strip()
        if not url:
            raise RuntimeError("获取网关地址失败：响应里没有 url")
        return url

    async def connect(self) -> None:
        """连接官方网关。

        契约与 NapCat 适配器一致：连接成功后一直阻塞，直到 disconnect()。
        """
        self._owner_loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        self._running = True
        self._connected = False

        await self._ensure_access_token()
        self._gateway_task = asyncio.create_task(self._gateway_loop())
        logger.info("官方通道已启动，正在连接网关……")
        try:
            await self._stopped.wait()
        finally:
            self._running = False

    async def _gateway_loop(self) -> None:
        delay = 3.0
        while self._running:
            try:
                url = await self._fetch_gateway_url()
                logger.info("官方网关地址：%s", url)
                await self._run_gateway(url)
                delay = 3.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("官方网关异常：%s", exc)
            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)

    async def _run_gateway(self, url: str) -> None:
        token = await self._ensure_access_token()
        async with websockets.connect(
            url, ping_interval=None, max_size=16 * 1024 * 1024
        ) as ws:
            self._ws = ws
            raw = await ws.recv()
            hello = self._decode(raw)
            interval_ms = 45000
            if hello.get("op") == OP_HELLO:
                interval_ms = int((hello.get("d") or {}).get("heartbeat_interval") or 45000)
            # 恢复旧会话还是全新握手，决定了下面要不要做「假恢复」检查。
            resuming = bool(self._gateway_session_id and self._last_seq is not None)
            await self._identify_or_resume(ws, token)
            self._resumed_connection = resuming
            self._connected_at = time.time()
            self._last_event_at = time.time()
            # 宽限期：刚连上先按「活着」算，等第一个 ACK 回来再收紧判定。
            self._last_ack_at = time.time()
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(ws, interval_ms / 1000.0)
            )
            self._connected = True
            logger.info("官方网关已连接（心跳周期 %.1f 秒）", interval_ms / 1000.0)
            try:
                async for raw in ws:
                    await self._handle_ws_payload(raw)
            finally:
                self._connected = False
                self._ws = None
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._heartbeat_task = None

    async def _identify_or_resume(self, ws: Any, token: str) -> None:
        auth = f"QQBot {token}"
        if self._gateway_session_id and self._last_seq is not None:
            payload = {
                "op": OP_RESUME,
                "d": {
                    "token": auth,
                    "session_id": self._gateway_session_id,
                    "seq": self._last_seq,
                },
            }
            logger.info("尝试恢复官方会话（seq=%s）", self._last_seq)
        else:
            payload = {
                "op": OP_IDENTIFY,
                "d": {
                    "token": auth,
                    "intents": self._intents,
                    "shard": [0, 1],
                    "properties": {},
                },
            }
        await ws.send(json.dumps(payload, ensure_ascii=False))

    async def _heartbeat_loop(self, ws: Any, interval: float) -> None:
        try:
            while True:
                await asyncio.sleep(max(5.0, interval))
                await ws.send(json.dumps({"op": OP_HEARTBEAT, "d": self._latest_s}))
                if self._gateway_stale(interval):
                    logger.warning(
                        "官方网关超过 %.0f 秒没有任何回包，判定连接已失效，主动断开重连",
                        interval * 3,
                    )
                    # 关掉连接，让读循环（async for）立刻结束并走重连流程。
                    # 只抛异常没用 —— 读循环正阻塞在 recv 上，不会自己醒。
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return
                if self._resumed_session_is_dead(interval):
                    logger.warning(
                        "恢复的官方会话长时间收不到事件，放弃旧 session_id，改用全新 Identify"
                    )
                    self._gateway_session_id = ""
                    self._last_seq = None
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("心跳发送失败：%s", exc)

    def _gateway_stale(self, interval: float) -> bool:
        """连接是不是已经半开（写得出、收不回）。

        官方网关每个心跳周期回一次 op 11 ACK。连续三个周期都没回，说明这条
        TCP 已经死了（写得进去但收不到包）—— 这种半开连接最阴险：不主动断开
        的话，面板一直显示「已连接」，但平台不再投递任何事件，群里 @ 机器人
        只会得到「服务不可用」。

        从来没收到过 ACK 就不判死：万一台上不实现 ACK，误杀一条健康连接
        比漏判一次更糟。这种情况下只提示一次，然后交给平台的 op 7 兜底。
        """
        if not self._ack_seen:
            if (
                self._last_ack_at
                and not self._ack_watchdog_logged
                and time.time() - self._last_ack_at > interval * 6
            ):
                self._ack_watchdog_logged = True
                logger.info("官方网关一直不回心跳 ACK，ACK 看门狗未启用")
            return False
        return (time.time() - self._last_ack_at) > interval * 3 + 5

    def _resumed_session_is_dead(self, interval: float) -> bool:
        """恢复的旧会话是不是其实已经废了。

        平台接受 Resume 并不代表它还会往这条连接上投递事件。之前就出现过：
        Resume 成功、ACK 也正常，但事件始终不来，直到重新 Identify 才恢复。
        所以「恢复后一段时间内一个事件都没有」就当作会话已废，重新握手。
        """
        if not self._resumed_connection:
            return False
        if time.time() - self._connected_at < 90:
            return False
        return self._last_event_at <= self._connected_at

    @staticmethod
    def _decode(raw: Any) -> dict:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    async def _handle_ws_payload(self, raw: Any) -> None:
        data = self._decode(raw)
        if not data:
            return
        op = data.get("op")
        seq = data.get("s")
        if isinstance(seq, int):
            self._latest_s = seq
        if op == OP_DISPATCH:
            event_type = str(data.get("t") or "")
            payload = data.get("d") or {}
            try:
                await self._handle_event(event_type, payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("处理官方事件 %s 失败：%s", event_type, exc, exc_info=True)
            # 官方建议处理完事件后再记录 s，重连时用它补发。
            if isinstance(seq, int):
                self._last_seq = seq
            return
        if op == OP_HEARTBEAT_ACK:
            # 收到 ACK 才说明这条连接真的还是活的 —— 看门狗全靠它判活。
            self._last_ack_at = time.time()
            self._ack_seen = True
            return
        if op == OP_RECONNECT:
            raise ConnectionError("官方网关要求重连")
        if op == OP_INVALID_SESSION:
            self._gateway_session_id = ""
            self._last_seq = None
            raise ConnectionError("官方会话失效，需要重新 identify")
        if op == OP_HELLO:
            return

    # ------------------------------------------------------------------
    # 事件解析
    # ------------------------------------------------------------------

    async def _handle_event(self, event_type: str, data: dict) -> None:
        if event_type == "READY":
            self._gateway_session_id = str(data.get("session_id") or "")
            user = data.get("user") or {}
            user_id = str(user.get("id") or "")
            if user_id:
                self._bot_ids.add(user_id)
            logger.info(
                "官方通道就绪：机器人=%s（session=%s）",
                user.get("username") or user_id or "-",
                self._gateway_session_id or "-",
            )
            return
        if event_type == "RESUMED":
            logger.info("官方会话已恢复")
            return
        self._last_event_at = time.time()
        if event_type == "GROUP_ADD_ROBOT":
            gid = str(data.get("group_openid") or "")
            self._record_seen_group(gid, "add")
            logger.info(
                "机器人被添加到新群：group_openid=%s —— 可在面板「发言设置 → 群组设置」"
                "给这个群单独配人格",
                gid or "-",
            )
            return
        if event_type == "GROUP_DEL_ROBOT":
            gid = str(data.get("group_openid") or "")
            self._record_seen_group(gid, "del")
            logger.info("机器人被移出群：group_openid=%s", gid or "-")
            return
        if event_type in GROUP_EVENTS:
            gid = str(data.get("group_openid") or "")
            if gid and gid not in self._announced_groups:
                self._announced_groups.add(gid)
                self._record_seen_group(gid, "message")
                logger.info(
                    "首次收到群消息：group_openid=%s（会话标识 group_%s）", gid, gid
                )
            if self._is_poke_scene(data):
                await self._emit_poke_notice(data, str(data.get("group_openid") or ""))
                return
            message = self._parse_group_message(data, event_type)
        elif event_type in C2C_EVENTS:
            if self._is_poke_scene(data):
                await self._emit_poke_notice(data, "")
                return
            message = self._parse_c2c_message(data)
        else:
            if event_type.startswith("GROUP_"):
                # 群成员变动、消息审核这些暂时没有业务，但打出来能拿到 group_openid
                logger.info("收到未处理的群事件：%s", event_type)
            else:
                logger.debug("忽略官方事件：%s", event_type)
            return
        if message is None:
            return
        if not self._remember_message_id(message.message_id):
            logger.debug("重复的官方消息，已忽略：%s", message.message_id)
            return
        self.messages_received += 1
        if self.on_message_callback:
            try:
                await self.on_message_callback(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("消息回调执行失败：%s", exc, exc_info=True)

    def _record_seen_group(self, group_openid: str, action: str) -> None:
        """记录「机器人被拉进/移出群」的 group_openid。

        开放平台没有「查询机器人所在群列表」的接口，群 openid 只能从事件里
        拿。落到 data/groups_seen.json，管理面板在没人发言时也能看到新群。
        """
        gid = str(group_openid or "").strip()
        if not gid:
            return
        path = Path("data") / "groups_seen.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            items: dict[str, Any] = {}
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8") or "{}")
                if isinstance(loaded, dict):
                    items = loaded
            items[gid] = {
                "action": action,
                "at": datetime.now().isoformat(timespec="seconds"),
            }
            path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("记录群 %s 失败：%s", gid, exc)

    def _remember_message_id(self, message_id: str) -> bool:
        message_id = str(message_id or "").strip()
        if not message_id:
            return True
        if message_id in self._seen_message_ids:
            return False
        if len(self._seen_message_order) >= self._seen_message_order.maxlen:
            oldest = self._seen_message_order.popleft()
            self._seen_message_ids.discard(oldest)
        self._seen_message_order.append(message_id)
        self._seen_message_ids.add(message_id)
        return True

    @staticmethod
    def _parse_ext(data: dict) -> dict[str, str]:
        """把 message_scene.ext 解析成字典。

        官方给的是 ``["msg_idx=REFIDX_xxx==", "auth_token=..."]`` 这种
        key=value 列表。
        """
        scene = data.get("message_scene") or {}
        ext: dict[str, str] = {}
        for item in scene.get("ext") or []:
            key, sep, value = str(item).partition("=")
            if sep:
                ext[key.strip()] = value.strip()
        return ext

    @staticmethod
    def _parse_timestamp(value: Any) -> float | None:
        """RFC3339 → Unix 秒。"""
        text = str(value or "").strip()
        if not text:
            return None
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()

    def _extract_at(self, content: str) -> tuple[list[dict], list[str]]:
        """把 ``<@xxx>`` 拆成 OneBot 风格的 at 段。"""
        segments: list[dict] = []
        ids: list[str] = []
        cursor = 0
        for match in AT_PATTERN.finditer(content):
            if match.start() > cursor:
                segments.append({"type": "text", "data": {"text": content[cursor:match.start()]}})
            at_id = match.group(1).strip()
            segments.append({"type": "at", "data": {"qq": at_id}})
            ids.append(at_id)
            cursor = match.end()
        if cursor < len(content):
            segments.append({"type": "text", "data": {"text": content[cursor:]}})
        return segments, ids

    def _attachment_segments(self, attachments: list) -> list[dict]:
        segments: list[dict] = []
        for item in attachments or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            content_type = str(item.get("content_type") or "").lower()
            filename = str(item.get("filename") or "")
            if content_type.startswith("image/"):
                segments.append(
                    {"type": "image", "data": {"url": url, "file": filename, "file_id": filename}}
                )
            elif content_type.startswith("video/"):
                segments.append(
                    {"type": "video", "data": {"url": url, "file": filename, "name": filename}}
                )
            elif content_type == "voice" or content_type.startswith("audio/"):
                voice_url = str(item.get("voice_wav_url") or url)
                segments.append(
                    {"type": "record", "data": {"url": voice_url, "file": filename, "name": filename}}
                )
            elif url or filename:
                segments.append(
                    {"type": "file", "data": {"url": url, "file": filename, "name": filename}}
                )
        return segments

    def _build_common_message(
        self,
        data: dict,
        *,
        message_type: str,
        group_id: str | None,
        sender_id: str,
        sender_name: str,
    ) -> Message:
        message_id = str(data.get("id") or "")
        content = str(data.get("content") or "")
        at_segments, at_ids = self._extract_at(content)
        segments: list[dict] = list(at_segments)

        mentions = [m for m in (data.get("mentions") or []) if isinstance(m, dict)]
        ark_data = data.get("ark_data") or {}
        msg_elements = data.get("msg_elements") or []

        # 卡片（message_type=3）：走 rich_content 的卡片解析，复用现成逻辑
        if isinstance(ark_data, dict) and ark_data:
            segments.append({"type": "json", "data": {"data": ark_data}})

        segments.extend(self._attachment_segments(data.get("attachments") or []))

        # 引用消息（message_type=103）：被引用的正文放在 msg_elements 里
        quoted_author_id = ""
        quoted_ref_idx = ""
        quoted_content = ""
        if isinstance(msg_elements, list) and msg_elements:
            first = msg_elements[0]
            if isinstance(first, dict):
                quoted_ref_idx = str(first.get("msg_idx") or "")
                quoted_content = str(first.get("content") or "")
                author = first.get("author") or {}
                if isinstance(author, dict):
                    quoted_author_id = str(
                        author.get("member_openid") or author.get("id") or ""
                    )
                    if author.get("bot"):
                        # 引用的是机器人自己的消息 → 用规范 bot id，上层才能
                        # 判断出「这条消息是在回复机器人」。
                        quoted_author_id = self._canonical_bot_id()
                        if author.get("id"):
                            self._bot_ids.add(str(author["id"]))
        if quoted_content:
            segments.append({"type": "text", "data": {"text": quoted_content}})

        ext = self._parse_ext(data)
        own_ref_idx = str(ext.get("msg_idx") or "")
        if own_ref_idx:
            self._ref_idx_by_msg_id[message_id] = own_ref_idx
            self._msg_id_by_ref_idx[own_ref_idx] = message_id
        if not quoted_ref_idx:
            quoted_ref_idx = str(ext.get("ref_msg_idx") or "")

        # @ 判定
        bot_candidates = self._bot_id_candidates()
        mentioned_me = False
        mentioned_user_ids: list[str] = []
        mentioned_others: list[str] = []
        for mention in mentions:
            uid = str(mention.get("member_openid") or mention.get("id") or "")
            if not uid:
                continue
            if mention.get("bot"):
                mentioned_me = True
                self._bot_ids.add(uid)
                canonical = self._canonical_bot_id()
                if canonical:
                    mentioned_user_ids.append(canonical)
                continue
            mentioned_user_ids.append(uid)
            mentioned_others.append(uid)
        if not mentioned_me and at_ids:
            if any(at_id in bot_candidates for at_id in at_ids):
                mentioned_me = True
                canonical = self._canonical_bot_id()
                if canonical and canonical not in mentioned_user_ids:
                    mentioned_user_ids.insert(0, canonical)
        if mentioned_me:
            canonical = self._canonical_bot_id()
            if canonical and canonical not in mentioned_user_ids:
                mentioned_user_ids.insert(0, canonical)
        # 内容里 @ 到的其它人也算「@ 了别人」
        for at_id in at_ids:
            if at_id == "all" or at_id in bot_candidates:
                continue
            if at_id not in mentioned_others:
                mentioned_others.append(at_id)
                if at_id not in mentioned_user_ids:
                    mentioned_user_ids.append(at_id)

        # 引用目标：优先用 REFIDX 找回被引用消息的 message_id
        reply_to_id = ""
        if quoted_ref_idx:
            reply_to_id = self._msg_id_by_ref_idx.get(quoted_ref_idx) or quoted_ref_idx
        reply_to_qq = quoted_author_id or None

        parsed = parse_message_segments(segments)
        rendered = render_segments(parsed)
        return Message(
            message_id=message_id,
            message_type=message_type,
            sender_id=sender_id,
            sender_name=sender_name or f"User_{sender_id}",
            group_id=group_id,
            content=rendered or "[无法识别的消息]",
            raw_content=content,
            mentioned_me=mentioned_me,
            mentioned_user_ids=list(dict.fromkeys(mentioned_user_ids)),
            mentioned_others=list(dict.fromkeys(mentioned_others)),
            reply_to_id=reply_to_id or None,
            reply_to_qq=reply_to_qq,
            segments=parsed,
            outer_text=render_outer_text(parsed),
            rich_only=is_rich_only(parsed),
            rich_type=primary_rich_type(parsed),
            timestamp=self._parse_timestamp(data.get("timestamp")),
        )

    # ------------------------------------------------------------------
    # 戳一戳（官方只有事件，没有接口）
    # ------------------------------------------------------------------

    @staticmethod
    def _is_poke_scene(data: dict) -> bool:
        """这条事件是不是「戳一戳」。

        官方没有戳一戳接口，但**会**把戳一戳当成一条带场景的消息推过来
        （``message_scene.scene`` 写着「戳一戳」）。识别出来至少能用普通
        文字回应一句，别让人以为机器人死了。
        """
        scene = data.get("message_scene") or {}
        if not isinstance(scene, dict):
            return False
        return str(scene.get("scene") or "").strip() in ("戳一戳", "poke", "POKE")

    async def _emit_poke_notice(self, data: dict, group_openid: str) -> None:
        """把官方的戳一戳事件转成 OneBot 风格的 notice，喂给戳一戳响应器。"""
        if not self.on_notice_callback:
            return
        author = data.get("author") or {}
        member_openid = ""
        if isinstance(author, dict):
            member_openid = str(author.get("member_openid") or author.get("id") or "").strip()
        notice = {
            "notice_type": "notify",
            "sub_type": "poke",
            "group_id": group_openid,
            "user_id": member_openid,
            "sender_id": member_openid,
            # 被戳的人就是机器人自己。官方通道下「机器人自己」在群里是
            # member_openid，和 AppID 不是一个命名空间；这里统一填 self_id
            # （AppID），和戳一戳响应器的判断口径保持一致。
            "target_id": self.self_id,
        }
        try:
            await self.on_notice_callback(notice)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("处理官方戳一戳事件失败：%s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 群成员显示名
    # ------------------------------------------------------------------

    def _remember_member_name(self, member_id: str, name: str) -> None:
        """记下 openid -> 用户名。

        官方的群成员接口是白名单能力（11253），但消息事件里本来就带
        ``author.username``，先从这儿攒一份，查不到成员资料时用它兜底。
        """
        member_id = str(member_id or "").strip()
        name = str(name or "").strip()
        if not member_id or not name:
            return
        if self._member_names.get(member_id) == name:
            return
        self._member_names[member_id] = name
        self._member_name_order.append(member_id)
        while len(self._member_name_order) > len(self._member_names):
            oldest = self._member_name_order.popleft()
            self._member_names.pop(oldest, None)

    def known_members(self) -> list[dict[str, str]]:
        """已知的群成员（openid + 用户名）。

        官方拿不到 QQ 号，面板要 @ 某人时只能用 openid，这个列表就是
        「openid -> 显示名」的对照表。
        """
        return [
            {"id": member_id, "name": name}
            for member_id, name in sorted(self._member_names.items(), key=lambda kv: kv[1])
        ]

    def _parse_group_message(self, data: dict, event_type: str) -> Message | None:
        group_openid = str(data.get("group_openid") or "").strip()
        if not group_openid:
            logger.warning("群消息缺少 group_openid，已丢弃")
            return None
        author = data.get("author") or {}
        sender_id = str(
            (author.get("member_openid") or author.get("id") or "") if isinstance(author, dict) else ""
        )
        sender_name = str(author.get("username") or "") if isinstance(author, dict) else ""
        self._remember_member_name(sender_id, sender_name)
        message = self._build_common_message(
            data,
            message_type="group",
            group_id=group_openid,
            sender_id=sender_id,
            sender_name=sender_name,
        )
        if event_type == "GROUP_AT_MESSAGE_CREATE":
            # 群里 @ 机器人：content 里已经去掉了 @ 前缀，但这条一定是对着
            # 机器人说的，必须标成 mentioned_me。
            message.mentioned_me = True
            canonical = self._canonical_bot_id()
            if canonical and canonical not in message.mentioned_user_ids:
                message.mentioned_user_ids.insert(0, canonical)
        self._remember_recent_event(message)
        return message

    def _parse_c2c_message(self, data: dict) -> Message | None:
        author = data.get("author") or {}
        sender_id = str(
            (author.get("user_openid") or author.get("id") or "") if isinstance(author, dict) else ""
        )
        sender_name = str(author.get("username") or "") if isinstance(author, dict) else ""
        self._remember_member_name(sender_id, sender_name)
        message = self._build_common_message(
            data,
            message_type="private",
            group_id=None,
            sender_id=sender_id,
            sender_name=sender_name,
        )
        self._remember_recent_event(message)
        return message

    def _remember_recent_event(self, message: Message) -> None:
        if not message.message_id:
            return
        self._recent_events[message.message_id] = (message.session_id, time.time())
        self._recent_event_order.append(message.message_id)
        while len(self._recent_event_order) > len(self._recent_events):
            oldest = self._recent_event_order.popleft()
            self._recent_events.pop(oldest, None)

    # ------------------------------------------------------------------
    # 发送消息
    # ------------------------------------------------------------------

    @staticmethod
    def _split_session(session_id: str) -> tuple[str, str]:
        """session_id -> ("group"|"private", openid)"""
        text = str(session_id or "").strip()
        if text.startswith("group_"):
            return "group", text[len("group_"):]
        if text.startswith("private_"):
            return "private", text[len("private_"):]
        return "", text

    def _messages_path(self, session_id: str) -> str:
        kind, target = self._split_session(session_id)
        if kind == "group":
            return f"/v2/groups/{quote(target, safe='')}/messages"
        return f"/v2/users/{quote(target, safe='')}/messages"

    def _files_path(self, session_id: str) -> str:
        kind, target = self._split_session(session_id)
        if kind == "group":
            return f"/v2/groups/{quote(target, safe='')}/files"
        return f"/v2/users/{quote(target, safe='')}/files"

    def _resolve_reply_target(
        self, session_id: str, reply_to_id: str | None
    ) -> tuple[str, str]:
        """决定这条回复用哪种引用方式。

        返回 ``(msg_id, ref_idx)``：
        * ``msg_id`` 非空 → 被动回复（带 msg_id + msg_seq，QQ 里会原生引用
          那条消息），只在官方给的窗口期内有效；
        * ``ref_idx`` 非空 → 主动消息 + message_reference（引用渲染，但不吃
          被动回复次数）。
        """
        text = str(reply_to_id or "").strip()
        if not text:
            return "", ""
        if text.startswith("REFIDX_"):
            return "", text
        info = self._recent_events.get(text)
        if info:
            event_session, arrived = info
            is_group = event_session.startswith("group_")
            window = PASSIVE_WINDOW_GROUP if is_group else PASSIVE_WINDOW_C2C
            max_reply = PASSIVE_MAX_REPLY_GROUP if is_group else PASSIVE_MAX_REPLY_C2C
            if time.time() - arrived <= window and self._msg_seq_used.get(text, 0) < max_reply:
                return text, ""
        ref_idx = self._ref_idx_by_msg_id.get(text)
        if ref_idx:
            return "", ref_idx
        return "", ""

    def _next_msg_seq(self, msg_id: str) -> int:
        seq = self._msg_seq_used.get(msg_id, 0) + 1
        self._msg_seq_used[msg_id] = seq
        return seq

    def _apply_reply(self, body: dict, session_id: str, reply_to_id: str | None) -> None:
        target_id, ref_idx = self._resolve_reply_target(session_id, reply_to_id)
        if target_id:
            body["msg_id"] = target_id
            body["msg_seq"] = self._next_msg_seq(target_id)
        elif ref_idx:
            body["message_reference"] = {"message_id": ref_idx}

    async def _post_message(self, session_id: str, body: dict, timeout: float = 30.0) -> dict:
        # 带 msg_id 的是被动回复（有人在等），额度不够可以排队等一下；
        # 不带的是主动消息（插话、日报、表情包），额度不够直接跳过。
        passive = bool(body.get("msg_id"))
        await self.rate_limiter.acquire(session_id, passive=passive)
        try:
            return await self._request_json(
                "POST", self._messages_path(session_id), json_body=body, timeout=timeout
            )
        except OfficialAPIError as exc:
            if exc.is_rate_limit:
                self.rate_limiter.note_official_limit(exc.err_code, str(exc))
                logger.warning("官方限频，这条消息被拒收：err_code=%s %s", exc.err_code, exc)
            raise

    def _record_outbound(self, data: Any) -> str:
        if not isinstance(data, dict):
            self.last_sent_message_id = ""
            return ""
        message_id = str(data.get("id") or "").strip()
        self.last_sent_message_id = message_id
        if message_id:
            self._message_senders[message_id] = self._canonical_bot_id()
            ext_info = data.get("ext_info") or {}
            ref_idx = str(ext_info.get("ref_idx") or "").strip()
            if ref_idx:
                self._ref_idx_by_msg_id[message_id] = ref_idx
                self._msg_id_by_ref_idx[ref_idx] = message_id
        return message_id

    async def send_message_with_id(
        self,
        session_id: str,
        content: str,
        reply_to_id: str | None = None,
    ) -> tuple[bool, str]:
        """发送文本消息，返回 (是否成功, 平台消息 ID)。"""
        if not self._connected:
            logger.error("官方通道未连接")
            return False, ""
        kind, target = self._split_session(session_id)
        if kind not in ("group", "private") or not target:
            logger.error("会话格式不正确：%s", session_id)
            return False, ""
        try:
            body: dict[str, Any] = {"msg_type": 0, "content": str(content or "")}
            self._apply_reply(body, session_id, reply_to_id)
            data = await self._post_message(session_id, body)
            message_id = self._record_outbound(data)
            self.messages_sent += 1
            return True, message_id
        except RateLimitedError as exc:
            logger.warning("发送被频控拦下：%s", exc)
            return False, ""
        except Exception as exc:
            logger.exception("发送消息失败：%r", exc)
            return False, ""

    async def send_message(
        self,
        session_id: str,
        content: str,
        reply_to_id: str | None = None,
    ) -> bool:
        success, _ = await self.send_message_with_id(
            session_id, content, reply_to_id=reply_to_id
        )
        return success

    async def send_group_message(
        self, group_id: str, content: str, reply_to_id: str | None = None
    ) -> bool:
        return await self.send_message(f"group_{group_id}", content, reply_to_id=reply_to_id)

    async def send_private_message(self, user_id: str, content: str) -> bool:
        return await self.send_message(f"private_{user_id}", content)

    # ------------------------------------------------------------------
    # 富媒体
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_image_name(blob: bytes) -> str:
        """按文件头猜扩展名；官方只认 png/jpg/gif/webp/bmp 这几种图片格式。"""
        if blob.startswith(b"\x89PNG"):
            return "image.png"
        if blob.startswith(b"\xff\xd8"):
            return "image.jpg"
        if blob.startswith(b"GIF8"):
            return "image.gif"
        if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
            return "image.webp"
        if blob.startswith(b"BM"):
            return "image.bmp"
        return "image.png"

    async def upload_file(
        self,
        session_id: str,
        blob: bytes,
        filename: str = "",
        file_type: int = FILE_TYPE_IMAGE,
    ) -> str:
        """上传文件并返回 file_info。

        5MB 以内走二进制直传，更大走分片上传（upload_prepare → PUT →
        upload_part_finish → 合并）。两条路都拿不到 file_info 时抛异常。
        """
        kind, target = self._split_session(session_id)
        if kind not in ("group", "private") or not target:
            raise RuntimeError(f"会话格式不正确：{session_id}")
        if not blob:
            raise RuntimeError("没有可上传的内容")
        files_path = self._files_path(session_id)
        name = filename or self._guess_image_name(blob)

        if len(blob) <= DIRECT_UPLOAD_LIMIT:
            try:
                data = await self._request_json(
                    "POST",
                    files_path,
                    json_body={
                        "file_type": file_type,
                        "file_name": name,
                        "file_data": base64.b64encode(blob).decode("ascii"),
                        "srv_send_msg": False,
                    },
                    timeout=120.0,
                )
                file_info = str(data.get("file_info") or "")
                if file_info:
                    return file_info
                logger.warning("二进制直传没有返回 file_info，改走分片上传")
            except Exception as exc:
                logger.warning("二进制直传失败，改走分片上传：%s", exc)
        return await self._upload_file_chunked(files_path, blob, name, file_type)

    async def _upload_file_chunked(
        self, files_path: str, blob: bytes, filename: str, file_type: int
    ) -> str:
        prepare_path = files_path.replace("/files", "/upload_prepare")
        finish_path = files_path.replace("/files", "/upload_part_finish")
        body = {
            "file_type": file_type,
            "file_size": str(len(blob)),
            "file_name": filename,
            "md5": hashlib.md5(blob).hexdigest(),
            "sha1": hashlib.sha1(blob).hexdigest(),
            "md5_10m": hashlib.md5(blob[:MD5_10M_SIZE]).hexdigest(),
        }
        data = await self._request_json(
            "POST", prepare_path, json_body=body, timeout=60.0
        )
        upload_id = str(data.get("upload_id") or "")
        if not upload_id:
            raise RuntimeError("预上传没有返回 upload_id")
        try:
            block_size = int(data.get("block_size") or (5 * 1024 * 1024))
        except (TypeError, ValueError):
            block_size = 5 * 1024 * 1024
        parts = [p for p in (data.get("parts") or []) if isinstance(p, dict)]
        if not parts:
            raise RuntimeError("预上传没有返回分片列表")

        for part in parts:
            index = int(part.get("index") or 0)
            url = str(part.get("presigned_url") or "")
            if not url:
                continue
            try:
                part_size = int(part.get("block_size") or block_size)
            except (TypeError, ValueError):
                part_size = block_size
            start = index * block_size
            chunk = blob[start:start + part_size]
            await self._put_bytes(url, chunk)
            await self._request_json(
                "POST",
                finish_path,
                json_body={
                    "upload_id": upload_id,
                    "part_index": index,
                    "block_size": str(len(chunk)),
                    "md5": hashlib.md5(chunk).hexdigest(),
                },
                timeout=60.0,
            )

        merged = await self._request_json(
            "POST",
            files_path,
            json_body={
                "file_type": file_type,
                "file_name": filename,
                "upload_id": upload_id,
                "srv_send_msg": False,
            },
            timeout=180.0,
        )
        file_info = str(merged.get("file_info") or "")
        if not file_info:
            raise RuntimeError("合并上传没有返回 file_info")
        return file_info

    async def send_image_with_id(
        self,
        session_id: str,
        image_bytes: bytes,
        reply_to_id: str | None = None,
    ) -> tuple[bool, str]:
        """上传图片后按富媒体消息发出。"""
        if not self._connected:
            logger.error("官方通道未连接")
            return False, ""
        if not image_bytes:
            return False, ""
        try:
            file_info = await self.upload_file(session_id, image_bytes)
            body: dict[str, Any] = {
                "msg_type": 7,
                "media": {"file_info": file_info},
            }
            self._apply_reply(body, session_id, reply_to_id)
            data = await self._post_message(session_id, body)
            message_id = self._record_outbound(data)
            self.messages_sent += 1
            return True, message_id
        except RateLimitedError as exc:
            logger.warning("发送图片被频控拦下：%s", exc)
            return False, ""
        except Exception as exc:
            logger.exception("发送图片失败：%r", exc)
            return False, ""

    async def send_image(
        self,
        session_id: str,
        image_bytes: bytes,
        reply_to_id: str | None = None,
    ) -> bool:
        success, _ = await self.send_image_with_id(
            session_id, image_bytes, reply_to_id=reply_to_id
        )
        return success

    def _render_at_prefix(self, at_users: list[str] | None) -> tuple[str, list[str]]:
        """把 @ 列表渲染成官方格式的前缀。

        官方群里 @ 某人写成 ``<@member_openid>``，@全体成员写成 ``<@all>``。
        返回 (前缀, 问题列表)：认不出来的 id 不静默丢弃，而是写进问题列表，
        由调用方决定要不要报出来。
        """
        prefix = ""
        problems: list[str] = []
        for raw in at_users or []:
            member_id = str(raw or "").strip()
            if not member_id:
                continue
            if member_id == "all":
                prefix += "<@all> "
                continue
            if member_id not in self._member_names:
                problems.append(
                    f"@{member_id} 不是已知的 member_openid"
                    f"（官方通道拿不到 QQ 号，无法映射），可能不会真的 @ 到人"
                )
            prefix += f"<@{member_id}> "
        return prefix, problems

    async def send_images(
        self,
        session_id: str,
        image_bytes_list: list[bytes],
        reply_to_id: str | None = None,
        text: str = "",
        at_users: list[str] | None = None,
        mode: str = "chunks",
        per_message: int = 9,
        timeout: float | None = 180.0,
    ) -> dict[str, Any]:
        """发多张图。

        官方通道没有「一条消息多张图」，也没有合并转发接口，所以 mode="forward"
        会降级成「一条消息一张图」。
        """
        if not self._connected:
            return {
                "success": False,
                "error": "官方通道未连接",
                "sent": 0,
                "total": 0,
                "message_ids": [],
                "errors": ["官方通道未连接"],
            }
        kind, target = self._split_session(session_id)
        if kind not in ("group", "private") or not target:
            return {
                "success": False,
                "error": "会话格式不正确",
                "sent": 0,
                "total": 0,
                "message_ids": [],
                "errors": ["会话格式不正确"],
            }
        payloads = [item for item in (image_bytes_list or []) if item]
        if not payloads:
            return {
                "success": False,
                "error": "没有可发送的图片",
                "sent": 0,
                "total": 0,
                "message_ids": [],
                "errors": ["没有可发送的图片"],
            }
        if str(mode or "").strip().lower() == "forward":
            logger.warning("官方通道不支持合并转发，降级为逐张发送")

        message_ids: list[str] = []
        errors: list[str] = []
        sent = 0
        # 官方群里 @ 某人写成 <@member_openid>。at_users 里如果塞的是 QQ 号，
        # 官方通道根本没有 QQ 号，也就无从映射 —— 这种要明确报出来，
        # 不能像以前那样静默丢掉，让人以为 @ 生效了。
        at_prefix, at_problems = self._render_at_prefix(at_users)
        errors.extend(at_problems)
        # 文字单独发一条：富媒体消息（msg_type=7）只带 media，再塞 content 会
        # 撞上「消息类型与内容不匹配」（错误码 22006）。
        body_text = f"{at_prefix}{text}" if at_prefix else text
        if body_text:
            ok, text_id = await self.send_message_with_id(
                session_id, body_text, reply_to_id=reply_to_id
            )
            if ok:
                message_ids.append(text_id)
            else:
                errors.append("文字部分发送失败")
        for payload in payloads:
            try:
                file_info = await self.upload_file(session_id, payload)
                body: dict[str, Any] = {
                    "msg_type": 7,
                    "media": {"file_info": file_info},
                }
                self._apply_reply(body, session_id, reply_to_id)
                data = await self._post_message(session_id, body, timeout=timeout or 180.0)
                message_id = self._record_outbound(data)
                if message_id:
                    message_ids.append(message_id)
                sent += 1
            except RateLimitedError as exc:
                logger.warning("发送图片被频控拦下：%s", exc)
                errors.append(str(exc))
            except Exception as exc:
                logger.warning("发送图片失败：%s", exc)
                errors.append(str(exc))
        return {
            "success": sent == len(payloads),
            "mode": "chunks",
            "sent": sent,
            "total": len(payloads),
            "message_count": len(message_ids),
            "message_ids": message_ids,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # 官方通道没有的能力：显式降级，不要假装成功
    # ------------------------------------------------------------------

    async def delete_message(self, message_id: str) -> bool:
        """撤回自己发出的消息（超过 2 分钟不可撤回）。"""
        message_id = str(message_id or "").strip()
        if not message_id:
            return False
        session_id = self._session_of_message(message_id)
        if not session_id:
            logger.warning("官方通道撤回失败：不知道这条消息发在哪个会话")
            return False
        kind, target = self._split_session(session_id)
        path = f"/v2/groups/{quote(target, safe='')}/messages/{quote(message_id, safe='')}"
        if kind != "group":
            path = f"/v2/users/{quote(target, safe='')}/messages/{quote(message_id, safe='')}"
        try:
            await self._request_json("DELETE", path, timeout=20.0)
            return True
        except Exception as exc:
            logger.warning("撤回消息失败：%s", exc)
            return False

    def _session_of_message(self, message_id: str) -> str:
        """从最近的上下文里找出这条消息发在哪个会话。"""
        for candidate, (session_id, _) in reversed(list(self._recent_events.items())):
            if candidate == message_id:
                return session_id
        return ""

    async def poke(self, session_id: str, target_id: str) -> bool:
        """官方通道没有戳一戳接口。

        上层正常情况下不会走到这里（适配器 ``supports_poke=False``，根本不会
        注册回戳回调，主动戳也会被跳过）。留着是为了兜底：万一被调到，明确
        返回 False，而不是假装成功。
        """
        if not self._poke_warned:
            self._poke_warned = True
            logger.warning("官方通道不支持戳一戳，已忽略（后续不再重复提示）")
        return False

    async def get_member_display_name(self, group_id: str, user_id: str) -> str:
        """查群成员显示名。

        群成员接口是白名单能力（错误码 11253），拿不到就退回到「消息事件里
        看到的那个用户名」——官方推事件时本来就带 ``author.username``，
        所以平时根本用不着白名单接口。真拿不到才返回空串，让上层按「没有
        名字」处理，而不是编一个。
        """
        group_id = str(group_id or "").strip()
        user_id = str(user_id or "").strip()
        if not group_id or not user_id:
            return ""
        try:
            data = await self._request_json(
                "GET",
                f"/v2/groups/{quote(group_id, safe='')}/members/{quote(user_id, safe='')}",
                timeout=6.0,
            )
        except Exception as exc:
            logger.debug("查询群成员信息失败（官方通道需要白名单权限）：%s", exc)
            return self._member_names.get(user_id, "")
        if not isinstance(data, dict):
            return self._member_names.get(user_id, "")
        for key in ("nickname", "member_name", "name"):
            value = str(data.get(key) or "").strip()
            if value:
                self._remember_member_name(user_id, value)
                return value
        return self._member_names.get(user_id, "")

    async def fetch_user_avatar(self, user_id: str, *, timeout: float = 6.0) -> bytes | None:
        """官方通道拿不到 QQ 号，头像接口无从谈起。"""
        return None

    async def call_api(
        self,
        action: str,
        params: dict,
        timeout: float | None = 10.0,
    ):
        """兼容 OneBot 的调用入口。

        官方通道没有 OneBot 动作表，只有少数几个能对上；其余返回 None 并
        记一条警告，让调用方（富媒体增强器）自己降级。
        """
        action = str(action or "")
        if action == "get_login_info":
            return {
                "user_id": self._canonical_bot_id(),
                "nickname": self._canonical_bot_id() or "官方机器人",
            }
        if action not in self._warned_actions:
            self._warned_actions.add(action)
            logger.warning("官方通道不支持 OneBot 动作：%s（已降级）", action)
        return None

    # ------------------------------------------------------------------
    # 富媒体增强（与 NapCat 适配器同签名）
    # ------------------------------------------------------------------

    async def enrich_message(
        self,
        message: Message,
        *,
        directed: bool = False,
        conversation_context: str = "",
        group_image_urls: list[dict] | None = None,
    ) -> Message:
        return await self.rich_media_enricher.enrich(
            message,
            directed=directed,
            conversation_context=conversation_context,
            group_image_urls=group_image_urls,
        )

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        owner = self._foreign_owner_loop()
        if owner is not None:
            future = asyncio.run_coroutine_threadsafe(self.disconnect(), owner)
            await asyncio.wrap_future(future)
            return
        self._running = False
        self._connected = False
        self._stopped.set()
        if self._gateway_task:
            self._gateway_task.cancel()
            try:
                await self._gateway_task
            except (asyncio.CancelledError, Exception):
                pass
            self._gateway_task = None
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._http is not None and not self._http.closed:
            try:
                await self._http.close()
            except Exception:
                pass
        self._http = None
        logger.info("官方通道已断开连接")

    def _foreign_owner_loop(self) -> asyncio.AbstractEventLoop | None:
        """面板线程直接操作连接时，要回到适配器所属的事件循环。"""
        owner = self._owner_loop
        if owner is None or owner is asyncio.get_running_loop():
            return None
        if not owner.is_running():
            raise ConnectionError("官方通道事件循环没有运行")
        return owner

    @property
    def is_connected(self) -> bool:
        return self._connected

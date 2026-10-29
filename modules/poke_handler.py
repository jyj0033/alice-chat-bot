"""戳一戳事件的处理。

NapCat 把"戳一戳"当通知事件推过来（``post_type=notice``、
``notice_type=notify``、``sub_type=poke``），字段语义是：

- 群戳：``user_id`` = 戳的人，``target_id`` = 被戳的人，另有 ``group_id``；
- 私戳：``target_id`` = 被戳的人，``sender_id``/``user_id`` = 戳的人。

也就是说 **``target_id`` 才是"被戳的那个"**，只有它等于机器人自己的号，
才说明"有人戳了我"。这条判断必须放在最前面，它同时挡住两类误触发：

1. 别人戳别人（target 跟自己无关）；
2. **机器人自己主动戳别人** —— 这种时候 target 是被戳的人、user_id 才是
   机器人自己。不挡住的话，机器人每戳一次人就会被自己当成"被戳"，
   然后自己回话、自己回戳，直接把自己玩成死循环。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from modules.llm.base import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)

# 不走 LLM 时的固定回应。
DEFAULT_CANNED_REPLIES = (
    "在的在的，戳我干嘛",
    "再戳一下试试？",
    "手欠是不是",
    "戳到了，然后呢",
    "别戳了，痒",
)

# 让 LLM 生成一句回应的提示词。刻意要求短、口语化，
# 免得模型把"戳一戳"当成需要解释的概念题，写出一堆废话。
POKE_PROMPT = (
    "{persona}\n\n"
    "刚刚{where}有人戳了你一下。\n"
    "请只输出一句回应：口语化、有情绪、15个字以内，符合你平时的性格。\n"
    "只输出这句话本身，不要引号、不要解释、不要描写动作。"
)


@dataclass
class PokeConfig:
    """戳一戳的行为开关，配置在 config.yaml 的 ``qq.poke`` 下。"""

    enabled: bool = True
    group_enabled: bool = True
    private_enabled: bool = True
    # 收到戳之后要不要回一句话
    reply: bool = True
    # 回话是否走 LLM（False = 用固定话术，更快也不花 token）
    use_llm: bool = True
    # 要不要戳回去
    poke_back: bool = True
    # 同一个会话里同一个人的回应冷却，防止连点刷屏
    cooldown_seconds: float = 30.0
    # 回戳冷却，比回应冷却长，避免和对方来回戳
    poke_back_cooldown_seconds: float = 60.0
    # 主动戳人的概率：群里每条消息都判定一次（不要求 bot 参与那条消息，
    # 也不要求和最后说不说话有关，可以只戳不说话），0.05 就是 5%
    proactive_probability: float = 0.05
    # 主动戳的冷却：同一个人在这个窗口内不会被主动戳
    proactive_cooldown_seconds: float = 60.0
    llm_timeout: float = 8.0
    max_tokens: int = 60
    canned: tuple[str, ...] = DEFAULT_CANNED_REPLIES

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PokeConfig":
        """从配置字典构造，类型不对的项直接忽略、用默认值。"""
        raw = dict(data or {})
        kwargs: dict[str, Any] = {}
        for name, field_info in cls.__dataclass_fields__.items():
            if name not in raw:
                continue
            value = raw[name]
            # __dataclass_fields__ 里存的是 Field 对象，默认值要取 .default。
            default = field_info.default
            if isinstance(default, bool):
                kwargs[name] = bool(value)
            elif isinstance(default, float):
                try:
                    kwargs[name] = float(value)
                except (TypeError, ValueError):
                    continue
            elif isinstance(default, int):
                try:
                    kwargs[name] = int(value)
                except (TypeError, ValueError):
                    continue
            elif isinstance(default, tuple):
                if isinstance(value, (list, tuple)):
                    items = tuple(
                        str(item).strip() for item in value if str(item).strip()
                    )
                    if items:
                        kwargs[name] = items
            else:
                kwargs[name] = value
        return cls(**kwargs)


@dataclass
class PokeEvent:
    """一次戳一戳。"""

    session_id: str
    is_group: bool
    group_id: str
    poker_id: str
    target_id: str
    raw: dict = field(default_factory=dict)


def parse_poke_event(raw: Any, self_id: str) -> Optional[PokeEvent]:
    """把 OneBot 通知解析成戳一戳事件。

    不是 poke、或者被戳的不是机器人自己，都返回 None。
    """
    if not isinstance(raw, dict):
        return None
    if str(raw.get("notice_type") or "") != "notify":
        return None
    if str(raw.get("sub_type") or "") != "poke":
        return None
    self_id = str(self_id or "").strip()
    if not self_id:
        return None

    group_id = str(raw.get("group_id") or "").strip()
    user_id = str(raw.get("user_id") or "").strip()
    target_id = str(raw.get("target_id") or "").strip()
    sender_id = str(raw.get("sender_id") or "").strip()

    # target_id 才是"被戳的人"：不是我就跟这次事件无关。
    if target_id != self_id:
        return None

    if group_id:
        # 群戳：user_id 就是戳的人。
        poker = user_id
        session_id = f"group_{group_id}"
        is_group = True
    else:
        # 私戳：sender_id 才是戳的人，user_id 是会话对端。
        poker = sender_id or user_id
        session_id = f"private_{poker or user_id}"
        is_group = False
    if not poker or not session_id:
        return None
    return PokeEvent(
        session_id=session_id,
        is_group=is_group,
        group_id=group_id,
        poker_id=poker,
        target_id=target_id,
        raw=dict(raw),
    )


def clean_reply(text: Any) -> str:
    """去掉引号、换行和口水话前缀，控制在 40 字以内。"""
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("\n", " ").replace("\r", " ").strip()
    cleaned = cleaned.strip("「」\"'“”‘’ ")
    for prefix in ("好的，", "好，", "哎呀，", "哈哈哈，", "嘿嘿，"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    if len(cleaned) > 40:
        cleaned = cleaned[:40].strip()
    return cleaned


class PokeResponder:
    """收到戳一戳之后的反应：回一句话，外加可选的戳回去。"""

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        self_id: str = "",
        provider: Any | None = None,
        persona: str = "",
        send_text: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        send_poke: Optional[Callable[[str, str], Awaitable[bool]]] = None,
    ):
        self.config = PokeConfig.from_dict(config)
        self.self_id = str(self_id or "").strip()
        self.provider = provider
        self.persona = persona
        self._send_text = send_text
        self._send_poke = send_poke
        # session+poker -> 上次回应的时间戳
        self._last_reply: dict[str, float] = {}
        # session+target -> 上次回戳的时间戳
        self._last_poke_back: dict[str, float] = {}
        # session+target -> 上次主动戳的时间戳（和回戳节奏互相独立）
        self._last_proactive: dict[str, float] = {}

    async def handle(self, raw: Any) -> dict[str, Any]:
        """处理一条通知事件。返回处理结果，方便调用方看日志。

        只要这次事件确实是"有人戳了机器人"，返回里就带 ``session_id``/
        ``poker_id``/``is_group``/``group_id``，调用方可以据此把这次戳一戳
        写进会话上下文；被跳过的情形用 ``skipped`` 标出原因。
        """
        if not self.config.enabled:
            return {"skipped": "disabled"}
        event = parse_poke_event(raw, self.self_id)
        if event is None:
            return {"skipped": "not_me"}
        result: dict[str, Any] = {
            "session_id": event.session_id,
            "poker_id": event.poker_id,
            "is_group": event.is_group,
            "group_id": event.group_id,
            "target_id": event.target_id,
            "skipped": None,
            "reply": "",
            "reply_sent": False,
            "poked_back": False,
        }
        if event.poker_id == self.self_id:
            # 自己戳自己（或者是自己发起的戳被回显）：不回话也不回戳，
            # 否则一次戳会把自己带成自触发循环。
            result["skipped"] = "self_poke"
            logger.info("[戳一戳] 忽略自己发起的戳（会话=%s）", event.session_id)
            return result
        if event.is_group and not self.config.group_enabled:
            result["skipped"] = "group_disabled"
            return result
        if not event.is_group and not self.config.private_enabled:
            result["skipped"] = "private_disabled"
            return result

        key = f"{event.session_id}:{event.poker_id}"
        now = time.time()
        if now - self._last_reply.get(key, 0.0) < self.config.cooldown_seconds:
            result["skipped"] = "cooldown"
            logger.info(
                "[戳一戳] %s 刚戳过，忽略这次（冷却 %.0f 秒）",
                event.poker_id,
                self.config.cooldown_seconds,
            )
            return result
        self._last_reply[key] = now
        if self.config.reply:
            text = await self._build_reply(event)
            result["reply"] = text
            if text and self._send_text:
                try:
                    result["reply_sent"] = bool(
                        await self._send_text(event.session_id, text)
                    )
                except Exception as exc:
                    logger.warning("[戳一戳] 回应发送失败：%s", exc)
        if self.config.poke_back:
            result["poked_back"] = await self._poke_back(event)
        logger.info(
            "[戳一戳] 会话=%s 戳的人=%s 回应=%r 发出=%s 回戳=%s",
            event.session_id,
            event.poker_id,
            result["reply"],
            result["reply_sent"],
            result["poked_back"],
        )
        return result

    async def maybe_poke(self, session_id: str, target_id: str) -> bool:
        """按概率主动戳一下某人（默认 5%）。

        在群消息进来的那一刻调用：每条群消息都掷一次骰子，不要求 bot
        参与那条消息，也和最后说不说话无关 —— 沉默的时候也可以只戳不说话。
        戳失败不影响后续流程，也不会重试 —— 它是调味，不是任务。
        """
        if not self.config.enabled or not self._send_poke:
            return False
        probability = float(self.config.proactive_probability or 0.0)
        if probability <= 0:
            return False
        target_id = str(target_id or "").strip()
        if not target_id or target_id == self.self_id:
            return False
        key = f"{session_id}:{target_id}"
        now = time.time()
        # 刚回戳过的人，短时间内不再被主动戳（别一分钟内连戳同一人两下）
        if (
            now - self._last_poke_back.get(key, 0.0)
            < self.config.poke_back_cooldown_seconds
        ):
            return False
        # 主动戳自己的节奏，和回戳冷却各管各的
        if (
            now - self._last_proactive.get(key, 0.0)
            < self.config.proactive_cooldown_seconds
        ):
            return False
        if random.random() >= probability:
            return False
        self._last_proactive[key] = now
        try:
            ok = bool(await self._send_poke(session_id, target_id))
        except Exception as exc:
            logger.warning("[主动戳] 失败：%s", exc)
            return False
        if ok:
            logger.info("[主动戳] 戳了 %s（会话 %s）", target_id, session_id)
        return ok

    async def _build_reply(self, event: PokeEvent) -> str:
        """生成回应：优先 LLM，失败或不启用就用固定话术。"""
        if self.config.use_llm and self.provider is not None:
            try:
                raw = await asyncio.wait_for(
                    self._llm_reply(event), timeout=self.config.llm_timeout
                )
                text = clean_reply(raw)
                if text:
                    return text
                logger.warning("[戳一戳] LLM 没给出可用回应，改用固定话术")
            except Exception as exc:
                logger.warning("[戳一戳] LLM 回应失败，改用固定话术：%s", exc)
        if not self.config.canned:
            return ""
        return random.choice(self.config.canned)

    async def _llm_reply(self, event: PokeEvent) -> str:
        persona = self.persona or "你是一个活泼、有点毒舌但不刻薄的群聊机器人。"
        request = ChatRequest(
            model=getattr(self.provider, "model", "") or "",
            messages=[
                ChatMessage(
                    role="user",
                    content=POKE_PROMPT.format(
                        persona=persona,
                        where="群里" if event.is_group else "私聊里",
                    ),
                )
            ],
            max_tokens=self.config.max_tokens,
            temperature=0.9,
        )
        response = await self.provider.chat(request)
        return str(getattr(response, "content", "") or "")

    async def _poke_back(self, event: PokeEvent) -> bool:
        """戳回去。冷却内、或者戳的人是机器人自己，都不戳。"""
        if not self._send_poke:
            return False
        if not event.poker_id or event.poker_id == self.self_id:
            return False
        key = f"{event.session_id}:{event.poker_id}"
        now = time.time()
        if (
            now - self._last_poke_back.get(key, 0.0)
            < self.config.poke_back_cooldown_seconds
        ):
            return False
        self._last_poke_back[key] = now
        try:
            return bool(await self._send_poke(event.session_id, event.poker_id))
        except Exception as exc:
            logger.warning("[戳一戳] 回戳失败：%s", exc)
            return False

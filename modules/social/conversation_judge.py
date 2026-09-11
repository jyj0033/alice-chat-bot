"""群聊目标与接话意图判断。

这里不负责生成最终回复，只负责把一条群消息放回最近的对话中，判断：

* 这句话主要是在对谁说；
* Alice 是否应该参与；
* 如果参与，应该回答、续接、附和还是补充信息；
* 回复应该引用哪条消息。

显式 @、回复段和消息顺序都作为模型输入证据保留，但不在这里写死“必回”
或“必静默”。真实群聊中的省略、转话题和多人插话需要结合上下文判断。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from modules.llm.base import ChatRequest

logger = logging.getLogger(__name__)


@dataclass
class ConversationJudgeResult:
    """一次目标/接话判断的可序列化结果。"""

    target: str = "unknown"  # bot / other / group / unknown
    intent: str = "silent"  # answer / follow_up / acknowledge / add_info / react / silent
    should_reply: bool = False
    confidence: float = 0.0
    reference_message_id: str = ""
    target_user_id: str = ""
    reason: str = ""
    available: bool = False
    error: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "intent": self.intent,
            "should_reply": bool(self.should_reply),
            "confidence": round(max(0.0, min(1.0, float(self.confidence))), 3),
            "reference_message_id": self.reference_message_id,
            "target_user_id": self.target_user_id,
            "reason": self.reason[:240],
            "available": bool(self.available),
            "error": self.error[:160],
            "evidence": self.evidence,
        }

    @classmethod
    def unavailable(cls, error: str = "") -> "ConversationJudgeResult":
        return cls(available=False, error=str(error or ""))


class ConversationJudge:
    """使用小而短的 LLM 调用判断群聊目标和接话意图。"""

    TARGETS = {"bot", "other", "group", "unknown"}
    INTENTS = {
        "answer",
        "follow_up",
        "acknowledge",
        "add_info",
        "react",
        "silent",
    }

    def __init__(
        self,
        provider=None,
        *,
        bot_id: str = "",
        bot_name: str = "爱丽丝",
        enabled: bool = True,
        timeout: float = 8.0,
        max_tokens: int = 220,
        context_messages: int = 16,
    ):
        self.provider = provider
        self.bot_id = str(bot_id or "")
        self.bot_name = str(bot_name or "爱丽丝")
        self.enabled = bool(enabled)
        try:
            self.timeout = max(1.0, float(timeout))
        except (TypeError, ValueError):
            self.timeout = 8.0
        try:
            self.max_tokens = max(120, min(500, int(max_tokens)))
        except (TypeError, ValueError):
            self.max_tokens = 220
        try:
            self.context_messages = max(6, min(30, int(context_messages)))
        except (TypeError, ValueError):
            self.context_messages = 16

    async def judge(
        self,
        current_message: Any,
        recent_messages: list[Any],
        *,
        heuristic_signals: dict[str, Any] | None = None,
    ) -> ConversationJudgeResult:
        """判断当前消息，不输出思维过程，只接受结构化 JSON。"""
        if not self.enabled:
            return ConversationJudgeResult.unavailable("disabled")
        if not self.provider:
            return ConversationJudgeResult.unavailable("provider_unavailable")

        prompt = self._build_prompt(
            current_message,
            recent_messages,
            heuristic_signals=heuristic_signals or {},
        )
        request = ChatRequest(
            model=getattr(self.provider, "model", "") or "",
            temperature=0.0,
            max_tokens=self.max_tokens,
            top_p=0.1,
        )
        request.add_system(
            "你是群聊对话目标判断器，不是回复生成器。"
            "请结合消息顺序、发送者、@对象、回复对象和语义判断当前消息的真正指向，"
            "并判断 Alice 是否应该参与。群聊中的省略主语、接着上一句、转头问别人、"
            "自言自语和多人插话都要结合上下文区分。"
            "不要因为出现问号、昵称或关键词就机械判定为问 Alice。"
            "也不要因为 Alice 刚回复过某人就默认下一句仍然是对 Alice 说。\n"
            "只输出一个 JSON 对象，不要输出 Markdown、解释、思维过程或额外文字。"
            "字段必须是：target（bot/other/group/unknown）、"
            "intent（answer/follow_up/acknowledge/add_info/react/silent）、"
            "should_reply（true/false）、confidence（0到1）、"
            "reference_message_id（相关消息ID，没有就空字符串）、"
            "target_user_id（如果主要对某个群友说则填QQ号，否则空字符串）、"
            "reason（不超过40字的简短依据）。"
        )
        request.add_user(prompt)

        try:
            response = await asyncio.wait_for(
                self.provider.chat(request), timeout=self.timeout
            )
            result = self.parse_result(getattr(response, "content", ""))
            result.available = True
            result.evidence = {
                "message_id": str(getattr(current_message, "message_id", "") or ""),
                "sender_id": str(getattr(current_message, "sender_id", "") or ""),
            }
            logger.info(
                "[目标判断] %s → 目标=%s，意图=%s，是否回复=%s，置信度=%.2f，理由=%s",
                result.evidence["message_id"] or "no-id",
                result.target,
                result.intent,
                result.should_reply,
                result.confidence,
                result.reason,
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[目标判断] 调用失败，回退旧决策：%s", exc)
            return ConversationJudgeResult.unavailable(str(exc))

    async def review_reply(
        self,
        current_message: Any,
        recent_messages: list[Any],
        reply: str,
        *,
        direction: str = "group",
    ) -> ConversationJudgeResult:
        """判断草稿是否只是复述前文；只在可疑草稿上调用。"""
        if not self.enabled or not self.provider:
            return ConversationJudgeResult.unavailable("provider_unavailable")

        history = self._render_messages(recent_messages)
        current = self._render_message(current_message, current=True)
        prompt = (
            "判断下面这条 Bot 草稿是不是把用户刚说的话或前文换一种说法重复了一遍，"
            "而没有真正接话、回答或增加信息。正常使用同一个关键词、对问题直接回答，"
            "不算复读；只有主要内容等价于复述/总结原话才算。\n\n"
            f"【最近上下文】\n{history or '（无）'}\n\n"
            f"【当前消息】\n{current}\n\n"
            f"【Bot草稿】\n{reply}\n\n"
            f"【回复方向】{direction}\n"
            "只输出 JSON：{\"is_paraphrase\":true/false,"
            "\"adds_information\":true/false,"
            "\"replacement_hint\":\"不超过30字\"}。不要输出解释或思维过程。"
        )
        request = ChatRequest(
            model=getattr(self.provider, "model", "") or "",
            temperature=0.0,
            max_tokens=min(self.max_tokens, 160),
            top_p=0.1,
        )
        request.add_system(
            "你是对话质量检查器。只做语义复读判断，不评价人格，不负责改写。"
            "只输出要求的 JSON，不要输出思维过程。"
        )
        request.add_user(prompt)
        try:
            response = await asyncio.wait_for(
                self.provider.chat(request), timeout=self.timeout
            )
            payload = self._parse_json(getattr(response, "content", ""))
            if not payload:
                raise ValueError("invalid_json_result")
            is_paraphrase = self._parse_bool(payload.get("is_paraphrase"), False)
            adds_information = self._parse_bool(
                payload.get("adds_information"), not is_paraphrase
            )
            return ConversationJudgeResult(
                intent="silent" if is_paraphrase and not adds_information else "react",
                should_reply=not (is_paraphrase and not adds_information),
                confidence=0.8 if is_paraphrase else 0.6,
                reason=str(payload.get("replacement_hint") or "")[:80],
                available=True,
                evidence={
                    "is_paraphrase": is_paraphrase,
                    "adds_information": adds_information,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[复读判断] 调用失败：%s", exc)
            return ConversationJudgeResult.unavailable(str(exc))

    async def review_meme_send(
        self,
        current_message: Any,
        recent_messages: list[Any],
        meme: dict[str, Any],
        *,
        reply: str = "",
        direction: str = "group",
    ) -> ConversationJudgeResult:
        """复核候选表情包是否适合当前语境。"""
        if not self.enabled or not self.provider:
            return ConversationJudgeResult.unavailable("provider_unavailable")

        current_id = str(getattr(current_message, "message_id", "") or "")
        history = self._render_messages(recent_messages, exclude_id=current_id)
        current = self._render_message(current_message, current=True)
        category = str((meme or {}).get("category") or "待整理")
        meaning = str((meme or {}).get("meaning") or "").strip()
        tags = "、".join(
            str(tag).strip()
            for tag in ((meme or {}).get("tags") or [])
            if str(tag).strip()
        )
        prompt = (
            "判断 Bot 这次是否真的应该自动发送候选表情包。不要因为模型已经提出发图"
            "就默认同意，必须结合当前消息和最近对话判断图片是否贴切、是否有自然的情绪"
            "或梗的对应关系，以及发送后是否会打断群聊。\n\n"
            f"【最近上下文】\n{history or '（无）'}\n\n"
            f"【当前消息】\n{current}\n\n"
            f"【Bot文字回复】\n{reply or '（只发图）'}\n\n"
            f"【候选表情包】分类={category}；含义={meaning or '未填写'}；标签={tags or '无'}\n"
            f"【回复方向】{direction}\n\n"
            "适合发送：图片含义与当前内容直接匹配，能自然表达反应、吐槽、安慰或接梗。"
            "不适合发送：只是普通事实/技术回答、图片含义不清或与当前话题无关、"
            "为了凑频率硬塞、或者会让 Bot 显得在复读和刷屏。\n"
            "只输出 JSON：{\"should_send_meme\":true/false,"
            "\"confidence\":0到1,\"reason\":\"不超过40字\"}。不要输出解释或思维过程。"
        )
        request = ChatRequest(
            model=getattr(self.provider, "model", "") or "",
            temperature=0.0,
            max_tokens=min(self.max_tokens, 160),
            top_p=0.1,
        )
        request.add_system(
            "你是自动发图的最终语义复核器，只判断候选表情是否适合，"
            "不负责生成文字。只输出要求的 JSON，不要输出思维过程。"
        )
        request.add_user(prompt)
        try:
            response = await asyncio.wait_for(
                self.provider.chat(request), timeout=self.timeout
            )
            payload = self._parse_json(getattr(response, "content", ""))
            if not payload or not any(
                key in payload for key in ("should_send_meme", "should_send")
            ):
                raise ValueError("missing_meme_review_field")
            should_send = self._parse_bool(
                payload.get("should_send_meme", payload.get("should_send")),
                False,
            )
            confidence = self._parse_float(payload.get("confidence"), 0.0)
            result = ConversationJudgeResult(
                target="group",
                intent="react" if should_send else "silent",
                should_reply=should_send,
                confidence=confidence,
                reason=str(payload.get("reason") or "")[:80],
                available=True,
                evidence={
                    "should_send_meme": should_send,
                    "meme_id": str((meme or {}).get("id") or ""),
                },
            )
            logger.info(
                "[表情复核] 候选=%s，是否发送=%s，置信度=%.2f，理由=%s",
                str((meme or {}).get("id") or "")[:10] or "无编号",
                should_send,
                result.confidence,
                result.reason,
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[表情复核] 调用失败：%s", exc)
            return ConversationJudgeResult.unavailable(str(exc))

    def _build_prompt(
        self,
        current_message: Any,
        recent_messages: list[Any],
        *,
        heuristic_signals: dict[str, Any],
    ) -> str:
        current_id = str(getattr(current_message, "message_id", "") or "")
        history = self._render_messages(recent_messages, exclude_id=current_id)
        signals = self._render_signals(heuristic_signals)
        current = self._render_message(current_message, current=True)
        return (
            f"【Bot】{self.bot_name}（QQ:{self.bot_id or '未知'}）\n"
            f"【历史对话】\n{history or '（没有可用历史）'}\n\n"
            f"【当前待判断消息】\n{current}\n\n"
            f"【程序提取的线索（仅供参考，不能替代语义判断）】\n{signals or '（无）'}\n\n"
            "请判断当前消息主要对谁说，以及 Alice 是否应该现在发言。"
        )

    def _render_messages(
        self,
        messages: list[Any],
        *,
        exclude_id: str = "",
    ) -> str:
        rows = []
        for message in list(messages or [])[-self.context_messages :]:
            message_id = str(getattr(message, "message_id", "") or "")
            if exclude_id and message_id and message_id == exclude_id:
                continue
            rows.append(self._render_message(message))
        return "\n".join(rows)

    def _render_message(self, message: Any, *, current: bool = False) -> str:
        sender_id = str(getattr(message, "sender_id", "") or "")
        sender_name = str(getattr(message, "sender_name", "") or "未知用户")
        if bool(getattr(message, "is_bot", False)) or (
            self.bot_id and sender_id == self.bot_id
        ):
            sender = f"{self.bot_name}(Bot)"
        else:
            sender = f"{sender_name}({sender_id or '无QQ'})"

        annotations = []
        message_id = str(getattr(message, "message_id", "") or "")
        if message_id:
            annotations.append(f"id={message_id}")
        reply_to_id = str(getattr(message, "reply_to_id", "") or "")
        reply_to_qq = str(getattr(message, "reply_to_qq", "") or "")
        if reply_to_id or reply_to_qq:
            annotations.append(
                "回复=" + "/".join(x for x in (reply_to_id, reply_to_qq) if x)
            )
        mentions = getattr(message, "mentioned_user_ids", None)
        if mentions is None:
            mentions = []
        mentions = [str(value) for value in mentions if str(value)]
        if mentions:
            annotations.append("@=" + ",".join(mentions))
        if bool(getattr(message, "directed_to_bot", False)):
            annotations.append("旧规则线索=可能对Bot")
        dynamic_target = str(getattr(message, "conversation_target", "") or "")
        if dynamic_target:
            annotations.append("已判断=" + dynamic_target)
        prefix = "当前 " if current else ""
        suffix = f" [{'；'.join(annotations)}]" if annotations else ""
        return f"{prefix}{sender}{suffix}：{str(getattr(message, 'content', '') or '')[:500]}"

    @staticmethod
    def _render_signals(signals: dict[str, Any]) -> str:
        entries = []
        for key in (
            "mentioned_me",
            "mentioned_others",
            "reply_to_me",
            "reply_to_qq",
            "continuation_hint",
            "trigger_reasons",
        ):
            value = signals.get(key)
            if value not in (None, "", [], False):
                entries.append(f"{key}={value}")
        return "；".join(entries)

    @classmethod
    def parse_result(cls, content: str) -> ConversationJudgeResult:
        payload = cls._parse_json(content)
        if not payload:
            raise ValueError("invalid_json_result")
        if not any(
            key in payload for key in ("target", "intent", "should_reply")
        ):
            raise ValueError("missing_judgement_fields")
        target = cls._normalize_target(payload.get("target"))
        intent = cls._normalize_intent(payload.get("intent"))
        should_reply = cls._parse_bool(
            payload.get("should_reply"),
            target == "bot" and intent != "silent",
        )
        if intent == "silent":
            should_reply = False
        confidence = cls._parse_float(payload.get("confidence"), 0.5)
        return ConversationJudgeResult(
            target=target,
            intent=intent,
            should_reply=should_reply,
            confidence=confidence,
            reference_message_id=str(
                payload.get("reference_message_id")
                or payload.get("reference_id")
                or payload.get("target_message_id")
                or ""
            )[:100],
            target_user_id=str(
                payload.get("target_user_id") or payload.get("target_qq") or ""
            )[:40],
            reason=str(payload.get("reason") or "")[:240],
        )

    @classmethod
    def _parse_json(cls, content: str) -> dict[str, Any]:
        text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    @classmethod
    def _normalize_target(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        aliases = {
            "alice": "bot",
            "assistant": "bot",
            "机器人": "bot",
            "爱丽丝": "bot",
            "other_user": "other",
            "user": "other",
            "别人": "other",
            "群友": "other",
            "everyone": "group",
            "all": "group",
            "群": "group",
            "不确定": "unknown",
            "未知": "unknown",
        }
        text = aliases.get(text, text)
        return text if text in cls.TARGETS else "unknown"

    @classmethod
    def _normalize_intent(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        aliases = {
            "reply": "answer",
            "回答": "answer",
            "续接": "follow_up",
            "followup": "follow_up",
            "附和": "acknowledge",
            "补充": "add_info",
            "反应": "react",
            "沉默": "silent",
            "不回复": "silent",
        }
        text = aliases.get(text, text)
        return text if text in cls.INTENTS else "silent"

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value or "").strip().lower()
        if text in {"true", "yes", "1", "是", "应该", "回复"}:
            return True
        if text in {"false", "no", "0", "否", "不", "静默"}:
            return False
        return default

    @staticmethod
    def _parse_float(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

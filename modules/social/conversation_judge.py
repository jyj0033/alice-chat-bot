"""群聊目标与接话意图判断。

这里不负责生成最终回复，只负责把一条群消息放回最近的对话中，判断：

* 这句话主要是在对谁说；
* Alice 是否应该参与；
* 如果参与，应该回答、续接、附和还是补充信息；
* 这句话是在提问、请求、调侃、接梗还是做身份试探；
* 回复应该引用哪条消息。

显式 @、回复段和消息顺序都作为模型输入证据保留。模型判断之后还会再套一层
硬规则：同一发言者刚在明确对别人说话、当前又没有 @/引用 Bot 时，不允许
因为出现「你」或问号就改判为对 Bot；对 Bot 的引用目标只能是当前消息。
面向全群的打招呼、开放提问和闲聊吐槽允许参与；互聊仍保持沉默。
明确把 Alice 拉进玩笑或行动的点名允许短反应，普通第三人称提及仍按上下文判断。
模型没点头时，group/unknown 还可以交给低概率随机插话，other 不行。
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

_VOCATIVE_RE = re.compile(
    r"(?:了|吧|啊|呀|呢|哦|哈|嘛)([\u4e00-\u9fff]{2,4})$"
)
_OTHER_TITLES = frozenset(
    {
        "团长",
        "队长",
        "老板",
        "老师",
        "班长",
        "组长",
        "馆长",
        "会长",
        "室长",
        "主席",
        "大哥",
        "大姐",
        "师兄",
        "师姐",
        "学长",
        "学姐",
    }
)


@dataclass
class ConversationJudgeResult:
    """一次目标/接话判断的可序列化结果。"""

    target: str = "unknown"  # bot / other / group / unknown
    intent: str = "silent"  # answer / follow_up / acknowledge / add_info / react / silent
    should_reply: bool = False
    confidence: float = 0.0
    # ``intent`` 控制是否进入回复流程；语用字段描述当前消息在语境中
    # 做的社交动作。两者分开，避免语用分类偶尔不准时影响原有的发言决策。
    pragmatic_intent: dict[str, Any] = field(default_factory=dict)
    target_confidence: float | None = None
    pragmatic_confidence: float | None = None
    reference_message_id: str = ""
    target_user_id: str = ""
    reason: str = ""
    available: bool = False
    error: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def clamp(value: Any, default: float = 0.0) -> float:
            try:
                return round(max(0.0, min(1.0, float(value))), 3)
            except (TypeError, ValueError):
                return default

        return {
            "target": self.target,
            "intent": self.intent,
            "should_reply": bool(self.should_reply),
            "confidence": clamp(self.confidence),
            "target_confidence": clamp(
                self.confidence
                if self.target_confidence is None
                else self.target_confidence
            ),
            "pragmatic_confidence": (
                None
                if self.pragmatic_confidence is None
                else clamp(self.pragmatic_confidence)
            ),
            "pragmatic_intent": dict(self.pragmatic_intent or {}),
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
    PRAGMATIC_ACTS = {
        "question",
        "request",
        "answer",
        "follow_up",
        "acknowledge",
        "share",
        "tease",
        "banter",
        "complaint",
        "correction",
        "praise",
        "comfort",
        "challenge",
        "identity_test",
        "silence",
    }
    PRAGMATIC_TONES = {
        "neutral",
        "serious",
        "playful",
        "sarcastic",
        "hostile",
        "uncertain",
    }
    EXPECTED_REPLIES = {
        "direct_answer",
        "short_reaction",
        "short_banter",
        "clarify",
        "defuse",
        "silent",
    }
    RISK_FLAGS = {
        "identity_bait",
        "ambiguous_addressee",
        "unsupported_assumption",
        "escalation_risk",
    }

    def __init__(
        self,
        provider=None,
        *,
        review_provider=None,
        meme_review_provider=None,
        bot_id: str = "",
        bot_name: str = "爱丽丝",
        enabled: bool = True,
        timeout: float = 8.0,
        max_tokens: int = 220,
        context_messages: int = 16,
    ):
        self.provider = provider
        # 目标判断、回复复核和表情复核可以使用不同的模型；未单独指定时
        # 依次回退到目标判断 Provider，保持旧配置行为不变。
        self.review_provider = review_provider or provider
        self.meme_review_provider = meme_review_provider or self.review_provider
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

    @staticmethod
    def allows_probabilistic_interjection(
        judgement: dict[str, Any] | ConversationJudgeResult | None,
        *,
        mentioned_others: bool = False,
        reply_to_other: bool = False,
    ) -> bool:
        """Judge 未强制参与时，是否还允许走低概率随机插话。

        互聊（target=other）、对 Bot 但判定沉默、以及「刚在对别人说」的延续否决
        一律不准。面向全群或目标不明、且当前没有 @/回复其他人时可以交给旧概率。
        """
        if isinstance(judgement, ConversationJudgeResult):
            data = judgement.to_dict()
        else:
            data = dict(judgement or {})
        if not data.get("available"):
            return True
        if data.get("should_reply"):
            return False
        target = str(data.get("target") or "")
        if target in {"other", "bot"}:
            return False
        evidence = data.get("evidence") or {}
        if evidence.get("continuity_veto"):
            return False
        if mentioned_others or reply_to_other:
            return False
        return target in {"group", "unknown"}

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
            "并判断 Alice 是否应该参与。Alice 是群成员，不是只在被点名时才说话的客服。"
            "群聊中的省略主语、接着上一句、转头问别人、自言自语和多人插话都要结合上下文区分。"
            "不要因为出现问号、昵称或关键词就机械判定为问 Alice。"
            "也不要因为 Alice 刚回复过某人就默认下一句仍然是对 Alice 说。\n"
            "【应当 should_reply=true】"
            "target=bot：明确@/回复/叫「爱丽丝」「小艾」「小爱」在对她说话或提问；"
            "target=group：面向全群的打招呼（早、早上好、孩子们）、"
            "开放提问（有无人、有无、谁来、找人聊天）、"
            "随口分享后适合接一句的闲聊吐槽，或明确把 Alice 拉进玩笑、惩罚或行动"
            "（如‘连带着爱丽丝一起处刑’、‘把小艾也算上’）；此类场景允许一条短反应，"
            "即使它同时承接了对其他群友的话。单纯第三人称提到 Alice 不算拉她参与。"
            "面向全群时 intent 用 acknowledge/react/answer/add_info，"
            "不要因为没@Alice 就改成 silent 或 should_reply=false。\n"
            "【应当 should_reply=false】"
            "target=other：正在回复/@其他群友，或同一人刚在对别人说、当前没有新的指向；"
            "广告、卡片、合并转发、纯链接、无法识别的消息；"
            "两人对聊的中间句、自言自语补完上一句。\n"
            "同一发言者最近一次明确@/回复的对象如果是其他群友，而当前消息没有新的"
            "@/引用 Bot，不能因为出现‘你’、问号或附近有图片就改判为对 Bot；"
            "应保持 target=other，should_reply=false。\n"
            "reference_message_id 只能填当前待判断消息的 id，不要填旁边的图片或别人的消息。\n"
            "能确定是在对某个群友说：target=other、should_reply=false。"
            "面向全群且适合随口接一句：target=group、should_reply=true。"
            "只有既不像对特定人说、也不像适合接的闲聊时，才用 target=unknown、should_reply=false。"
            "不要把面向全群的话一律判成 silent。\n"
            "只输出一个 JSON 对象，不要输出 Markdown、解释、思维过程或额外文字。"
            "字段必须是：target（bot/other/group/unknown）、"
            "intent（answer/follow_up/acknowledge/add_info/react/silent）、"
            "should_reply（true/false）、confidence（0到1）、"
            "target_confidence（目标判断置信度，0到1）、"
            "pragmatic_confidence（语用判断置信度，0到1）、"
            "pragmatic_intent（对象：act/tone/expected_reply/literal/risk_flags）、"
            "reference_message_id（当前消息ID，没有就空字符串）、"
            "target_user_id（如果主要对某个群友说则填QQ号，否则空字符串）、"
            "reason（不超过40字的简短依据）。\n"
            "pragmatic_intent.act 只能是 question/request/answer/follow_up/"
            "acknowledge/share/tease/banter/complaint/correction/praise/comfort/"
            "challenge/identity_test/silence；"
            "tone 只能是 neutral/serious/playful/sarcastic/hostile/uncertain；"
            "expected_reply 只能是 direct_answer/short_reaction/short_banter/"
            "clarify/defuse/silent；literal 是 true/false；"
            "risk_flags 只能从 identity_bait/ambiguous_addressee/"
            "unsupported_assumption/escalation_risk 中选择。\n"
            "语用意图描述这句话在当前语境中做的社交动作，不要把它当成事实判断。"
            "target/should_reply 仍负责是否进入回复流程；语气拿不准时用 uncertain，"
            "不要为了凑字段臆测。"
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
                "target_continuity": bool(
                    self._build_target_continuity_hint(
                        current_message,
                        recent_messages,
                        self._build_user_name_map(
                            [*(recent_messages or []), current_message]
                        ),
                    )
                ),
            }
            result = self.apply_safety_overrides(
                result,
                current_message,
                recent_messages,
                heuristic_signals or {},
            )
            logger.info(
                "[目标判断] %s → 目标=%s，意图=%s，语用=%s/%s，是否回复=%s，"
                "置信度=%.2f/%.2f，理由=%s",
                result.evidence["message_id"] or "no-id",
                result.target,
                result.intent,
                (result.pragmatic_intent or {}).get("act", "-"),
                (result.pragmatic_intent or {}).get("tone", "-"),
                result.should_reply,
                result.target_confidence
                if result.target_confidence is not None
                else result.confidence,
                result.pragmatic_confidence
                if result.pragmatic_confidence is not None
                else 0.0,
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
        conversation_judgement: dict[str, Any] | ConversationJudgeResult | None = None,
    ) -> ConversationJudgeResult:
        """复核草稿是否符合消息意图、证据和基本表达质量。"""
        provider = self.review_provider or self.provider
        if not self.enabled or not provider:
            return ConversationJudgeResult.unavailable("provider_unavailable")

        current_id = str(getattr(current_message, "message_id", "") or "")
        known_users = self._build_user_name_map(
            [*(recent_messages or []), current_message]
        )
        history = self._render_messages(
            recent_messages,
            exclude_id=current_id,
            known_users=known_users,
        )
        current = self._render_message(
            current_message,
            current=True,
            known_users=known_users,
        )
        if isinstance(conversation_judgement, ConversationJudgeResult):
            judgement = conversation_judgement.to_dict()
        else:
            judgement = dict(conversation_judgement or {})
        judgement_contract = self._render_judgement_contract(judgement)
        prompt = (
            "判断下面这条 Bot 草稿是否真正理解并接住了当前消息。重点检查两类问题：\n"
            "1. 是不是把用户刚说的话或前文换一种说法重复了一遍，而没有真正接话、回答或增加信息；\n"
            "2. 是不是把‘好像/似乎/可能’这类不确定转述当成了已确认事实；\n"
            "3. 回复是否在句法上完整；草稿是不是在分析对话、执行规则、描述回复过程，"
            "而不是直接对群友说话。换一种措辞仍然算元话语。"
            "一本正经承认自己是机器人/AI，或把图片内部描述复述成台词，也不是自然回应。"
            "群聊插话遇到这些通常应当丢弃或重写。"
            "如果上下文只有[图片]、[视频]或[无法识别的消息]占位，或图片摘要标注‘视觉识别不确定’，"
            "没有足够客观证据，"
            "Bot不能假装知道画面内容；例如用户说‘好像在夸你’，直接回‘谢谢’可能就是无根据的确认，"
            "更合适的做法通常是澄清、谨慎回应或保持沉默。\n"
            "正常使用同一个关键词、对问题直接回答不算复读；只有主要内容等价于复述/总结原话才算。\n\n"
            f"【最近上下文】\n{history or '（无）'}\n\n"
            f"【当前消息】\n{current}\n\n"
            f"【Bot草稿】\n{reply}\n\n"
            f"【回复方向】{direction}\n"
            f"{judgement_contract}"
            "如果提供了消息语用意图，还要检查草稿是否完成了对应的社交动作："
            "调侃不能被按字面认真解释，轻松玩笑不能无故升级成攻击，"
            "对 Bot 的称呼不能倒扣给发言人；命中 identity_bait 时不要承认自己是"
            "机器人、AI、程序，也不要用身体设定暴露身份。语用置信度偏低时，"
            "允许多种自然接法；只有明确的对象错误、身份暴露、事实臆断或语气升级才判不匹配。\n"
            "只输出 JSON：{\"is_paraphrase\":true/false,"
            "\"adds_information\":true/false,"
            "\"unsupported_assumption\":true/false,"
            "\"needs_clarification\":true/false,"
            "\"incomplete\":true/false,"
            "\"meta_commentary\":true/false,"
            "\"on_topic\":true/false,"
            "\"pragmatic_mismatch\":true/false,"
            "\"mismatch_types\":[\"identity_exposure\",\"literalized_banter\","
            "\"wrong_addressee\",\"tone_drift\",\"strategy_mismatch\"],"
            "\"replacement_hint\":\"不超过30字\"}。不要输出解释或思维过程。"
        )
        request = ChatRequest(
            model=getattr(provider, "model", "") or "",
            temperature=0.0,
            max_tokens=min(self.max_tokens, 160),
            top_p=0.1,
        )
        request.add_system(
            "你是对话质量检查器。检查语义质量和消息语用契约，不评价人格，不负责改写。"
            "只输出要求的 JSON，不要输出思维过程。"
        )
        request.add_user(prompt)
        try:
            response = await asyncio.wait_for(
                provider.chat(request), timeout=self.timeout
            )
            payload = self._parse_json(getattr(response, "content", ""))
            if not payload or not any(
                key in payload
                for key in (
                    "is_paraphrase",
                    "adds_information",
                    "unsupported_assumption",
                    "needs_clarification",
                    "incomplete",
                    "meta_commentary",
                    "on_topic",
                    "pragmatic_mismatch",
                    "mismatch_types",
                )
            ):
                raise ValueError("invalid_json_result")
            is_paraphrase = self._parse_bool(payload.get("is_paraphrase"), False)
            adds_information = self._parse_bool(
                payload.get("adds_information"), not is_paraphrase
            )
            unsupported_assumption = self._parse_bool(
                payload.get("unsupported_assumption"), False
            )
            needs_clarification = self._parse_bool(
                payload.get("needs_clarification"), False
            )
            incomplete = self._parse_bool(payload.get("incomplete"), False)
            meta_commentary = self._parse_bool(
                payload.get("meta_commentary"), False
            )
            on_topic = self._parse_bool(payload.get("on_topic"), True)
            pragmatic_mismatch = self._parse_bool(
                payload.get("pragmatic_mismatch"), False
            )
            mismatch_types = self._normalize_mismatch_types(
                payload.get("mismatch_types")
            )
            if self._looks_like_identity_exposure(
                current_message,
                reply,
                judgement,
            ) and "identity_exposure" not in mismatch_types:
                mismatch_types.append("identity_exposure")
            pragmatic_mismatch = pragmatic_mismatch or bool(mismatch_types)
            quality_issue = (
                (is_paraphrase and not adds_information)
                or unsupported_assumption
                or needs_clarification
                or incomplete
                or meta_commentary
                or not on_topic
                or pragmatic_mismatch
            )
            return ConversationJudgeResult(
                intent="silent" if quality_issue else "react",
                should_reply=not quality_issue,
                confidence=0.8 if quality_issue else 0.6,
                reason=str(payload.get("replacement_hint") or "")[:80],
                available=True,
                evidence={
                    "is_paraphrase": is_paraphrase,
                    "adds_information": adds_information,
                    "unsupported_assumption": unsupported_assumption,
                    "needs_clarification": needs_clarification,
                    "incomplete": incomplete,
                    "meta_commentary": meta_commentary,
                    "on_topic": on_topic,
                    "pragmatic_mismatch": pragmatic_mismatch,
                    "mismatch_types": mismatch_types,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[复读判断] 调用失败：%s", exc)
            return ConversationJudgeResult.unavailable(str(exc))

    @classmethod
    def _render_judgement_contract(cls, judgement: dict[str, Any]) -> str:
        """把前置判断压成短契约，交给复核器核对而不是重新猜测。"""
        if not judgement or not judgement.get("available"):
            return ""
        pragmatic = judgement.get("pragmatic_intent") or {}
        if not isinstance(pragmatic, dict):
            pragmatic = {}
        lines = [
            "【预先判断的消息意图（仅作为复核契约）】",
            f"target={str(judgement.get('target') or 'unknown')}",
            f"intent={str(judgement.get('intent') or 'silent')}",
            f"should_reply={bool(judgement.get('should_reply'))}",
        ]
        for key in ("act", "tone", "expected_reply", "literal", "risk_flags"):
            if key in pragmatic and pragmatic.get(key) not in (None, "", []):
                lines.append(f"{key}={pragmatic.get(key)}")
        pragmatic_confidence = judgement.get("pragmatic_confidence")
        if pragmatic_confidence is not None:
            lines.append(f"pragmatic_confidence={pragmatic_confidence}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _normalize_mismatch_types(value: Any) -> list[str]:
        allowed = {
            "identity_exposure",
            "literalized_banter",
            "wrong_addressee",
            "tone_drift",
            "strategy_mismatch",
        }
        aliases = {
            "身份暴露": "identity_exposure",
            "把玩笑当真": "literalized_banter",
            "称呼对象错误": "wrong_addressee",
            "语气漂移": "tone_drift",
            "策略不匹配": "strategy_mismatch",
        }
        if isinstance(value, str):
            value = re.split(r"[,，、/\s]+", value)
        if not isinstance(value, (list, tuple, set)):
            return []
        result = []
        for item in value:
            text = str(item or "").strip().lower()
            text = aliases.get(text, text)
            if text in allowed and text not in result:
                result.append(text)
        return result[:5]

    @staticmethod
    def _looks_like_identity_exposure(
        current_message: Any,
        reply: str,
        judgement: dict[str, Any],
    ) -> bool:
        """拦截身份诱导下的明显自我暴露，避免完全依赖复核模型。"""
        current_text = str(
            getattr(current_message, "outer_text", "")
            or getattr(current_message, "content", "")
            or ""
        ).lower()
        reply_text = str(reply or "").lower()
        pragmatic = judgement.get("pragmatic_intent") or {}
        raw_flags = pragmatic.get("risk_flags") if isinstance(pragmatic, dict) else []
        if isinstance(raw_flags, str):
            raw_flags = [raw_flags]
        risk_flags = set(raw_flags or [])
        english_identity_term = bool(
            re.search(r"(?<![a-z])(?:ai|bot)(?![a-z])", current_text)
        )
        identity_context = bool(
            risk_flags.intersection({"identity_bait"})
            or any(term in current_text for term in ("人机", "机器人", "人工智能"))
            or english_identity_term
        )
        if not identity_context:
            return False
        if re.search(
            r"(?:我|本人)\s*(?:是|就是|属于|作为)\s*(?:机器人|人工智能|语言模型|ai|bot|人机)",
            reply_text,
            flags=re.IGNORECASE,
        ):
            return True
        # 例如当前问“你有皮肤嘛”，回复“我没有，但我知道疼”也是把玩笑
        # 按 Bot 的身体设定回答，应该回到轻松接梗或装傻策略。
        return "皮肤" in current_text and bool(
            re.search(r"(?:我|本人)\s*(?:没|没有|不具备|没有什么)", reply_text)
        )

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
        provider = self.meme_review_provider or self.review_provider or self.provider
        if not self.enabled or not provider:
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
            f"【Bot文字回复】\n{reply or '（只发图；空白表示图片已经足够表达，不是回复缺失）'}\n\n"
            f"【候选表情包】分类={category}；含义={meaning or '未填写'}；标签={tags or '无'}\n"
            f"【回复方向】{direction}\n\n"
            "适合发送：图片含义与当前内容直接匹配，能自然表达反应、吐槽、安慰或接梗；"
            "如果图片已经完整表达 Bot 想说的内容，只发图片即可，不需要补一条同义文字。"
            "不适合发送：只是普通事实/技术回答、图片含义不清或与当前话题无关、"
            "为了凑频率硬塞、或者会让 Bot 显得在复读和刷屏。\n"
            "只输出 JSON：{\"should_send_meme\":true/false,"
            "\"confidence\":0到1,\"reason\":\"不超过40字\"}。不要输出解释或思维过程。"
        )
        request = ChatRequest(
            model=getattr(provider, "model", "") or "",
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
                provider.chat(request), timeout=self.timeout
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
        known_users = self._build_user_name_map(
            [*(recent_messages or []), current_message]
        )
        history = self._render_messages(
            recent_messages,
            exclude_id=current_id,
            known_users=known_users,
        )
        signals = self._render_signals(heuristic_signals)
        current = self._render_message(
            current_message,
            current=True,
            known_users=known_users,
        )
        continuity = self._build_target_continuity_hint(
            current_message,
            recent_messages,
            known_users,
        )
        continuity_block = (
            f"【同一发言者的目标延续线索（语义证据，不是硬规则）】\n{continuity}\n\n"
            if continuity
            else ""
        )
        return (
            f"【Bot】{self.bot_name}（QQ:{self.bot_id or '未知'}）\n"
            f"【历史对话】\n{history or '（没有可用历史）'}\n\n"
            f"{continuity_block}"
            f"【当前待判断消息】\n{current}\n\n"
            f"【程序提取的线索（仅供参考，不能替代语义判断）】\n{signals or '（无）'}\n\n"
            "请判断当前消息主要对谁说，以及 Alice 是否应该现在发言。"
        )

    def _render_messages(
        self,
        messages: list[Any],
        *,
        exclude_id: str = "",
        known_users: dict[str, str] | None = None,
    ) -> str:
        rows = []
        for message in list(messages or [])[-self.context_messages :]:
            message_id = str(getattr(message, "message_id", "") or "")
            if exclude_id and message_id and message_id == exclude_id:
                continue
            rows.append(self._render_message(message, known_users=known_users))
        return "\n".join(rows)

    def _render_message(
        self,
        message: Any,
        *,
        current: bool = False,
        known_users: dict[str, str] | None = None,
    ) -> str:
        known_users = known_users or {}
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
            reply_ref = "/".join(
                x for x in (reply_to_id, reply_to_qq) if x
            )
            if reply_to_qq:
                reply_ref += f"({self._format_user_reference(reply_to_qq, known_users)})"
            annotations.append("回复=" + reply_ref)
        mentions = getattr(message, "mentioned_user_ids", None)
        if mentions is None:
            mentions = []
        mentions = [str(value) for value in mentions if str(value)]
        if mentions:
            annotations.append(
                "@="
                + ",".join(
                    self._format_user_reference(value, known_users)
                    for value in mentions
                )
            )
        if bool(getattr(message, "directed_to_bot", False)):
            annotations.append("旧规则线索=可能对Bot")
        dynamic_target = str(getattr(message, "conversation_target", "") or "")
        if dynamic_target:
            annotations.append("已判断=" + dynamic_target)
        prefix = "当前 " if current else ""
        suffix = f" [{'；'.join(annotations)}]" if annotations else ""
        return f"{prefix}{sender}{suffix}：{str(getattr(message, 'content', '') or '')[:500]}"

    def _build_user_name_map(self, messages: list[Any]) -> dict[str, str]:
        """从当前判断窗口建立 QQ→昵称映射，避免模型只看到难以辨认的数字。"""
        users: dict[str, str] = {}
        for message in messages or []:
            user_id = str(getattr(message, "sender_id", "") or "")
            user_name = str(getattr(message, "sender_name", "") or "").strip()
            if user_id and user_name and user_name != "未知用户":
                users[user_id] = user_name[:40]
        if self.bot_id:
            users[self.bot_id] = self.bot_name
        return users

    def _format_user_reference(
        self,
        user_id: str,
        known_users: dict[str, str] | None = None,
    ) -> str:
        """渲染 @/回复对象，同时保留原始 QQ 号作为硬证据。"""
        user_id = str(user_id or "")
        if not user_id:
            return "未知用户"
        known_users = known_users or {}
        user_name = known_users.get(user_id, "")
        if user_name:
            return f"{user_id}({user_name})"
        return f"{user_id}(未知用户)"

    def _build_target_continuity_hint(
        self,
        current_message: Any,
        recent_messages: list[Any],
        known_users: dict[str, str] | None = None,
    ) -> str:
        """提取同一发言者最近一次明确指向，交给模型做动态语义判断。"""
        known_users = known_users or {}
        all_messages = list(recent_messages or []) + [current_message]
        message_senders = {
            str(getattr(message, "message_id", "") or ""): str(
                getattr(message, "sender_id", "") or ""
            )
            for message in all_messages
            if str(getattr(message, "message_id", "") or "")
        }
        sender_id = str(getattr(current_message, "sender_id", "") or "")
        current_id = str(getattr(current_message, "message_id", "") or "")

        latest = None
        for message in list(recent_messages or []):
            if str(getattr(message, "message_id", "") or "") == current_id:
                continue
            if str(getattr(message, "sender_id", "") or "") != sender_id:
                continue
            targets = self._explicit_target_ids(message, message_senders)
            if targets:
                latest = (message, targets)

        if not latest:
            return ""

        target_message, target_ids = latest
        target_text = "、".join(
            self._format_user_reference(target_id, known_users)
            for target_id in target_ids
        )
        target_message_id = str(
            getattr(target_message, "message_id", "") or "无编号"
        )
        current_targets = self._explicit_target_ids(
            current_message, message_senders
        )
        if current_targets:
            current_note = (
                "当前消息已经出现新的@/回复对象，请优先按当前对象和当前语义判断，"
                "不要机械沿用上一条。"
            )
        else:
            current_note = (
                "当前消息没有新的@/回复对象；请判断它是否仍在延续这次指向。"
            )
        return (
            f"同一发言者最近一次明确@/回复的对象是：{target_text}（消息id={target_message_id}）。"
            f"{current_note}如果当前内容更像发给该群友、自己补完上一句或已经换话题，"
            "不能因为出现‘你’、问句或 Bot 刚才说过话就自动改判为对 Bot；"
            "指向确实不清时，使用 target=unknown 并保持 should_reply=false。"
        )

    def _explicit_target_ids(
        self,
        message: Any,
        message_senders: dict[str, str] | None = None,
    ) -> list[str]:
        """提取一条消息显式 @ 或回复的对象，去重并排除 @全体。"""
        targets = []
        for value in getattr(message, "mentioned_user_ids", None) or []:
            value = str(value or "")
            if value and value.lower() not in {"all", "everyone"}:
                targets.append(value)
        reply_to_qq = str(getattr(message, "reply_to_qq", "") or "")
        if reply_to_qq:
            targets.append(reply_to_qq)
        elif getattr(message, "reply_to_id", None) and message_senders:
            replied_sender = message_senders.get(
                str(getattr(message, "reply_to_id", "") or "")
            )
            if replied_sender:
                targets.append(replied_sender)
        return list(dict.fromkeys(targets))

    def apply_safety_overrides(
        self,
        result: ConversationJudgeResult,
        current_message: Any,
        recent_messages: list[Any] | None = None,
        heuristic_signals: dict[str, Any] | None = None,
    ) -> ConversationJudgeResult:
        """钉死引用目标，并否决「刚在对别人说却被改判成问 Bot」的结果。"""
        signals = heuristic_signals or {}
        recent_messages = list(recent_messages or [])
        current_id = str(getattr(current_message, "message_id", "") or "")
        all_messages = [*recent_messages, current_message]
        message_senders = {
            str(getattr(message, "message_id", "") or ""): str(
                getattr(message, "sender_id", "") or ""
            )
            for message in all_messages
            if str(getattr(message, "message_id", "") or "")
        }

        if current_id:
            if result.target == "bot":
                if (
                    result.reference_message_id
                    and result.reference_message_id != current_id
                ):
                    result.evidence["dropped_reference_message_id"] = (
                        result.reference_message_id
                    )
                result.reference_message_id = current_id
                result.evidence["pinned_reference"] = True
            elif result.should_reply and not result.reference_message_id:
                result.reference_message_id = current_id

        points_to_bot = self._current_points_to_bot(
            current_message, message_senders, signals
        )
        continuity = self._same_sender_continuity(
            current_message, recent_messages, message_senders
        )
        if (
            result.target == "bot"
            and not points_to_bot
            and continuity
            and continuity.get("points_to_other")
            and not continuity.get("points_to_bot")
        ):
            other_id = str(continuity.get("target_user_id") or "")
            result.evidence["model_target"] = result.target
            result.evidence["continuity_veto"] = True
            result.evidence["continuity_source"] = str(
                continuity.get("source_id") or ""
            )
            result.target = "other" if other_id else "unknown"
            result.should_reply = False
            result.intent = "silent"
            if other_id:
                result.target_user_id = other_id
            result.reason = "硬规则：同一人刚在对别人说，当前没有@/引用Bot"
            logger.info(
                "[目标判断] 延续否决 %s：上一对象=%s",
                current_id or "no-id",
                other_id or continuity.get("judged") or "other",
            )

        if self._is_playful_bot_inclusion(current_message):
            result.evidence["playful_bot_inclusion"] = True
            result.evidence["model_target"] = result.target
            result.target = "group"
            result.intent = "react"
            result.should_reply = True
            result.confidence = max(result.confidence, 0.72)
            result.target_confidence = max(
                result.target_confidence
                if result.target_confidence is not None
                else 0.0,
                0.72,
            )
            result.pragmatic_confidence = max(
                result.pragmatic_confidence
                if result.pragmatic_confidence is not None
                else 0.0,
                0.72,
            )
            result.pragmatic_intent = {
                **(result.pragmatic_intent or {}),
                "act": "banter",
                "tone": "playful",
                "expected_reply": "short_banter",
                "literal": False,
            }
            result.target_user_id = ""
            if current_id:
                result.reference_message_id = current_id
            result.reason = "群友把爱丽丝拉进玩笑，适合短接一句"
            logger.info(
                "[目标判断] 玩笑点名拉入群聊 %s，允许短反应",
                current_id or "no-id",
            )
        return result

    def _is_playful_bot_inclusion(self, message: Any) -> bool:
        """识别明确把 Bot 拉进玩笑/行动的短句，避免只按普通第三人称提及处理。"""
        if bool(getattr(message, "rich_only", False)):
            return False
        text = str(
            getattr(message, "outer_text", "")
            or getattr(message, "content", "")
            or ""
        )
        if not text:
            return False

        names = {
            str(name or "").strip()
            for name in (self.bot_name, "爱丽丝", "小艾", "小爱")
            if str(name or "").strip()
        }
        if not names:
            return False
        name_pattern = "(?:" + "|".join(
            re.escape(name) for name in sorted(names, key=len, reverse=True)
        ) + ")"
        inclusion_pattern = re.compile(
            rf"(?:连带(?:着)?|算上|加上|带上|拉上|拖上|捎上).{{0,8}}{name_pattern}"
            rf"|{name_pattern}.{{0,8}}(?:也)?(?:一起|算上|加上|带上|拉上|拖上|捎上|挨罚|被罚|遭殃|处刑|上名单)"
        )
        return bool(inclusion_pattern.search(text))

    def _current_points_to_bot(
        self,
        message: Any,
        message_senders: dict[str, str],
        signals: dict[str, Any],
    ) -> bool:
        """只认显式 @/引用/点名，不把旧的 directed_to_bot 分析标记当成证据。"""
        if signals.get("mentioned_me") or signals.get("reply_to_me"):
            return True
        if bool(getattr(message, "mentioned_me", False)):
            return True
        targets = self._explicit_target_ids(message, message_senders)
        if self.bot_id and self.bot_id in targets:
            return True
        content = str(getattr(message, "content", "") or "")
        for name in (self.bot_name, "爱丽丝", "小艾"):
            if name and name in content:
                return True
        return False

    def _same_sender_continuity(
        self,
        current_message: Any,
        recent_messages: list[Any],
        message_senders: dict[str, str],
    ) -> dict[str, Any] | None:
        """同一发言者最近一次明确指向：显式 @/回复优先，其次已落盘的动态判断。"""
        sender_id = str(getattr(current_message, "sender_id", "") or "")
        current_id = str(getattr(current_message, "message_id", "") or "")
        latest: dict[str, Any] | None = None
        for message in recent_messages or []:
            if str(getattr(message, "message_id", "") or "") == current_id:
                continue
            if str(getattr(message, "sender_id", "") or "") != sender_id:
                continue
            explicit = [
                target
                for target in self._explicit_target_ids(message, message_senders)
                if target
            ]
            judged = str(getattr(message, "conversation_target", "") or "")
            if not explicit and judged not in {"bot", "other"}:
                continue
            other_ids = [target for target in explicit if target != self.bot_id]
            if explicit:
                points_to_bot = bool(self.bot_id and self.bot_id in explicit)
                points_to_other = bool(other_ids) and not points_to_bot
            else:
                points_to_bot = judged == "bot"
                points_to_other = judged == "other"
            latest = {
                "source_id": str(getattr(message, "message_id", "") or ""),
                "target_ids": other_ids,
                "target_user_id": other_ids[0] if other_ids else "",
                "points_to_other": points_to_other,
                "points_to_bot": points_to_bot,
                "judged": judged,
            }
        return latest

    @staticmethod
    def name_tokens(name: str) -> set[str]:
        text = re.sub(r"[（(].*?[）)]", " ", str(name or ""))
        tokens: set[str] = set()
        for part in re.split(r"[\s\-_|/／,，.。!！?？~～·]+", text):
            part = part.strip()
            if len(part) >= 2:
                tokens.add(part)
            if len(part) >= 4:
                tokens.add(part[-2:])
        return tokens

    @classmethod
    def followup_diverts_directed_reply(
        cls,
        current_message: Any,
        later_messages: list[Any],
        *,
        bot_id: str = "",
        bot_names: list[str] | None = None,
        known_users: dict[str, str] | None = None,
    ) -> str:
        """同一人在思考期间改口对别人说时，返回撤稿原因。"""
        later_messages = [message for message in (later_messages or []) if message is not None]
        if not later_messages:
            return ""
        bot_id = str(bot_id or "")
        bot_name_set = {
            str(name).strip()
            for name in (bot_names or [])
            if str(name).strip()
        }
        sender_id = str(getattr(current_message, "sender_id", "") or "")
        known_users = dict(known_users or {})
        other_name_tokens: set[str] = set()
        for user_id, user_name in known_users.items():
            if not user_id or user_id in {sender_id, bot_id}:
                continue
            other_name_tokens.update(cls.name_tokens(user_name))
        other_name_tokens -= bot_name_set

        for message in later_messages:
            if str(getattr(message, "sender_id", "") or "") != sender_id:
                continue
            explicit = []
            for value in getattr(message, "mentioned_user_ids", None) or []:
                value = str(value or "")
                if value and value.lower() not in {"all", "everyone"}:
                    explicit.append(value)
            reply_to_qq = str(getattr(message, "reply_to_qq", "") or "")
            if reply_to_qq:
                explicit.append(reply_to_qq)
            others = [
                target
                for target in dict.fromkeys(explicit)
                if target and target != bot_id
            ]
            if others:
                return "同一人思考期间@/回复了别人"
            judged = str(getattr(message, "conversation_target", "") or "")
            if judged == "other":
                return "同一人思考期间已被判断为对别人说"
            content = str(getattr(message, "content", "") or "").strip()
            if any(token and token in content for token in other_name_tokens):
                return "同一人思考期间点了别人的名字"
            compact = re.sub(r"[。！？!?…\s]+$", "", content)
            match = _VOCATIVE_RE.search(compact)
            vocative = match.group(1) if match else ""
            if vocative and vocative not in bot_name_set and (
                vocative in _OTHER_TITLES or vocative in other_name_tokens
            ):
                return f"同一人思考期间改口喊了{vocative}"
        return ""

    @staticmethod
    def _render_signals(signals: dict[str, Any]) -> str:
        entries = []
        for key in (
            "mentioned_me",
            "mentioned_user_ids",
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
        target_confidence = cls._parse_float(
            payload.get("target_confidence"), confidence
        )
        pragmatic_payload = payload.get("pragmatic_intent")
        pragmatic_confidence_value = payload.get("pragmatic_confidence")
        if isinstance(pragmatic_payload, dict) and pragmatic_confidence_value is None:
            pragmatic_confidence_value = pragmatic_payload.get("confidence")
        pragmatic_confidence = (
            None
            if pragmatic_confidence_value is None
            else cls._parse_float(pragmatic_confidence_value, confidence)
        )
        return ConversationJudgeResult(
            target=target,
            intent=intent,
            should_reply=should_reply,
            confidence=confidence,
            pragmatic_intent=cls._normalize_pragmatic_intent(
                pragmatic_payload, payload
            ),
            target_confidence=target_confidence,
            pragmatic_confidence=pragmatic_confidence,
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

    @classmethod
    def _normalize_pragmatic_intent(
        cls,
        value: Any,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """归一化语用字段，并兼容模型返回扁平字段或旧 JSON。"""
        payload = payload or {}
        raw = dict(value) if isinstance(value, dict) else {}
        if not raw:
            raw = {
                key: payload.get(key)
                for key in (
                    "act",
                    "pragmatic_act",
                    "tone",
                    "expected_reply",
                    "reply_strategy",
                    "literal",
                    "risk_flags",
                )
                if payload.get(key) is not None
            }

        def normalize_choice(
            raw_value: Any,
            allowed: set[str],
            aliases: dict[str, str],
        ) -> str:
            text = str(raw_value or "").strip().lower()
            text = aliases.get(text, text)
            return text if text in allowed else ""

        act = normalize_choice(
            raw.get("act") or raw.get("pragmatic_act"),
            cls.PRAGMATIC_ACTS,
            {
                "提问": "question",
                "问题": "question",
                "请求": "request",
                "回答": "answer",
                "续话": "follow_up",
                "附和": "acknowledge",
                "分享": "share",
                "调侃": "tease",
                "玩笑": "banter",
                "吐槽": "complaint",
                "抱怨": "complaint",
                "纠正": "correction",
                "夸奖": "praise",
                "安慰": "comfort",
                "挑战": "challenge",
                "身份试探": "identity_test",
                "沉默": "silence",
            },
        )
        tone = normalize_choice(
            raw.get("tone"),
            cls.PRAGMATIC_TONES,
            {
                "中性": "neutral",
                "认真": "serious",
                "严肃": "serious",
                "轻松": "playful",
                "玩笑": "playful",
                "讽刺": "sarcastic",
                "敌意": "hostile",
                "不确定": "uncertain",
            },
        )
        expected_reply = normalize_choice(
            raw.get("expected_reply") or raw.get("reply_strategy"),
            cls.EXPECTED_REPLIES,
            {
                "直接回答": "direct_answer",
                "短反应": "short_reaction",
                "接梗": "short_banter",
                "短接梗": "short_banter",
                "澄清": "clarify",
                "缓和": "defuse",
                "沉默": "silent",
            },
        )

        normalized: dict[str, Any] = {}
        if act:
            normalized["act"] = act
        if tone:
            normalized["tone"] = tone
        if expected_reply:
            normalized["expected_reply"] = expected_reply

        literal_value = raw.get("literal")
        if literal_value is not None:
            normalized["literal"] = cls._parse_bool(literal_value, False)

        flags = raw.get("risk_flags")
        if isinstance(flags, str):
            flags = re.split(r"[,，、/\s]+", flags)
        if not isinstance(flags, (list, tuple, set)):
            flags = []
        flag_aliases = {
            "身份诱导": "identity_bait",
            "身份试探": "identity_bait",
            "指代不清": "ambiguous_addressee",
            "无依据": "unsupported_assumption",
            "升级冲突": "escalation_risk",
        }
        normalized_flags = []
        for flag in flags:
            text = str(flag or "").strip().lower()
            text = flag_aliases.get(text, text)
            if text in cls.RISK_FLAGS and text not in normalized_flags:
                normalized_flags.append(text)
        if normalized_flags:
            normalized["risk_flags"] = normalized_flags
        return normalized

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

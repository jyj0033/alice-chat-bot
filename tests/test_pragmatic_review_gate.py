"""语义复核的「语用意图不匹配」不再一票否决。

线上 2026-11-13 13:32 群里有人 @爱丽丝 问小说推荐，Bot 草稿是

    怕报错书名：你更偏幕后布局还是灵气复苏？

这是向群友提问补全信息，是正确接话（复核提示词里也这么写着），
但复核器把它判成 strategy_mismatch，首轮闸拦下 → 重答 → 重答复核又拦 →
「重答未通过，放弃本轮回复」→ 群里一条没发。

这里锁两件事：
1. strategy_mismatch（以及没给出任何类型的情况）不再一票否决；
2. 四种「明确错误」类型仍然照拦，别把闸门拆没了。
"""

import json
import unittest

from core.adapter.base import Message
from modules.llm.base import ChatResponse
from modules.social import conversation_judge as judge_module
from modules.social.conversation_judge import (
    ConversationJudge,
    HARD_PRAGMATIC_TYPES,
    pragmatic_mismatch_verdict,
)

# 线上 13:32 那条被 @ 的真实草稿
REAL_DRAFT = "怕报错书名：你更偏幕后布局还是灵气复苏？"
REAL_QUESTION = "有类似我在东京掀起百鬼夜行这种日式，幕后黑手，灵气复苏流的小说嘛？"


class _Provider:
    model = "offline-pragmatic-gate-test"

    def __init__(self, content):
        self.content = content
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        return ChatResponse(content=self.content, model=self.model)


def _message(content=REAL_QUESTION):
    return Message(
        message_id="current",
        message_type="group",
        group_id="g1",
        sender_id="u1",
        sender_name="龍飛月",
        content=content,
    )


def _payload(**overrides):
    base = {
        "is_paraphrase": False,
        "adds_information": True,
        "unsupported_assumption": False,
        "needs_clarification": False,
        "incomplete": False,
        "meta_commentary": False,
        "on_topic": True,
        "pragmatic_mismatch": False,
        "mismatch_types": [],
    }
    base.update(overrides)
    return base


class PragmaticVerdictTests(unittest.TestCase):
    """共用策略函数本身的判定。"""

    def test_strategy_mismatch_is_not_a_veto(self):
        blocked, soft = pragmatic_mismatch_verdict(
            {"pragmatic_mismatch": True, "mismatch_types": ["strategy_mismatch"]}
        )
        self.assertFalse(blocked)
        self.assertTrue(soft)

    def test_hard_types_still_block(self):
        for kind in sorted(HARD_PRAGMATIC_TYPES):
            with self.subTest(kind=kind):
                blocked, soft = pragmatic_mismatch_verdict(
                    {"pragmatic_mismatch": True, "mismatch_types": [kind]}
                )
                self.assertTrue(blocked)
                self.assertFalse(soft)

    def test_hard_type_wins_over_strategy_mismatch(self):
        blocked, soft = pragmatic_mismatch_verdict(
            {
                "pragmatic_mismatch": True,
                "mismatch_types": ["strategy_mismatch", "tone_drift"],
            }
        )
        self.assertTrue(blocked)
        self.assertFalse(soft)

    def test_mismatch_without_types_is_soft(self):
        blocked, soft = pragmatic_mismatch_verdict({"pragmatic_mismatch": True})
        self.assertFalse(blocked)
        self.assertTrue(soft)

    def test_no_mismatch_is_neither_blocked_nor_soft(self):
        blocked, soft = pragmatic_mismatch_verdict({"pragmatic_mismatch": False})
        self.assertFalse(blocked)
        self.assertFalse(soft)

    def test_types_given_as_a_string(self):
        blocked, soft = pragmatic_mismatch_verdict(
            {"pragmatic_mismatch": True, "mismatch_types": "tone_drift"}
        )
        self.assertTrue(blocked)
        self.assertFalse(soft)

    def test_empty_evidence(self):
        blocked, soft = pragmatic_mismatch_verdict(None)
        self.assertFalse(blocked)
        self.assertFalse(soft)


class ReviewReplyGateTests(unittest.IsolatedAsyncioTestCase):
    """复核器这一层：strategy_mismatch 不能再把 should_reply 打成 False。"""

    async def _review(self, payload, reply=REAL_DRAFT):
        provider = _Provider(json.dumps(payload, ensure_ascii=False))
        judge = ConversationJudge(provider)
        result = await judge.review_reply(_message(), [], reply, direction="group")
        return result

    async def test_strategy_mismatch_only_keeps_the_draft(self):
        result = await self._review(
            _payload(pragmatic_mismatch=True, mismatch_types=["strategy_mismatch"])
        )
        self.assertTrue(result.available)
        self.assertTrue(result.should_reply)
        self.assertTrue(result.evidence["pragmatic_mismatch"])
        self.assertFalse(result.evidence["pragmatic_hard_mismatch"])

    async def test_hard_mismatch_type_still_rejects(self):
        result = await self._review(
            _payload(pragmatic_mismatch=True, mismatch_types=["tone_drift"])
        )
        self.assertTrue(result.available)
        self.assertFalse(result.should_reply)
        self.assertTrue(result.evidence["pragmatic_hard_mismatch"])

    async def test_mismatch_without_types_is_soft(self):
        result = await self._review(_payload(pragmatic_mismatch=True))
        self.assertTrue(result.should_reply)
        self.assertFalse(result.evidence["pragmatic_hard_mismatch"])

    async def test_clarification_question_is_not_a_quality_defect(self):
        """13:32 那条草稿：含「书名」+ 问号 → 澄清型反问，不算草稿自身缺陷。"""
        result = await self._review(
            _payload(
                is_paraphrase=True,
                adds_information=False,
                needs_clarification=True,
                pragmatic_mismatch=True,
                mismatch_types=["strategy_mismatch"],
            )
        )
        self.assertTrue(result.evidence["clarification_question"])
        self.assertFalse(result.evidence["needs_clarification"])
        self.assertFalse(result.evidence["is_paraphrase"])
        self.assertTrue(result.should_reply)

    async def test_real_draft_passes_the_gate_even_when_flagged(self):
        """线上那条草稿 + 复核器给 strategy_mismatch → 必须放行。"""
        result = await self._review(
            _payload(pragmatic_mismatch=True, mismatch_types=["strategy_mismatch"]),
            reply=REAL_DRAFT,
        )
        self.assertTrue(result.should_reply)


class RetryGateTests(unittest.IsolatedAsyncioTestCase):
    """重答复核那道闸同样不能被 strategy_mismatch 打掉。"""

    async def test_retry_review_accepts_strategy_mismatch_only(self):
        from main import GroupChatBot

        provider = _Provider(
            json.dumps(
                _payload(pragmatic_mismatch=True, mismatch_types=["strategy_mismatch"]),
                ensure_ascii=False,
            )
        )
        judge = ConversationJudge(provider)
        accepted, failure = await GroupChatBot._validate_retried_reply(
            judge, _message(), [], REAL_DRAFT, "group", None
        )
        self.assertTrue(accepted)
        self.assertEqual(failure, "")

    async def test_retry_review_rejects_hard_mismatch(self):
        from main import GroupChatBot

        provider = _Provider(
            json.dumps(
                _payload(pragmatic_mismatch=True, mismatch_types=["identity_exposure"]),
                ensure_ascii=False,
            )
        )
        judge = ConversationJudge(provider)
        accepted, failure = await GroupChatBot._validate_retried_reply(
            judge, _message(), [], REAL_DRAFT, "group", None
        )
        self.assertFalse(accepted)
        self.assertTrue(failure)


class GateWiringTests(unittest.TestCase):
    """两道闸必须共用同一份策略，别各自抄一份。"""

    def test_main_uses_the_shared_verdict(self):
        import main as main_module

        self.assertIs(
            main_module.pragmatic_mismatch_verdict,
            judge_module.pragmatic_mismatch_verdict,
        )


if __name__ == "__main__":
    unittest.main()

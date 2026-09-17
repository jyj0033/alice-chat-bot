"""二次生成后的回复复核测试。"""

import unittest
from types import SimpleNamespace

from main import GroupChatBot


class _FakeJudge:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def review_reply(self, message, recent_messages, reply, *, direction):
        self.calls.append((message, recent_messages, reply, direction))
        return self.result


class RetryReplyValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_a_retry_marked_as_meta_commentary(self):
        judge = _FakeJudge(SimpleNamespace(
            available=True,
            should_reply=False,
            reason="回复过程说明",
            evidence={"meta_commentary": True},
        ))

        accepted, reason = await GroupChatBot._validate_retried_reply(
            judge,
            object(),
            [],
            '先把"今天又老了一岁"和生日这事理',
            "group",
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "回复过程说明")
        self.assertEqual(len(judge.calls), 1)

    async def test_fails_closed_when_retry_review_is_unavailable(self):
        judge = _FakeJudge(SimpleNamespace(
            available=False,
            should_reply=False,
            reason="provider_unavailable",
            evidence={},
        ))

        accepted, reason = await GroupChatBot._validate_retried_reply(
            judge, object(), [], "我不太确定", "group"
        )

        self.assertFalse(accepted)
        self.assertEqual(reason, "重答复核不可用")

    async def test_accepts_a_retry_that_passes_review(self):
        judge = _FakeJudge(SimpleNamespace(
            available=True,
            should_reply=True,
            reason="",
            evidence={},
        ))

        accepted, reason = await GroupChatBot._validate_retried_reply(
            judge, object(), [], "今天真是生日吗？", "group"
        )

        self.assertTrue(accepted)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()

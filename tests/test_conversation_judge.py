"""群聊目标判断器的离线测试。"""

import unittest

from core.adapter.base import Message
from modules.llm.base import ChatResponse
from modules.memory.context import ContextMessage
from modules.social.conversation_judge import (
    ConversationJudge,
)


class _FakeProvider:
    model = "judge-test-model"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        content = self.responses.pop(0)
        return ChatResponse(content=content, model=self.model)


class ConversationJudgeTests(unittest.IsolatedAsyncioTestCase):
    def _message(self, **kwargs):
        values = {
            "message_id": "m2",
            "message_type": "group",
            "sender_id": "u2",
            "sender_name": "小红",
            "group_id": "g1",
            "content": "你觉得这个怎么改？",
            "mentioned_user_ids": ["u1"],
            "reply_to_id": "m1",
            "reply_to_qq": "u1",
        }
        values.update(kwargs)
        return Message(**values)

    async def test_judge_parses_json_and_keeps_relationship_evidence(self):
        provider = _FakeProvider(
            '{"target":"other","intent":"add_info",'
            '"should_reply":true,"confidence":0.82,'
            '"reference_message_id":"m1","target_user_id":"u1",'
            '"reason":"在给小明补充方案"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message()
        history = [
            ContextMessage(
                sender_id="u1",
                sender_name="小明",
                content="这个报错怎么改",
                message_id="m1",
            ),
            current,
        ]

        result = await judge.judge(
            current,
            history,
            heuristic_signals={
                "mentioned_others": ["u1"],
                "reply_to_qq": "u1",
            },
        )

        self.assertTrue(result.available)
        self.assertEqual(result.target, "other")
        self.assertEqual(result.intent, "add_info")
        self.assertTrue(result.should_reply)
        self.assertEqual(result.reference_message_id, "m1")
        self.assertEqual(result.target_user_id, "u1")
        prompt = provider.requests[0].messages[-1].content
        self.assertIn("小明", prompt)
        self.assertIn("回复=m1/u1", prompt)
        self.assertIn("@=u1", prompt)

    async def test_invalid_judge_output_falls_back_as_unavailable(self):
        provider = _FakeProvider("这不是 JSON，也没有结构化结果")
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.judge(self._message(), [])

        self.assertFalse(result.available)
        self.assertIn("invalid_json", result.error)

    async def test_prompt_keeps_previous_explicit_target_for_same_sender(self):
        provider = _FakeProvider(
            '{"target":"unknown","intent":"silent",'
            '"should_reply":false,"confidence":0.9,"reason":"指向不明"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m3",
            sender_id="u2",
            sender_name="小红",
            content="这个男人好像在夸你什么",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )
        history = [
            ContextMessage(
                sender_id="u1",
                sender_name="小明",
                content="发张图看看",
                message_id="m0",
            ),
            ContextMessage(
                sender_id="u2",
                sender_name="小红",
                content="[无法识别的消息]",
                message_id="m1",
                mentioned_user_ids=("u1",),
            ),
            ContextMessage(
                sender_id="u2",
                sender_name="小红",
                content="[图片]",
                message_id="m2",
            ),
            current,
        ]

        result = await judge.judge(current, history)

        self.assertTrue(result.evidence["target_continuity"])
        prompt = provider.requests[0].messages[-1].content
        self.assertIn("@=u1(小明)", prompt)
        self.assertIn("最近一次明确@/回复的对象", prompt)
        self.assertIn("不能因为出现‘你’", prompt)

    async def test_invalid_review_output_does_not_claim_a_review(self):
        provider = _FakeProvider("暂时看不懂")
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(), [], "你这个问题可以这样处理", direction="group"
        )

        self.assertFalse(result.available)

    async def test_review_paraphrase_result_is_structured(self):
        provider = _FakeProvider(
            '{"is_paraphrase":true,"adds_information":false,'
            '"replacement_hint":"只是换句话说"}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(), [], "换句话说就是这个意思", direction="group"
        )

        self.assertTrue(result.available)
        self.assertFalse(result.should_reply)
        self.assertEqual(result.intent, "silent")
        self.assertTrue(result.evidence["is_paraphrase"])

    async def test_review_flags_unsupported_media_assumption(self):
        provider = _FakeProvider(
            '{"is_paraphrase":false,"adds_information":true,'
            '"unsupported_assumption":true,"needs_clarification":true,'
            '"replacement_hint":"图片内容未确认"}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(content="这个男人好像在夸你什么"),
            [
                ContextMessage(
                    sender_id="u2",
                    sender_name="小红",
                    content="[图片]",
                    message_id="m1",
                )
            ],
            "谢了哈～",
            direction="group",
        )

        self.assertTrue(result.available)
        self.assertFalse(result.should_reply)
        self.assertTrue(result.evidence["unsupported_assumption"])
        self.assertTrue(result.evidence["needs_clarification"])

    async def test_review_meme_send_result_is_structured(self):
        provider = _FakeProvider(
            '{"should_send_meme":true,"confidence":0.91,"reason":"和当前吐槽很贴"}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_meme_send(
            self._message(),
            [],
            {
                "id": "meme-1",
                "category": "吐槽",
                "meaning": "翻白眼看热闹",
                "tags": ["自动收集"],
            },
            reply="这也太离谱了",
            direction="group",
        )

        self.assertTrue(result.available)
        self.assertTrue(result.should_reply)
        self.assertEqual(result.intent, "react")
        self.assertTrue(result.evidence["should_send_meme"])
        prompt = provider.requests[0].messages[-1].content
        self.assertIn("翻白眼看热闹", prompt)
        self.assertIn("这也太离谱了", prompt)

    async def test_invalid_meme_review_output_is_not_approved(self):
        provider = _FakeProvider('{"reason":"无法判断"}')
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_meme_send(
            self._message(),
            [],
            {"id": "meme-1", "category": "吐槽", "meaning": "看热闹"},
        )

        self.assertFalse(result.available)

    def test_parse_result_accepts_partial_but_meaningful_json(self):
        result = ConversationJudge.parse_result(
            '{"target":"bot","intent":"answer"}'
        )

        self.assertEqual(result.target, "bot")
        self.assertTrue(result.should_reply)

    async def test_continuity_vetoes_false_bot_target_without_mention(self):
        """同一人刚在回复别人时，不能因为「你都认识？」就改判成问 Bot。"""
        provider = _FakeProvider(
            '{"target":"bot","intent":"answer","should_reply":true,'
            '"confidence":0.85,"reference_message_id":"pic1",'
            '"reason":"上一句发图，结合语境是在问Alice"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m3",
            sender_id="u2",
            sender_name="luguan",
            content="这里面出现的你都认识？",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )
        history = [
            ContextMessage(
                sender_id="u3",
                sender_name="Silver_Wing",
                content="[图片]",
                message_id="pic1",
            ),
            ContextMessage(
                sender_id="u1",
                sender_name="江雨衿",
                content="怎么没看见无职",
                message_id="m1",
            ),
            ContextMessage(
                sender_id="u2",
                sender_name="luguan",
                content="不知道",
                message_id="m2",
                reply_to_id="m1",
                reply_to_qq="u1",
            ),
            current,
        ]

        result = await judge.judge(current, history)

        self.assertTrue(result.available)
        self.assertNotEqual(result.target, "bot")
        self.assertFalse(result.should_reply)
        self.assertTrue(result.evidence.get("continuity_veto"))
        self.assertEqual(result.reference_message_id, "m3")

    async def test_judged_other_target_also_vetoes_false_bot(self):
        """即使没有 reply 段，上一句动态判断已是 other 也要否决。"""
        provider = _FakeProvider(
            '{"target":"bot","intent":"answer","should_reply":true,'
            '"confidence":0.9,"reason":"在问Alice"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m3",
            sender_id="u2",
            sender_name="luguan",
            content="这里面出现的你都认识？",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )
        history = [
            ContextMessage(
                sender_id="u2",
                sender_name="luguan",
                content="不知道",
                message_id="m2",
                conversation_target="other",
            ),
            current,
        ]

        result = await judge.judge(current, history)

        self.assertFalse(result.should_reply)
        self.assertNotEqual(result.target, "bot")
        self.assertTrue(result.evidence.get("continuity_veto"))

    async def test_explicit_bot_mention_is_not_vetoed(self):
        """当前句真的 @ 了 Bot，即使上一句在对别人说，也允许回答。"""
        provider = _FakeProvider(
            '{"target":"bot","intent":"answer","should_reply":true,'
            '"confidence":0.9,"reference_message_id":"pic1","reason":"点名Alice"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m3",
            sender_id="u2",
            sender_name="luguan",
            content="爱丽丝你看这个",
            mentioned_user_ids=["bot"],
            reply_to_id=None,
            reply_to_qq=None,
        )
        history = [
            ContextMessage(
                sender_id="u2",
                sender_name="luguan",
                content="不知道",
                message_id="m2",
                reply_to_qq="u1",
            ),
            current,
        ]

        result = await judge.judge(
            current,
            history,
            heuristic_signals={"mentioned_me": True},
        )

        self.assertEqual(result.target, "bot")
        self.assertTrue(result.should_reply)
        self.assertFalse(result.evidence.get("continuity_veto"))
        self.assertEqual(result.reference_message_id, "m3")
        self.assertEqual(result.evidence.get("dropped_reference_message_id"), "pic1")

    async def test_bot_reference_is_pinned_to_current_message(self):
        """对 Bot 的引用不能落在旁边的图片上。"""
        provider = _FakeProvider(
            '{"target":"bot","intent":"answer","should_reply":true,'
            '"confidence":0.8,"reference_message_id":"pic1","reason":"在问图"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m2",
            sender_id="u2",
            sender_name="小红",
            content="爱丽丝这是谁",
            mentioned_user_ids=["bot"],
            reply_to_id=None,
            reply_to_qq=None,
        )
        history = [
            ContextMessage(
                sender_id="u3",
                sender_name="Silver_Wing",
                content="[图片]",
                message_id="pic1",
            ),
            current,
        ]

        result = await judge.judge(
            current,
            history,
            heuristic_signals={"mentioned_me": True},
        )

        self.assertEqual(result.target, "bot")
        self.assertEqual(result.reference_message_id, "m2")

    def test_followup_vocative_diverts_directed_reply(self):
        current = self._message(
            message_id="m3",
            sender_id="u2",
            sender_name="luguan",
            content="这里面出现的你都认识？",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )
        later = [
            ContextMessage(
                sender_id="u2",
                sender_name="luguan",
                content="那你也可以进去了团长",
                message_id="m4",
            )
        ]
        reason = ConversationJudge.followup_diverts_directed_reply(
            current,
            later,
            bot_id="bot",
            bot_names=["爱丽丝", "小艾"],
            known_users={"u1": "江雨衿", "u2": "luguan"},
        )
        self.assertIn("团长", reason)

    def test_followup_other_name_diverts_directed_reply(self):
        current = self._message(
            message_id="m3",
            sender_id="u2",
            sender_name="luguan",
            content="这里面出现的你都认识？",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )
        later = [
            ContextMessage(
                sender_id="u2",
                sender_name="luguan",
                content="雨衿你看这个",
                message_id="m4",
            )
        ]
        reason = ConversationJudge.followup_diverts_directed_reply(
            current,
            later,
            bot_id="bot",
            bot_names=["爱丽丝"],
            known_users={"u1": "江       雨衿（永别了，牢笼！）", "u2": "luguan"},
        )
        self.assertIn("名字", reason)

    def test_allows_probabilistic_interjection_for_group_and_unknown(self):
        group = {
            "available": True,
            "target": "group",
            "should_reply": False,
            "evidence": {},
        }
        unknown = {
            "available": True,
            "target": "unknown",
            "should_reply": False,
            "evidence": {},
        }
        self.assertTrue(ConversationJudge.allows_probabilistic_interjection(group))
        self.assertTrue(ConversationJudge.allows_probabilistic_interjection(unknown))

    def test_allows_probabilistic_interjection_blocks_other_and_veto(self):
        other = {
            "available": True,
            "target": "other",
            "should_reply": False,
            "evidence": {},
        }
        vetoed = {
            "available": True,
            "target": "unknown",
            "should_reply": False,
            "evidence": {"continuity_veto": True},
        }
        bot_silent = {
            "available": True,
            "target": "bot",
            "should_reply": False,
            "evidence": {},
        }
        replied_other = {
            "available": True,
            "target": "group",
            "should_reply": False,
            "evidence": {},
        }
        self.assertFalse(ConversationJudge.allows_probabilistic_interjection(other))
        self.assertFalse(ConversationJudge.allows_probabilistic_interjection(vetoed))
        self.assertFalse(ConversationJudge.allows_probabilistic_interjection(bot_silent))
        self.assertFalse(
            ConversationJudge.allows_probabilistic_interjection(
                replied_other, reply_to_other=True
            )
        )
        self.assertFalse(
            ConversationJudge.allows_probabilistic_interjection(
                replied_other, mentioned_others=True
            )
        )

    async def test_system_prompt_allows_group_chime_in(self):
        provider = _FakeProvider(
            '{"target":"group","intent":"acknowledge","should_reply":true,'
            '"confidence":0.7,"reason":"群里打招呼"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m1",
            content="早",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )

        result = await judge.judge(current, [])

        system = provider.requests[0].messages[0].content
        self.assertIn("面向全群", system)
        self.assertIn("打招呼", system)
        self.assertIn("开放提问", system)
        self.assertNotIn("证据不足，优先 target=unknown", system)
        self.assertEqual(result.target, "group")
        self.assertTrue(result.should_reply)

    def test_unrelated_later_message_does_not_divert(self):
        current = self._message(
            message_id="m3",
            sender_id="u2",
            sender_name="luguan",
            content="爱丽丝帮我看看",
            mentioned_user_ids=["bot"],
            reply_to_id=None,
            reply_to_qq=None,
        )
        later = [
            ContextMessage(
                sender_id="u2",
                sender_name="luguan",
                content="就是刚才那个报错",
                message_id="m4",
            )
        ]
        reason = ConversationJudge.followup_diverts_directed_reply(
            current,
            later,
            bot_id="bot",
            bot_names=["爱丽丝"],
            known_users={"u1": "江雨衿", "u2": "luguan"},
        )
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()

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

    async def test_playful_inclusion_of_alice_allows_a_short_reaction(self):
        provider = _FakeProvider(
            '{"target":"other","intent":"silent",'
            '"should_reply":false,"confidence":0.9,"reason":"在接别人话"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m3",
            content="应该连带着爱丽丝一起处刑~",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )

        result = await judge.judge(current, [])

        self.assertTrue(result.should_reply)
        self.assertEqual(result.target, "group")
        self.assertEqual(result.intent, "react")
        self.assertTrue(result.evidence["playful_bot_inclusion"])
        self.assertEqual(result.reference_message_id, "m3")
        prompt = provider.requests[0].messages[-1].content
        self.assertIn("连带着爱丽丝一起处刑", prompt)

    async def test_plain_third_person_bot_mention_stays_silent(self):
        provider = _FakeProvider(
            '{"target":"other","intent":"silent",'
            '"should_reply":false,"confidence":0.9,"reason":"群友之间的对话"}'
        )
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = self._message(
            message_id="m3",
            content="宫园薰说爱丽丝今天不在线。",
            mentioned_user_ids=[],
            reply_to_id=None,
            reply_to_qq=None,
        )

        result = await judge.judge(current, [])

        self.assertFalse(result.should_reply)
        self.assertEqual(result.target, "other")
        self.assertNotIn("playful_bot_inclusion", result.evidence)

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

    async def test_clarification_question_is_not_flagged_as_needing_clarification(self):
        """用户请求本身信息不足时，反问澄清是正确接话，不能被判「需要澄清」。

        生产事故：@ 机器人「爱丽丝，帮我找个小说」，草稿「找哪本？书名或剧情
        关键词给我发下」被判 needs_clarification 打回，重答还是反问又被打回，
        整轮静默。
        """
        provider = _FakeProvider(
            '{"is_paraphrase":false,"adds_information":false,'
            '"needs_clarification":true,'
            '"replacement_hint":"请提供书名或剧情关键词"}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(content="爱丽丝，帮我找个小说"),
            [],
            "找哪本？书名或剧情关键词给我发下",
            direction="to_bot",
        )

        self.assertTrue(result.available)
        self.assertTrue(result.should_reply)
        self.assertEqual(result.intent, "react")
        self.assertFalse(result.evidence["needs_clarification"])
        self.assertTrue(result.evidence["clarification_question"])

    async def test_clarification_question_exempts_paraphrase_flag(self):
        """澄清问句天然没新增信息，不能因为「复读」再被判死一次。"""
        provider = _FakeProvider(
            '{"is_paraphrase":true,"adds_information":false,'
            '"needs_clarification":false,"replacement_hint":""}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(content="爱丽丝，帮我找个小说"),
            [],
            "书名叫啥？或者讲了啥剧情，你说说看",
            direction="to_bot",
        )

        self.assertTrue(result.available)
        self.assertTrue(result.should_reply)
        self.assertFalse(result.evidence["is_paraphrase"])
        self.assertTrue(result.evidence["adds_information"])

    async def test_vague_draft_still_needs_clarification(self):
        """含糊、没有问句的草稿仍然要被打回，豁免不能放宽成什么都放行。"""
        provider = _FakeProvider(
            '{"is_paraphrase":false,"adds_information":true,'
            '"needs_clarification":true,"replacement_hint":"说清楚是哪本"}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(content="爱丽丝，帮我找个小说"),
            [],
            "那个东西不太行",
            direction="to_bot",
        )

        self.assertTrue(result.available)
        self.assertFalse(result.should_reply)
        self.assertTrue(result.evidence["needs_clarification"])
        self.assertFalse(result.evidence["clarification_question"])

    async def test_review_prompt_scopes_needs_clarification_to_the_draft(self):
        """提示词要写清 needs_clarification 只针对草稿自身。"""
        provider = _FakeProvider(
            '{"is_paraphrase":false,"adds_information":true,'
            '"needs_clarification":false,"replacement_hint":""}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        await judge.review_reply(
            self._message(), [], "行", direction="group"
        )

        prompt = provider.requests[0].messages[-1].content
        self.assertIn("needs_clarification 只针对", prompt)
        self.assertIn("是正确接话", prompt)

    def test_clarification_question_detection(self):
        """澄清型反问的识别：必须同时含问号和疑问词。"""
        self.assertTrue(
            ConversationJudge._looks_like_clarification_question(
                "找哪本？书名或剧情关键词给我发下"
            )
        )
        self.assertTrue(
            ConversationJudge._looks_like_clarification_question(
                "想找什么类型的小说？"
            )
        )
        self.assertFalse(
            ConversationJudge._looks_like_clarification_question(
                "这顿看着不错，人均得多少"
            )
        )
        self.assertFalse(
            ConversationJudge._looks_like_clarification_question("那就没意思了")
        )
        self.assertFalse(ConversationJudge._looks_like_clarification_question(""))

    async def test_review_prompt_forbids_turning_hedged_identity_into_fact(self):
        """视觉识别给出的身份带「疑似」，复核必须拦住把它说成确证。

        保守规则：摘要写「疑似《崩坏3》的琪亚娜」，草稿就不能说「这是琪亚娜」。
        """
        provider = _FakeProvider(
            '{"is_paraphrase":false,"adds_information":true,'
            '"unsupported_assumption":true,"needs_clarification":true,'
            '"replacement_hint":"身份只是疑似"}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        await judge.review_reply(
            self._message(content="看看这个"),
            [
                ContextMessage(
                    sender_id="u2",
                    sender_name="小红",
                    content="一张图，画面是白发绿眼角色在做憋笑表情（看不出具体是谁）。",
                    message_id="m1",
                )
            ],
            "这不是琪亚娜吗～",
            direction="group",
        )

        sent = "\n".join(
            str(getattr(message, "content", "")) for message in provider.requests[0].messages
        )
        self.assertIn("疑似/看起来像/可能是/看不出具体是谁", sent)
        self.assertIn("都是没认准", sent)
        self.assertIn("草稿就不能说", sent)

    async def test_review_flags_meta_commentary_and_incomplete_reply(self):
        provider = _FakeProvider(
            '{"is_paraphrase":false,"adds_information":true,'
            '"unsupported_assumption":false,"needs_clarification":false,'
            '"incomplete":true,"meta_commentary":true,"on_topic":true,'
            '"replacement_hint":"直接接当前话题"}'
        )
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(content="大家在聊游戏"),
            [],
            "他们聊游戏的，没我事。",
            direction="group",
        )

        self.assertTrue(result.available)
        self.assertFalse(result.should_reply)
        self.assertTrue(result.evidence["incomplete"])
        self.assertTrue(result.evidence["meta_commentary"])
        prompt = provider.requests[0].messages[-1].content
        self.assertIn("换一种措辞仍然算元话语", prompt)

    async def test_review_payload_without_quality_fields_is_unavailable(self):
        provider = _FakeProvider('{"replacement_hint":"看起来还行"}')
        judge = ConversationJudge(provider, bot_id="bot")

        result = await judge.review_reply(
            self._message(), [], "看起来还行", direction="group"
        )

        self.assertFalse(result.available)

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

    async def test_judged_other_target_is_not_reused_as_hard_evidence(self):
        """模型自己的旧推断不能不断续期、覆盖本轮的语义判断。"""
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

        self.assertTrue(result.should_reply)
        self.assertEqual(result.target, "bot")
        self.assertFalse(result.evidence.get("continuity_veto"))

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


class ExplicitAddressOverrideTests(unittest.IsolatedAsyncioTestCase):
    """明确 @ 机器人却被打上 intent=silent 时，不能整轮沉默。

    生产上真实踩过：模型判 target=bot、理由写着「适合轻松回应」，却把
    intent 标成 silent；代码里 intent=silent 会强制 should_reply=False，
    于是明确点名的人被晾在一边。真实回放 50 条被 @ 的消息，9 条落进这个坑，
    其中 8 条语用动作是调侃/纠正/夸赞/请求（本来就该回），只有 1 条是
    「[无法识别的消息]」（确实不该回）。
    """

    def _message(self, **kwargs):
        values = {
            "message_id": "m9",
            "message_type": "group",
            "sender_id": "u2",
            "sender_name": "小红",
            "group_id": "g1",
            "content": "我爱你",
            "mentioned_me": True,
            "mentioned_user_ids": ["bot"],
            "reply_to_id": None,
            "reply_to_qq": None,
        }
        values.update(kwargs)
        return Message(**values)

    def _payload(self, act, expected_reply, tone="playful"):
        return (
            '{"target":"bot","intent":"silent","should_reply":false,'
            '"confidence":0.94,"target_confidence":1.0,"pragmatic_confidence":0.94,'
            f'"pragmatic_intent":{{"act":"{act}","tone":"{tone}",'
            f'"expected_reply":"{expected_reply}"}},'
            '"reason":"当前消息明确@爱丽丝并表达爱意，适合轻松回应。"}'
        )

    async def _judge(self, payload, message=None, signals=None):
        provider = _FakeProvider(payload)
        judge = ConversationJudge(provider, bot_id="bot", bot_name="爱丽丝")
        current = message if message is not None else self._message()
        result = await judge.judge(
            current,
            [current],
            heuristic_signals=signals if signals is not None else {"mentioned_me": True},
        )
        return result, provider

    async def test_tease_with_reply_expected_is_not_silenced(self):
        result, _ = await self._judge(self._payload("tease", "short_banter"))
        self.assertTrue(result.should_reply)
        self.assertEqual(result.intent, "react")
        self.assertEqual(result.target, "bot")
        self.assertTrue(result.evidence.get("explicit_address_override"))

    async def test_request_with_direct_answer_expected_is_not_silenced(self):
        result, _ = await self._judge(self._payload("request", "direct_answer", tone="neutral"))
        self.assertTrue(result.should_reply)
        self.assertEqual(result.intent, "react")

    async def test_unrecognizable_message_stays_silent(self):
        # act=silence、expected_reply=silent：内容都没识别出来，回了也是瞎回
        result, _ = await self._judge(self._payload("silence", "silent", tone="uncertain"))
        self.assertFalse(result.should_reply)
        self.assertEqual(result.intent, "silent")
        self.assertFalse(result.evidence.get("explicit_address_override"))

    async def test_expected_reply_silent_stays_silent(self):
        result, _ = await self._judge(self._payload("banter", "silent"))
        self.assertFalse(result.should_reply)
        self.assertEqual(result.intent, "silent")

    async def test_third_person_mention_is_not_overridden(self):
        # 只是正文里出现「爱丽丝」，不等于点名她
        message = self._message(
            mentioned_me=False,
            mentioned_user_ids=[],
            content="爱丽丝刚才说的那个梗挺好笑",
        )
        result, _ = await self._judge(
            self._payload("tease", "short_banter"), message=message, signals={}
        )
        self.assertFalse(result.should_reply)
        self.assertEqual(result.intent, "silent")

    async def test_other_target_is_not_overridden(self):
        payload = (
            '{"target":"other","intent":"silent","should_reply":false,'
            '"confidence":0.9,"pragmatic_intent":{"act":"tease","tone":"playful",'
            '"expected_reply":"short_banter"},"reason":"在跟别人说话"}'
        )
        result, _ = await self._judge(payload)
        self.assertFalse(result.should_reply)
        self.assertEqual(result.target, "other")

    async def test_closing_acknowledgement_is_untouched(self):
        # intent=acknowledge、should_reply=false 是模型明确的收尾判断，不归兜底管
        payload = (
            '{"target":"bot","intent":"acknowledge","should_reply":false,'
            '"confidence":0.9,"pragmatic_intent":{"act":"acknowledge","tone":"neutral",'
            '"expected_reply":"silent"},"reason":"好，已构成收束"}'
        )
        result, _ = await self._judge(payload)
        self.assertFalse(result.should_reply)
        self.assertEqual(result.intent, "acknowledge")

    async def test_system_prompt_states_explicit_requirement(self):
        result, provider = await self._judge(self._payload("tease", "short_banter"))
        system = provider.requests[0].messages[0].content
        self.assertIn("intent 不许用 silent", system)
        self.assertIn("明确 @/回复 Alice", system)


if __name__ == "__main__":
    unittest.main()

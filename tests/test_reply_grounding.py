# -*- coding: utf-8 -*-
"""回复接地性回归：看不见图却断言具体内容、复核挂掉就放行、切题却答错问题。

三条都是线上真实事故驱动的：

1. 14:07 收到「一张图，看不清画面」后回「国庆在家吃鸳鸯锅，爽啊」——鸳鸯锅是编的，
   群友当场纠正说不是鸳鸯锅；20:15「你那个108的早坏了，这臭表倒是不错」同类。
2. 20:15:29 上游返回 529，语义复核整层不可用，首轮草稿**未经任何复核直接发到群里**
   （重答复核早就按「不可用即不通过」处理，首轮却是 `available and (...)`）。
3. 11:21 群友问「第二次摧城怎么弄出来的？」（问大招机制），草稿答配队怎么搭——话题对、
   答案错，而复核器的 on_topic 仍然是 true。
"""

import json
import unittest
from types import SimpleNamespace

from core.adapter.base import Message
from core.adapter.rich_content import describe_media_in_words
from main import GroupChatBot
from modules.llm.base import ChatResponse
from modules.reply.generator import ReplyGenerator
from modules.social.conversation_judge import ConversationJudge

UNSEEN = describe_media_in_words("image", "")
SEEN = describe_media_in_words("image", "一只橘猫趴在桌上")


class _Provider:
    model = "offline-grounding-test"

    def __init__(self, content):
        self.content = content
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        return ChatResponse(content=self.content, model=self.model)


def _message(content="这个怎么弄的？"):
    return Message(
        message_id="current", message_type="group", group_id="g1",
        sender_id="u1", sender_name="小明", content=content,
    )


def _review(available=True, reason="", evidence=None):
    return SimpleNamespace(
        available=available, should_reply=available, reason=reason,
        evidence=evidence or {},
    )


class UnseenMediaClaimTests(unittest.TestCase):
    """修复 1：画面没看清时，不许凭空冒出具体名词。"""

    def test_real_fabricated_food_and_model_names_are_flagged(self):
        """线上两条真实事故样本。"""
        self.assertEqual(
            ReplyGenerator.introduces_unseen_content("国庆在家吃鸳鸯锅，爽啊", UNSEEN, []),
            ["鸳鸯锅"],
        )
        self.assertEqual(
            ReplyGenerator.introduces_unseen_content(
                "你那个108的早坏了，这臭表倒是不错", UNSEEN, []
            ),
            ["臭表"],
        )

    def test_ordinary_replies_to_an_unseen_image_are_not_flagged(self):
        """误报会让每条图回复都多跑一次复核，必须压住。"""
        cases = (
            ("哈哈这张图挺好笑的", UNSEEN, []),
            ("这图啥意思啊", UNSEEN, []),
            ("笑死，绷不住了", UNSEEN, []),
            ("我昨天也去吃了顿好的", UNSEEN, []),
            ("这也太狠了吧", UNSEEN, ["刚才那波操作是真的狠"]),
            ("鸳鸯锅好吃", "今天群里在聊吃的", []),
            ("国庆出去玩了吗", "国庆快到了", []),
            ("这只猫趴得也太舒服了吧", SEEN, []),
        )
        for reply, current, sources in cases:
            with self.subTest(reply=reply):
                self.assertEqual(
                    ReplyGenerator.introduces_unseen_content(reply, current, sources), []
                )

    def test_prompt_forbids_specific_nouns_when_blurry(self):
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        request = generator._build_request("", UNSEEN, direction="group")
        guidance = "\n".join(i.content for i in request.messages if i.role == "system")
        self.assertIn("看不清画面", guidance)
        self.assertIn("菜名、品牌、型号、款式、价格", guidance)
        self.assertIn("统统不许出现", guidance)

    def test_detector_is_not_wired_to_the_dead_review_chain(self):
        """needs_semantic_review / needs_reply_quality_review 在生产里没有调用点。

        把检测挂在那条链上等于死代码——这类「写了但不生效」的检查最危险，
        因为它看起来在防护。真正生效的富媒体硬拦截是 main.py 的
        claims_unseen_media 发送前检查。
        """
        self.assertFalse(
            ReplyGenerator.needs_semantic_review("国庆在家吃鸳鸯锅，爽啊", UNSEEN, [])
        )


class MediaClaimEvidenceTests(unittest.IsolatedAsyncioTestCase):
    """修复 1 的落地：检测结果要真的进到复核提示词里。"""

    async def test_suspicious_nouns_reach_the_review_prompt_as_data(self):
        provider = _Provider('{"on_topic":true}')
        judge = ConversationJudge(provider)
        await judge.review_reply(
            _message(), [], "国庆在家吃鸳鸯锅，爽啊",
            media_claim_evidence={
                "本轮消息里媒体没有看清": True,
                "草稿中查不到出处的名词": ["鸳鸯锅"],
            },
        )
        prompt = provider.requests[0].messages[-1].content
        self.assertIn("程序侧已检出的可疑词", prompt)
        self.assertIn("鸳鸯锅", prompt)
        self.assertIn("不是结论", prompt)

    async def test_no_suspicious_nouns_means_no_evidence_block(self):
        provider = _Provider('{"on_topic":true}')
        judge = ConversationJudge(provider)
        await judge.review_reply(_message(), [], "这图啥意思啊")
        prompt = provider.requests[0].messages[-1].content
        self.assertNotIn("程序侧已检出的可疑词", prompt)

    def test_bot_builds_evidence_only_when_nouns_appear(self):
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.reply_generator = ReplyGenerator(llm_provider=None)
        self.assertIsNone(
            bot._media_claim_evidence("这图啥意思啊", UNSEEN, [])
        )
        evidence = bot._media_claim_evidence("国庆在家吃鸳鸯锅，爽啊", UNSEEN, [])
        self.assertEqual(evidence["草稿中查不到出处的名词"], ["鸳鸯锅"])

    async def test_legacy_judge_without_the_kwarg_still_works(self):
        """老判断器没有 media_claim_evidence 形参时不能炸。"""

        class _Legacy:
            def __init__(self):
                self.calls = 0

            async def review_reply(self, message, recent, reply, *, direction):
                self.calls += 1
                return _review(evidence={"on_topic": True})

        judge = _Legacy()
        await GroupChatBot._review_reply_with_context(
            judge, _message(), [], "答复", direction="group",
            media_claim_evidence={"草稿中查不到出处的名词": ["鸳鸯锅"]},
        )
        self.assertEqual(judge.calls, 1)


class FirstReviewVerdictTests(unittest.TestCase):
    """修复 2：复核器挂掉时不放行，宁可沉默。"""

    def test_unavailable_review_never_passes(self):
        """20:15:29 上游 529，草稿就是从这里漏到群里的。"""
        verdict, label, _ = GroupChatBot._first_review_verdict(
            _review(available=False, reason="provider_unavailable"), {}
        )
        self.assertEqual(verdict, "unavailable")
        self.assertEqual(label, "provider_unavailable")
        self.assertNotEqual(verdict, "pass")

    def test_unavailable_beats_an_otherwise_clean_draft(self):
        """证据里一个旗标都没亮，也必须拦。"""
        review = _review(available=False, reason="上游异常")
        verdict, _, flags = GroupChatBot._first_review_verdict(review, {})
        self.assertEqual(verdict, "unavailable")
        self.assertFalse(any(flags.values()))

    def test_clean_draft_passes(self):
        verdict, label, _ = GroupChatBot._first_review_verdict(
            _review(), {"on_topic": True}
        )
        self.assertEqual(verdict, "pass")
        self.assertEqual(label, "")

    def test_flagged_draft_is_retried_with_a_readable_label(self):
        verdict, label, flags = GroupChatBot._first_review_verdict(
            _review(), {"meta_commentary": True}
        )
        self.assertEqual(verdict, "retry")
        self.assertEqual(label, "元话语")
        self.assertTrue(flags["meta_commentary_issue"])

    def test_identity_exposure_label_wins(self):
        verdict, label, _ = GroupChatBot._first_review_verdict(
            _review(),
            {"on_topic": False, "mismatch_types": ["identity_exposure"]},
        )
        self.assertEqual(verdict, "retry")
        self.assertEqual(label, "暴露身份设定")

    def test_mismatch_types_may_be_a_bare_string(self):
        verdict, label, _ = GroupChatBot._first_review_verdict(
            _review(), {"on_topic": False, "mismatch_types": "tone_drift"}
        )
        self.assertEqual(verdict, "retry")
        self.assertEqual(label, "语气偏离")

    def test_paraphrase_that_adds_information_is_not_a_rewrite_trigger(self):
        verdict, _, _ = GroupChatBot._first_review_verdict(
            _review(), {"is_paraphrase": True, "adds_information": True}
        )
        self.assertEqual(verdict, "pass")


class AnswersTheQuestionTests(unittest.IsolatedAsyncioTestCase):
    """修复 3：切题不等于答了被问的那个问题。"""

    async def test_on_topic_but_wrong_question_is_rejected(self):
        """摧城：问大招机制，答配队怎么搭——on_topic 仍然是 true。"""
        provider = _Provider(
            '{"on_topic":true,"answers_the_question":false,'
            '"mismatch_types":[]}'
        )
        judge = ConversationJudge(provider)
        result = await judge.review_reply(
            _message("第二次摧城怎么弄出来的？"), [],
            "配队就是往导电/音感仪那套里塞……",
        )
        self.assertTrue(result.available)
        self.assertFalse(result.should_reply)
        self.assertTrue(result.evidence["on_topic"])
        self.assertFalse(result.evidence["answers_the_question"])

    async def test_missing_field_defaults_to_answered(self):
        """新增检查项缺席不能把所有正常回复判死。"""
        provider = _Provider('{"on_topic":true,"is_paraphrase":false}')
        judge = ConversationJudge(provider)
        result = await judge.review_reply(_message(), [], "大招要能量满了再放")
        self.assertTrue(result.should_reply)
        self.assertTrue(result.evidence["answers_the_question"])

    async def test_non_question_chat_keeps_the_check_quiet(self):
        provider = _Provider('{"on_topic":true,"answers_the_question":true}')
        judge = ConversationJudge(provider)
        result = await judge.review_reply(_message("今天累死了"), [], "摸摸鱼")
        self.assertTrue(result.should_reply)

    def test_verdict_retries_when_the_wrong_question_was_answered(self):
        verdict, label, flags = GroupChatBot._first_review_verdict(
            _review(), {"on_topic": True, "answers_the_question": False}
        )
        self.assertEqual(verdict, "retry")
        self.assertEqual(label, "答的不是被问的那个问题")
        self.assertTrue(flags["answers_wrong_question_issue"])

    def test_hint_tells_the_generator_to_answer_the_actual_question(self):
        generator = ReplyGenerator(llm_provider=None)
        hint = generator._build_reply_review_hint(
            {"answers_the_question": False}
        )
        self.assertIn("只回答那一个问题", hint)
        self.assertIn("不要用同一话题的别的内容代替答案", hint)

    async def test_review_prompt_separates_topic_from_answer(self):
        provider = _Provider('{"on_topic":true}')
        judge = ConversationJudge(provider)
        await judge.review_reply(_message("第二次摧城怎么弄出来的？"), [], "配队……")
        prompt = provider.requests[0].messages[-1].content
        self.assertIn("必须区分", prompt)
        self.assertIn("answers_the_question", prompt)
        self.assertIn("摧城", prompt)


if __name__ == "__main__":
    unittest.main()

"""群聊发言权与发送前复核的离线场景测试。"""

from datetime import datetime, timedelta
from types import SimpleNamespace
import asyncio
import unittest
from unittest.mock import patch

from modules.memory.context import ContextManager, ContextMessage
from modules.reply.generator import ReplyGenerator
from modules.social.awareness import SocialAwarenessManager, SocialContext, TriggerDetector
from modules.social.conversation_floor import (
    ActionPlan,
    ActionType,
    ConversationFloorManager,
)
from modules.social.enhanced_decider import EnhancedSpeakingDecider


class ConversationFloorTests(unittest.TestCase):
    def setUp(self):
        self.manager = ConversationFloorManager()
        self.now = datetime.now()

    def message(
        self,
        sender: str,
        content: str,
        seconds: int,
        *,
        message_id: str,
        reply_to_id: str = "",
        reply_to_qq: str = "",
        directed: bool = False,
    ) -> ContextMessage:
        return ContextMessage(
            sender_id=sender,
            sender_name=sender,
            content=content,
            timestamp=self.now + timedelta(seconds=seconds),
            message_id=message_id,
            reply_to_id=reply_to_id or None,
            reply_to_qq=reply_to_qq or None,
            directed_to_bot=directed,
        )

    def test_direct_question_gets_answer_plan_and_zero_interruption_cost(self):
        current = self.message(
            "u1", "你觉得今晚吃什么？", 0, message_id="m1", directed=True
        )

        floor, plan = self.manager.analyze(
            current,
            [current],
            bot_id="bot",
            directed_to_bot=True,
            is_question=True,
        )

        self.assertTrue(floor.bot_has_floor)
        self.assertEqual(floor.interruption_cost, 0.0)
        self.assertEqual(plan.action, ActionType.ANSWER)
        self.assertTrue(plan.directed)

    def test_two_people_in_fast_dialogue_produce_silent_plan(self):
        messages = [
            self.message("u1", "今晚吃啥", 0, message_id="m1"),
            self.message("u2", "火锅吧", 1, message_id="m2"),
            self.message("u1", "太辣了", 2, message_id="m3"),
            self.message("u2", "那就烤肉", 3, message_id="m4"),
        ]

        floor, plan = self.manager.analyze(messages[-1], messages, bot_id="bot")

        self.assertTrue(floor.two_person_thread)
        self.assertTrue(floor.fast_burst)
        self.assertEqual(plan.action, ActionType.SILENT)

    def test_message_replying_to_another_user_produces_silent_plan(self):
        first = self.message("u1", "这个怎么弄", 0, message_id="m1")
        current = self.message(
            "u2",
            "我教你",
            2,
            message_id="m2",
            reply_to_id="m1",
            reply_to_qq="u1",
        )

        floor, plan = self.manager.analyze(current, [first, current], bot_id="bot")

        self.assertEqual(floor.interruption_cost, 0.95)
        self.assertEqual(plan.action, ActionType.SILENT)

    def test_short_expressive_message_prefers_react(self):
        current = self.message("u1", "笑死哈哈哈", 0, message_id="m1")

        _, plan = self.manager.analyze(current, [current], bot_id="bot")

        self.assertEqual(plan.action, ActionType.REACT)
        self.assertLessEqual(plan.max_chars, 10)

    def test_normal_words_containing_expressive_characters_do_not_fake_react(self):
        current = self.message("u1", "今晚吃牛肉火锅", 0, message_id="m1")

        _, plan = self.manager.analyze(current, [current], bot_id="bot")

        self.assertEqual(plan.action, ActionType.REPLY)

    def test_guard_cancels_when_another_user_replies_to_target(self):
        target = self.message("u1", "这个报错怎么修？", 0, message_id="m1")
        _, plan = self.manager.analyze(
            target,
            [target],
            bot_id="bot",
            topic_relevance=0.8,
            is_question=True,
        )
        answer = self.message(
            "u2",
            "升级依赖就可以",
            2,
            message_id="m2",
            reply_to_id="m1",
            reply_to_qq="u1",
        )

        cancel, reason = self.manager.should_cancel(
            plan, [target, answer], bot_id="bot"
        )

        self.assertTrue(cancel)
        self.assertIn("已有群友回复", reason)

    def test_guard_cancels_when_topic_has_moved(self):
        target = self.message("u1", "今晚吃火锅", 0, message_id="m1")
        _, plan = self.manager.analyze(target, [target], bot_id="bot")
        newer = [
            self.message("u2", "新版本游戏更新了", 2, message_id="m2"),
            self.message("u3", "新地图挺好玩", 3, message_id="m3"),
            self.message("u2", "晚上一起开黑", 4, message_id="m4"),
        ]

        cancel, reason = self.manager.should_cancel(
            plan, [target, *newer], bot_id="bot"
        )

        self.assertTrue(cancel)
        self.assertIn("切换话题", reason)

    def test_guard_keeps_direct_reply_even_when_new_messages_arrive(self):
        target = self.message(
            "u1", "帮我看看？", 0, message_id="m1", directed=True
        )
        _, plan = self.manager.analyze(
            target,
            [target],
            bot_id="bot",
            directed_to_bot=True,
            is_question=True,
        )
        newer = self.message("u2", "路过", 2, message_id="m2")

        cancel, _ = self.manager.should_cancel(
            plan, [target, newer], bot_id="bot"
        )

        self.assertFalse(cancel)

    def test_react_cancels_when_same_sender_continues_after_image(self):
        """发图的人自己又续了句话（如"换这个"），对图的简短反应已过期。"""
        target = self.message("u1", "[图片]", 0, message_id="m1")
        _, plan = self.manager.analyze(
            target,
            [target],
            bot_id="bot",
            rich_message_only=True,
            rich_type="image",
        )
        self.assertEqual(plan.action, ActionType.REACT)
        follow_up = self.message("u1", "换这个", 2, message_id="m2")

        cancel, reason = self.manager.should_cancel(
            plan, [target, follow_up], bot_id="bot"
        )

        self.assertTrue(cancel)
        self.assertIn("续话", reason)

    def test_react_kept_when_other_user_chimes_in(self):
        """图主没续话，只是其他群友零星说话，简短反应仍可补一句。"""
        target = self.message("u1", "[图片]", 0, message_id="m1")
        _, plan = self.manager.analyze(
            target,
            [target],
            bot_id="bot",
            rich_message_only=True,
            rich_type="image",
        )
        newer = self.message("u2", "路过", 2, message_id="m2")

        cancel, _ = self.manager.should_cancel(
            plan, [target, newer], bot_id="bot"
        )

        self.assertFalse(cancel)

    def test_decider_respects_silent_action_plan(self):
        plan = ActionPlan(
            action=ActionType.SILENT,
            target_message_id="m1",
            target_user_id="u1",
            confidence=0.9,
            interruption_cost=0.95,
            reason="两位群友正在连续对聊",
            tone="保持旁观",
            max_chars=0,
            wait_multiplier=1.0,
            directed=False,
            is_question=False,
            target_timestamp=self.now,
        )
        context = SocialContext(
            message_content="你说得对",
            sender_id="u1",
            group_id="g1",
            session_id="group_g1",
        )
        # 即使“关键词+问句”达到旧强制阈值，明确对聊别人仍应优先旁观。
        context.extra["trigger"] = {
            "forced_trigger": True,
            "priority": 0.7,
            "reasons": ["关键词「bot」", "直接提问"],
        }
        context.extra["action_plan"] = plan

        decision = EnhancedSpeakingDecider().decide(context)

        self.assertFalse(decision.should_speak)
        self.assertEqual(decision.probability, 0.0)

    def test_taboo_group_topic_does_not_trigger_unsolicited_interjection(self):
        """禁忌话题在群友互聊时不主动插话，明确问 bot 时交给生成器做边界回复。"""
        awareness = SocialAwarenessManager(taboo_topics=["政治宗教"])
        context = SocialContext(
            message_content="群里又聊到政治宗教了",
            sender_id="u1",
            group_id="g1",
            session_id="group_g1",
        )
        awareness.analyze(context)
        self.assertTrue(context.extra["taboo_topic"])
        self.assertEqual(context.topic_relevance, 0.0)

        decision = EnhancedSpeakingDecider().decide(context)
        self.assertFalse(decision.should_speak)
        self.assertIn("禁忌", decision.reason)

    def test_action_plan_constrains_prompt_and_final_length(self):
        plan = {
            "action": "react",
            "tone": "只做很短的群友式反应",
            "max_chars": 6,
        }
        generator = ReplyGenerator(llm_provider=None)
        request = generator._build_request(
            context_prompt="[刚刚] 小明：笑死哈哈哈",
            current_message="笑死哈哈哈",
            direction="group",
            action_plan=plan,
        )

        self.assertTrue(
            any("很短的即时反应" in message.content for message in request.messages)
        )
        # 按词边界截断：不切进词中间（"确实有点太离谱了"→"确实有点太"）
        cut = generator._limit_action_length("确实有点太离谱了", 6)
        self.assertLessEqual(len(cut), 6)
        self.assertEqual(cut, "确实有点太")

    def test_short_reaction_prompt_prioritizes_complete_event(self):
        generator = ReplyGenerator(llm_provider=None)
        request = generator._build_request(
            context_prompt="[刚刚] 牢森：我一个吕布还能打不过夏侯惇了？\n"
            "[刚刚] 森：猝，享年4级",
            current_message="森：猝，享年4级",
            direction="group",
            action_plan={"action": "reply", "tone": "自然", "max_chars": 18},
        )

        self.assertTrue(
            any("最近2到4条消息" in message.content for message in request.messages)
        )
        self.assertTrue(
            any("不要只抓最新消息" in message.content for message in request.messages)
        )

    def test_surface_reaction_guard_catches_number_only_comment(self):
        plan = {"action": "reply", "max_chars": 18}
        context = (
            "[刚刚] 牢森：我一个吕布还能打不过夏侯惇了？\n"
            "[刚刚] 森：猝，享年4级"
        )

        self.assertTrue(
            ReplyGenerator._is_surface_reaction(
                "4级也太惨了",
                context_prompt=context,
                current_message="森：猝，享年4级",
                direction="group",
                action_plan=plan,
            )
        )
        self.assertFalse(
            ReplyGenerator._is_surface_reaction(
                "放完狠话马上就没了，这反差也太快",
                context_prompt=context,
                current_message="森：猝，享年4级",
                direction="group",
                action_plan=plan,
            )
        )

    def test_settle_window_has_a_bounded_idle_and_total_wait(self):
        manager = ConversationFloorManager(
            settle_window_seconds=0.7,
            settle_max_seconds=2.4,
        )

        self.assertEqual(manager.settle_window_seconds, 0.7)
        self.assertEqual(manager.settle_max_seconds, 2.4)

    def test_group_settle_resets_for_new_message_but_directed_does_not_wait(self):
        from main import GroupChatBot

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.context_manager = ContextManager()
        bot.conversation_floor_manager = ConversationFloorManager(
            settle_window_seconds=0.2,
            settle_max_seconds=0.4,
        )
        target = self.message("u1", "放狠话", 0, message_id="m1")
        newer = self.message("u2", "然后翻车", 1, message_id="m2")
        bot.context_manager.get_window("group_g1").add(target)
        calls = 0

        async def fake_sleep(_seconds):
            nonlocal calls
            calls += 1
            if calls == 1:
                bot.context_manager.get_window("group_g1").add(newer)

        async def exercise():
            with patch("main.asyncio.sleep", new=fake_sleep):
                await bot._wait_for_group_settle(
                    "group_g1", SimpleNamespace(directed=False)
                )

        asyncio.run(exercise())
        self.assertEqual(calls, 2)

        calls = 0

        async def exercise_directed():
            with patch("main.asyncio.sleep", new=fake_sleep):
                await bot._wait_for_group_settle(
                    "group_g1", SimpleNamespace(directed=True)
                )

        asyncio.run(exercise_directed())
        self.assertEqual(calls, 0)

    def test_reply_plan_refreshes_to_latest_undirected_message(self):
        """思考期间群聊前进但未过期时，计划应跟随最新消息而不是旧目标。"""
        from main import GroupChatBot

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.config = {"qq": {"self_id": "bot"}}
        bot.context_manager = ContextManager()
        bot.conversation_floor_manager = ConversationFloorManager()
        bot.social_awareness = SocialAwarenessManager()
        bot.trigger_detector = TriggerDetector(bot_nickname="爱丽丝")

        original = self.message("u1", "今晚吃什么", 0, message_id="m1")
        newer = self.message("u2", "笑死哈哈哈", 2, message_id="m2")
        window = bot.context_manager.get_window("group_g1")
        window.add(original)
        window.add(newer)

        plan = ActionPlan(
            action=ActionType.REPLY,
            target_message_id="m1",
            target_user_id="u1",
            confidence=0.5,
            interruption_cost=0.28,
            reason="存在自然接话机会",
            tone="像普通群友一样随意接一句",
            max_chars=26,
            wait_multiplier=1.0,
            directed=False,
            is_question=False,
            target_timestamp=original.timestamp,
        )

        refreshed, cancelled = bot._refresh_action_plan_after_wait(
            SimpleNamespace(session_id="group_g1"), "group", plan
        )

        self.assertFalse(cancelled)
        self.assertEqual(refreshed.target_message_id, "m2")
        self.assertEqual(refreshed.action, ActionType.REACT)


if __name__ == "__main__":
    unittest.main()

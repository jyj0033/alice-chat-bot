"""群聊发言权与发送前复核的离线场景测试。"""

from datetime import datetime, timedelta
import unittest

from modules.memory.context import ContextMessage
from modules.reply.generator import ReplyGenerator
from modules.social.awareness import SocialContext
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
        self.assertEqual(generator._limit_action_length("确实有点太离谱了", 6), "确实有点太离")


if __name__ == "__main__":
    unittest.main()

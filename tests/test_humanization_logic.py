"""不依赖线上 QQ/LLM 的拟人状态逻辑测试。"""

import unittest
import sys
import types

from modules.memory.context import ContextManager
from modules.personality.emotional_state import Emotion, EmotionalManager
from modules.reply.generator import ReplyGenerator
from modules.social.attention import AttentionManager
from modules.social.awareness import SocialContext
from modules.social.enhanced_decider import EnhancedSpeakingDecider
from modules.social.fatigue import FatigueManager


class HumanizationLogicTests(unittest.TestCase):
    @staticmethod
    def _qq_adapter_class():
        """本地未安装运行时依赖时，也能单测纯消息解析逻辑。"""
        try:
            from core.adapter.qq_adapter import QQAdapter
        except ModuleNotFoundError as exc:
            if exc.name != "websockets":
                raise
            fake_websockets = types.ModuleType("websockets")
            fake_websockets.WebSocketServerProtocol = object
            fake_websockets.exceptions = types.SimpleNamespace(
                ConnectionClosed=Exception
            )
            sys.modules["websockets"] = fake_websockets
            from core.adapter.qq_adapter import QQAdapter
        return QQAdapter

    def test_private_message_is_always_addressed(self):
        decider = EnhancedSpeakingDecider()
        context = SocialContext(
            message_content="在吗",
            sender_id="u1",
            session_id="private_u1",
        )
        context.extra["is_private"] = True
        context.extra["trigger"] = {}

        decision = decider.decide(context)

        self.assertTrue(decision.should_speak)
        self.assertEqual(decision.probability, 0.98)

    def test_conversation_is_committed_only_after_successful_send(self):
        decider = EnhancedSpeakingDecider()
        context = SocialContext(
            message_content="@你 在吗",
            sender_id="u1",
            group_id="g1",
            session_id="group_g1",
            mentioned_me=True,
        )
        context.extra["trigger"] = {
            "forced_trigger": True,
            "priority": 0.7,
            "reasons": ["被@"],
        }

        decision = decider.decide(context)

        self.assertTrue(decision.should_speak)
        self.assertFalse(decider.is_conversation_with("group_g1", "u1"))

        decider.on_bot_reply(
            "group_g1", "g1", probability=decision.probability, user_id="u1"
        )
        self.assertTrue(decider.is_conversation_with("group_g1", "u1"))

    def test_unrelated_group_messages_do_not_create_fatigue(self):
        decider = EnhancedSpeakingDecider()
        for _ in range(20):
            decider.on_message(
                session_id="group_g1",
                group_id="g1",
                user_id="u2",
                is_directed_to_bot=False,
            )

        state = decider.fatigue_manager.get_state("group_g1")
        self.assertEqual(state.conversation_rounds, 0)
        self.assertEqual(state.fatigue_level, 0.0)

        for _ in range(3):
            decider.on_bot_reply(
                "group_g1", "g1", probability=0.95, user_id="u2"
            )
        self.assertEqual(state.conversation_rounds, 3)
        self.assertGreater(state.fatigue_level, 0.0)

    def test_emotion_enters_negative_state_and_returns_to_baseline(self):
        manager = EmotionalManager(decay_halflife=10)
        state = manager.get_state("group_g1")

        state.update("boring_topic", intensity=1.0)
        self.assertEqual(state.current_emotion, Emotion.BORED)
        self.assertLess(state.engagement, state.baseline_engagement)

        state._last_update -= 100
        state.update("normal_message", intensity=0.0)
        self.assertEqual(state.current_emotion, Emotion.NEUTRAL)
        self.assertAlmostEqual(state.energy, state.baseline_energy, places=2)
        self.assertAlmostEqual(
            state.engagement, state.baseline_engagement, places=2
        )

    def test_configured_initial_attention_is_used(self):
        attention = AttentionManager(initial_attention=0.1)

        self.assertEqual(attention.get_group_state("g1").base_attention, 0.1)
        self.assertAlmostEqual(attention.get_user_attention("g1", "u1"), 0.1)

    def test_web_enabled_switches_really_disable_social_state_systems(self):
        attention = AttentionManager(enabled=False)
        attention.on_message_received("g1", "u1", mentioned_bot=True)
        self.assertEqual(attention.get_effective_attention("g1", "u1"), 0.5)
        self.assertFalse(attention._group_states)

        fatigue = FatigueManager(enabled=False)
        fatigue.on_message("group_g1", is_bot_message=True)
        self.assertEqual(fatigue.get_probability_penalty("group_g1"), 0.0)
        self.assertFalse(fatigue.should_close_conversation("group_g1"))

        emotion = EmotionalManager(
            enabled=False,
            positive_keywords=["谢谢"],
            positive_boost=0.4,
        )
        self.assertEqual(emotion.detect_emotion_keywords("谢谢"), (0.0, ""))

    def test_group_probability_override_is_used(self):
        decider = EnhancedSpeakingDecider(base_probability=0.02)
        default_context = SocialContext(
            group_id="g1", session_id="group_g1", sender_id="u1"
        )
        override_context = SocialContext(
            group_id="g2", session_id="group_g2", sender_id="u1"
        )
        default_context.extra["trigger"] = {}
        override_context.extra["trigger"] = {}
        override_context.extra["group_base_probability"] = 0.8

        default_probability = decider._calculate_probability(
            default_context, 0.0, {}
        )
        override_probability = decider._calculate_probability(
            override_context, 0.0, {}
        )
        self.assertGreater(override_probability, default_probability)

        override_context.extra["group_base_probability"] = "invalid"
        invalid_probability = decider._calculate_probability(
            override_context, 0.0, {}
        )
        self.assertAlmostEqual(invalid_probability, default_probability)

    def test_topic_interest_changes_participation_probability(self):
        decider = EnhancedSpeakingDecider()
        interested = SocialContext(
            group_id="g1",
            session_id="group_g1",
            sender_id="u1",
            group_activity=0.5,
            topic_relevance=0.8,
        )
        bored = SocialContext(
            group_id="g2",
            session_id="group_g2",
            sender_id="u1",
            group_activity=0.5,
            topic_relevance=0.2,
        )
        interested.extra["trigger"] = {}
        bored.extra["trigger"] = {}

        interested_probability = decider._calculate_probability(
            interested, 0.0, {}
        )
        bored_probability = decider._calculate_probability(bored, 0.0, {})

        self.assertGreater(interested_probability, bored_probability)

    def test_reply_request_does_not_duplicate_stale_trigger_message(self):
        generator = ReplyGenerator(llm_provider=None)
        trigger = "这句话只应出现一次"

        request = generator._build_request(
            context_prompt=f"[刚刚] 小明(对你说)：{trigger}\n[刚刚] 小明：补充一句",
            current_message=trigger,
            direction="to_bot",
        )
        user_text = request.messages[-1].content

        self.assertEqual(user_text.count(trigger), 1)
        self.assertNotIn("【当前消息】", user_text)

    def test_context_marks_target_and_resolves_reply_message_id(self):
        manager = ContextManager()
        manager.add_message(
            "group_g1",
            "u1",
            "小明",
            "爱丽丝在吗",
            message_id="m1",
            directed_to_bot=True,
        )
        manager.add_message(
            "group_g1",
            "u2",
            "小红",
            "他刚才还在",
            message_id="m2",
            reply_to_id="m1",
        )

        text = manager.get_window("group_g1").build_conversation_text("爱丽丝")

        self.assertIn("小明(对你说)：爱丽丝在吗", text)
        # 引用对象在窗口内时，顺带给出其内容，让 LLM 明白"回@谁"具体引用了什么
        self.assertIn("小红(回@小明：爱丽丝在吗)：他刚才还在", text)

    def test_qq_parser_keeps_non_text_meaning_and_does_not_treat_at_all_as_me(self):
        adapter = self._qq_adapter_class()({"self_id": "42"})
        message = adapter._parse_message(
            {
                "message_id": 1,
                "message_type": "group",
                "group_id": 100,
                "user_id": 7,
                "sender": {"nickname": "昵称", "card": "群名片"},
                "message": [
                    {"type": "at", "data": {"qq": "all"}},
                    {"type": "image", "data": {"file": "x.jpg"}},
                ],
            }
        )

        self.assertFalse(message.mentioned_me)
        self.assertEqual(message.content, "[图片]")
        self.assertEqual(message.sender_name, "群名片")

    def test_qq_parser_resolves_reply_sender_from_recent_message_id(self):
        adapter = self._qq_adapter_class()({"self_id": "42"})
        adapter._parse_message(
            {
                "message_id": 10,
                "message_type": "group",
                "group_id": 100,
                "user_id": 7,
                "sender": {"nickname": "小明"},
                "message": "原消息",
            }
        )
        reply = adapter._parse_message(
            {
                "message_id": 11,
                "message_type": "group",
                "group_id": 100,
                "user_id": 8,
                "sender": {"nickname": "小红"},
                "message": [
                    {"type": "reply", "data": {"id": "10"}},
                    {"type": "text", "data": {"text": "回复内容"}},
                ],
            }
        )

        self.assertEqual(reply.reply_to_qq, "7")
        self.assertEqual(reply.content, "回复内容")


if __name__ == "__main__":
    unittest.main()

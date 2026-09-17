"""回复处境：上下文格式、跨群去重、被赶闭嘴、模板泄漏。"""
import unittest

from core.adapter.rich_content import describe_media_in_words
from modules.memory.context import (
    ContextManager,
    asks_about_unseen_media,
    is_unresolved_media,
    opaque_media_content,
)
from modules.reply.generator import ReplyGenerator
from modules.social.session_awareness import SessionAwareness


class ReplySituationTests(unittest.TestCase):
    def test_history_is_xml_not_speakable_chat(self):
        manager = ContextManager()
        manager.add_message(
            "g1", "u1", "小明", "在吗", message_id="m1", directed_to_bot=True
        )
        manager.add_message("g1", "bot", "爱丽丝", "在", is_bot=True, message_id="b1")
        text = manager.get_window("g1").build_conversation_text("爱丽丝", bot_id="bot")
        self.assertNotIn("[刚刚]", text)
        self.assertNotIn("(对你说)", text)
        self.assertIn("<m ", text)
        self.assertIn('to="you"', text)
        self.assertIn('self="1"', text)

    def test_unseen_image_is_opaque(self):
        self.assertTrue(is_unresolved_media("[图片] [image]"))
        self.assertFalse(is_unresolved_media("[图片，内容：一只猫]"))
        self.assertEqual(opaque_media_content("[图片]"), "一张图，看不清画面。")
        self.assertEqual(
            opaque_media_content("[图片，内容：一只橘猫趴在桌上]"),
            "一张图，画面是一只橘猫趴在桌上。",
        )
        self.assertTrue(asks_about_unseen_media("这里面出现的你都认识？"))

        manager = ContextManager()
        manager.add_message("g1", "u1", "小明", "[图片] [image]", message_id="p1")
        text = manager.get_window("g1").build_conversation_text("爱丽丝")
        self.assertIn("一张图，看不清画面。", text)
        self.assertNotIn("[图片] [image]", text)
        self.assertNotIn("<unseen", text)

    def test_describe_media_in_words(self):
        self.assertEqual(
            describe_media_in_words("image", "一只橘猫趴在桌上"),
            "一张图，画面是一只橘猫趴在桌上。",
        )
        self.assertEqual(describe_media_in_words("mface"), "一张表情，看不清画面。")

    def test_described_image_stays_natural_language(self):
        manager = ContextManager()
        manager.add_message(
            "g1", "u1", "小明", "一张图，画面是白发角色闭眼捂脸。", message_id="p1"
        )
        text = manager.get_window("g1").build_conversation_text("爱丽丝")
        self.assertIn("一张图，画面是白发角色闭眼捂脸。", text)

    def test_prompt_echo_is_detected(self):
        self.assertTrue(
            ReplyGenerator.looks_like_prompt_echo("[刚刚]龍飛月(对你说)：是")
        )
        self.assertTrue(
            ReplyGenerator.looks_like_prompt_echo('<m t="14:08" from="龍飛月">是</m>')
        )
        self.assertFalse(ReplyGenerator.looks_like_prompt_echo("困得理直气壮"))

    def test_judgement_guide_uses_self_deprecating_banter_for_playful_inclusion(self):
        guide = ReplyGenerator._build_judgement_guide(
            {
                "available": True,
                "target": "group",
                "intent": "react",
                "should_reply": True,
                "evidence": {"playful_bot_inclusion": True},
            }
        )

        self.assertIn("短的自嘲或接梗", guide)
        self.assertIn("别把玩笑当真", guide)

    def test_confused_short_probe_does_not_include_yes_or_come(self):
        self.assertTrue(ReplyGenerator._is_confused_short_probe("啥？"))
        self.assertFalse(ReplyGenerator._is_confused_short_probe("是"))
        self.assertFalse(ReplyGenerator._is_confused_short_probe("来吗？"))
        self.assertFalse(ReplyGenerator._is_confused_short_probe("三件事"))

    def test_silence_request_and_mute(self):
        awareness = SessionAwareness(mute_seconds=30)
        self.assertTrue(
            awareness.is_silence_request(
                "爱丽丝出去，别瞎说话",
                bot_names=["爱丽丝", "小爱"],
            )
        )
        self.assertFalse(
            awareness.is_silence_request(
                "我们一起把爱丽丝做掉吧",
                bot_names=["爱丽丝"],
            )
        )
        awareness.mute("group_1", seconds=60)
        self.assertTrue(awareness.is_muted("group_1"))
        self.assertFalse(awareness.is_muted("group_2"))

    def test_cross_group_same_utterance_claimed_once(self):
        awareness = SessionAwareness(utterance_window=120)
        text = "早上八点睡到一点睡了五个小时"
        self.assertTrue(awareness.claim_utterance("group_a", "u1", text))
        self.assertFalse(awareness.claim_utterance("group_b", "u1", text))
        self.assertTrue(awareness.claim_utterance("group_a", "u1", text))
        self.assertTrue(awareness.claim_utterance("group_b", "u2", text))
        self.assertTrue(awareness.claim_utterance("group_b", "u1", "是"))

    def test_reply_brief_pins_quote_and_own_words(self):
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        text = generator._format_current_message_context(
            "是",
            {
                "sender_name": "龍飛月",
                "quoted_sender": "you",
                "quoted_content": "你这是白天补觉吧",
                "own_recent": ["你这是白天补觉吧"],
            },
        )
        self.assertIn("发送者：龍飛月", text)
        self.assertIn("回你的「你这是白天补觉吧」", text)
        self.assertNotIn("消息ID", text)
        self.assertNotIn("动态判断", text)


if __name__ == "__main__":
    unittest.main()

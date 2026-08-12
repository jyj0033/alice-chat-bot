"""不依赖线上 QQ/LLM 的拟人状态逻辑测试。"""

import unittest
import sys
import types

from modules.memory.context import ContextManager
from modules.personality.emotional_state import Emotion, EmotionalManager
from modules.reply.generator import ReplyGenerator
from modules.social.attention import AttentionManager
from modules.social.awareness import SocialContext, TopicAnalyzer, TriggerDetector
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

    # === "被嫌弃降级" 触发判定 ===

    def test_sticker_description_does_not_trigger_frustration(self):
        """[表情包，内容：...无语...] 是对图片情绪的描述，不是群友嫌弃 bot。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        sticker = (
            '[表情包，内容：白发角色闭眼捂脸、表情痛苦崩溃，'
            '表达极度崩溃、无语自闭的情绪，回应仿生猫连歪三次抽卡的惨状]'
        )
        self.assertFalse(generator._is_user_frustrated("g1", "", sticker))

    def test_real_frustration_triggers(self):
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        self.assertTrue(generator._is_user_frustrated("g1", "", "你在说什么呢"))

    def test_tail_frustration_fires_once_per_session(self):
        """上下文尾巴命中降级，但同 session 短时间内不反复道歉。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        generator._last_frustrated.clear()
        tail = (
            "[最近对话]\n[今天 00:00] 小明: 歪三次还抽\n"
            "[今天 00:00] 爱丽丝: 这卡池有毒吧\n"
            "[今天 00:00] 小明: TMD爱丽丝\n"
            "[今天 00:00] 小明: 就这"
        )
        self.assertTrue(generator._is_user_frustrated("g1", tail, ""))
        self.assertFalse(generator._is_user_frustrated("g1", tail, ""))
        self.assertTrue(generator._is_user_frustrated("g2", tail, ""))

    def test_bot_own_marker_words_do_not_trigger_frustration(self):
        """bot 自己的回复常带「离谱」等词，不能把自己的话当成被嫌弃。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        generator._last_frustrated.clear()
        tail = (
            "[最近对话]\n[刚刚] 小明：百度热搜什么时候能正常点\n"
            "[刚刚] 爱丽丝(你)：百度热搜是真的离谱\n"
            "[刚刚] 小红：确实\n"
            "[刚刚] 小明：晚上吃什么"
        )
        self.assertFalse(generator._is_user_frustrated("g1", tail, "晚上吃什么"))

    def test_bot_own_lines_are_marked_as_self(self):
        """对话记录里 bot 自己的发言标注「(你)」，并在头部说明，
        避免 LLM 把自己的历史发言当成别的群友说的话。"""
        manager = ContextManager()
        manager.add_message("group_g1", "u1", "小明", "这个是冰之女王", message_id="m1")
        manager.add_message(
            "group_g1", "bot", "爱丽丝", "那个我好像还没抽", is_bot=True
        )
        text = manager.get_window("group_g1").build_conversation_text("爱丽丝")
        self.assertIn("爱丽丝(你)：那个我好像还没抽", text)

        prompt = manager.build_context_prompt("group_g1", bot_name="爱丽丝")
        self.assertIn("你自己说过的话", prompt)

    def test_implicit_direction_allows_silence(self):
        """延续对话推断出的指向（to_bot_implicit）允许 LLM 判断后沉默；
        明确对bot说（to_bot）则不给沉默选项。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        implicit = generator._build_participation_guide("to_bot_implicit")
        self.assertIn("<silent>", implicit)
        self.assertIn("补完自己上一句", implicit)
        explicit = generator._build_participation_guide("to_bot")
        self.assertNotIn("<silent>", explicit)

    def test_bystander_banter_does_not_trigger_frustration(self):
        """群友互聊里的吐槽（bot 没参与、没被指向）不该让 bot 道歉。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        generator._last_frustrated.clear()
        tail = (
            "[最近对话]\n[刚刚] 小明：这boss设计有病吧\n"
            "[刚刚] 小红：无语了\n"
            "[刚刚] 小明：就这就这\n"
            "[刚刚] 小红：太离谱了"
        )
        # 尾巴没有任何一行指向 bot，bot 也没发过言 → 不降级
        self.assertFalse(generator._is_user_frustrated("g1", tail, ""))
        # bot 只是随机插话（direction=group）时，当前消息的吐槽词也不降级
        self.assertFalse(
            generator._is_user_frustrated("g1", "", "太离谱了", direction="group")
        )
        # 但明确对 bot 说的质疑仍然降级
        self.assertTrue(
            generator._is_user_frustrated("g1", "", "你在说什么呢", direction="to_bot")
        )

    def test_directed_line_in_tail_triggers_frustration(self):
        """尾巴里标注"(对你说)"的质疑行，即使 bot 不是刚发言也算数。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        generator._last_frustrated.clear()
        tail = (
            "[最近对话]\n[刚刚] 小明：随便聊聊\n"
            "[刚刚] 小红(对你说)：你在说什么呢\n"
            "[刚刚] 小明：哈哈\n"
            "[刚刚] 小红：草"
        )
        self.assertTrue(generator._is_user_frustrated("g3", tail, ""))

    # === 话题关键词与紧急词判定 ===

    def test_descriptive_topic_config_matches_keywords(self):
        """配置里的描述式话题（带括号/斜杠）应拆成关键词参与匹配。"""
        analyzer = TopicAnalyzer(
            interested_topics=[
                "游戏（什么类型都聊，手游端游主机都OK）",
                "技术/编程",
            ],
            bored_topics=["微商/广告/引流", "无脑键政"],
        )
        self.assertGreater(analyzer.analyze_relevance("昨晚肝了一晚上手游"), 0.5)
        self.assertGreater(analyzer.analyze_relevance("最近在学编程"), 0.5)
        self.assertLess(analyzer.analyze_relevance("有人在群里发广告"), 0.5)
        self.assertLess(analyzer.analyze_relevance("别在群里键政了"), 0.5)
        self.assertAlmostEqual(analyzer.analyze_relevance("今晚吃火锅吗"), 0.5)

    def test_meme_urgency_words_are_not_emergency(self):
        """「他急了」「笑死救命」是玩梗，不能当成紧急求助强制回复。"""
        detector = TriggerDetector(bot_nickname="爱丽丝")
        for text in ("他急了他急了", "笑死救命", "救命哈哈哈这也太逗了"):
            context = SocialContext(message_content=text)
            result = detector.detect(context)
            self.assertFalse(context.is_emergency, text)
            self.assertNotIn("紧急", result["reasons"], text)

    def test_genuine_emergency_still_detected(self):
        detector = TriggerDetector(bot_nickname="爱丽丝")
        context = SocialContext(message_content="有人晕倒了救命")
        result = detector.detect(context)
        self.assertTrue(context.is_emergency)
        self.assertIn("紧急", result["reasons"])


if __name__ == "__main__":
    unittest.main()

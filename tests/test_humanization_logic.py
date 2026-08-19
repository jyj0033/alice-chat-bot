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

    def test_profile_refusal_text_is_detected(self):
        """LLM 素材不足时的"无法提炼"拒绝文本不能存成用户画像。"""
        from main import GroupChatBot
        self.assertTrue(GroupChatBot._is_profile_refusal(
            "无法提炼。提供的日常发言内容过于零散、不连贯"
        ))
        self.assertTrue(GroupChatBot._is_profile_refusal("信息不足，难以判断"))
        # 拒绝语藏在句中、开头伪装成正常画像的情况也要识别
        self.assertTrue(GroupChatBot._is_profile_refusal(
            "该用户在群内发言内容多为无明确语义的短句、提问或玩笑话，"
            "缺乏可识别的职业、兴趣等稳定信息，从现有发言中难以提炼出明确特征"
        ))
        self.assertFalse(GroupChatBot._is_profile_refusal(
            "互联网行业从业者，喜欢打游戏，经常吐槽工作"
        ))

    # === 语气词后处理 ===

    def test_filler_never_prepended_to_colloquial_openings(self):
        """已经带口语色彩的开头不再叠语气词（避免「这个哈哈哈哈」「那个嗯…」）。"""
        from modules.personality.speaking_style import (
            SpeakingStyle,
            SpeakingStyleManager,
        )
        # 频率拉满，只验证护栏是否拦住
        style = SpeakingStyle(
            filler_words=["呃", "嗯", "那个", "这个"], filler_frequency=1.0
        )
        manager = SpeakingStyleManager(style)
        for text in (
            "哈哈哈赵云乱杀也太爽了",
            "嗯感觉有点憨",
            "那个等等蕾米是谁啊",
            "确实，推送算法有时候真的看不懂",
            "草这也太离谱了吧",
            "对，乌鸦那个配色确实像",
        ):
            self.assertEqual(manager._apply_fillers(text), text, text)

    def test_filler_skips_short_replies(self):
        """「来了来了」这类短句自成语气，前面加语气词只会拖沓。"""
        from modules.personality.speaking_style import (
            SpeakingStyle,
            SpeakingStyleManager,
        )
        style = SpeakingStyle(filler_words=["那个"], filler_frequency=1.0)
        manager = SpeakingStyleManager(style)
        self.assertEqual(manager._apply_fillers("来了来了"), "来了来了")
        # 平铺直叙的长句才是这层该服务的场景
        self.assertEqual(
            manager._apply_fillers("限定池是夏活的安洁莉娜和珊比"),
            "那个限定池是夏活的安洁莉娜和珊比",
        )

    def test_filler_frequency_defaults_low(self):
        """默认频率要足够低：历史上 0.2 导致 1/3 回复以语气词开头。"""
        from modules.personality.speaking_style import (
            SpeakingStyle,
            create_default_style,
        )
        self.assertLessEqual(SpeakingStyle().filler_frequency, 0.1)
        self.assertLessEqual(create_default_style().filler_frequency, 0.1)

    def test_style_guide_tells_llm_fillers_are_occasional(self):
        """发给 LLM 的风格指导不能只列语气词（会被读成"每句都要用"）。"""
        from modules.personality.speaking_style import (
            SpeakingStyle,
            SpeakingStyleManager,
        )
        guide = SpeakingStyleManager(
            SpeakingStyle(filler_words=["呃", "嗯", "那个"])
        ).get_style_guide()
        self.assertIn("偶尔", guide)
        self.assertIn("直接说事", guide)

    # === 复读检测 ===

    def test_parroting_group_message_is_detected(self):
        """表情包摘要会引用前文原话，短反应档下 LLM 容易抓那句引文当发言。"""
        recent = [
            "没玩爽就结束了",
            '[表情包，内容：一只橙色胖猫眯眼歪头，表情不屑，配字"拽"。'
            '回应上面"没玩爽就结束了"，表达还没尽兴、很扫兴的情绪。]',
        ]
        self.assertTrue(ReplyGenerator.is_parroting("没玩爽就结束了哈哈", recent))
        self.assertTrue(ReplyGenerator.is_parroting("那个没玩爽就结束了…", recent))

    def test_short_agreement_is_not_parroting(self):
        """「确实」「哈哈」这类正常附和不能被当成复读。"""
        recent = ["确实太离谱了", "哈哈哈笑死", "这波操作真的强"]
        for reply in ("确实", "哈哈确实", "笑死", "牛啊", "确实强"):
            self.assertFalse(ReplyGenerator.is_parroting(reply, recent), reply)

    def test_original_reply_is_not_parroting(self):
        """自己组织的话即使用了对方提到的词，也不算复读。"""
        recent = ["满愿三个阶段一共才射了6箭", "没玩爽就结束了"]
        for reply in ("对面强度不太够吧这", "才6箭啊那确实亏", "满愿这活动设计有问题"):
            self.assertFalse(ReplyGenerator.is_parroting(reply, recent), reply)

    # === 省略号与笑声频率 ===

    def test_ellipsis_not_appended_after_tone_marks(self):
        """句尾已有语气标记时不再追加省略号（避免「真人！...」「哈哈...」）。"""
        from modules.personality.speaking_style import (
            SpeakingStyle,
            SpeakingStyleManager,
        )
        style = SpeakingStyle(use_ellipsis=True, ellipsis_frequency=1.0)
        manager = SpeakingStyleManager(style)
        for text in (
            "还没从工资的悲伤里走出来？",
            "我不是ai我是真人！",
            "我看得我汗流浃背哈哈",
            "这什么鬼~",
            "已经结束了…",
            "行吧",  # 太短
        ):
            self.assertEqual(manager._apply_punctuation(text), text, text)
        # 平铺直叙的长句仍会偶尔加
        self.assertEqual(
            manager._apply_punctuation("同感氪金一时爽月末火葬场"),
            "同感氪金一时爽月末火葬场...",
        )

    def test_ellipsis_frequency_defaults_low(self):
        from modules.personality.speaking_style import SpeakingStyle
        self.assertLessEqual(SpeakingStyle().ellipsis_frequency, 0.08)

    def test_consecutive_laughter_is_damped(self):
        """刚笑过的会话里，去掉下一条回复开头的笑声。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        generator._last_laugh.clear()
        # 第一条正常保留
        self.assertEqual(
            generator._damp_laughter("g1", "哈哈逃课才是真谛"), "哈哈逃课才是真谛"
        )
        # 紧接着的第二条去掉开头笑声
        self.assertEqual(generator._damp_laughter("g1", "哈哈哈哈这什么鬼"), "这什么鬼")
        # 换个会话不受影响
        self.assertEqual(generator._damp_laughter("g2", "哈哈好惨"), "哈哈好惨")

    def test_mid_sentence_laughter_is_kept(self):
        """句中的笑声更自然，不动它。"""
        generator = ReplyGenerator(llm_provider=None, bot_name="爱丽丝")
        generator._last_laugh.clear()
        generator._damp_laughter("g1", "哈哈好惨")
        self.assertEqual(
            generator._damp_laughter("g1", "发工资叫必要损失费哈哈哈"),
            "发工资叫必要损失费哈哈哈",
        )

    # === 人格提示词 ===

    def _persona(self, **overrides):
        from modules.personality.personality import Personality
        config = {
            "name": "爱丽丝",
            "nickname": "小爱",
            "age_range": "20-24",
            "traits": {"extraversion": 0.7, "agreeableness": 0.75},
            "background": "普通大学生，游戏重度爱好者。",
            "interested_topics": [
                "游戏（什么类型都聊，手游端游主机都OK）",
                "技术/编程",
                "动漫（追番、新番）",
            ],
            "bored_topics": ["微商/广告/引流"],
            "taboo_topics": ["政治宗教"],
            "catchphrases": ["话说", "好像", "感觉", "有点", "绝了", "笑死"],
            "humor_style": "self-deprecating",
            **overrides,
        }
        return Personality.from_dict(config).build_persona_prompt()

    def test_persona_prompt_reads_like_a_person_not_a_spec(self):
        """人格提示词不能是「字段名 + 数值」的规格书——那会把模型推回助手腔。"""
        prompt = self._persona()
        for spec_marker in ("===", "agreeableness", "extraversion", "(值:", "行为准则"):
            self.assertNotIn(spec_marker, prompt, spec_marker)
        self.assertIn("你叫爱丽丝", prompt)

    def test_persona_does_not_encourage_long_sentences(self):
        """人格说「句子可以稍长」会和"不超过30字"的硬约束直接打架。"""
        prompt = self._persona(traits={"extraversion": 0.7})
        self.assertNotIn("句子可以稍长", prompt)
        self.assertIn("一句话说完就停", prompt)

    def test_persona_does_not_instruct_bot_to_be_confused(self):
        """别主动要求"犯迷糊、说错话"——真人偶尔理解错梗是自然发生的。"""
        prompt = self._persona()
        self.assertNotIn("犯迷糊、说错话", prompt)
        self.assertIn("不知道的事就说不知道", prompt)

    def test_persona_topics_strip_config_syntax(self):
        """话题配置里的括号补充和斜杠不能原样出现在人格介绍里。"""
        prompt = self._persona()
        self.assertNotIn("什么类型都聊", prompt)
        self.assertNotIn("技术/编程", prompt)
        # 每个配置话题至少有一个词进入提示词，不因截断丢失
        for word in ("游戏", "技术", "动漫"):
            self.assertIn(word, prompt)

    def test_persona_without_nickname_has_no_alias_line(self):
        """没有别名时不能凭空写出"熟人喊你…"，也不能留下空称呼。"""
        prompt = self._persona(nickname="", age_range="18")
        self.assertIn("你叫爱丽丝，18岁。", prompt)
        self.assertNotIn("熟人喊你", prompt)

    def test_nickname_trigger_ignores_empty_alias(self):
        """别名为空时，触发检测不能把空串当成被点名。"""
        detector = TriggerDetector(bot_nickname="爱丽丝", nicknames=[""])
        result = detector.detect(SocialContext(message_content="今晚吃什么"))
        self.assertFalse(any(r.startswith("昵称") for r in result["reasons"]))
        named = detector.detect(SocialContext(message_content="爱丽丝在吗"))
        self.assertTrue(any(r.startswith("昵称") for r in named["reasons"]))

    # === MBTI 倾向解析 ===

    # === 自述判别（切词误伤 / 反讽 / 讲别人） ===

    def test_real_self_statements_are_kept(self):
        from main import GroupChatBot
        for text in (
            "因为我是水母", "我是电催", "我在芜湖", "我是玩周瑜的",
            "我家已经淹了", "给我家乌龟吃", "谁跟我住的近", "薰姐和我是无业游民",
        ):
            self.assertTrue(GroupChatBot._is_self_statement(text), text)

    def test_dismissive_reply_is_not_self_statement(self):
        """「我喜欢个🥚」形式像自述，意思是否定，不能当成"喜欢某物"。"""
        from main import GroupChatBot
        for text in ("我喜欢个🥚", "我喜欢个屁", "我爱个鬼", "我喜欢才怪"):
            self.assertFalse(GroupChatBot._is_self_statement(text), text)

    def test_mis_segmented_pronoun_is_not_self_statement(self):
        """「对我|是吧」「帮我|在群里」「我|是说」是切词误伤，不是自述。"""
        from main import GroupChatBot
        for text in (
            "就这么对我是吧", "我是说专武", "我看看我是要抽谁",
            "爱丽丝，帮我在群里所有人的手机里下载原神",
            "我在看我朋友聊", "我在想要不要拿这个潜能",
        ):
            self.assertFalse(GroupChatBot._is_self_statement(text), text)

    def test_talking_about_others_is_not_self_statement(self):
        """讲朋友、同事的话不是关于自己的事实。"""
        from main import GroupChatBot
        for text in ("和我朋友一样啊", "我同事走的也很快"):
            self.assertFalse(GroupChatBot._is_self_statement(text), text)

    def test_high_importance_does_not_bypass_self_statement_check(self):
        """importance 会被检索反馈抬高，不能让它把普通消息顶成"自述"。"""
        from main import GroupChatBot
        bot = GroupChatBot.__new__(GroupChatBot)

        class _Mem:
            content = "我在想要不要拿这个潜能"
            importance = 0.95
            metadata: dict = {}

        self.assertFalse(bot._is_personal_fact(_Mem()))

    # === 群聊黑话 ===

    def test_slang_lines_are_parsed(self):
        from main import GroupChatBot
        raw = (
            "<think>找一下</think>\n"
            "舟舟老师 | 菜地里定时刷新的稀有物品 | 舟舟老师又出来了\n"
            "- 农活 | 转发领皮肤口令的行为\n"
            "霸之意志 | 群里调侃某人硬撑的说法 | 又开始霸之意志了\n"
        )
        items = GroupChatBot._parse_slang_lines(raw)
        self.assertEqual([i["term"] for i in items], ["舟舟老师", "农活", "霸之意志"])
        self.assertEqual(items[0]["example"], "舟舟老师又出来了")
        self.assertEqual(items[1]["example"], "")

    def test_common_internet_slang_is_filtered(self):
        """全网通用词不是"这个群的"黑话，收进词表没意义。"""
        from main import GroupChatBot
        raw = "yyds | 永远的神\n绝了 | 表示厉害\n豆撅子 | 群友自制的下饭菜"
        items = GroupChatBot._parse_slang_lines(raw)
        self.assertEqual([i["term"] for i in items], ["豆撅子"])

    def test_glossary_guide_only_lists_matched_terms(self):
        from modules.reply.generator import ReplyGenerator
        guide = ReplyGenerator._build_glossary_guide([
            {"term": "舟舟老师", "meaning": "菜地里刷新的稀有物品"},
            {"term": "农活", "meaning": "转发领皮肤口令"},
        ])
        self.assertIn("「舟舟老师」＝菜地里刷新的稀有物品", guide)
        self.assertIn("不要按字面理解", guide)
        self.assertEqual(ReplyGenerator._build_glossary_guide([]), "")
        self.assertEqual(ReplyGenerator._build_glossary_guide(None), "")

    def test_slang_storage_roundtrip(self):
        """建表、去重、人工词条不被自动提取覆盖、命中匹配。"""
        import tempfile, os
        from modules.memory.storage import MemoryStorage
        path = os.path.join(tempfile.mkdtemp(), "slang.db")
        store = MemoryStorage(path)

        store.upsert_slang("舟舟老师", "菜地稀有物品", session="group_1", source="auto")
        store.upsert_slang("舟舟老师", "被人工改过的释义", session="group_1", source="manual")
        rows = store.list_slang(session="group_1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["meaning"], "被人工改过的释义")

        # 人工校订过的词条不该被后续自动提取冲掉
        store.upsert_slang("舟舟老师", "自动生成的新释义", session="group_1", source="auto")
        self.assertEqual(store.list_slang(session="group_1")[0]["meaning"], "被人工改过的释义")

        # 只返回文本里真正出现的词，且长词优先
        store.upsert_slang("舟舟", "某群友", session="group_1")
        store.upsert_slang("农活", "转发口令", session="group_1")
        matched = store.match_slang("今天舟舟老师又刷出来了", session="group_1")
        self.assertEqual([m["term"] for m in matched], ["舟舟老师", "舟舟"])

        # 停用后不再注入
        store.update_slang(rows[0]["id"], enabled=0)
        self.assertNotIn(
            "舟舟老师",
            [m["term"] for m in store.match_slang("舟舟老师来了", session="group_1")],
        )

    def test_mbti_response_is_parsed(self):
        from main import GroupChatBot
        raw = (
            "<think>先看发言频率</think>\n"
            "E/I: I | 0.65 | 很少主动开话题，多是被点到才回\n"
            "S/N: S | 0.6 | 聊的多是具体操作和数值\n"
            "T/F: F | 0.55 | 常安慰群友，情绪表达多\n"
            "J/P: P | 0.7 | 作息随意，临时起意组队。\n"
        )
        parsed = GroupChatBot._parse_mbti_response(raw)
        self.assertEqual(parsed["type"], "ISFP")
        self.assertEqual(len(parsed["dimensions"]), 4)
        first = parsed["dimensions"][0]
        self.assertEqual((first["axis"], first["letter"]), ("E/I", "I"))
        self.assertAlmostEqual(first["confidence"], 0.65)
        self.assertNotIn("。", parsed["dimensions"][3]["reason"][-1:])

    def test_incomplete_mbti_response_is_rejected(self):
        """缺维度就不显示，宁可没有也不猜。"""
        from main import GroupChatBot
        self.assertIsNone(GroupChatBot._parse_mbti_response("E/I: I | 0.6 | 话少"))
        self.assertIsNone(GroupChatBot._parse_mbti_response(""))
        self.assertIsNone(GroupChatBot._parse_mbti_response("无法判断"))

    def test_mbti_confidence_is_clamped(self):
        """置信度一栏容错：百分比、负数、非数字都不能让整条解析失败。"""
        from main import GroupChatBot
        raw = (
            "E/I: E | 1.8 | a\nS/N: N | 65% | b\n"
            "T/F: T | abc | c\nJ/P: J | 0.5 | d"
        )
        parsed = GroupChatBot._parse_mbti_response(raw)
        self.assertEqual(parsed["type"], "ENTJ")
        vals = [d["confidence"] for d in parsed["dimensions"]]
        self.assertEqual(vals, [0.02, 0.65, 0.0, 0.5])

    def test_dialect_first_person_counts_as_self_description(self):
        """习惯说「俺」「咱」的人，自述不能永远匹配不上。"""
        from main import GroupChatBot
        kws = GroupChatBot._PERSONAL_KEYWORDS
        for word in ("我在", "俺在", "咱家", "俺住", "咱喜欢"):
            self.assertIn(word, kws, word)
        self.assertTrue(GroupChatBot._is_self_statement("俺在芜湖"))

    def test_promo_forward_text_is_excluded(self):
        """群里转发的领取口令模板不是本人发言，不该进记忆也不该参与画像。"""
        from main import GroupChatBot
        ad = "【王者荣耀送新史诗皮肤】好农活!价值747农友钱，复制粘贴这条口令接取：D9KXHK"
        self.assertTrue(GroupChatBot._is_promo_text(ad))
        # 正常聊到兑换码不能被误杀
        for normal in ("这游戏兑换码在哪领啊", "俺要去排位了", "今天口令红包抢到没"):
            self.assertFalse(GroupChatBot._is_promo_text(normal), normal)

    def test_profile_summary_normalization(self):
        """剥掉元话语前缀，并把累加变长的画像收进长度上限（按句子边界）。"""
        from main import GroupChatBot
        self.assertEqual(
            GroupChatBot._normalize_profile_summary("更新后画像：在芜湖做电催，常加班"),
            "在芜湖做电催，常加班",
        )
        self.assertEqual(
            GroupChatBot._normalize_profile_summary("综合来看：总结：常玩原神"),
            "常玩原神",
        )
        long_text = "在芜湖做电催工作，常加班且单休；" * 12
        out = GroupChatBot._normalize_profile_summary(long_text)
        self.assertLessEqual(len(out), 120)
        # 按句子边界收尾，不能停在半句
        self.assertTrue(out.endswith(("；", "。", "单休")), out)

    def test_profile_prefix_is_stripped_for_incremental_update(self):
        """已有画像要去掉「【用户画像 名字】」前缀后喂回模型做增量修订。"""
        from main import GroupChatBot
        self.assertEqual(
            GroupChatBot._strip_profile_prefix("【用户画像 龍飛月】在芜湖做电催，常加班"),
            "在芜湖做电催，常加班",
        )
        self.assertEqual(GroupChatBot._strip_profile_prefix(""), "")
        # 没有前缀时原样返回
        self.assertEqual(GroupChatBot._strip_profile_prefix("在芜湖做电催"), "在芜湖做电催")

    def test_profile_quality_warnings_flag_suspicious_wording(self):
        """推测、性别不明、空话要打标记，交给页面展示以便人工核对原句。"""
        from main import GroupChatBot
        w = GroupChatBot._profile_quality_warnings(
            "他/她可能家境一般，似乎在做AI相关工作"
        )
        self.assertTrue(any("推测用词" in x for x in w))
        self.assertTrue(any("性别不明确" in x for x in w))
        self.assertTrue(
            GroupChatBot._profile_quality_warnings("此人日常关注游戏话题")
        )
        # 具体、确定的画像不该被打标
        self.assertEqual(
            GroupChatBot._profile_quality_warnings(
                "在芜湖做电话催收，加班多，常聊原神和工作吐槽"
            ),
            [],
        )

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

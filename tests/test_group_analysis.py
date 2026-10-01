import json
import io
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules.group_analysis import GroupDailyAnalysis
from modules.memory.storage import Memory


def _message(content, sender_id, sender_name, created_at, **metadata):
    return Memory(
        content=content,
        memory_type="group_analysis",
        created_at=created_at,
        metadata={
            "sender_id": sender_id,
            "sender_name": sender_name,
            **metadata,
        },
    )


class _Provider:
    def __init__(self, payload):
        self.payload = payload
        self.request = None

    async def chat(self, request):
        self.request = request
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


class _ScriptedProvider:
    """按脚本依次返回 (content, finish_reason)，并记录每次请求（用于验证预算爬升）。"""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.script) - 1)
        content, finish_reason = self.script[index]
        return SimpleNamespace(content=content, finish_reason=finish_reason)


def _full_report_payload():
    """一份结构完整的日报，逆天语录来自 GroupAnalysisTests.setUp 的真实消息。"""
    return {
        "title": "今晚的群聊小剧场",
        "subtitle": "大家从开黑聊到了配队",
        "summary": "今晚约了开黑，大家讨论了配队。",
        "topics": [{
            "name": "开黑安排",
            "detail": "讨论上线时间和游戏安排",
            "sender_ids": ["u1", "u2"],
        }],
        "profiles": [{
            "sender_id": "u1",
            "title": "抽象配队师",
            "mbti": "今日观察派",
            "reason": "吐槽配队",
        }],
        "unhinged_quotes": [{
            "content": "我可以，八点半上线",
            "sender_id": "u2",
            "score": 90,
            "reason": "接得太顺了",
        }],
    }


class _AnalysisStorage:
    def __init__(self, messages):
        self.messages = messages
        self.reports = []

    async def get_group_analysis_messages(self, session, since=None, until=None, limit=500):
        return self.messages[:limit]

    async def store_group_analysis_report(self, memory):
        self.reports.append(memory)


class _QQ:
    def __init__(self):
        self.sent = []

    async def send_message(self, session, content):
        self.sent.append((session, content))
        return True


class GroupAnalysisTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        base = datetime(2026, 8, 24, 20, 0)
        self.messages = [
            _message("今晚开黑打原神吗😀", "u1", "甲", base),
            _message("我可以，八点半上线", "u2", "乙", base + timedelta(minutes=1)),
            _message("这波配队太抽象了", "u1", "甲", base + timedelta(minutes=2)),
            _message("爱丽丝：收到", "bot", "爱丽丝", base + timedelta(minutes=3), is_bot=True),
            _message("/群分析", "u2", "乙", base + timedelta(minutes=4), is_command=True),
        ]

    def test_statistics_ignore_bot_and_legacy_command(self):
        stats = GroupDailyAnalysis.build_statistics(self.messages)
        self.assertEqual(stats["message_count"], 3)
        self.assertEqual(stats["participant_count"], 2)
        self.assertEqual(stats["reply_count"], 0)
        self.assertEqual(stats["emoji_count"], 1)
        self.assertEqual(stats["top_users"][0]["name"], "甲")
        ordinary_message = _message("日报", "u3", "丙", datetime(2026, 8, 24, 21, 0))
        self.assertEqual(len(GroupDailyAnalysis.human_messages([ordinary_message])), 1)

    async def test_llm_result_is_normalized_and_unhinged_quote_must_have_source(self):
        provider = _Provider({
            "title": "今晚的群聊小剧场",
            "subtitle": "大家从开黑聊到了配队",
            "summary": "今晚约了开黑，大家讨论了配队。",
            "topics": [{
                "name": "开黑安排",
                "detail": "讨论上线时间和游戏安排",
                "sender_ids": ["u1", "u2", "unknown"],
            }],
            # 金句区已下线：模型即使给了 quotes 也不该出现在返回里。
            "quotes": [
                {"content": "这波配队太抽象了", "sender_id": "u1", "reason": "过时的金句区"},
            ],
            "unhinged_quotes": [
                {
                    "content": "这波配队太抽象了",
                    "sender_id": "u1",
                    "score": 91,
                    "reason": "好家伙，原来还能这么说",
                },
                {
                    "content": "我可以，八点半上线",
                    "sender_id": "u2",
                    "score": 60,
                    "reason": "这句只是普通接话，够不上门槛",
                },
                {
                    "content": "模型编的逆天原话",
                    "sender_id": "u2",
                    "score": 99,
                    "reason": "不应保留",
                },
            ],
            "profiles": [{
                "sender_id": "u1",
                "title": "抽象配队师",
                "mbti": "今日观察派",
                "reason": "吐槽配队",
            }],
            "quality_review": {
                "title": "轻松开黑局",
                "subtitle": "今晚的群聊温度",
                "summary": "大家接话很顺，话题也没有冷场。",
                "dimensions": [
                    {"name": "接话度", "percentage": 70, "comment": "回复都接得上。"},
                    {"name": "跑题度", "percentage": 30, "comment": "偶尔拐得很快。"},
                ],
            },
            "atmosphere": "轻松闲聊",
        })
        report = await GroupDailyAnalysis.analyze(
            self.messages,
            provider=provider,
            max_topics=3,
            max_unhinged_quotes=4,
            unhinged_min_score=75,
            max_titles=3,
            bot_name="爱丽丝",
            bot_persona="我说话很短，偶尔会吐槽。",
        )
        self.assertEqual(report["summary"], "今晚约了开黑，大家讨论了配队。")
        self.assertEqual(report["topics"][0]["sender_ids"], ["u1", "u2"])
        # 金句整块下线：返回结构里不再有 quotes 键，模型硬塞的也不消费。
        self.assertNotIn("quotes", report)
        # 99 分那条是编的（原句匹配失败），60 分那条低于门槛分 → 只剩 91 分。
        self.assertEqual(len(report["unhinged_quotes"]), 1)
        self.assertEqual(report["unhinged_quotes"][0]["score"], 91)
        self.assertEqual(report["unhinged_quotes"][0]["content"], "这波配队太抽象了")
        self.assertEqual(report["titles"][0]["sender_id"], "u1")
        self.assertEqual(report["title"], "今晚的群聊小剧场")
        self.assertEqual(report["subtitle"], "大家从开黑聊到了配队")
        self.assertEqual(report["profiles"][0]["mbti"], "今日观察派")
        self.assertEqual(report["quality_review"]["dimensions"][0]["percentage"], 70.0)
        prompt = provider.request.messages[-1].content
        self.assertIn("【群聊原文】", prompt)
        self.assertIn("第一人称", prompt)
        self.assertIn('"quality_review"', prompt)
        self.assertIn('"profiles"', prompt)
        self.assertIn('"unhinged_quotes"', prompt)
        self.assertNotIn('"quotes"', prompt)
        self.assertIn("逆天语录", prompt)
        self.assertIn("短反应", prompt)
        self.assertIn("多用互联网黑话", prompt)
        self.assertIn("话题又拐回来了", prompt)
        self.assertIn("宁缺毋滥", prompt)
        # 逆天语录必须是分档标准，不能退回成「最离谱、最反差」这种笼统说法
        for tier in ("情绪越界", "争议暴论", "一本正经的胡说八道", "逻辑跳脱"):
            self.assertIn(tier, prompt)
        self.assertIn("至少列出5个", prompt)
        self.assertIn("我说话很短", prompt)
        rendered = GroupDailyAnalysis.render_report(report)
        self.assertNotIn("我忍不住记下的几句", rendered)
        self.assertIn("好家伙，原来还能这么说", rendered)
        self.assertIn("逆天现场", rendered)

    async def test_provider_failure_keeps_local_statistics(self):
        report = await GroupDailyAnalysis.analyze(self.messages, provider=None)
        self.assertEqual(report["statistics"]["message_count"], 3)
        self.assertEqual(report["topics"], [])
        self.assertTrue(report["analysis_error"])
        rendered = GroupDailyAnalysis.render_report(report, "今日")
        self.assertIn("我按看到的消息做的本地统计", rendered)
        self.assertNotIn("📒", rendered)

    def test_looks_like_report_rejects_truncated_inner_fragment(self):
        # 截断时 _parse_json 会退而返回内层碎片（单个 profiles 项）。
        # 那种碎片能在 `if parsed` 上成真，必须由结构完整性判据挡掉，
        # 否则日报会静默变成只剩统计的一页，逆天语录凭空消失。
        fragment = {
            "sender_id": "u1",
            "title": "接话担当",
            "mbti": "今日观察",
            "reason": "接话",
        }
        self.assertFalse(GroupDailyAnalysis._looks_like_report(fragment))
        self.assertFalse(GroupDailyAnalysis._looks_like_report({}))
        self.assertFalse(GroupDailyAnalysis._looks_like_report(None))
        # 只有 summary、没有任何列表区 → 也算不完整
        self.assertFalse(
            GroupDailyAnalysis._looks_like_report({"summary": "大家好安静。"})
        )
        self.assertTrue(
            GroupDailyAnalysis._looks_like_report(
                {"summary": "今晚聊了很久。", "topics": [], "profiles": [], "unhinged_quotes": []}
            )
        )
        # 今天一条逆天语录都没有（空数组）仍然算完整，不能被误判成截断。
        self.assertTrue(
            GroupDailyAnalysis._looks_like_report(
                {"summary": "今晚聊了很久。", "topics": [{"name": "x", "detail": "y"}], "unhinged_quotes": []}
            )
        )

    async def test_truncated_report_retries_with_a_bigger_output_budget(self):
        partial = (
            '{"title": "今晚", "subtitle": "夜谈", "summary": "大家聊了很久。",'
            ' "topics": [{"name": "话题", "detail": "细节", "sender_ids": ["u1"]}],'
            ' "profiles": [{"sender_id": "u1", "title": "接话担当", "mbti": "观察",'
            ' "reason": "接话"}],'
            ' "unhinged_quotes": [{"content": "我可以，八点半上线", "sender_id": "u2",'
        )
        provider = _ScriptedProvider([
            (partial, "incomplete"),
            (json.dumps(_full_report_payload(), ensure_ascii=False), "stop"),
        ])
        report = await GroupDailyAnalysis.analyze(
            self.messages,
            provider=provider,
            max_tokens=1000,
            retries=2,
            bot_name="爱丽丝",
        )
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.requests[0].max_tokens, 1000)
        self.assertGreater(provider.requests[1].max_tokens, 1000)
        self.assertEqual(report["analysis_error"], "")
        self.assertNotIn("quotes", report)
        self.assertEqual(len(report["unhinged_quotes"]), 1)
        self.assertEqual(report["unhinged_quotes"][0]["score"], 90)

    async def test_every_attempt_truncated_degrades_loudly(self):
        partial = '{"title": "今晚", "profiles": [{"sender_id": "u1", "title": "接话担当"'
        provider = _ScriptedProvider([(partial, "incomplete"), (partial, "incomplete")])
        report = await GroupDailyAnalysis.analyze(
            self.messages, provider=provider, max_tokens=1000, retries=2
        )
        # 不能静默产出一份只有标题、没有逆天语录的「半份日报」
        self.assertTrue(report["analysis_error"])
        self.assertNotIn("quotes", report)
        self.assertEqual(report["unhinged_quotes"], [])
        self.assertEqual(report["statistics"]["message_count"], 3)

    async def test_low_score_unhinged_quotes_are_dropped(self):
        """门槛分是代码侧硬闸门：分数不够的候选一律不出现（宁缺毋滥）。"""
        provider = _Provider({
            "title": "今晚",
            "summary": "大家随便聊了几句。",
            "topics": [{"name": "闲聊", "detail": "没什么重点", "sender_ids": ["u1"]}],
            "profiles": [{
                "sender_id": "u1",
                "title": "围观群众",
                "mbti": "",
                "reason": "今天没怎么说话",
            }],
            "unhinged_quotes": [
                {
                    "content": "这波配队太抽象了",
                    "sender_id": "u1",
                    "score": 74,
                    "reason": "差一分",
                },
                {
                    "content": "我可以，八点半上线",
                    "sender_id": "u2",
                    "score": 60,
                    "reason": "普通接话",
                },
            ],
        })
        report = await GroupDailyAnalysis.analyze(
            self.messages, provider=provider, unhinged_min_score=75
        )
        self.assertEqual(report["unhinged_quotes"], [])
        # 空数组不该被判成「结构不完整」而触发降级
        self.assertEqual(report["analysis_error"], "")
        self.assertEqual(report["title"], "今晚")

    async def test_report_image_is_a_png(self):
        report = await GroupDailyAnalysis.analyze(self.messages, provider=None)
        image = GroupDailyAnalysis.render_report_image(report)
        self.assertIsNotNone(image)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    async def test_report_fetches_and_uses_cached_avatars(self):
        # 头像只画在带 sender_id 的画像卡和逆天语录卡上。用真实 LLM 结构产出这两块，
        # 否则 provider=None 的空报告里没有任何绘制点，渲染结果和有无头像必然相同。
        report = await GroupDailyAnalysis.analyze(
            self.messages, provider=_Provider(_full_report_payload())
        )
        self.assertTrue(
            report["profiles"] or report["unhinged_quotes"],
            "报告里必须存在会消费头像的区块，否则下面的头像断言是空转",
        )
        avatar_image = Image.new("RGB", (64, 64), (244, 128, 168))
        avatar_buffer = io.BytesIO()
        avatar_image.save(avatar_buffer, format="PNG")
        payload = avatar_buffer.getvalue()
        calls = []

        async def fetcher(sender_id):
            calls.append(sender_id)
            return payload

        with tempfile.TemporaryDirectory() as temp_dir:
            avatars = await GroupDailyAnalysis.fetch_avatars(
                report,
                fetcher,
                temp_dir,
                max_count=2,
                cache_days=7,
            )
            self.assertEqual(set(avatars), {"u1", "u2"})
            self.assertEqual(set(calls), {"u1", "u2"})
            self.assertTrue(all(Path(path).is_file() for path in avatars.values()))

            calls.clear()
            cached = await GroupDailyAnalysis.fetch_avatars(
                report,
                fetcher,
                temp_dir,
                max_count=2,
                cache_days=7,
            )
            self.assertEqual(cached, avatars)
            self.assertEqual(calls, [])

            report["avatars"] = avatars
            with_avatar = GroupDailyAnalysis.render_report_image(report)
            report.pop("avatars")
            without_avatar = GroupDailyAnalysis.render_report_image(report)
            self.assertNotEqual(with_avatar, without_avatar)

    def test_rich_report_image_is_a_png(self):
        report = {
            "bot_name": "爱丽丝",
            "title": "晚饭后的小剧场",
            "subtitle": "话题拐了几个弯，大家还没散场",
            "summary": "我看到大家先聊周末安排，后来又认真复盘了一局游戏。",
            "atmosphere": "像吃完饭还坐在桌边继续聊天。",
            "statistics": {
                "message_count": 20,
                "participant_count": 3,
                "total_characters": 240,
                "reply_count": 8,
                "peak_hour": 21,
                "hourly_activity": {"19": 4, "20": 7, "21": 9},
                "top_users": [
                    {"sender_id": "u1", "name": "甲", "message_count": 9},
                    {"sender_id": "u2", "name": "乙", "message_count": 7},
                    {"sender_id": "u3", "name": "丙", "message_count": 4},
                ],
                "sender_names": {"u1": "甲", "u2": "乙", "u3": "丙"},
            },
            "topics": [{
                "name": "周末安排",
                "sender_ids": ["u1", "u2"],
                "detail": "甲先提起周末安排，乙接着补充了时间，最后大家决定先集合再看具体去哪儿。",
            }],
            "profiles": [{
                "sender_id": "u1",
                "title": "接话担当",
                "mbti": "今日观察",
                "reason": "我注意到甲今天一直在接住大家的话头。",
            }],
            "unhinged_quotes": [
                {
                    "sender_id": "u3",
                    "content": "这也能接上，真的太抽象了",
                    "score": 97,
                    "reason": "这句的走向我是真没猜到",
                },
                {
                    "sender_id": "u1",
                    "content": "我宣布今天的计划先不计划",
                    "score": 88,
                    "reason": "好家伙，计划自己先消失了",
                },
            ],
            "quality_review": {
                "title": "没有散场的闲聊",
                "subtitle": "今晚",
                "summary": "计划没有完全落地，但聊天一直有人接。",
                "dimensions": [{
                    "name": "接话度",
                    "percentage": 100,
                    "comment": "大家都在认真接前一句。",
                }],
            },
        }
        image = GroupDailyAnalysis.render_report_image(report)
        self.assertIsNotNone(image)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_editorial_report_handles_long_fields(self):
        long_text = "这是一段故意拉长的内容，用来验证日报的正文、标签和卡片边界都会自然换行，不会覆盖其他文字或跑出容器。"
        report = {
            "bot_name": "爱丽丝",
            "title": "这是一个足够长的日报标题，用来验证主视觉区域的换行行为",
            "subtitle": long_text,
            "summary": long_text,
            "atmosphere": long_text,
            "statistics": {
                "message_count": 99,
                "participant_count": 4,
                "total_characters": 9999,
                "reply_count": 40,
                "peak_hour": 21,
                "hourly_activity": {"21": 99},
                "top_users": [
                    {"sender_id": "u1", "name": "甲"},
                    {"sender_id": "u2", "name": "乙"},
                ],
                "sender_names": {"u1": "甲", "u2": "乙"},
            },
            "topics": [{
                "name": "一个很长的热门话题标题，用来验证标题区域",
                "sender_ids": ["u1", "u2"],
                "detail": long_text,
            }],
            "profiles": [{
                "sender_id": "u1",
                "title": "一个很长的称号标签",
                "mbti": "一个很长的轻量标签",
                "reason": long_text,
            }],
            "unhinged_quotes": [{
                "sender_id": "u2",
                "content": long_text,
                "score": 99,
                "reason": long_text,
            }],
            "quality_review": {
                "title": "一个很长的聊天质量主题标题",
                "subtitle": long_text,
                "summary": long_text,
                "dimensions": [{
                    "name": "一个很长的维度名称",
                    "percentage": 100,
                    "comment": long_text,
                }],
            },
        }
        image = GroupDailyAnalysis.render_report_image(report)
        self.assertIsNotNone(image)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    async def test_bot_pipeline_saves_and_sends_report(self):
        from main import GroupChatBot

        provider = _Provider({"summary": "大家约了晚上开黑。"})
        bot = GroupChatBot.__new__(GroupChatBot)
        # 功能路由会读 config.llm_routing；真实运行由 _load_config 保证存在，
        # 这里补空配置让替身能走到回退到 get_active_provider 的分支。
        bot.config = {}
        bot._group_analysis_config = {
            "enabled": True,
            "min_messages": 2,
            "max_messages": 100,
            "max_prompt_chars": 10000,
            "max_topics": 2,
            "max_unhinged_quotes": 2,
            "max_titles": 2,
            "max_tokens": 300,
            "max_report_chars": 2000,
            "send_report": True,
        }
        bot._group_analysis_write_tasks = set()
        bot._group_analysis_tasks = {}
        bot._memory_tasks = set()
        bot.memory_storage = _AnalysisStorage(self.messages[:3])
        bot.qq_adapter = _QQ()
        bot.personality = SimpleNamespace(
            name="爱丽丝",
            build_persona_prompt=lambda: "我说话简短，喜欢吐槽。",
        )
        bot.get_active_provider = lambda: provider

        await bot._run_group_analysis("group_123", 1)
        self.assertEqual(len(bot.memory_storage.reports), 1)
        self.assertEqual(bot.qq_adapter.sent[0][0], "group_123")
        self.assertIn("群聊日报", bot.qq_adapter.sent[0][1])


if __name__ == "__main__":
    unittest.main()

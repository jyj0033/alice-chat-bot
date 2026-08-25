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

    async def test_llm_result_is_normalized_and_quote_must_have_source(self):
        provider = _Provider({
            "title": "今晚的群聊小剧场",
            "subtitle": "大家从开黑聊到了配队",
            "summary": "今晚约了开黑，大家讨论了配队。",
            "topics": [{
                "name": "开黑安排",
                "detail": "讨论上线时间和游戏安排",
                "sender_ids": ["u1", "u2", "unknown"],
            }],
            "quotes": [
                {"content": "这波配队太抽象了", "sender_id": "u1", "reason": "吐槽有画面"},
                {"content": "模型编的不存在原话", "sender_id": "u2", "reason": "不应保留"},
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
                    "reason": "这句至少是真的",
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
            max_quotes=3,
            max_titles=3,
            bot_name="爱丽丝",
            bot_persona="我说话很短，偶尔会吐槽。",
        )
        self.assertEqual(report["summary"], "今晚约了开黑，大家讨论了配队。")
        self.assertEqual(report["topics"][0]["sender_ids"], ["u1", "u2"])
        self.assertEqual(len(report["quotes"]), 1)
        self.assertEqual(report["quotes"][0]["content"], "这波配队太抽象了")
        self.assertEqual(len(report["unhinged_quotes"]), 2)
        self.assertEqual(report["unhinged_quotes"][0]["score"], 91)
        self.assertEqual(report["unhinged_quotes"][1]["score"], 60)
        self.assertEqual(report["titles"][0]["sender_id"], "u1")
        self.assertEqual(report["title"], "今晚的群聊小剧场")
        self.assertEqual(report["subtitle"], "大家从开黑聊到了配队")
        self.assertEqual(report["profiles"][0]["mbti"], "今日观察派")
        self.assertEqual(report["quality_review"]["dimensions"][0]["percentage"], 70.0)
        self.assertIn("【群聊原文】", provider.request.messages[-1].content)
        self.assertIn("第一人称", provider.request.messages[-1].content)
        self.assertIn('"quality_review"', provider.request.messages[-1].content)
        self.assertIn('"profiles"', provider.request.messages[-1].content)
        self.assertIn('"unhinged_quotes"', provider.request.messages[-1].content)
        self.assertIn("逆天语录", provider.request.messages[-1].content)
        self.assertIn("短反应", provider.request.messages[-1].content)
        self.assertIn("多用互联网黑话", provider.request.messages[-1].content)
        self.assertIn("话题又拐回来了", provider.request.messages[-1].content)
        self.assertIn("我说话很短", provider.request.messages[-1].content)
        rendered = GroupDailyAnalysis.render_report(report)
        self.assertNotIn("我当时想说", rendered)
        self.assertIn("吐槽有画面", rendered)
        self.assertIn("逆天现场", rendered)

    async def test_provider_failure_keeps_local_statistics(self):
        report = await GroupDailyAnalysis.analyze(self.messages, provider=None)
        self.assertEqual(report["statistics"]["message_count"], 3)
        self.assertEqual(report["topics"], [])
        self.assertTrue(report["analysis_error"])
        rendered = GroupDailyAnalysis.render_report(report, "今日")
        self.assertIn("我按看到的消息做的本地统计", rendered)
        self.assertNotIn("📒", rendered)

    async def test_report_image_is_a_png(self):
        report = await GroupDailyAnalysis.analyze(self.messages, provider=None)
        image = GroupDailyAnalysis.render_report_image(report)
        self.assertIsNotNone(image)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

    async def test_report_fetches_and_uses_cached_avatars(self):
        report = await GroupDailyAnalysis.analyze(self.messages, provider=None)
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
            "quotes": [{
                "sender_id": "u2",
                "content": "先别决定，先复盘一下",
                "reason": "把一个普通安排拐成了复盘会议。",
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
            "quotes": [{
                "sender_id": "u2",
                "content": long_text,
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
        bot._group_analysis_config = {
            "enabled": True,
            "min_messages": 2,
            "max_messages": 100,
            "max_prompt_chars": 10000,
            "max_topics": 2,
            "max_quotes": 2,
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

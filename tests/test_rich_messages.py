"""富媒体解析、NapCat 回执和拟人策略的离线测试。"""

import asyncio
import json
import unittest

from core.adapter.base import Message
from core.adapter.rich_content import (
    MessageSegment,
    parse_message_segments,
    render_outer_text,
    render_segments,
)
from core.adapter.rich_media import RichMediaEnricher, _validate_http_url
from modules.memory.context import ContextMessage
from modules.social.conversation_floor import ActionType, ConversationFloorManager


class RichMessageParsingTests(unittest.TestCase):
    def test_image_url_is_structured_but_not_exposed_to_context(self):
        secret_url = "https://cdn.example.com/a.jpg?token=top-secret"
        segments = parse_message_segments([
            {"type": "text", "data": {"text": "看看这个 "}},
            {"type": "image", "data": {"file": "a.jpg", "url": secret_url}},
        ])

        self.assertEqual(render_segments(segments), "看看这个 [图片]")
        self.assertEqual(render_outer_text(segments), "看看这个")
        self.assertEqual(segments[-1].url, secret_url)
        self.assertNotIn("top-secret", render_segments(segments))

    def test_plain_url_keeps_domain_and_hides_path_and_query(self):
        segments = parse_message_segments(
            "帮我看看 https://example.com/private/path?token=secret 怎么样"
        )

        self.assertEqual(
            render_segments(segments),
            "帮我看看 [链接：example.com] 怎么样",
        )
        self.assertEqual(render_outer_text(segments), "帮我看看 [链接] 怎么样")

    def test_miniapp_json_extracts_safe_title(self):
        payload = {
            "app": "com.tencent.miniapp_01",
            "prompt": "[小程序] 腾讯文档",
            "meta": {
                "detail": {
                    "title": "本周排班表",
                    "qqdocurl": "https://docs.qq.com/safe?id=secret",
                }
            },
        }
        segments = parse_message_segments([
            {"type": "json", "data": {"data": json.dumps(payload, ensure_ascii=False)}}
        ])

        self.assertEqual(segments[0].type, "miniapp")
        self.assertEqual(render_segments(segments), "[小程序：本周排班表]")
        self.assertEqual(render_outer_text(segments), "")
        self.assertNotIn("secret", render_segments(segments))

    def test_cq_encoded_miniapp_is_parsed_without_exposing_json(self):
        raw = (
            '[CQ:json,data={&quot;app&quot;:&quot;com.tencent.miniapp_01&quot;'
            '&#44;&quot;prompt&quot;:&quot;&#91;小程序&#93; 测试入口&quot;}]'
        )
        segments = parse_message_segments(raw)

        self.assertEqual(segments[0].type, "miniapp")
        self.assertEqual(render_segments(segments), "[小程序：测试入口]")
        self.assertEqual(render_outer_text(segments), "")

    def test_forwarded_bot_name_is_not_outer_trigger_text(self):
        segments = parse_message_segments([
            {
                "type": "forward",
                "data": {
                    "id": "forward-1",
                    "content": [
                        {
                            "type": "node",
                            "data": {
                                "nickname": "小明",
                                "content": [
                                    {"type": "text", "data": {"text": "爱丽丝在吗？"}}
                                ],
                            },
                        }
                    ],
                },
            }
        ])

        self.assertEqual(render_outer_text(segments), "")
        self.assertEqual(render_segments(segments), "[合并转发]")

    def test_ssrf_guard_rejects_local_targets_and_nonstandard_ports(self):
        for url in (
            "http://127.0.0.1/admin",
            "http://169.254.169.254/latest/meta-data",
            "http://localhost/",
            "https://example.com:8443/private",
            "http://user:pass@example.com/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_http_url(url)


class RichMediaEnricherTests(unittest.IsolatedAsyncioTestCase):
    async def test_forward_expands_with_bounded_readable_excerpts(self):
        calls = []

        async def api_call(action, params, timeout):
            calls.append((action, params, timeout))
            return {
                "messages": [
                    {
                        "sender": {"nickname": "小明"},
                        "message": [{"type": "text", "data": {"text": "周六吃火锅"}}],
                    },
                    {
                        "sender": {"nickname": "小红"},
                        "message": [{"type": "image", "data": {"file": "menu.jpg"}}],
                    },
                ]
            }

        segment = MessageSegment(
            type="forward",
            summary="[合并转发]",
            file_id="f-1",
            data={"id": "f-1"},
        )
        message = Message(
            message_id="m1",
            message_type="group",
            sender_id="u1",
            sender_name="发送者",
            group_id="g1",
            content="[合并转发]",
            segments=[segment],
            outer_text="",
            rich_only=True,
            rich_type="forward",
        )
        enricher = RichMediaEnricher({}, api_call)

        await enricher.enrich(message, directed=False)

        self.assertEqual(calls[0][0], "get_forward_msg")
        self.assertEqual(calls[0][1], {"message_id": "f-1"})
        self.assertIn("共2条", message.content)
        self.assertIn("小明：周六吃火锅", message.content)
        self.assertIn("小红：[图片]", message.content)
        self.assertEqual(message.outer_text, "")

    async def test_link_preview_runs_only_for_directed_message_by_default(self):
        segment = MessageSegment(
            type="link",
            summary="[链接：example.com]",
            url="https://example.com/article",
        )
        message = Message(
            message_id="m1",
            message_type="group",
            sender_id="u1",
            sender_name="发送者",
            group_id="g1",
            content=segment.summary,
            segments=[segment],
        )
        enricher = RichMediaEnricher({}, lambda *_: None)
        calls = 0

        async def fake_fetch(url):
            nonlocal calls
            calls += 1
            return "文章标题", "一段摘要"

        enricher._fetch_preview = fake_fetch
        await enricher.enrich(message, directed=False)
        self.assertEqual(calls, 0)

        await enricher.enrich(message, directed=True)
        self.assertEqual(calls, 1)
        self.assertIn("文章标题", message.content)


class RichMessageFloorTests(unittest.TestCase):
    def _analyze(self, rich_type: str):
        current = ContextMessage(
            sender_id="u1",
            sender_name="小明",
            content=f"[{rich_type}]",
            message_id="m1",
        )
        return ConversationFloorManager().analyze(
            current,
            [current],
            rich_message_only=True,
            rich_type=rich_type,
        )[1]

    def test_pure_media_only_allows_short_reaction(self):
        plan = self._analyze("image")
        self.assertEqual(plan.action, ActionType.REACT)
        self.assertLessEqual(plan.max_chars, 8)

    def test_pure_link_and_forward_default_to_silent(self):
        self.assertEqual(self._analyze("link").action, ActionType.SILENT)
        self.assertEqual(self._analyze("forward").action, ActionType.SILENT)


class NapCatApiResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_api_matches_echo_and_returns_data(self):
        # 延迟导入，复用项目测试环境对可选 websockets 依赖的处理。
        try:
            from core.adapter.qq_adapter import QQAdapter
        except ModuleNotFoundError as exc:
            if exc.name != "websockets":
                raise
            import sys
            import types
            fake = types.ModuleType("websockets")
            fake.WebSocketServerProtocol = object
            fake.exceptions = types.SimpleNamespace(ConnectionClosed=Exception)
            sys.modules["websockets"] = fake
            from core.adapter.qq_adapter import QQAdapter

        sent = []
        sent_event = asyncio.Event()

        class Client:
            async def send(self, payload):
                sent.append(json.loads(payload))
                sent_event.set()

        adapter = QQAdapter({"self_id": "42"})
        adapter._clients.add(Client())
        task = asyncio.create_task(
            adapter.call_api("get_forward_msg", {"message_id": "f1"}, 1.0)
        )
        await asyncio.wait_for(sent_event.wait(), timeout=1.0)
        echo = sent[0]["echo"]
        await adapter._handle_message(json.dumps({
            "status": "ok",
            "retcode": 0,
            "data": {"messages": [1, 2]},
            "echo": echo,
        }))

        self.assertEqual(await task, {"messages": [1, 2]})
        self.assertFalse(adapter._pending_api)


if __name__ == "__main__":
    unittest.main()

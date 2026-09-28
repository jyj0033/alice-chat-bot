"""富媒体解析、NapCat 回执和拟人策略的离线测试。"""

import asyncio
import base64
import json
import unittest

from core.adapter.base import Message
from core.adapter.rich_content import (
    MessageSegment,
    parse_message_segments,
    render_outer_text,
    render_segments,
)
from core.adapter.rich_media import (
    RichMediaEnricher,
    VisionResult,
    _shrink_image_bytes,
    _strip_scene_prefix,
    _strip_thinking,
    _validate_http_url,
)
from modules.memory.context import ContextMessage
from modules.social.conversation_floor import ActionType, ConversationFloorManager


class RichMessageParsingTests(unittest.TestCase):
    def test_image_url_is_structured_but_not_exposed_to_context(self):
        secret_url = "https://cdn.example.com/a.jpg?token=top-secret"
        segments = parse_message_segments([
            {"type": "text", "data": {"text": "看看这个 "}},
            {"type": "image", "data": {"file": "a.jpg", "url": secret_url}},
        ])

        self.assertEqual(render_segments(segments), "看看这个 一张图，看不清画面。")
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
    def test_vision_result_requires_structured_evidence(self):
        result = RichMediaEnricher._parse_vision_result(
            '{"description":"黑发双马尾人物，手里拿着杯子",'
            '"confidence":0.91,"uncertain":false}'
        )
        self.assertIsInstance(result, VisionResult)
        self.assertEqual(result.description, "黑发双马尾人物，手里拿着杯子")
        self.assertEqual(result.confidence, 0.91)
        self.assertFalse(result.uncertain)

        # 兼容旧视觉模型的纯文本输出，但必须带不确定标记，交给后续复核。
        legacy = RichMediaEnricher._parse_vision_result("一张看不清的图片")
        self.assertEqual(legacy.description, "一张看不清的图片")
        self.assertTrue(legacy.uncertain)

    def test_image_prompt_asks_to_identify_before_describing(self):
        """prompt 必须先要求认人/认作品，只描述画面等于没识别。"""
        enricher = RichMediaEnricher({}, lambda *_: None)
        prompt = enricher._build_image_prompt(
            MessageSegment(type="mface"),
            "小明说：这个不铲还偷偷练？",
        )

        self.assertIn("先认出这张图是什么", prompt)
        self.assertIn("作品名 + 角色名", prompt)
        self.assertIn("看不出具体是谁", prompt)
        self.assertIn("绝对不要编造角色名、作品名、出处", prompt)
        self.assertIn("不要把前文人物、事件、评价或原话写进图片描述", prompt)
        self.assertIn('"confidence"', prompt)
        # 旧的「只写直接可观察的事实」写法会把识别能力一起禁掉
        self.assertNotIn("只写图片中直接可观察的事实", prompt)


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
        self.assertIn("小红：一张图，看不清画面。", message.content)
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


class VisionThinkingAndPayloadTests(unittest.TestCase):
    """端点先吐 <think> 再给 JSON，且不接受超过 10MiB 的媒体。"""

    def test_thinking_block_is_stripped(self):
        self.assertEqual(_strip_thinking("<think>a\nb</think>正文"), "正文")
        # 被截断时只有开标签，也要把整段思考吃掉
        self.assertEqual(_strip_thinking("正文<think>想了一半就断了"), "正文")
        self.assertEqual(_strip_thinking("<thinking>x</thinking>ok"), "ok")

    def test_description_is_read_from_json_after_thinking(self):
        raw = (
            "<think>The image shows a silver-haired girl with cat ears; "
            "could be from a known game.</think>\n"
            '{"description":"《崩坏：星穹铁道》风格的角色表情包",'
            '"confidence":0.9,"uncertain":false}'
        )
        result = RichMediaEnricher._parse_vision_result(raw)
        self.assertEqual(result.description, "《崩坏：星穹铁道》风格的角色表情包")
        self.assertFalse(result.uncertain)

    def test_thinking_is_never_used_as_a_description(self):
        """整段输出被思考块吃光时，视为识别失败，不能把英文推理写进上下文。"""
        truncated = "<think>The user wants a brief objective description of what is</think>"
        self.assertIsNone(RichMediaEnricher._parse_vision_result(truncated))

        unclosed = "<think>The image shows an anime-style character with teal hair"
        self.assertIsNone(RichMediaEnricher._parse_vision_result(unclosed))

    def test_last_json_object_wins(self):
        """思考块里也可能出现大括号，真正的答案在最后。"""
        raw = (
            '<think>maybe {"draft":"wrong"} or others</think>'
            '{"description":"熊猫头借口龙表情包","confidence":0.95,"uncertain":false}'
        )
        result = RichMediaEnricher._parse_vision_result(raw)
        self.assertEqual(result.description, "熊猫头借口龙表情包")

    def test_fenced_json_is_accepted(self):
        raw = (
            "<think>thinking…</think>\n```json\n"
            '{"description":"《蔚蓝档案》风格表情包","confidence":0.7,"uncertain":false}\n```'
        )
        result = RichMediaEnricher._parse_vision_result(raw)
        self.assertEqual(result.description, "《蔚蓝档案》风格表情包")

    def test_oversized_image_is_shrunk_below_the_endpoint_limit(self):
        """端点硬上限 10MiB；QQ 原图常见 11~18MB，必须压下去。"""
        from PIL import Image

        import io as _io
        import os as _os

        # 必须用噪声：纯色/渐变图压出来只有几百 KB，根本走不到缩放分支。
        picture = Image.frombytes("RGB", (1800, 1000), _os.urandom(1800 * 1000 * 3))
        buffer = _io.BytesIO()
        picture.save(buffer, format="JPEG", quality=100)
        original = buffer.getvalue()
        self.assertGreater(len(original), 2_000_000)

        shrunk, media_type = _shrink_image_bytes(
            original,
            "image/jpeg",
            max_side=800,
            quality=85,
            max_bytes=2_000_000,
        )
        self.assertEqual(media_type, "image/jpeg")
        self.assertLessEqual(len(shrunk), 2_000_000)
        self.assertLess(len(shrunk), len(original))
        with Image.open(_io.BytesIO(shrunk)) as check:
            # 断言的是传入的 max_side，确认参数真的透传下去了
            self.assertLessEqual(max(check.size), 800)

    def test_animated_gif_is_flattened_to_one_static_frame(self):
        from PIL import Image

        import io as _io

        # 这张 GIF 只有 3KB，重编码成 JPEG 反而更大——旧逻辑「变大就用原图」会让它
        # 原样发给端点，等于没拍平。动图必须强制转单帧 JPEG。
        frames = [Image.new("RGB", (900, 900), color) for color in ((255, 0, 0), (0, 255, 0))]
        buffer = _io.BytesIO()
        frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=200)
        original = buffer.getvalue()
        self.assertGreater(getattr(Image.open(_io.BytesIO(original)), "n_frames", 1), 1)

        shrunk, media_type = _shrink_image_bytes(
            original, "image/gif", max_side=1600, quality=85, max_bytes=2_000_000
        )
        self.assertEqual(media_type, "image/jpeg")
        with Image.open(_io.BytesIO(shrunk)) as check:
            self.assertEqual(getattr(check, "n_frames", 1), 1)

    def test_small_jpeg_is_passed_through_untouched(self):
        from PIL import Image

        import io as _io

        buffer = _io.BytesIO()
        Image.new("RGB", (320, 240), (10, 20, 30)).save(buffer, format="JPEG", quality=90)
        original = buffer.getvalue()
        shrunk, media_type = _shrink_image_bytes(
            original, "image/jpeg", max_side=1600, quality=85, max_bytes=2_000_000
        )
        self.assertEqual(shrunk, original)
        self.assertEqual(media_type, "image/jpeg")

    def test_broken_bytes_fall_back_to_the_original(self):
        """预处理不能因为一张坏图把整个识别流程整死。"""
        broken = b"\xff\xd8\xff\xe0not-really-a-jpeg"
        shrunk, media_type = _shrink_image_bytes(
            broken, "image/jpeg", max_side=1600, quality=85, max_bytes=2_000_000
        )
        self.assertEqual(shrunk, broken)
        self.assertEqual(media_type, "image/jpeg")

    def test_truncated_json_still_yields_the_description(self):
        """线上实测：长输出被 max_tokens 截断，JSON 不闭合，description 已成完整。"""
        raw = (
            "<think>这个是憋笑梗图</think>"
            '{"description":"一张 chibi 风格表情包，白发绿眼角色在做「憋笑」表情",'
            '"confidence":0.55'
        )
        result = RichMediaEnricher._parse_vision_result(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result.description, "一张 chibi 风格表情包，白发绿眼角色在做「憋笑」表情")
        self.assertAlmostEqual(result.confidence, 0.55)

    def test_truncated_json_never_leaks_raw_json_as_description(self):
        """线上实测的坏样子：整串 JSON 原文被当成描述写进群聊上下文。"""
        raw = (
            '{"description":"一张 chibi 风格表情包，画面中是一个白发绿眼的二次元角色",'
            '"confidence":0.55,"uncertain":false'
        )
        result = RichMediaEnricher._parse_vision_result(raw)
        self.assertIsNotNone(result)
        self.assertNotIn("{", result.description)
        self.assertNotIn("confidence", result.description)

    def test_json_fragment_without_description_is_a_failure(self):
        """连 description 的值都没吐出来时，宁可不写描述，也不写一坨 JSON。"""
        for raw in ('{"description":', '{"description":"', '{"confid'):
            with self.subTest(raw=raw):
                self.assertIsNone(RichMediaEnricher._parse_vision_result(raw))

    def test_salvaged_json_escapes_are_decoded(self):
        raw = '{"description":"第一行\\n第二行 \\"带引号\\"","confidence":0.7'
        result = RichMediaEnricher._parse_vision_result(raw)
        self.assertEqual(result.description, '第一行\n第二行 "带引号"')

    def test_scene_prefix_is_not_repeated_by_the_template(self):
        """模型也爱以「画面是…」开头，会和「一张图，画面是{}。」拼成「画面是画面是」。"""
        self.assertEqual(_strip_scene_prefix("画面是一个动漫角色"), "一个动漫角色")
        self.assertEqual(_strip_scene_prefix("图中是一位女孩"), "一位女孩")
        self.assertEqual(_strip_scene_prefix("这张图是熊猫头梗图"), "熊猫头梗图")
        # 没有引导语时不要乱删内容
        self.assertEqual(_strip_scene_prefix("熊猫头借口龙表情包"), "熊猫头借口龙表情包")
        self.assertEqual(_strip_scene_prefix(""), "")

    def test_get_image_fallback_also_shrinks(self):
        """QQ 图床直连失败时会走 get_image 读本地文件，这条路同样必须压缩。

        探针把「下载」换成了读本地文件，覆盖不到这个分支；而线上 QQ 图床
        带时效与防盗链，它其实才是常态路径。
        """
        from PIL import Image

        import io as _io
        import os as _os
        import tempfile

        picture = Image.frombytes("RGB", (1800, 1000), _os.urandom(1800 * 1000 * 3))
        buffer = _io.BytesIO()
        picture.save(buffer, format="JPEG", quality=100)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(buffer.getvalue())
            temp_path = handle.name
        self.addCleanup(lambda: _os.unlink(temp_path))

        async def fake_api_call(action, params, timeout):
            return {"path": temp_path} if action == "get_image" else {}

        enricher = RichMediaEnricher({}, fake_api_call)
        segment = MessageSegment(type="image", file="some-ref")
        data_url = asyncio.run(enricher._download_image_data_url_via_get_image(segment))
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        payload = base64.b64decode(data_url.split(",", 1)[1])
        self.assertLessEqual(len(payload), enricher.image_vision_max_payload)

    def test_wired_into_data_url_conversion(self):
        from PIL import Image

        import io as _io

        enricher = RichMediaEnricher({}, lambda *_: None)
        self.assertEqual(enricher.image_vision_max_payload, 2_000_000)
        self.assertEqual(enricher.image_vision_max_tokens, 2000)

        picture = Image.new("RGB", (3000, 2000), (200, 30, 30))
        buffer = _io.BytesIO()
        picture.save(buffer, format="PNG")
        data_url = enricher._to_vision_data_url(buffer.getvalue(), "image/png")
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        payload = base64.b64decode(data_url.split(",", 1)[1])
        self.assertLessEqual(len(payload), enricher.image_vision_max_payload)
        self.assertLess(len(payload), 10 * 1024 * 1024)


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
        self.assertLessEqual(plan.max_chars, 14)

    def test_pure_link_and_forward_default_to_silent(self):
        self.assertEqual(self._analyze("link").action, ActionType.SILENT)
        self.assertEqual(self._analyze("forward").action, ActionType.SILENT)


class NapCatApiResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_returns_ack_id_and_maps_bot_sender(self):
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
            adapter.send_message_with_id("group_123", "接着说", reply_to_id="m1")
        )
        await asyncio.wait_for(sent_event.wait(), timeout=1.0)
        request = sent[0]
        self.assertEqual(request["params"]["message"][0]["type"], "reply")

        await adapter._handle_message(json.dumps({
            "status": "ok",
            "retcode": 0,
            "data": {"message_id": 321},
            "echo": request["echo"],
        }))

        self.assertEqual(await task, (True, "321"))
        self.assertEqual(adapter.last_sent_message_id, "321")
        self.assertEqual(adapter._message_senders["321"], "42")

    async def test_duplicate_inbound_message_id_is_ignored(self):
        from core.adapter.qq_adapter import QQAdapter

        adapter = QQAdapter({"self_id": "42"})

        self.assertTrue(adapter._remember_message_id("m1"))
        self.assertFalse(adapter._remember_message_id("m1"))

    async def test_send_image_waits_for_napcat_ack(self):
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
            adapter.send_image("group_123", b"not-a-real-png")
        )
        await asyncio.wait_for(sent_event.wait(), timeout=1.0)

        request = sent[0]
        self.assertEqual(request["action"], "send_group_msg")
        self.assertEqual(request["params"]["group_id"], 123)
        self.assertTrue(request["params"]["message"][0]["data"]["file"].startswith("base64://"))
        self.assertIn("echo", request)
        self.assertFalse(task.done())

        await adapter._handle_message(json.dumps({
            "status": "ok",
            "retcode": 0,
            "data": {"message_id": 987},
            "echo": request["echo"],
        }))

        self.assertTrue(await task)
        self.assertEqual(adapter.messages_sent, 1)

    async def test_send_image_reports_napcat_failure(self):
        from core.adapter.qq_adapter import QQAdapter

        sent = []
        sent_event = asyncio.Event()

        class Client:
            async def send(self, payload):
                sent.append(json.loads(payload))
                sent_event.set()

        adapter = QQAdapter({"self_id": "42"})
        adapter._clients.add(Client())
        task = asyncio.create_task(adapter.send_image("group_123", b"image"))
        await asyncio.wait_for(sent_event.wait(), timeout=1.0)
        request = sent[0]
        await adapter._handle_message(json.dumps({
            "status": "failed",
            "retcode": 1200,
            "message": "图片发送失败",
            "echo": request["echo"],
        }))

        self.assertFalse(await task)
        self.assertEqual(adapter.messages_sent, 0)

    async def test_rejected_gif_retries_as_first_frame_png(self):
        from io import BytesIO

        from PIL import Image

        from core.adapter.qq_adapter import QQAdapter

        gif_output = BytesIO()
        Image.new("P", (2, 2), 3).save(gif_output, format="GIF")
        gif_bytes = gif_output.getvalue()

        adapter = QQAdapter({"self_id": "42"})
        adapter._clients.add(object())
        calls = []

        async def fake_call_api(action, params):
            calls.append((action, params))
            if len(calls) == 1:
                raise RuntimeError("NapCat 不支持该 GIF")
            return {"message_id": 988}

        adapter.call_api = fake_call_api

        self.assertTrue(await adapter.send_image("group_123", gif_bytes))
        self.assertEqual(len(calls), 2)
        first_file = calls[0][1]["message"][0]["data"]["file"]
        second_file = calls[1][1]["message"][0]["data"]["file"]
        self.assertTrue(base64.b64decode(first_file.removeprefix("base64://")).startswith(b"GIF"))
        self.assertTrue(base64.b64decode(second_file.removeprefix("base64://")).startswith(b"\x89PNG"))
        self.assertEqual(adapter.messages_sent, 1)

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

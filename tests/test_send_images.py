"""批量原图发送的离线测试。"""

import base64
import tempfile
import unittest
from pathlib import Path

from core.adapter.qq_adapter import QQAdapter


class _FakeBot:
    """只记录调用的假 Bot。"""

    def __init__(self):
        self.calls = []

    async def send_images(self, session_id, images, *, text="", mode="chunks", per_message=9):
        self.calls.append(
            {
                "session_id": session_id,
                "images": list(images),
                "text": text,
                "mode": mode,
                "per_message": per_message,
            }
        )
        return {"success": True, "sent": len(images), "mode": mode}


def _adapter(call_api_result=None):
    adapter = QQAdapter({"self_id": "1", "ws_port": 3001})
    adapter._clients = {object()}  # 假装 NapCat 已连接
    adapter.last_call = None
    adapter.calls = []

    async def _fake_call_api(action, params, timeout=None):
        adapter.last_call = (action, params, timeout)
        adapter.calls.append((action, params, timeout))
        return call_api_result if call_api_result is not None else {"message_id": 999}

    adapter.call_api = _fake_call_api
    return adapter


def _image_count(params):
    return sum(1 for seg in params["message"] if seg["type"] == "image")


class SendImagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunks_split_by_qq_limit(self):
        adapter = _adapter()
        payloads = [b"img-%d" % index for index in range(37)]
        result = await adapter.send_images("private_1021803090", payloads)

        self.assertTrue(result["success"])
        self.assertEqual(result["sent"], 37)
        self.assertEqual(result["message_count"], 5)
        self.assertEqual(len(adapter.calls), 5)

    async def test_chunk_sizes_never_exceed_nine(self):
        adapter = _adapter()
        payloads = [b"x" for _ in range(37)]
        await adapter.send_images("private_1021803090", payloads)

        sizes = [_image_count(params) for _, params, _ in adapter.calls]
        self.assertEqual(sizes, [9, 9, 9, 9, 1])

    async def test_per_message_is_clamped_to_qq_limit(self):
        adapter = _adapter()
        await adapter.send_images("private_1", [b"x"] * 20, per_message=20)
        action, params, _ = adapter.calls[0]
        self.assertEqual(action, "send_private_msg")
        self.assertEqual(_image_count(params), 9)
        self.assertEqual(len(adapter.calls), 3)  # 20 张 → 9 + 9 + 2

    async def test_payload_keeps_original_bytes(self):
        adapter = _adapter()
        raw = b"\xff\xd8\xff\xe0" + b"original-jpeg-bytes"
        await adapter.send_images("private_1", [raw])
        _, params, _ = adapter.calls[0]
        segment = params["message"][0]
        self.assertEqual(segment["type"], "image")
        encoded = segment["data"]["file"]
        self.assertTrue(encoded.startswith("base64://"))
        self.assertEqual(base64.b64decode(encoded[len("base64://") :]), raw)

    async def test_text_segment_only_in_first_message(self):
        adapter = _adapter()
        await adapter.send_images("private_1", [b"a"] * 10, text="漫画原图")
        self.assertEqual(adapter.calls[0][1]["message"][0]["type"], "text")
        self.assertTrue(
            all(seg["type"] != "text" for seg in adapter.calls[1][1]["message"])
        )

    async def test_reply_segment_only_in_first_message(self):
        adapter = _adapter()
        await adapter.send_images(
            "private_1", [b"a"] * 12, reply_to_id="msg-1"
        )
        self.assertEqual(adapter.calls[0][1]["message"][0]["type"], "reply")
        self.assertTrue(
            all(seg["type"] != "reply" for seg in adapter.calls[1][1]["message"])
        )

    async def test_forward_mode_builds_one_node_per_image(self):
        adapter = _adapter(call_api_result={"message_id": 555, "res_id": "res-1"})
        result = await adapter.send_images(
            "private_1021803090", [b"a", b"b", b"c"], mode="forward"
        )
        action, params, _ = adapter.calls[0]
        self.assertEqual(action, "send_private_forward_msg")
        self.assertEqual(params["user_id"], 1021803090)
        self.assertEqual(len(params["messages"]), 3)
        for node in params["messages"]:
            self.assertEqual(node["type"], "node")
            segment = node["data"]["content"][0]
            self.assertEqual(segment["type"], "image")
            self.assertTrue(segment["data"]["file"].startswith("base64://"))
        self.assertTrue(result["success"])
        self.assertEqual(result["message_count"], 1)
        # 发送后应再调一次 get_forward_msg 核对图片数量
        self.assertEqual(adapter.calls[1][0], "get_forward_msg")

    async def test_forward_mode_for_group(self):
        adapter = _adapter(call_api_result={"message_id": 1})
        await adapter.send_images("group_724786753", [b"a"], mode="forward")
        action, params, _ = adapter.calls[0]
        self.assertEqual(action, "send_group_forward_msg")
        self.assertEqual(params["group_id"], 724786753)

    async def test_empty_payload_is_rejected(self):
        adapter = _adapter()
        result = await adapter.send_images("private_1", [])
        self.assertFalse(result["success"])
        self.assertEqual(result["sent"], 0)

    async def test_invalid_session_is_rejected(self):
        adapter = _adapter()
        result = await adapter.send_images("1021803090", [b"a"])
        self.assertFalse(result["success"])

    async def test_disconnected_napcat_is_rejected(self):
        adapter = _adapter()
        adapter._clients = set()
        result = await adapter.send_images("private_1", [b"a"])
        self.assertFalse(result["success"])
        self.assertIn("NapCat", result["error"])

    def test_count_forward_images_handles_nested_nodes(self):
        nodes = [
            {"data": {"content": [{"type": "image"}, {"type": "text"}]}},
            {
                "data": {
                    "content": [
                        {"type": "node", "data": {"content": [{"type": "image"}]}}
                    ]
                }
            },
        ]
        self.assertEqual(QQAdapter._count_forward_images(nodes), 2)


class SendImagesEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_reads_files_and_calls_bot(self):
        import dashboard
        from starlette.requests import Request

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "02.jpeg").write_bytes(b"jpeg-bytes")
            original_dir = dashboard.SEND_MEDIA_DIR
            original_bot = dashboard.bot_instance
            dashboard.SEND_MEDIA_DIR = base
            fake = _FakeBot()
            dashboard.bot_instance = fake
            try:
                body = (
                    b'{"session": "private_1021803090", "paths": ["02.jpeg"],'
                    b' "mode": "forward"}'
                )

                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}

                request = Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/api/send/images",
                        "headers": [],
                        "query_string": b"",
                        "server": ("test", 80),
                        "client": ("test", 1),
                        "scheme": "http",
                        "root_path": "",
                        "http_version": "1.1",
                    },
                    receive,
                )
                result = await dashboard.send_images(request)
                self.assertTrue(result["success"])
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(fake.calls[0]["session_id"], "private_1021803090")
                self.assertEqual(fake.calls[0]["images"], [b"jpeg-bytes"])
                self.assertEqual(fake.calls[0]["mode"], "forward")
            finally:
                dashboard.SEND_MEDIA_DIR = original_dir
                dashboard.bot_instance = original_bot

    def test_path_traversal_is_rejected(self):
        import dashboard

        with tempfile.TemporaryDirectory() as tmp:
            original_dir = dashboard.SEND_MEDIA_DIR
            dashboard.SEND_MEDIA_DIR = Path(tmp)
            try:
                with self.assertRaises(ValueError):
                    dashboard._resolve_send_media_path("../../etc/passwd")
            finally:
                dashboard.SEND_MEDIA_DIR = original_dir


if __name__ == "__main__":
    unittest.main()

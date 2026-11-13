"""官方适配器跨事件循环调用（面板线程 → bot 主循环）的离线测试。

背景（2026-11-13 线上事故）：
面板（uvicorn 自带事件循环）和 bot 主循环不是同一个 loop，而官方适配器的
aiohttp session 建在主循环上。面板线程直接调 ``send_message_with_id`` 会抛：

    RuntimeError: Timeout context manager should be used inside a task

aiohttp 的超时上下文管理器要求在 asyncio Task 里执行，跨 loop 复用 session
就会踩到。适配器已有 ``_foreign_owner_loop()`` + ``run_coroutine_threadsafe``
这套机制（disconnect() 一直在用），但发送路径漏了，于是**面板上手动发消息
必然失败**。

覆盖：
  1. 面板线程发文本 → 自动切回适配器所属循环执行
  2. 面板线程发图片 → 同样切回
  3. 在适配器自己的循环里调用 → 不切换，直接执行
  4. 未连接时 → 返回失败，不发送
  5. 面板线程传素材 / 发图文卡片 → 同样要切回

  第 5 条是 2026-11-13 下午踩的第二个坑：素材上传和图文卡片是新加的路径，
  一开始忘了加跨循环保护，面板「手动发卡片」直接炸
  ``RuntimeError: Timeout context manager should be used inside a task``。
"""
import asyncio
import threading
import unittest
from pathlib import Path

from core.adapter.official_adapter import OfficialQQAdapter


class _FakeAdapter(OfficialQQAdapter):
    """跳过真实初始化，只保留跨循环判断所需的状态。"""

    def __init__(self, owner_loop, connected=True):
        # 不调用父类 __init__：那会去建连、起协程。
        self._owner_loop = owner_loop
        self._connected = connected
        self._http = None
        self._access_token = ""
        self._token_expires_at = 0.0
        self._msg_seq_used = {}
        self.messages_sent = 0
        self.last_sent_message_id = ""
        self._ref_idx_by_msg_id = {}
        self._msg_id_by_ref_idx = {}
        self._message_senders = {}
        # 素材索引：不落盘，测试里只需要它别炸。
        self._media_registry = {}
        self._media_registry_loaded = True
        self._media_registry_path = Path("unused")
        # self_id / _app_id 都是从 config 派生的，不能直接赋值。
        self.config = {"self_id": "bot-1", "official": {"app_id": "app-1"}}
        self.self_id = "bot-1"
        self.calls = []

    async def _post_message(self, session_id, body, timeout=30.0):
        self.calls.append((session_id, dict(body)))
        return {"id": "msg-out-1"}

    async def upload_file(self, session_id, image_bytes):
        self.calls.append(("upload", session_id))
        return "file-info-1"

    async def _request_json(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if path.endswith("/prepare"):
            return {
                "upload_id": "upload-1",
                "presigned_url": "https://cos.example.com/put",
            }
        if path.endswith("/complete"):
            return {
                "image_id": "img-1",
                "url": "qqbot-media://image/v1/img-1",
                "preview_url": "https://cos.example.com/preview",
            }
        return {}

    async def _put_bytes(self, url, blob, timeout=120.0, content_type=""):
        self.calls.append(("put", url, content_type))
        return {"ETag": '"abc123"'}

    def _save_media_registry(self):
        """测试里不落盘。"""
        self.calls.append(("save_registry", "", {}))


def _run_in_other_loop(coro_factory, other_loop):
    """把协程丢到另一个 loop 里跑，返回其结果。

    模拟面板：uvicorn 的事件循环和 bot 主循环不是同一个。
    用 ``run_until_complete`` 驱动那个 loop，而不是在它里面同步等 future
    （那会死锁：被等的协程正等着这个 loop 推进）。
    """
    return other_loop.run_until_complete(coro_factory())


class CrossLoopSendTests(unittest.TestCase):
    """面板线程（另一个 loop）调用发送时必须切回适配器所属循环。"""

    def test_foreign_loop_send_text_reroutes_to_owner_loop(self):
        owner_loop = asyncio.new_event_loop()
        other_loop = asyncio.new_event_loop()
        owner_thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        owner_thread.start()
        try:
            adapter = _FakeAdapter(owner_loop)

            ok, mid = _run_in_other_loop(
                lambda: adapter.send_message_with_id("group_g1", "你好"), other_loop
            )

            self.assertTrue(ok, "跨循环发送没有成功")
            self.assertEqual(mid, "msg-out-1")
            self.assertEqual(len(adapter.calls), 1)
            self.assertEqual(adapter.calls[0][0], "group_g1")
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            owner_thread.join(timeout=5)
            owner_loop.close()
            other_loop.close()

    def test_foreign_loop_send_image_reroutes_to_owner_loop(self):
        owner_loop = asyncio.new_event_loop()
        other_loop = asyncio.new_event_loop()
        owner_thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        owner_thread.start()
        try:
            adapter = _FakeAdapter(owner_loop)

            ok, mid = _run_in_other_loop(
                lambda: adapter.send_image_with_id("group_g1", b"fake-image-bytes"),
                other_loop,
            )

            self.assertTrue(ok, "跨循环发图没有成功")
            self.assertEqual(mid, "msg-out-1")
            self.assertTrue(any(c[0] == "upload" for c in adapter.calls))
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            owner_thread.join(timeout=5)
            owner_loop.close()
            other_loop.close()

    def test_foreign_loop_media_upload_reroutes_to_owner_loop(self):
        """面板线程传素材：aiohttp session 在主循环上，必须切回去。"""
        owner_loop = asyncio.new_event_loop()
        other_loop = asyncio.new_event_loop()
        owner_thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        owner_thread.start()
        try:
            adapter = _FakeAdapter(owner_loop)

            media = _run_in_other_loop(
                lambda: adapter.upload_media_image(b"\x89PNG" + b"0" * 32, "a.png"),
                other_loop,
            )

            self.assertEqual(media["image_id"], "img-1")
            self.assertTrue(any(c[0] == "put" for c in adapter.calls))
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            owner_thread.join(timeout=5)
            owner_loop.close()
            other_loop.close()

    def test_foreign_loop_send_card_reroutes_to_owner_loop(self):
        owner_loop = asyncio.new_event_loop()
        other_loop = asyncio.new_event_loop()
        owner_thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        owner_thread.start()
        try:
            adapter = _FakeAdapter(owner_loop)

            ok, mid = _run_in_other_loop(
                lambda: adapter.send_card_with_id(
                    "group_g1",
                    title="标题",
                    description="描述",
                    url="https://example.com",
                    pic_url="https://example.com/a.png",
                ),
                other_loop,
            )

            self.assertTrue(ok, "跨循环发图文卡片没有成功")
            self.assertEqual(mid, "msg-out-1")
            body = adapter.calls[0][1]
            self.assertEqual(body["msg_type"], 8)
            self.assertEqual(body["card"]["type"], "tuwen")
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            owner_thread.join(timeout=5)
            owner_loop.close()
            other_loop.close()

    def test_same_loop_send_text_does_not_reroute(self):
        """在适配器自己的循环里调用时直接执行，不做多余的切换。"""
        owner_loop = asyncio.new_event_loop()

        async def scenario():
            adapter = _FakeAdapter(asyncio.get_running_loop())
            return await adapter.send_message_with_id("group_g1", "你好")

        try:
            ok, mid = owner_loop.run_until_complete(scenario())
            self.assertTrue(ok)
            self.assertEqual(mid, "msg-out-1")
        finally:
            owner_loop.close()

    def test_not_connected_returns_false_without_sending(self):
        """未连接时不能因为跨循环逻辑就去发。"""
        owner_loop = asyncio.new_event_loop()

        async def scenario():
            adapter = _FakeAdapter(asyncio.get_running_loop(), connected=False)
            return await adapter.send_message_with_id("group_g1", "你好")

        try:
            ok, mid = owner_loop.run_until_complete(scenario())
            self.assertFalse(ok)
            self.assertEqual(mid, "")
        finally:
            owner_loop.close()


class ForeignOwnerLoopTests(unittest.TestCase):
    """_foreign_owner_loop 的判断本身。"""

    def test_returns_none_in_owner_loop(self):
        owner_loop = asyncio.new_event_loop()

        async def scenario():
            adapter = _FakeAdapter(asyncio.get_running_loop())
            return adapter._foreign_owner_loop()

        try:
            self.assertIsNone(owner_loop.run_until_complete(scenario()))
        finally:
            owner_loop.close()

    def test_returns_owner_loop_from_other_loop(self):
        owner_loop = asyncio.new_event_loop()
        other_loop = asyncio.new_event_loop()
        owner_thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        owner_thread.start()
        try:
            adapter = _FakeAdapter(owner_loop)

            async def probe():
                return adapter._foreign_owner_loop()

            result = _run_in_other_loop(probe, other_loop)
            self.assertIs(result, owner_loop)
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            owner_thread.join(timeout=5)
            owner_loop.close()
            other_loop.close()


if __name__ == "__main__":
    unittest.main()

"""官方通道的图文卡片、素材管理和 @全体成员测试。

覆盖官方 2026-10-28 上新的三样东西：

* **图文卡片**（``msg_type=8``）：标题 + 描述 + 图 + 链接；
* **素材管理**（``qqbot-media://``）：图片存平台，Markdown 里直接引用。

不连真实网关：用假的 ``_request_json`` / ``_put_bytes`` 顶掉 HTTP。
"""

import json
import tempfile
import unittest
from pathlib import Path

from core.adapter.official_adapter import OfficialQQAdapter


GROUP_OPENID = "B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5"
APP_ID = "1020899999"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"0" * 64


def _adapter(tmpdir: str, **overrides):
    config = {
        "self_id": APP_ID,
        "official": {
            "app_id": APP_ID,
            "app_secret": "secret-value",
            # 状态文件指到临时目录，别把测试产物写进仓库的 data/
            "media_registry_file": str(Path(tmpdir) / "media.json"),
        },
    }
    config.update(overrides)
    adapter = OfficialQQAdapter(config)
    adapter.calls = []
    adapter.responses = {}
    adapter.puts = []

    async def _fake_request(method, path, **kwargs):
        adapter.calls.append((method, path, kwargs))
        key = (method, path)
        if key in adapter.responses:
            value = adapter.responses[key]
            if isinstance(value, Exception):
                raise value
            return value
        if path.endswith("/messages"):
            return {"id": "ROBOT1.0_sent"}
        return {}

    async def _fake_put(url, blob, timeout=120.0, content_type=""):
        adapter.puts.append((url, blob, content_type))
        return {"ETag": '"deadbeef"'}

    adapter._request_json = _fake_request
    adapter._put_bytes = _fake_put
    adapter._connected = True
    return adapter


def _media_responses(adapter, image_id="IMG_1"):
    adapter.responses[("POST", "/v2/media-images/prepare")] = {
        "upload_id": "upload_1",
        "presigned_url": "https://cos.example.com/put",
        "expires_in": 3600,
    }
    adapter.responses[("POST", "/v2/media-images/complete")] = {
        "image_id": image_id,
        "url": f"qqbot-media://image/v1/{image_id}",
        "ttl": 7776000,
        "preview_url": f"https://cos.example.com/preview/{image_id}",
    }


class CardTests(unittest.IsolatedAsyncioTestCase):
    """图文卡片（官方 msg_type=8）。"""

    async def test_card_body_matches_official_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)

            ok, message_id = await adapter.send_card_with_id(
                f"group_{GROUP_OPENID}",
                title="爱丽丝的群聊日报",
                description="今天群里聊了 42 条消息",
                url="https://example.com/report",
                pic_url="https://example.com/report.png",
            )

            self.assertTrue(ok)
            self.assertTrue(message_id)
            method, path, kwargs = adapter.calls[-1]
            self.assertEqual(method, "POST")
            self.assertEqual(path, f"/v2/groups/{GROUP_OPENID}/messages")
            body = kwargs["json_body"]
            self.assertEqual(body["msg_type"], 8)
            self.assertEqual(body["card"]["type"], "tuwen")
            content = body["card"]["content"]
            self.assertEqual(content["title"], "爱丽丝的群聊日报")
            self.assertEqual(content["description"], "今天群里聊了 42 条消息")
            self.assertEqual(content["url"], "https://example.com/report")
            self.assertEqual(content["pic_url"], "https://example.com/report.png")

    async def test_card_rejects_overlong_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)

            ok, _ = await adapter.send_card_with_id(
                f"group_{GROUP_OPENID}",
                title="标" * 51,
                description="描述",
                url="https://example.com",
                pic_url="https://example.com/a.png",
            )

            self.assertFalse(ok)
            self.assertFalse([c for c in adapter.calls if c[1].endswith("/messages")])

    async def test_card_rejects_overlong_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)

            ok, _ = await adapter.send_card_with_id(
                f"group_{GROUP_OPENID}",
                title="标题",
                description="描" * 201,
                url="https://example.com",
                pic_url="https://example.com/a.png",
            )

            self.assertFalse(ok)

    async def test_card_rejects_non_https_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)

            ok, _ = await adapter.send_card_with_id(
                f"group_{GROUP_OPENID}",
                title="标题",
                description="描述",
                url="http://example.com",
                pic_url="https://example.com/a.png",
            )
            self.assertFalse(ok)

            ok, _ = await adapter.send_card_with_id(
                f"group_{GROUP_OPENID}",
                title="标题",
                description="描述",
                url="https://example.com",
                pic_url="ftp://example.com/a.png",
            )
            self.assertFalse(ok)
            self.assertFalse([c for c in adapter.calls if c[1].endswith("/messages")])

    async def test_card_needs_title_and_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)

            ok, _ = await adapter.send_card_with_id(
                f"group_{GROUP_OPENID}",
                title="   ",
                description="描述",
                url="https://example.com",
                pic_url="https://example.com/a.png",
            )
            self.assertFalse(ok)

    async def test_card_helper_returns_bool(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)

            self.assertTrue(
                await adapter.send_card(
                    f"group_{GROUP_OPENID}",
                    title="标题",
                    description="描述",
                    url="https://example.com",
                    pic_url="https://example.com/a.png",
                )
            )

    async def test_card_with_image_uses_media_preview_url(self):
        """卡片图片地址必须是 https，只能从素材库预览链接拿。"""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            ok, _ = await adapter.send_card_with_image(
                f"group_{GROUP_OPENID}",
                PNG_BYTES,
                title="日报",
                description="描述",
                purpose="group_report:test",
            )

            self.assertTrue(ok)
            body = adapter.calls[-1][2]["json_body"]
            self.assertEqual(body["msg_type"], 8)
            self.assertTrue(body["card"]["content"]["pic_url"].startswith("https://"))


class MediaTests(unittest.IsolatedAsyncioTestCase):
    """素材管理（qqbot-media://）。"""

    async def test_upload_does_prepare_put_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            result = await adapter.upload_media_image(PNG_BYTES, "a.png")

            self.assertEqual(result["image_id"], "IMG_1")
            self.assertEqual(result["url"], "qqbot-media://image/v1/IMG_1")
            self.assertEqual(result["preview_url"], "https://cos.example.com/preview/IMG_1")
            # 直传那一步带了 Content-Type，和声明的一致
            self.assertEqual(len(adapter.puts), 1)
            self.assertEqual(adapter.puts[0][2], "image/png")
            # complete 用的是 COS 返回的 ETag
            complete = [c for c in adapter.calls if c[1].endswith("/complete")][0]
            self.assertEqual(complete[2]["json_body"]["upload_id"], "upload_1")
            self.assertEqual(complete[2]["json_body"]["md5"], "deadbeef")

    async def test_prepare_declares_size_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            await adapter.upload_media_image(PNG_BYTES, "a.png")

            prepare = [c for c in adapter.calls if c[1].endswith("/prepare")][0]
            body = prepare[2]["json_body"]
            self.assertEqual(body["file_name"], "a.png")
            self.assertEqual(body["file_size"], len(PNG_BYTES))
            self.assertEqual(body["content_type"], "image/png")
            self.assertEqual(len(body["md5"]), 32)
            self.assertEqual(len(body["sha1"]), 40)

    async def test_jpeg_is_normalized_to_jpeg(self):
        """官方要求扩展名和 Content-Type 一致，jpg 统一改成 jpeg。"""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            await adapter.upload_media_image(JPEG_BYTES, "photo.jpg")

            prepare = [c for c in adapter.calls if c[1].endswith("/prepare")][0]
            body = prepare[2]["json_body"]
            self.assertEqual(body["file_name"], "photo.jpeg")
            self.assertEqual(body["content_type"], "image/jpeg")

    async def test_same_purpose_deletes_the_previous_image(self):
        """素材库上限 100 张，同用途的旧图要先删掉。"""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            await adapter.upload_media_image(PNG_BYTES, "a.png", purpose="group_report:x")
            await adapter.upload_media_image(PNG_BYTES, "b.png", purpose="group_report:x")

            deletes = [c for c in adapter.calls if c[0] == "DELETE"]
            self.assertEqual(len(deletes), 1)
            self.assertEqual(deletes[0][1], "/v2/media-images/IMG_1/delete")

    async def test_registry_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            await adapter.upload_media_image(PNG_BYTES, "a.png", purpose="group_report:x")

            saved = json.loads(
                (Path(tmp) / "media.json").read_text(encoding="utf-8")
            )
            entries = saved["group_report:x"]
            self.assertIsInstance(entries, list)
            self.assertEqual(entries[0]["image_id"], "IMG_1")

    async def test_keep_retains_recent_images_of_the_same_purpose(self):
        """日报要留最近几张，删素材时别把刚发的图也删了。"""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            for index in range(3):
                adapter.responses[("POST", "/v2/media-images/complete")] = {
                    "image_id": f"IMG_{index + 1}",
                    "url": f"qqbot-media://image/v1/IMG_{index + 1}",
                    "ttl": 7776000,
                    "preview_url": f"https://cos.example.com/preview/IMG_{index + 1}",
                }
                await adapter.upload_media_image(
                    PNG_BYTES, "a.png", purpose="group_report:x", keep=2
                )

            deletes = [c for c in adapter.calls if c[0] == "DELETE"]
            # 传了三张、留两张 → 只删掉最旧的那张
            self.assertEqual(len(deletes), 1)
            self.assertEqual(deletes[0][1], "/v2/media-images/IMG_1/delete")
            saved = json.loads(
                (Path(tmp) / "media.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["image_id"] for item in saved["group_report:x"]],
                ["IMG_2", "IMG_3"],
            )

    async def test_oversized_image_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            with self.assertRaises(RuntimeError):
                await adapter.upload_media_image(
                    b"\x89PNG" + b"0" * (10 * 1024 * 1024 + 1)
                )
            self.assertFalse([c for c in adapter.calls if c[1].endswith("/prepare")])

    async def test_list_media_images_passes_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            adapter.responses[("GET", "/v2/media-images")] = {"images": [], "total": 0}

            await adapter.list_media_images()

            self.assertIsNone(adapter.calls[-1][2]["params"])

            await adapter.list_media_images("next-page")
            self.assertEqual(adapter.calls[-1][2]["params"], {"cursor": "next-page"})

    async def test_markdown_image_ref_always_carries_size(self):
        """安卓端不给尺寸就只显示替代文字，尺寸必须显式写。"""
        ref = OfficialQQAdapter.markdown_image_ref("qqbot-media://image/v1/tok", "日报")
        self.assertEqual(ref, "![日报 #0px #0px](qqbot-media://image/v1/tok)")

        sized = OfficialQQAdapter.markdown_image_ref(
            "qqbot-media://image/v1/tok", "日报", 800, 1200
        )
        self.assertEqual(sized, "![日报 #800px #1200px](qqbot-media://image/v1/tok)")

        self.assertEqual(OfficialQQAdapter.markdown_image_ref(""), "")


class MarkdownWithImageTests(unittest.IsolatedAsyncioTestCase):
    """「文字 + 图」的 Markdown 消息（图片走素材库引用）。

    这是带图长驻的正经做法：``qqbot-media://`` 引用 90 天滑动有效，
    官方发送时会转存，已发出去的消息不受素材删除影响。
    """

    async def test_uploads_media_and_appends_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            ok, message_id = await adapter.send_markdown_with_image(
                f"group_{GROUP_OPENID}",
                PNG_BYTES,
                "## 群聊日报\n\n> 消息 12 条",
                purpose="group_report:test",
                alt="群聊日报",
                width=800,
                height=1200,
            )

            self.assertTrue(ok)
            self.assertTrue(message_id)
            body = adapter.calls[-1][2]["json_body"]
            self.assertEqual(body["msg_type"], 2)
            content = body["markdown"]["content"]
            self.assertIn("## 群聊日报", content)
            self.assertIn("![群聊日报 #800px #1200px]", content)
            self.assertIn("qqbot-media://image/v1/IMG_1", content)

    async def test_uses_media_reference_not_preview_url(self):
        """预览链接只有 1 小时有效期，长驻的图必须用 qqbot-media:// 引用。"""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            _media_responses(adapter)

            await adapter.send_markdown_with_image(
                f"group_{GROUP_OPENID}", PNG_BYTES, "正文"
            )

            content = adapter.calls[-1][2]["json_body"]["markdown"]["content"]
            self.assertNotIn("cos.example.com", content)

    async def test_upload_failure_does_not_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(tmp)
            adapter.responses[("POST", "/v2/media-images/prepare")] = RuntimeError(
                "网络炸了"
            )

            ok, _ = await adapter.send_markdown_with_image(
                f"group_{GROUP_OPENID}", PNG_BYTES, "正文"
            )

            self.assertFalse(ok)
            self.assertFalse([c for c in adapter.calls if c[1].endswith("/messages")])


if __name__ == "__main__":
    unittest.main()

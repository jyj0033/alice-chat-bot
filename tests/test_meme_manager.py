"""本地表情库的离线测试。"""

import base64
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

from core.adapter.base import Message
from core.adapter.rich_content import MessageSegment, parse_message_segments
from modules.meme_manager import MemeManager
from modules.reply.generator import ReplyGenerator


def _png_bytes(color=(124, 108, 255)):
    buffer = BytesIO()
    Image.new("RGB", (240, 160), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(color=(124, 108, 255)):
    buffer = BytesIO()
    Image.new("RGB", (240, 160), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _gif_bytes():
    buffer = BytesIO()
    Image.new("P", (240, 160), 3).save(buffer, format="GIF")
    return buffer.getvalue()


class MemeManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.manager = MemeManager(
            {"storage_path": self.temp_dir.name},
            base_dir=Path.cwd(),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_deduplicates_and_update_moves_file(self):
        content = _png_bytes()
        first = self.manager.add_bytes(content, category="吐槽", description="第一次保存")
        duplicate = self.manager.add_bytes(content, category="开心")

        self.assertFalse(first["duplicate"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(self.manager.stats()["total"], 1)
        self.assertEqual(self.manager.get(first["id"])["category"], "吐槽")

        pending = self.manager.add_bytes(_png_bytes((10, 20, 30)), category="待整理")
        self.manager.add_bytes(_png_bytes((10, 20, 30)), category="开心", meaning="开心的反应")
        self.assertEqual(self.manager.get(pending["id"])["category"], "开心")

        updated = self.manager.update(first["id"], category="开心", tags=["反应"])
        self.assertEqual(updated["category"], "开心")
        self.assertEqual(updated["tags"], ["反应"])
        item_and_bytes = self.manager.get_bytes(first["id"])
        self.assertIsNotNone(item_and_bytes)
        self.assertEqual(item_and_bytes[1], content)
        self.assertTrue((Path(self.temp_dir.name) / "开心" / first["filename"]).is_file())

    def test_sub_type_does_not_promote_plain_image_to_mface(self):
        segments = parse_message_segments([
            {"type": "image", "data": {"file": "ordinary.jpg", "sub_type": "1"}},
        ])
        self.assertEqual(segments[0].type, "image")

        marketface = parse_message_segments([
            {"type": "image", "data": {"file": "marketface", "sub_type": "1"}},
        ])
        self.assertEqual(marketface[0].type, "mface")

    def test_data_url_directive_and_delete(self):
        content = _png_bytes((255, 120, 160))
        data_url = "data:image/png;base64," + base64.b64encode(content).decode("ascii")
        item = self.manager.add_data_url(
            data_url,
            category="卖萌",
            meaning="被夸之后有点害羞，但其实很开心",
        )
        self.assertEqual(item["meaning"], "被夸之后有点害羞，但其实很开心")

        cleaned, category = self.manager.extract_directive("好耶 [[表情：卖萌]]")
        self.assertEqual(cleaned, "好耶")
        self.assertEqual(category, "卖萌")
        cleaned, category = self.manager.extract_directive("收到 &&meme:随机&&")
        self.assertEqual(cleaned, "收到")
        self.assertEqual(category, "")
        cleaned, category = self.manager.extract_directive(
            f"好，就发这张 [[表情:编号:{item['id'][:10]}]]"
        )
        self.assertEqual(cleaned, "好，就发这张")
        self.assertEqual(category, f"@id:{item['id'][:10]}")
        self.assertEqual(self.manager.resolve(item["id"][:10])["id"], item["id"])

        self.manager.update_config({
            "storage_path": self.temp_dir.name,
            "auto_send_enabled": True,
        })
        guide = self.manager.build_prompt_guide()
        self.assertIn("被夸之后有点害羞", guide)
        self.assertIn(item["id"][:10], guide)
        self.assertIn("编号:短编号", guide)

        self.assertTrue(self.manager.delete(item["id"]))
        self.assertIsNone(self.manager.get(item["id"]))
        self.assertFalse(self.manager.delete(item["id"]))

    def test_new_entries_only_accept_png_or_jpeg_and_send_as_png(self):
        with self.assertRaises(ValueError):
            self.manager.add_bytes(_gif_bytes(), category="吐槽")

        for content in (_png_bytes(), _jpeg_bytes(), _gif_bytes()):
            converted = self.manager.to_png_bytes(content)
            with Image.open(BytesIO(converted)) as image:
                self.assertEqual(image.format, "PNG")

    def test_auto_collect_skips_gif_content(self):
        gif = _gif_bytes()
        data_url = "data:image/gif;base64," + base64.b64encode(gif).decode("ascii")

        class Enricher:
            async def _download_image_data_url(self, segment):
                return data_url

        manager = MemeManager(
            {
                "storage_path": self.temp_dir.name,
                "auto_collect_enabled": True,
                "collect_cooldown_seconds": 0,
            },
            base_dir=Path.cwd(),
        )
        message = Message(
            message_id="gif-1",
            message_type="group",
            sender_id="u-gif",
            sender_name="小明",
            group_id="123",
            content="哈哈这个梗图",
            outer_text="哈哈这个梗图",
            segments=[MessageSegment(type="image", file="reaction.gif")],
        )
        adapter = SimpleNamespace(self_id="bot", rich_media_enricher=Enricher())

        import asyncio

        self.assertEqual(asyncio.run(manager.collect_message(message, adapter)), [])
        self.assertEqual(manager.stats()["total"], 0)

    def test_send_meme_passes_png_bytes_to_adapter(self):
        item = self.manager.add_bytes(_jpeg_bytes(), category="吐槽")
        sent: list[bytes] = []

        class Adapter:
            async def send_image(self, session_id, image_bytes, reply_to_id=None):
                sent.append(image_bytes)
                return True

        from main import GroupChatBot

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.meme_manager = self.manager
        bot.qq_adapter = Adapter()

        import asyncio

        result = asyncio.run(
            bot.send_meme("group_123", meme_id=item["id"])
        )
        self.assertTrue(result["success"])
        self.assertEqual(len(sent), 1)
        with Image.open(BytesIO(sent[0])) as image:
            self.assertEqual(image.format, "PNG")

    def test_send_meme_returns_outbound_message_id(self):
        item = self.manager.add_bytes(_png_bytes(), category="吐槽")

        class Adapter:
            async def send_image_with_id(self, session_id, image_bytes, reply_to_id=None):
                return True, "img-42"

        from main import GroupChatBot

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.meme_manager = self.manager
        bot.qq_adapter = Adapter()

        import asyncio

        result = asyncio.run(bot.send_meme("group_123", meme_id=item["id"]))
        self.assertTrue(result["success"])
        self.assertEqual(result["message_id"], "img-42")

    def test_auto_collect_reuses_enricher_and_respects_scope(self):
        content = _png_bytes((20, 180, 130))
        data_url = "data:image/png;base64," + base64.b64encode(content).decode("ascii")

        class Enricher:
            async def _download_image_data_url(self, segment):
                return data_url

        manager = MemeManager(
            {
                "storage_path": self.temp_dir.name,
                "auto_collect_enabled": True,
                "collect_scope": ["123"],
                "collect_cooldown_seconds": 0,
            },
            base_dir=Path.cwd(),
        )
        message = Message(
            message_id="m1",
            message_type="group",
            sender_id="u1",
            sender_name="小明",
            group_id="123",
            content="这张可以",
            outer_text="这是一个表情包，哈哈",
            segments=[MessageSegment(type="image")],
        )
        adapter = SimpleNamespace(self_id="bot", rich_media_enricher=Enricher())

        import asyncio

        result = asyncio.run(manager.collect_message(message, adapter))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category"], "开心")
        self.assertEqual(result[0]["source_session"], "group_123")

        message.group_id = "999"
        self.assertEqual(asyncio.run(manager.collect_message(message, adapter)), [])

    def test_plain_images_require_signal_and_screenshots_are_skipped(self):
        content = _png_bytes((220, 220, 220))
        data_url = "data:image/png;base64," + base64.b64encode(content).decode("ascii")
        calls = 0

        class Enricher:
            async def _download_image_data_url(self, segment):
                nonlocal calls
                calls += 1
                return data_url

        manager = MemeManager(
            {
                "storage_path": self.temp_dir.name,
                "auto_collect_enabled": True,
                "collect_cooldown_seconds": 0,
            },
            base_dir=Path.cwd(),
        )
        message = Message(
            message_id="m2",
            message_type="group",
            sender_id="u2",
            sender_name="小红",
            group_id="123",
            content="看一下",
            outer_text="看一下",
            segments=[MessageSegment(type="image", summary="[图片]")],
        )
        adapter = SimpleNamespace(self_id="bot", rich_media_enricher=Enricher())

        import asyncio

        self.assertEqual(asyncio.run(manager.collect_message(message, adapter)), [])
        self.assertEqual(calls, 0)

        message.outer_text = "哈哈这张好好笑"
        self.assertEqual(asyncio.run(manager.collect_message(message, adapter)), [])
        self.assertEqual(calls, 0)

        message.outer_text = "看一下"
        message.segments[0].summary = "[图片，内容：一张聊天截图]"
        self.assertEqual(asyncio.run(manager.collect_message(message, adapter)), [])
        self.assertEqual(calls, 0)

        message.segments[0].summary = "[图片，内容：一个人在翻白眼的梗图]"
        result = asyncio.run(manager.collect_message(message, adapter))
        self.assertEqual(len(result), 1)
        self.assertEqual(calls, 1)
        self.assertEqual(result[0]["category"], "吐槽")

    def test_auto_category_falls_back_when_meaning_is_unclear(self):
        self.assertEqual(MemeManager._infer_category("一张普通风景图"), "待整理")
        self.assertEqual(MemeManager._infer_category("哈哈但又有点无语"), "待整理")

    def test_meme_meaning_keeps_visual_facts_and_drops_chat_context(self):
        summary = (
            "[表情包，内容：一只眯眼橘猫配字\"拽\"，群友在吐槽对方偷偷练了，"
            "表达一种\"你可拉倒吧别装了\"的调侃和无语感。]"
        )
        self.assertEqual(
            MemeManager._meaning_from_content(summary),
            "一只眯眼橘猫配字\"拽\"",
        )

        segment = MessageSegment(
            type="mface",
            summary="[表情包，内容：结合前文判断是在嘲讽某人]",
            data={"objective_summary": "一只眯眼橘猫，眯眼，配字\"拽\""},
        )
        self.assertEqual(
            MemeManager._meaning_from_content(
                segment.data["objective_summary"]
            ),
            "一只眯眼橘猫，眯眼，配字\"拽\"",
        )

    def test_private_collection_is_off_by_default(self):
        manager = MemeManager(
            {"storage_path": self.temp_dir.name, "auto_collect_enabled": True},
            base_dir=Path.cwd(),
        )
        self.assertFalse(manager._scope_allows("private_1", "private"))
        self.assertTrue(manager._scope_allows("group_1", "group"))

    def test_internal_directive_survives_short_action_limit(self):
        result = ReplyGenerator._limit_action_length(
            "这句很长，需要截短 [[表情:编号:abcdef1234]]",
            6,
        )
        self.assertTrue(result.endswith("[[表情:编号:abcdef1234]]"))
        self.assertLessEqual(len(result.removesuffix(" [[表情:编号:abcdef1234]]")), 6)


if __name__ == "__main__":
    unittest.main()

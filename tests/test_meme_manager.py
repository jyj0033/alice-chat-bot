"""本地表情库的离线测试。"""

import base64
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

from core.adapter.base import Message
from core.adapter.rich_content import MessageSegment
from modules.meme_manager import MemeManager
from modules.reply.generator import ReplyGenerator


def _png_bytes(color=(124, 108, 255)):
    buffer = BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format="PNG")
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

        updated = self.manager.update(first["id"], category="开心", tags=["反应"])
        self.assertEqual(updated["category"], "开心")
        self.assertEqual(updated["tags"], ["反应"])
        item_and_bytes = self.manager.get_bytes(first["id"])
        self.assertIsNotNone(item_and_bytes)
        self.assertEqual(item_and_bytes[1], content)
        self.assertTrue((Path(self.temp_dir.name) / "开心" / first["filename"]).is_file())

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
            segments=[MessageSegment(type="image")],
        )
        adapter = SimpleNamespace(self_id="bot", rich_media_enricher=Enricher())

        import asyncio

        result = asyncio.run(manager.collect_message(message, adapter))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category"], "待整理")
        self.assertEqual(result[0]["source_session"], "group_123")

        message.group_id = "999"
        self.assertEqual(asyncio.run(manager.collect_message(message, adapter)), [])

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

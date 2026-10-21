"""表情包大模型归类的离线测试。"""

import inspect
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from modules.llm.base import ChatResponse
from modules.meme_categorizer import (
    ANALYZE_PROMPT,
    BLOCKED_TAG,
    PENDING_TAG,
    MemeCategorizer,
    normalize_send_advice,
    normalize_sentiment,
    parse_analysis,
    parse_category,
    sniff_media_type,
)
from modules.meme_manager import MemeManager


def _png_bytes(color=(124, 108, 255)):
    buffer = BytesIO()
    Image.new("RGB", (240, 160), color).save(buffer, format="PNG")
    return buffer.getvalue()


ANALYSIS_TEXT = (
    "【核心含义】阴阳怪气地嘲讽\n"
    "【情绪倾向】负面\n"
    "【适用场景】对方说废话时\n"
    "【是否建议发送】是"
)


class _FakeProvider:
    def __init__(self, content="", error=None):
        self.content = content
        self.error = error
        self.requests = []
        self.model = "fake-model"

    async def chat(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ChatResponse(content=self.content, model="fake-model", usage={})


class AnalysisParsingTests(unittest.TestCase):
    def test_parses_all_four_fields(self):
        analysis = parse_analysis(ANALYSIS_TEXT)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.core_meaning, "阴阳怪气地嘲讽")
        self.assertEqual(analysis.sentiment, "负面")
        self.assertEqual(analysis.scene, "对方说废话时")
        self.assertEqual(analysis.should_send, "是")

    def test_missing_core_meaning_returns_none(self):
        self.assertIsNone(parse_analysis("这是一段普通文字"))
        self.assertIsNone(parse_analysis(""))

    def test_tolerates_markdown_prefix_and_colon(self):
        analysis = parse_analysis(
            "- 【核心含义】：极度无语\n* 【情绪倾向】：负面\n"
            "【适用场景】：被噎住时\n【是否建议发送】：谨慎"
        )
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.core_meaning, "极度无语")
        self.assertEqual(analysis.should_send, "谨慎")

    def test_normalize_sentiment(self):
        self.assertEqual(normalize_sentiment("正面情绪"), "正面")
        self.assertEqual(normalize_sentiment("偏负面"), "负面")
        self.assertEqual(normalize_sentiment("中性"), "中性")
        self.assertEqual(normalize_sentiment("说不好"), "中性")

    def test_normalize_send_advice(self):
        self.assertEqual(normalize_send_advice("是"), "是")
        self.assertEqual(normalize_send_advice("否"), "否")
        self.assertEqual(normalize_send_advice("谨慎"), "谨慎")
        self.assertEqual(normalize_send_advice("不建议发送"), "否")
        self.assertEqual(normalize_send_advice(""), "谨慎")

    def test_analyze_prompt_keeps_the_required_format(self):
        """提示词是这个功能的规格：四个字段 + 不解释画面细节。"""
        for field in ("【核心含义】", "【情绪倾向】", "【适用场景】", "【是否建议发送】"):
            self.assertIn(field, ANALYZE_PROMPT)
        self.assertIn("绝不解释画面细节", ANALYZE_PROMPT)
        self.assertIn("50字", ANALYZE_PROMPT)


class CategoryParsingTests(unittest.TestCase):
    def test_accepts_exact_category(self):
        self.assertEqual(parse_category("吐槽", ["开心", "吐槽"]), "吐槽")

    def test_accepts_category_with_surrounding_noise(self):
        self.assertEqual(parse_category("「吐槽」", ["开心", "吐槽"]), "吐槽")
        self.assertEqual(parse_category("分类是：无语", ["开心", "无语"]), "无语")

    def test_rejects_unknown_category(self):
        self.assertEqual(parse_category("火星语", ["开心", "吐槽"]), "")
        self.assertEqual(parse_category("", ["开心"]), "")

    def test_rejects_when_no_candidates(self):
        self.assertEqual(parse_category("吐槽", []), "")


class MemeCategorizerTests(unittest.IsolatedAsyncioTestCase):
    def _categorizer(self, vision_content=ANALYSIS_TEXT, text_content="吐槽"):
        vision = _FakeProvider(vision_content)
        text = _FakeProvider(text_content)
        categorizer = MemeCategorizer(vision, text, ["开心", "无语", "吐槽", "困倦"])
        # 压缩图片依赖 PIL 和真实图片字节，这里只测流程，绕过压缩。
        categorizer._to_data_url = lambda data, media_type: "data:image/png;base64,AAAA"
        return categorizer, vision, text

    async def test_process_returns_category_and_analysis(self):
        categorizer, vision, text = self._categorizer()
        result = await categorizer.process(_png_bytes(), "image/png")
        self.assertIsNotNone(result)
        category, analysis = result
        self.assertEqual(category, "吐槽")
        self.assertEqual(analysis.core_meaning, "阴阳怪气地嘲讽")
        # 视觉那次必须真的把图带上了。
        content = vision.requests[0].messages[0].content
        self.assertTrue(any(part.get("type") == "image_url" for part in content))

    async def test_process_returns_none_when_category_is_invalid(self):
        categorizer, _vision, _text = self._categorizer(text_content="不存在的分类")
        self.assertIsNone(await categorizer.process(_png_bytes(), "image/png"))

    async def test_process_returns_none_when_analysis_is_unusable(self):
        categorizer, _vision, _text = self._categorizer(vision_content="我随便说说")
        self.assertIsNone(await categorizer.process(_png_bytes(), "image/png"))

    async def test_analyze_returns_none_when_provider_fails(self):
        categorizer, _vision, _text = self._categorizer()
        categorizer.vision_provider = _FakeProvider(error=RuntimeError("端点炸了"))
        self.assertIsNone(await categorizer.analyze(_png_bytes(), "image/png"))

    async def test_categorize_prompt_carries_category_descriptions(self):
        """分类说明要带进 prompt，否则「困倦」和「委屈」分不清。"""
        categorizer, _vision, text = self._categorizer()
        categorizer.category_descriptions = {"困倦": "犯困、睡着或准备休息。"}
        await categorizer.categorize(parse_analysis(ANALYSIS_TEXT))
        prompt = text.requests[0].messages[0].content
        self.assertIn("困倦：犯困、睡着或准备休息。", prompt)
        self.assertIn("吐槽", prompt)

    async def test_unavailable_categorizer_does_nothing(self):
        categorizer = MemeCategorizer(None, None, [])
        self.assertFalse(categorizer.available)
        self.assertIsNone(await categorizer.process(_png_bytes(), "image/png"))

    def test_sniff_media_type(self):
        self.assertEqual(sniff_media_type(_png_bytes()), "image/png")
        self.assertEqual(sniff_media_type(b"\xff\xd8\xff\xe0", "x.jpg"), "image/jpeg")
        self.assertEqual(sniff_media_type(b"not-an-image", "x.gif"), "image/gif")


class PendingCategorizationQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.manager = MemeManager(
            {"storage_path": self.temp_dir.name},
            base_dir=Path.cwd(),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_only_pending_tagged_items_are_queued(self):
        # 功能上线前积压的旧素材：待整理但没有「待归类」标记。
        legacy = self.manager.add_bytes(_png_bytes((1, 2, 3)), category="待整理")
        # 新收集、还没归类成功的。
        fresh = self.manager.add_bytes(
            _png_bytes((4, 5, 6)), category="待整理", tags=["自动收集", PENDING_TAG]
        )
        # 已经归好类的。
        done = self.manager.add_bytes(
            _png_bytes((7, 8, 9)), category="吐槽", tags=["自动收集"]
        )

        queued = self.manager.list_pending_categorization(10)
        queued_ids = {item["id"] for item in queued}
        self.assertIn(fresh["id"], queued_ids)
        self.assertNotIn(legacy["id"], queued_ids)
        self.assertNotIn(done["id"], queued_ids)

    def test_queue_respects_limit(self):
        for index in range(5):
            self.manager.add_bytes(
                _png_bytes((index, index, index)),
                category="待整理",
                tags=[PENDING_TAG],
            )
        self.assertEqual(len(self.manager.list_pending_categorization(2)), 2)

    def test_blocked_items_are_excluded_from_auto_send(self):
        self.manager.add_bytes(
            _png_bytes((30, 40, 50)), category="吐槽", tags=[BLOCKED_TAG]
        )
        self.assertIsNone(self.manager.choose(category="吐槽"))

    def test_update_moves_file_to_the_new_category(self):
        item = self.manager.add_bytes(
            _png_bytes((60, 70, 80)), category="待整理", tags=[PENDING_TAG]
        )
        old_path = self.manager._image_path(self.manager._catalog["memes"][item["id"]])

        updated = self.manager.update(
            item["id"], category="吐槽", meaning="阴阳怪气地嘲讽", tags=["情绪:负面"]
        )

        self.assertEqual(updated["category"], "吐槽")
        self.assertEqual(updated["meaning"], "阴阳怪气地嘲讽")
        self.assertFalse(old_path.exists())
        self.assertTrue((Path(self.manager.root) / "吐槽" / item["filename"]).is_file())
        # 归类完成后不再留在待归类队列里。
        self.assertEqual(self.manager.list_pending_categorization(10), [])


class MainEntryStartsCategorizeLoopTests(unittest.TestCase):
    """容器实际走 main()，后台循环必须挂在那里。"""

    def test_main_entry_creates_the_meme_categorize_loop(self):
        import main as main_module

        source = inspect.getsource(main_module.main)
        self.assertIn(
            "asyncio.create_task(bot._meme_categorize_loop())",
            source,
            "main() 入口必须创建表情包自动归类循环",
        )

    def test_bot_exposes_an_async_categorize_loop(self):
        from main import GroupChatBot

        self.assertTrue(inspect.iscoroutinefunction(GroupChatBot._meme_categorize_loop))


if __name__ == "__main__":
    unittest.main()

"""Dashboard 静态结构测试，不依赖 FastAPI 运行时。"""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import unittest


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])


class DashboardSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "templates" / "dashboard.html"
        cls.html = path.read_text(encoding="utf-8")
        cls.parser = _IdCollector()
        cls.parser.feed(cls.html)

    def test_dashboard_ids_are_unique(self):
        duplicates = [
            item for item, count in Counter(self.parser.ids).items() if count > 1
        ]
        self.assertEqual(duplicates, [])

    def test_new_backend_features_have_management_controls(self):
        required_ids = {
            "page-behavior",
            "floor-active-window",
            "rich-enabled",
            "rich-forward-enabled",
            "rich-links-enabled",
            # 图片识别（视觉模型）控件，OCR 已由视觉模型取代
            "image-to-text-scope",
            "image-to-text-context",
            "image-group-enabled",
            # 视觉模型独立配置（LLM 页）
            "vision-enabled",
            # 群聊纪要查询/管理（记忆页）
            "memory-type-filter",
            "digest-enabled",
            "attention-spillover-enabled",
            "cooldown-enabled",
            "qq-ws-host",
            "qq-ws-port",
            "provider-name",
            "personality-age",
            "personality-taboo",
            "embed-base-url",
            "style-common-words",
            "style-max-reply",
            "typing-min-length",
            # 联网搜索（LLM 页）
            "search-enabled",
            "search-primary",
            "search-bocha-api-key",
            "search-doubao-api-key",
        }
        self.assertTrue(required_ids.issubset(set(self.parser.ids)))
        self.assertNotIn("qq-ws-url", self.parser.ids)
        # OCR 已被视觉模型取代，不应再暴露单独配置
        self.assertNotIn("rich-ocr-enabled", self.parser.ids)

    def test_session_messages_are_html_escaped(self):
        self.assertIn("escapeHtml(m.sender)", self.html)
        self.assertIn("escapeHtml(m.content)", self.html)
        self.assertIn("escapeHtml(data.reply)", self.html)
        self.assertIn("escapeHtml(p.base_url", self.html)

    def test_new_provider_uses_provider_api_payload(self):
        self.assertIn(
            "JSON.stringify(isNew ? data : { llm: { [name]: data } })",
            self.html,
        )


if __name__ == "__main__":
    unittest.main()

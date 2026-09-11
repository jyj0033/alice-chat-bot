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
        cls.dashboard_source = (path.parent.parent / "dashboard.py").read_text(encoding="utf-8")
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
            "judge-enabled",
            "judge-provider-id",
            "judge-timeout",
            "judge-max-tokens",
            "judge-context-messages",
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
            "group-analysis-enabled",
            "group-analysis-auto",
            "group-analysis-times",
            "group-analysis-min",
            "group-analysis-retention",
            "group-analysis-session",
            "group-analysis-reports",
            "style-common-words",
            "style-max-reply",
            "typing-min-length",
            "memory-pagination",
            "memory-prev",
            "memory-next",
            "memory-page-meta",
            "profile-pagination",
            "profile-prev",
            "profile-next",
            "profile-page-meta",
            "profile-search",
            # 表情包素材库
            "page-memes",
            "meme-enabled",
            "meme-auto-collect",
            "meme-auto-send",
            "meme-upload-file",
            "meme-upload-meaning",
            "meme-collect-plain",
            "meme-skip-screenshots",
            "meme-max-dimension",
            "meme-category-filter",
            "meme-pagination",
            "meme-prev",
            "meme-next",
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

    def test_memory_list_uses_server_pagination_and_search_debounce(self):
        self.assertIn("page_size", self.html)
        self.assertIn("scheduleLoadMemories", self.html)
        self.assertIn("AbortController", self.html)
        self.assertIn("共 ${memoryState.total} 条", self.html)

    def test_profile_list_uses_server_pagination(self):
        self.assertIn("PROFILE_PAGE_SIZE = 30", self.html)
        self.assertIn("/profiles?", self.html)
        self.assertIn("scheduleLoadProfiles", self.html)
        self.assertIn("共 ${profileState.total} 个", self.html)

    def test_meme_library_has_upload_send_and_pagination(self):
        self.assertIn("/api/memes", self.dashboard_source)
        self.assertIn("/api/memes/upload", self.dashboard_source)
        self.assertIn("/api/memes/send", self.dashboard_source)
        self.assertIn("loadMemePage", self.html)
        self.assertIn("uploadMeme", self.html)
        self.assertIn("sendMeme", self.html)
        self.assertIn("updateMemeMeaning", self.html)
        self.assertIn("图片客观描述", self.html)
        self.assertIn("MEME_PAGE_SIZE = 48", self.html)

    def test_group_analysis_controls_and_report_api_are_exposed(self):
        self.assertIn("/api/group-analysis/reports", self.dashboard_source)
        self.assertIn("/api/group-analysis/sessions", self.dashboard_source)
        self.assertIn("/api/group-analysis/run", self.dashboard_source)
        self.assertIn("runGroupAnalysis", self.html)
        self.assertNotIn("/群分析 [天数]", self.html)

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

"""LLM token 消耗统计的离线测试。"""

import asyncio
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.adapter.base import Message
from modules.llm.base import ChatRequest, ChatResponse
from modules.llm.usage import (
    TokenUsageCollector,
    TaggedProvider,
    site_label,
    tag_provider,
    unwrap_provider,
)
from modules.memory.storage import MemoryStorage


class _FakeProvider:
    """只提供 chat() 的假 provider。"""

    def __init__(self, model="fake-model", usage=None):
        self.model = model
        self.config = {"name": model}
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        return ChatResponse(content="ok", model=self.model, usage=dict(self.usage))


class TokenUsageCollectorTests(unittest.IsolatedAsyncioTestCase):
    def test_record_accumulates_by_day_site_and_model(self):
        collector = TokenUsageCollector()
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        collector.record("reply", "m1", usage)
        collector.record("reply", "m1", usage)
        collector.record("vision", "m1", usage)

        self.assertEqual(collector.pending_calls, 3)
        self.assertEqual(len(collector._pending), 2)
        reply_row = collector._pending[(date.today().isoformat(), "reply", "m1")]
        self.assertEqual(reply_row["calls"], 2)
        self.assertEqual(reply_row["prompt_tokens"], 200)
        self.assertEqual(reply_row["total_tokens"], 240)

    def test_disabled_collector_records_nothing(self):
        collector = TokenUsageCollector(enabled=False)
        collector.record("reply", "m1", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        self.assertEqual(collector.pending_calls, 0)

    def test_empty_usage_is_ignored(self):
        collector = TokenUsageCollector()
        collector.record("reply", "m1", {})
        collector.record("reply", "m1", None)
        collector.record("reply", "m1", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertEqual(collector.pending_calls, 0)

    async def test_flush_writes_rows_to_storage(self):
        storage = SimpleNamespace(record_llm_token_usage=AsyncMock(return_value=None))
        collector = TokenUsageCollector(storage=storage)
        collector.record("reply", "m1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

        written = await collector.flush()

        self.assertEqual(written, 1)
        storage.record_llm_token_usage.assert_awaited_once()
        kwargs = storage.record_llm_token_usage.await_args.kwargs
        self.assertEqual(kwargs["site"], "reply")
        self.assertEqual(kwargs["calls"], 1)
        self.assertEqual(kwargs["total_tokens"], 15)
        # 落库后内存里的计数要清掉，否则会重复记账。
        self.assertEqual(collector.pending_calls, 0)

    async def test_flush_without_storage_keeps_counters(self):
        collector = TokenUsageCollector()
        collector.record("reply", "m1", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        written = await collector.flush()
        self.assertEqual(written, 0)
        self.assertEqual(collector.pending_calls, 1)

    def test_site_label_falls_back_to_raw_key(self):
        self.assertEqual(site_label("reply"), "回复生成")
        self.assertEqual(site_label("vision"), "视觉识别")
        self.assertEqual(site_label("some_new_site"), "some_new_site")
        self.assertEqual(site_label(""), "未分类")


class TaggedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_tagged_provider_records_usage_under_its_site(self):
        collector = TokenUsageCollector()
        raw = _FakeProvider(model="m-1")
        tagged = tag_provider(raw, "reply", collector)

        response = await tagged.chat(ChatRequest())

        self.assertEqual(response.content, "ok")
        self.assertEqual(collector.pending_calls, 1)
        key = next(iter(collector._pending))
        self.assertEqual(key[1], "reply")
        self.assertEqual(key[2], "m-1")

    async def test_same_provider_can_be_tagged_for_two_sites_without_double_counting(self):
        """同一个底层模型被两个功能点共用时，消耗要分开记，不能记两遍。"""
        collector = TokenUsageCollector()
        raw = _FakeProvider()
        reply_tagged = tag_provider(raw, "reply", collector)
        # 拿已经套过壳的 provider 再套一层：必须先剥壳，否则同一次调用会记两遍。
        review_tagged = tag_provider(reply_tagged, "reply_review", collector)

        await reply_tagged.chat(ChatRequest())
        await review_tagged.chat(ChatRequest())

        self.assertEqual(collector.pending_calls, 2)
        sites = {key[1] for key in collector._pending}
        self.assertEqual(sites, {"reply", "reply_review"})

    def test_retagging_same_site_reuses_the_wrapper(self):
        collector = TokenUsageCollector()
        raw = _FakeProvider()
        first = tag_provider(raw, "reply", collector)
        again = tag_provider(first, "reply", collector)
        self.assertIs(first, again)

    def test_unwrap_returns_the_raw_provider(self):
        raw = _FakeProvider()
        tagged = tag_provider(raw, "reply", TokenUsageCollector())
        self.assertIs(unwrap_provider(tagged), raw)
        self.assertIs(unwrap_provider(raw), raw)

    def test_attributes_are_delegated_to_the_raw_provider(self):
        raw = _FakeProvider(model="m-9")
        tagged = tag_provider(raw, "reply", TokenUsageCollector())
        self.assertEqual(tagged.model, "m-9")
        self.assertEqual(tagged.config, {"name": "m-9"})

    def test_disabled_collector_returns_the_raw_provider(self):
        raw = _FakeProvider()
        collector = TokenUsageCollector(enabled=False)
        self.assertIs(tag_provider(raw, "reply", collector), raw)

    def test_none_provider_stays_none(self):
        self.assertIsNone(tag_provider(None, "reply", TokenUsageCollector()))


class TokenUsageStorageTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        import os
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "usage.db")
        self.storage = MemoryStorage(self.path)

    def tearDown(self):
        try:
            self.storage.conn.close()
        except Exception:
            pass

    def test_records_accumulate_for_the_same_key(self):
        self.storage.record_llm_token_usage("2026-10-21", "reply", "m1", 1, 100, 20, 120)
        self.storage.record_llm_token_usage("2026-10-21", "reply", "m1", 2, 200, 40, 240)

        rows = self.storage.get_llm_token_usage(7)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["calls"], 3)
        self.assertEqual(rows[0]["prompt_tokens"], 300)
        self.assertEqual(rows[0]["completion_tokens"], 60)
        self.assertEqual(rows[0]["total_tokens"], 360)

    def test_different_sites_and_days_are_separate_rows(self):
        self.storage.record_llm_token_usage("2026-10-21", "reply", "m1", 1, 10, 5, 15)
        self.storage.record_llm_token_usage("2026-10-21", "vision", "m1", 1, 20, 8, 28)
        self.storage.record_llm_token_usage("2026-10-20", "reply", "m1", 1, 30, 9, 39)

        rows = self.storage.get_llm_token_usage(30)
        self.assertEqual(len(rows), 3)

    def test_cleanup_removes_rows_outside_the_retention_window(self):
        old_day = (date.today() - timedelta(days=30)).isoformat()
        self.storage.record_llm_token_usage(old_day, "reply", "m1", 1, 10, 5, 15)
        self.storage.record_llm_token_usage(date.today().isoformat(), "reply", "m1", 1, 10, 5, 15)

        removed = self.storage.cleanup_llm_token_usage(7)

        self.assertEqual(removed, 1)
        rows = self.storage.get_llm_token_usage(7)
        self.assertEqual(len(rows), 1)

    def test_get_only_returns_the_requested_window(self):
        old_day = (date.today() - timedelta(days=10)).isoformat()
        self.storage.record_llm_token_usage(old_day, "reply", "m1", 1, 10, 5, 15)
        self.storage.record_llm_token_usage(date.today().isoformat(), "reply", "m1", 1, 10, 5, 15)

        self.assertEqual(len(self.storage.get_llm_token_usage(3)), 1)
        self.assertEqual(len(self.storage.get_llm_token_usage(30)), 2)


class BotProviderTaggingTests(unittest.TestCase):
    """Bot 侧：按功能点选 provider 时要带上标签，且不能影响回退判断。"""

    def _bot(self):
        from main import GroupChatBot
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.config = {"llm_routing": {}}
        bot.llm_providers = {}
        bot.active_provider_id = "primary"
        bot.token_usage = TokenUsageCollector()
        return bot

    def test_role_provider_is_tagged_with_the_role_name(self):
        bot = self._bot()
        raw = _FakeProvider(model="primary-model")
        bot.llm_providers["primary"] = raw

        provider = bot._get_provider_for_role("reply")

        self.assertIsInstance(provider, TaggedProvider)
        self.assertEqual(provider.site, "reply")

    def test_fallback_is_not_created_when_reply_uses_the_active_model(self):
        """reply 指向激活模型时不该有备用模型；套壳后要比到底层 provider。"""
        bot = self._bot()
        raw = _FakeProvider(model="primary-model")
        bot.llm_providers["primary"] = raw

        self.assertIsNone(bot._reply_fallback_provider())

    def test_fallback_is_created_and_tagged_when_reply_routes_elsewhere(self):
        bot = self._bot()
        bot.config = {"llm_routing": {"reply": "other"}}
        primary = _FakeProvider(model="primary-model")
        other = _FakeProvider(model="other-model")
        bot.llm_providers["primary"] = primary
        bot.llm_providers["other"] = other

        fallback = bot._reply_fallback_provider()

        self.assertIsInstance(fallback, TaggedProvider)
        self.assertEqual(fallback.site, "reply_fallback")
        self.assertIs(unwrap_provider(fallback), primary)

    def test_disabled_usage_returns_the_raw_provider(self):
        bot = self._bot()
        bot.token_usage = TokenUsageCollector(enabled=False)
        raw = _FakeProvider()
        bot.llm_providers["primary"] = raw

        provider = bot._get_provider_for_role("reply")
        self.assertNotIsInstance(provider, TaggedProvider)
        self.assertIs(provider, raw)


class GeneratorSiteTaggingTests(unittest.TestCase):
    """回复生成器内部：检索判断和带资料生成要分别记账。"""

    def _generator(self, collector=None):
        from modules.reply.generator import ReplyGenerator
        gen = ReplyGenerator.__new__(ReplyGenerator)
        gen.usage_collector = collector if collector is not None else TokenUsageCollector()
        return gen

    def test_usage_tagged_wraps_with_the_given_site(self):
        gen = self._generator()
        raw = _FakeProvider()
        tagged = gen._usage_tagged(raw, "search_answer")
        self.assertIsInstance(tagged, TaggedProvider)
        self.assertEqual(tagged.site, "search_answer")

    def test_usage_tagged_without_collector_returns_the_raw_provider(self):
        gen = self._generator(collector=None)
        gen.usage_collector = None
        raw = _FakeProvider()
        self.assertIs(gen._usage_tagged(raw, "search"), raw)

    def test_usage_tagged_with_disabled_collector_returns_the_raw_provider(self):
        gen = self._generator(TokenUsageCollector(enabled=False))
        raw = _FakeProvider()
        self.assertIs(gen._usage_tagged(raw, "search"), raw)


if __name__ == "__main__":
    unittest.main()

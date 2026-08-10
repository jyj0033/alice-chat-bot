"""联网搜索测试：后端解析、主备回退、ReplyGenerator 工具循环。"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from modules.llm.base import ChatRequest, ChatResponse
from modules.reply.generator import ReplyGenerator
from modules.search import SearchClient
from modules.search.base import SearchResult
from modules.search.bocha import BochaSearch
from modules.search.doubao import DoubaoSearch


class SearchBackendParseTests(unittest.TestCase):
    def test_bocha_parses_webpages(self):
        payload = {"code": 200, "data": {"webPages": {"value": [
            {
                "name": "标题A", "url": "https://a.com", "snippet": "片段",
                "summary": "正文摘要AAA", "siteName": "搜狐", "datePublished": "2025-01-01T00:00:00+08:00",
            }
        ]}}}
        results = BochaSearch._parse(payload)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "标题A")
        self.assertEqual(results[0].summary, "正文摘要AAA")
        self.assertEqual(results[0].site, "搜狐")

    def test_doubao_parses_webresults(self):
        payload = {"ResponseMetadata": {}, "Result": {"WebResults": [
            {
                "Title": "标题B", "Url": "https://b.com", "Snippet": "片段",
                "Summary": "摘要BBB", "SiteName": "网易", "PublishTime": "2025-06-01T00:00:00+08:00",
            }
        ]}}
        results = DoubaoSearch._parse(payload)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "标题B")
        self.assertEqual(results[0].summary, "摘要BBB")

    def test_doubao_error_response_returns_empty(self):
        payload = {"ResponseMetadata": {"Error": {"Code": "invalid_api_key"}}, "Result": None}
        self.assertEqual(DoubaoSearch._parse(payload), [])


class SearchClientFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_to_secondary_when_primary_fails(self):
        client = SearchClient({
            "enabled": True,
            "primary": "bocha",
            "backends": {
                "bocha": {"api_key": "k1", "count": 3},
                "doubao": {"api_key": "k2", "count": 3},
            },
        })
        # 主后端失败 → 回退豆包
        client._backends["bocha"].search = AsyncMock(return_value=[])
        client._backends["doubao"].search = AsyncMock(return_value=[
            SearchResult(title="结果", url="https://x.com", snippet="s"),
        ])
        results = await client.search("测试", session_id="g1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "结果")
        client._backends["doubao"].search.assert_awaited_once()

    async def test_rate_limit_blocks_second_search(self):
        client = SearchClient({
            "enabled": True,
            "min_interval_seconds": 3600,
            "backends": {"bocha": {"api_key": "k1"}},
        })
        client._backends["bocha"].search = AsyncMock(return_value=[SearchResult(title="t")])
        await client.search("第一次", session_id="g1")
        client._backends["bocha"].search = AsyncMock(return_value=[SearchResult(title="t2")])
        # 间隔未到 → 直接返回空，不调用后端
        results = await client.search("第二次", session_id="g1")
        self.assertEqual(results, [])
        client._backends["bocha"].search.assert_not_awaited()


class ReplyGeneratorToolLoopTests(unittest.IsolatedAsyncioTestCase):
    def _make_generator(self, tool_llm, search_client):
        gen = ReplyGenerator(
            llm_provider=MagicMock(),  # 搜索路径不走主 LLM
            tool_llm_provider=tool_llm,
            search_client=search_client,
            search_trigger_keywords=["新闻", "天气"],
        )
        return gen

    async def test_search_enabled_only_when_keyword_hits(self):
        gen = self._make_generator(MagicMock(), MagicMock())
        gen.search_client.available = True
        self.assertTrue(gen._search_enabled("今天有什么新闻"))
        self.assertFalse(gen._search_enabled("晚饭吃什么"))

    async def test_tool_loop_calls_search_and_returns_final_reply(self):
        tool_llm = MagicMock()
        # 第一轮：请求 web_search；第二轮：直接回答
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(content="", model="test", tool_calls=[{
                "id": "call_1", "name": "web_search",
                "arguments": {"query": "最新新闻"},
            }]),
            ChatResponse(content="今天的新闻是xxx", model="test"),
        ])
        search_client = MagicMock()
        search_client.available = True
        search_client.search = AsyncMock(return_value=[
            SearchResult(title="某新闻", url="https://n.com", snippet="snippet"),
        ])
        search_client.format_results = lambda r: "搜索到的资料：\n1. 某新闻"

        gen = self._make_generator(tool_llm, search_client)
        reply = await gen.generate(
            context_prompt="[刚刚] 小明：今天有什么新闻？",
            current_message="今天有什么新闻",
            direction="group",
            session_id="group_1",
        )
        self.assertIn("xxx", reply)
        search_client.search.assert_awaited_once()
        # 第二轮请求应包含 tool 回放与 tool 结果消息
        self.assertEqual(tool_llm.chat.call_count, 2)
        roles = [m.role for m in tool_llm.chat.await_args.args[0].messages]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)

    async def test_no_search_when_keyword_misses(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="嗯嗯", model="test"))
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(content="随便吃点", model="test"))
        search_client = MagicMock()
        search_client.available = True
        gen = ReplyGenerator(
            llm_provider=main_llm,
            tool_llm_provider=tool_llm,
            search_client=search_client,
            search_trigger_keywords=["新闻", "天气"],
        )
        reply = await gen.generate(
            context_prompt="[刚刚] 小明：晚饭吃啥",
            current_message="晚饭吃啥",
            direction="group",
        )
        # 未命中关键词 → 走主 LLM，不触发搜索路径
        tool_llm.chat.assert_not_awaited()
        main_llm.chat.assert_awaited_once()
        self.assertIsNotNone(reply)


if __name__ == "__main__":
    unittest.main()

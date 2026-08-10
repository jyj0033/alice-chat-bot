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
    def _make_generator(self, tool_llm, search_client, main_llm=None):
        gen = ReplyGenerator(
            llm_provider=main_llm or MagicMock(),
            tool_llm_provider=tool_llm,
            search_client=search_client,
        )
        if not getattr(tool_llm, "model", None):
            tool_llm.model = "test"
        return gen

    # === LLM 判断是否需要搜索 ===

    async def test_judge_yes_when_llm_says_yes(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="YES", model="test"))
        gen = self._make_generator(tool_llm, MagicMock())
        gen.search_client.available = True
        self.assertTrue(await gen._judge_need_search("绝区零现在开的谁的池子", "[刚刚] 小明：绝区零现在开的谁的池子"))
        tool_llm.chat.assert_awaited_once()

    async def test_judge_no_when_llm_says_no(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="NO", model="test"))
        gen = self._make_generator(tool_llm, MagicMock())
        gen.search_client.available = True
        self.assertFalse(await gen._judge_need_search("晚饭吃啥", "[刚刚] 小明：晚饭吃啥"))

    async def test_judge_negative_phrase_not_misparsed(self):
        """「不需要」这类否定不能被「需要」误判成要搜索。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="不需要搜索", model="test"))
        gen = self._make_generator(tool_llm, MagicMock())
        gen.search_client.available = True
        self.assertFalse(await gen._judge_need_search("随便聊聊", "[刚刚] 小明：随便聊聊"))

    async def test_judge_media_description_never_searches(self):
        """表情包/图片识别描述不判断（富媒体内容，不是用户问句）。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="YES", model="test"))
        gen = self._make_generator(tool_llm, MagicMock())
        gen.search_client.available = True
        self.assertFalse(await gen._judge_need_search(
            "[表情包，内容：白发红瞳的动漫角色闭眼咧嘴笑着…，回复：笑死]", ""))
        tool_llm.chat.assert_not_awaited()

    async def test_judge_failure_conservative_no(self):
        """判断调用异常 → 按不需要搜索处理，保证回复不中断。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=RuntimeError("boom"))
        gen = self._make_generator(tool_llm, MagicMock())
        gen.search_client.available = True
        self.assertFalse(await gen._judge_need_search("今天天气", "[刚刚] 小明：今天天气"))

    # === 搜索路径（判断 YES） ===

    async def test_search_injects_results_into_clean_context_call(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(content="YES", model="test"),          # 判断
            ChatResponse(content="今天的新闻是xxx", model="test"),  # 带资料生成
        ])
        search_client = MagicMock()
        search_client.available = True
        search_client.is_time_sensitive = lambda q: True
        search_client.all_future = lambda r: False
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
        # 两次工具 LLM 调用：判断 + 带资料生成；资料已注入最后一条消息
        self.assertEqual(tool_llm.chat.call_count, 2)
        last_content = tool_llm.chat.await_args.args[0].messages[-1].content
        self.assertIn("搜索到的资料", last_content)

    async def test_no_results_falls_back_to_main_llm(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="YES", model="test"))
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(content="没搜到，正常回答", model="test"))
        search_client = MagicMock()
        search_client.available = True
        search_client.is_time_sensitive = lambda q: True
        search_client.all_future = lambda r: False
        search_client.search = AsyncMock(return_value=[])

        gen = self._make_generator(tool_llm, search_client, main_llm)
        reply = await gen.generate(
            context_prompt="[刚刚] 小明：今天有什么新闻？",
            current_message="今天有什么新闻",
            direction="group",
        )
        self.assertIn("没搜到", reply)
        # 判断调用一次；搜索空 → 不再带资料生成，主 LLM 兜底
        self.assertEqual(tool_llm.chat.call_count, 1)
        main_llm.chat.assert_awaited_once()

    # === 判断 NO → 不搜索 ===

    async def test_no_search_when_llm_judges_no(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="NO", model="test"))
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(content="随便吃点", model="test"))
        search_client = MagicMock()
        search_client.available = True
        search_client.search = AsyncMock(return_value=[])

        gen = self._make_generator(tool_llm, search_client, main_llm)
        reply = await gen.generate(
            context_prompt="[刚刚] 小明：晚饭吃啥",
            current_message="晚饭吃啥",
            direction="group",
        )
        # LLM 判定不需要搜索 → 走主 LLM，不触发搜索后端
        self.assertIsNotNone(reply)
        main_llm.chat.assert_awaited_once()
        search_client.search.assert_not_awaited()

    # === 工具 LLM 带资料生成失败 → 主 LLM 带同一份资料重答 ===

    async def test_tool_llm_think_only_retries_with_data_on_main_llm(self):
        """工具 LLM 只输出 <think> 思考块（无实质内容）→ 主 LLM 带资料重答，不丢搜索结果。

        回归 2026-08-10 21:10：搜索成功（doubao 5 条）但工具 LLM 返回纯思考块，
        _clean_thinking_process 清空后走了"回退无资料主 LLM"，凭记忆答错。
        """
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(content="YES", model="test"),  # 判断
            ChatResponse(content="<think>崩铁现在up的是流萤，记不太清了</think>", model="test"),
        ])
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(
            content="崩铁现在up的是流萤。", model="test"))
        search_client = MagicMock()
        search_client.available = True
        search_client.is_time_sensitive = lambda q: True
        search_client.all_future = lambda r: False
        search_client.search = AsyncMock(return_value=[
            SearchResult(title="崩铁当前卡池", url="https://s.com", snippet="流萤"),
        ])
        search_client.format_results = lambda r: "搜索到的资料：\n1. 崩铁当前卡池"

        gen = self._make_generator(tool_llm, search_client, main_llm)
        reply = await gen.generate(
            context_prompt="[刚刚] 小明：崩铁现在up角色是谁？",
            current_message="崩铁现在up角色是谁",
            direction="group",
        )
        self.assertIn("流萤", reply)
        # 主 LLM 带同一份资料重答，且不带资料的无资料重答不应发生
        main_llm.chat.assert_awaited_once()
        last_content = main_llm.chat.await_args.args[0].messages[-1].content
        self.assertIn("搜索到的资料", last_content)

    async def test_tool_llm_empty_content_retries_with_data(self):
        """工具 LLM 返回空 content（resp 非 None 但无内容）→ 同样主 LLM 带资料重答。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(content="YES", model="test"),  # 判断
            ChatResponse(content="", model="test"),
        ])
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(
            content="今天是晴天。", model="test"))
        search_client = MagicMock()
        search_client.available = True
        search_client.is_time_sensitive = lambda q: True
        search_client.all_future = lambda r: False
        search_client.search = AsyncMock(return_value=[
            SearchResult(title="天气", url="https://w.com", snippet="晴"),
        ])
        search_client.format_results = lambda r: "搜索到的资料：\n1. 天气"

        gen = self._make_generator(tool_llm, search_client, main_llm)
        reply = await gen.generate(
            context_prompt="[刚刚] 小明：今天天气",
            current_message="今天天气",
            direction="group",
        )
        self.assertIn("晴天", reply)
        main_llm.chat.assert_awaited_once()
        last_content = main_llm.chat.await_args.args[0].messages[-1].content
        self.assertIn("搜索到的资料", last_content)


if __name__ == "__main__":
    unittest.main()

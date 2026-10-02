"""LLM provider 响应解析的离线测试。"""

import unittest
from unittest.mock import AsyncMock

from modules.llm.base import ChatRequest
from modules.llm.claude_provider import ClaudeProvider
from modules.llm.openai_provider import OpenAIProvider
from modules.llm.responses_provider import (
    ResponsesProvider,
    _collect_search_evidence,
)


class OpenAIProviderResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_json_string_response_raises_actionable_error(self):
        """base_url 配错（少 /v1）时网关可能 200 返回 HTML，SDK 会给出字符串。

        这时应报出提示检查 base_url 的错误，而不是
        'str' object has no attribute 'choices'。
        """
        provider = OpenAIProvider({
            "api_key": "test-key",
            "base_url": "https://gateway.example.com",  # 故意缺 /v1
            "model": "test-model",
        })
        provider.client.chat.completions.create = AsyncMock(
            return_value="<!DOCTYPE html><html><head>网关首页</head></html>"
        )

        with self.assertRaises(ValueError) as ctx:
            await provider.chat(ChatRequest(messages=[], model="test-model"))

        self.assertIn("/v1", str(ctx.exception))

    async def test_default_chat_request_uses_provider_model(self):
        """ChatRequest 不显式传 model 时，必须用 provider 配置的模型。

        以前默认 "gpt-4o" 恒为真会压过 provider 配置：纪要/画像/连接测试
        这类调用会向端点请求 gpt-4o，在按 token 鉴权的网关上直接 403。
        """
        provider = OpenAIProvider({
            "api_key": "test-key",
            "base_url": "https://gateway.example.com/v1",
            "model": "my/custom-model",
        })
        mock_create = AsyncMock(side_effect=RuntimeError("stop"))
        provider.client.chat.completions.create = mock_create

        with self.assertRaises(RuntimeError):
            await provider.chat(ChatRequest(messages=[]))

        self.assertEqual(mock_create.call_args.kwargs["model"], "my/custom-model")


class ClaudeProviderFormattingTests(unittest.TestCase):
    def test_system_messages_are_kept_in_top_level_system_field(self):
        provider = ClaudeProvider({
            "api_key": "test-key",
            "base_url": "https://gateway.example.com",
            "model": "test-model",
        })
        request = ChatRequest()
        request.add_system("人格约束")
        request.add_system("目标判断结果")
        request.add_user("当前消息")

        self.assertEqual(
            provider._format_system_prompt(request.messages),
            "人格约束\n\n目标判断结果",
        )
        self.assertEqual(
            provider.format_claude_messages(request.messages),
            [{"role": "user", "content": "当前消息"}],
        )


class ResponsesSearchEvidenceTests(unittest.TestCase):
    """服务端检索留痕。

    线上真实问题：一条联网回复给出「音感仪叠满再切出去开大比较顺」这种
    需要反复实战才能总结的细节，而同一条回复又自称「只打了一次，不敢乱说」。
    检索到的真实资料和凭空编造，在回复正文里长得一模一样——不��痕就永远
    无法事后区分。所以这里固化「声明了搜索就必须能说出搜了什么」。
    """

    def _provider(self) -> ResponsesProvider:
        return ResponsesProvider({
            "api_key": "test-key",
            "base_url": "https://gateway.example.com/v1",
            "model": "test-model",
        })

    def _payload_with_search(self):
        return {
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "心月狐 摧城 怎么用"},
                },
                {
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "配队往导电/音感仪那套里塞",
                        "annotations": [
                            {"type": "url_citation", "url": "https://a.example.com/g1",
                             "title": "心月狐攻略"},
                            {"type": "url_citation", "url": "https://b.example.com/g2",
                             "title": "鸣潮版本说明"},
                        ],
                    }],
                },
            ],
        }

    def test_extracts_query_and_citations(self):
        queries, citations = _collect_search_evidence(self._payload_with_search()["output"])

        self.assertEqual(queries, ["心月狐 摧城 怎么用"])
        self.assertEqual(len(citations), 2)
        self.assertIn(("心月狐攻略", "https://a.example.com/g1"), citations)
        self.assertIn(("鸣潮版本说明", "https://b.example.com/g2"), citations)

    def test_duplicate_citations_are_collapsed(self):
        """同一来源被引用多次时只留一条，否则日志会被同一个页面刷屏。"""
        item = {
            "type": "message",
            "content": [{
                "type": "output_text",
                "annotations": [
                    {"url": "https://a.example.com/g1", "title": "心月狐攻略"},
                    {"url": "https://a.example.com/g1", "title": "心月狐攻略"},
                    {"url": "https://c.example.com/g3", "title": "另一篇"},
                ],
            }],
        }
        _, citations = _collect_search_evidence([item])

        self.assertEqual(len(citations), 2)

    def test_declared_search_with_sources_logs_evidence(self):
        with self.assertLogs("modules.llm.responses_provider", level="INFO") as ctx:
            self._provider()._parse(self._payload_with_search(), declared_search=True)

        joined = "\n".join(ctx.output)
        self.assertIn("服务端检索留痕", joined)
        self.assertIn("心月狐 摧城 怎么用", joined)
        self.assertIn("心月狐攻略", joined)
        self.assertIn("https://a.example.com/g1", joined)

    def test_declared_search_without_any_source_warns(self):
        """声明了搜索却拿不回来源 = 这次细节完全出自模型生成，必须报警。"""
        payload = {
            "status": "completed",
            "output_text": "音感仪叠满再切出去开大比较顺",
            "output": [{"type": "reasoning"}, {"type": "message",
                     "content": [{"type": "output_text", "text": "同上"}]}],
        }
        with self.assertLogs("modules.llm.responses_provider", level="WARNING") as ctx:
            self._provider()._parse(payload, declared_search=True)

        joined = "\n".join(ctx.output)
        self.assertIn("不可溯源", joined)

    def test_no_search_declared_stays_quiet(self):
        """绝大多数轮次没有声明搜索，不能给它们加噪音日志。

        这里不断言"没有任何日志"——`_parse` 本来就会打 INFO 级的停止日志；
        只断言检索留痕那一条没有出现。
        """
        with self.assertLogs("modules.llm.responses_provider", level="INFO") as ctx:
            self._provider()._parse(self._payload_with_search(), declared_search=False)

        self.assertNotIn("服务端检索留痕", "\n".join(ctx.output))

    def test_malformed_items_do_not_crash(self):
        """网关返回结构不规整时不能把整轮回复搞崩。"""
        items = [
            "不是字典",
            {"type": "web_search_call", "action": "不是字典"},
            {"type": "web_search_call"},
            {"type": "message", "content": "不是列表"},
            {"type": "message", "content": [{"annotations": ["脏数据", {"url": "https://ok.example.com"}]}]},
            {"type": "message", "content": [{"annotations": [{"title": "只有标题"}]}]},
        ]
        queries, citations = _collect_search_evidence(items)

        self.assertEqual(queries, [])
        self.assertEqual(
            citations, [("", "https://ok.example.com"), ("只有标题", "")]
        )

    def test_declared_server_search_detects_tools(self):
        detect = ResponsesProvider._declared_server_search
        self.assertTrue(detect({"tools": [{"type": "web_search"}]}))
        self.assertTrue(detect({"tools": [{"type": "function", "name": "x"},
                                          {"type": "web_search"}]}))
        self.assertFalse(detect({"tools": []}))
        self.assertFalse(detect({}))
        self.assertFalse(detect({"tools": [{"type": "function"}]}))

    def test_search_call_query_falls_back_to_top_level(self):
        """部分网关把 query 放在 web_search_call 顶层而非 action 里。"""
        queries, _ = _collect_search_evidence([
            {"type": "web_search_call", "query": "顶层 query"},
        ])
        self.assertEqual(queries, ["顶层 query"])


if __name__ == "__main__":
    unittest.main()

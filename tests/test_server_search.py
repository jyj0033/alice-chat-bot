"""服务端联网搜索（MiniMax Server Tools / OpenAI Responses）的离线测试。

覆盖两处关键设计：
  1. ResponsesProvider 只按能力放行 web_search，不主动追加；
     是否联网由上层按需声明，避免闲聊也触发多轮搜索把上下文撑爆。
  2. ReplyGenerator 的按需声明：判断器说需要查资料才把
     {"type":"web_search"} 加进本次请求，且判断器本身要认「服务端搜索
     也算有联网能力」——否则生产 search.enabled=false 会让它永远判不检索。
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from modules.llm.base import ChatResponse
from modules.llm.responses_provider import MIN_OUTPUT_TOKENS, ResponsesProvider
from modules.reply.generator import ReplyGenerator

SEND_MEME = {
    "type": "function",
    "function": {
        "name": "send_meme",
        "description": "发一张表情包",
        "parameters": {"type": "object", "properties": {}},
    },
}
WEB_SEARCH = {"type": "web_search"}


class ResponsesProviderToolTests(unittest.TestCase):
    """web_search 的声明闸门与输出预算下限。"""

    def _provider(self, web_search: bool) -> ResponsesProvider:
        return ResponsesProvider({
            "api_key": "test-key",
            "base_url": "https://api.minimax.cn/v1",
            "model": "MiniMax-M3.1-Flash-Preview",
            "web_search": web_search,
        })

    def test_web_search_passes_through_only_when_declared(self):
        """上层声明了才放行；没声明绝不能自动追加。"""
        provider = self._provider(True)
        self.assertIn(WEB_SEARCH, provider._build_tools([SEND_MEME, WEB_SEARCH]))
        self.assertNotIn(WEB_SEARCH, provider._build_tools([SEND_MEME]))
        self.assertNotIn(WEB_SEARCH, provider._build_tools([]))

    def test_web_search_dropped_when_capability_off(self):
        """能力开关关闭时，即使调用方传了也要剔除（防配置与调用不一致误开联网）。"""
        provider = self._provider(False)
        built = provider._build_tools([SEND_MEME, WEB_SEARCH])
        self.assertNotIn(WEB_SEARCH, built)
        self.assertEqual([t.get("name") for t in built], ["send_meme"])

    def test_supports_server_search_follows_capability(self):
        self.assertIs(self._provider(True).supports_server_search, True)
        self.assertIs(self._provider(False).supports_server_search, False)

    def test_output_budget_floor(self):
        """max_tokens=500 时输出上限要抬到下限，给 reasoning 留余量。

        实测出现过「整 1200 预算被 reasoning 吃光、一条 message 都没产出」，
        所以下限必须是 2000。
        """
        provider = ResponsesProvider({
            "api_key": "k", "base_url": "https://api.minimax.cn/v1",
            "model": "m", "max_tokens": 500,
        })
        self.assertEqual(MIN_OUTPUT_TOKENS, 2000)
        self.assertEqual(provider.max_output_tokens, MIN_OUTPUT_TOKENS)

    def test_parse_prefers_top_level_output_text(self):
        """message 片段里 annotations 可达几十条、体积虚高，正文只认顶层字段。"""
        provider = self._provider(False)
        payload = {
            "status": "completed",
            "output_text": "<say>答案</say>",
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "思" * 200}]},
                {"type": "web_search_call", "action": {"type": "search", "query": "崩铁 卡池"}},
                {
                    "type": "function_call", "name": "send_meme",
                    "arguments": '{"category": "笑哭"}', "call_id": "c1",
                },
                {
                    "type": "message",
                    "content": [{
                        "type": "output_text", "text": "<say>答案</say>",
                        "annotations": [{"type": "url_citation", "title": f"来源{i}"} for i in range(50)],
                    }],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

        resp = provider._parse(payload)

        self.assertEqual(resp.content, "<say>答案</say>")
        self.assertEqual(resp.finish_reason, "completed")
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0]["name"], "send_meme")
        self.assertEqual(resp.tool_calls[0]["arguments"], {"category": "笑哭"})
        self.assertEqual(resp.usage["prompt_tokens"], 10)


class ServerSearchDeclarationLogicTests(unittest.TestCase):
    """_should_declare_server_search 的真值表。"""

    def test_truth_table(self):
        decide = ReplyGenerator._should_declare_server_search
        need = {"need_search": True}
        skip = {"need_search": False}

        self.assertTrue(decide(True, need))
        self.assertFalse(decide(True, skip))
        self.assertFalse(decide(False, need))
        self.assertFalse(decide(False, skip))
        self.assertFalse(decide(True, None))
        self.assertFalse(decide(True, {}))
        # 非严格 True（如 MagicMock 的假真值）不得被当成具备能力
        self.assertFalse(decide("yes", need))


class _RecordingProvider:
    """记录每次请求，并按调用场景返回判断器结果或正文。"""

    model = "test"

    def __init__(self, decision: str, supports_server_search: bool = True):
        self.requests = []
        self._decision = decision
        self.supports_server_search = supports_server_search

    @staticmethod
    def _blob(request) -> str:
        return "\n".join(str(getattr(m, "content", "")) for m in request.messages)

    async def chat(self, request):
        self.requests.append(request)
        if "检索判断器" in self._blob(request):
            return ChatResponse(content=self._decision, model=self.model)
        return ChatResponse(content="<say>好嘞</say>", model=self.model)


class ServerSearchDeclarationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """走完整 generate() 流程，断言 web_search 只在需要时出现在主请求里。"""

    _NEED = (
        '<decision>{"need_search":true,"topic":"游戏版本",'
        '"query":"崩铁 当前卡池","freshness":"current"}</decision>'
    )
    _SKIP = (
        '<decision>{"need_search":false,"topic":"","query":"","freshness":"stable"}</decision>'
    )

    async def _run(self, decision: str, supports: bool = True) -> _RecordingProvider:
        provider = _RecordingProvider(decision, supports_server_search=supports)
        generator = ReplyGenerator(llm_provider=provider)
        with patch("modules.reply.generator.asyncio.sleep", new_callable=AsyncMock):
            await generator.generate(
                context_prompt="[刚刚] 小明(对你说)：崩铁现在up是谁",
                current_message="崩铁现在up是谁",
                direction="to_bot",
            )
        return provider

    @staticmethod
    def _main_request(provider: _RecordingProvider):
        for request in provider.requests:
            if "检索判断器" not in _RecordingProvider._blob(request):
                return request
        raise AssertionError("没有捕获到主回复请求")

    @staticmethod
    def _judge_called(provider: _RecordingProvider) -> bool:
        return any("检索判断器" in _RecordingProvider._blob(r) for r in provider.requests)

    async def test_declares_web_search_when_judge_says_needed(self):
        provider = await self._run(self._NEED)
        # 服务端搜索可用时，判断器必须真的被调用（不能因为 search.enabled=false 就跳过）
        self.assertTrue(self._judge_called(provider), "判断器未被调用，按需声明会失效")
        self.assertIn(WEB_SEARCH, self._main_request(provider).tools)

    async def test_does_not_declare_web_search_for_chitchat(self):
        provider = await self._run(self._SKIP)
        self.assertTrue(self._judge_called(provider))
        self.assertNotIn(WEB_SEARCH, self._main_request(provider).tools)

    async def test_no_declaration_when_provider_lacks_capability(self):
        provider = await self._run(self._NEED, supports=False)
        self.assertNotIn(WEB_SEARCH, self._main_request(provider).tools)


if __name__ == "__main__":
    unittest.main()

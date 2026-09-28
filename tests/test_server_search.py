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
from datetime import datetime
from unittest.mock import AsyncMock, patch

from modules.llm.base import ChatMessage, ChatRequest, ChatResponse
from modules.llm.responses_provider import MIN_OUTPUT_TOKENS, ResponsesProvider
from modules.reply.generator import SERVER_SEARCH_TEMPERATURE, ReplyGenerator

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


class ResponsesReasoningEffortTests(unittest.TestCase):
    """reasoning 档位：默认 low，显式配置可关闭。

    背景：不传 reasoning 时模型自适应推理长度，辅助任务（如「从 64 条词表里
    挑出不合语境的条目」）会把整个 max_output_tokens 烧在 reasoning 上，
    返回 status=incomplete 且一条 message 都没有——线上实测约 71% 失败。
    """

    def _provider(self, **extra) -> ResponsesProvider:
        config = {
            "api_key": "k",
            "base_url": "https://api.minimax.cn/v1",
            "model": "MiniMax-M3.1-Flash-Preview",
            "max_tokens": 500,
        }
        config.update(extra)
        return ResponsesProvider(config)

    @staticmethod
    def _body(provider: ResponsesProvider) -> dict:
        return provider._build_body(ChatRequest(messages=[
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
        ]))

    def test_defaults_to_low_effort(self):
        provider = self._provider()
        self.assertEqual(provider.reasoning_effort, "low")
        self.assertEqual(self._body(provider)["reasoning"], {"effort": "low"})

    def test_can_be_disabled_explicitly(self):
        provider = self._provider(reasoning_effort=None)
        self.assertIsNone(provider.reasoning_effort)
        self.assertNotIn("reasoning", self._body(provider))

    def test_configurable_and_normalized(self):
        provider = self._provider(reasoning_effort="MINIMAL")
        self.assertEqual(provider.reasoning_effort, "minimal")
        self.assertEqual(self._body(provider)["reasoning"], {"effort": "minimal"})

    def test_body_carries_instructions_input_and_budget(self):
        body = self._body(self._provider())
        self.assertEqual(body["instructions"], "s")
        self.assertEqual(body["input"], [{"role": "user", "content": "u"}])
        self.assertEqual(body["max_output_tokens"], MIN_OUTPUT_TOKENS)


class ServerSearchGuardrailTests(unittest.TestCase):
    """声明服务端搜索的那一轮：压温度 + 补时效取舍约束。

    背景（2026-09-28 线上探针，见 scripts/probe_search_stability.py）：
    服务端检索会把抓到的网页正文塞进上下文。模型不知道「当前版本号」时会把它
    猜出来的版本号写进检索词（实测猜过 3.1 / 3.7 / 2.4），第一轮检索因此系统性
    跑偏——猜 3.7 那次 20 条来源全是《鸣潮》的。再叠加 0.8 温度采样，同一问题
    4 次得到 4 个互不相同的答案。
    """

    class _StubLLM:
        def __init__(self, config):
            self.config = config
            self.model = "stub"

    def _generator(self, **config):
        base = {"temperature": 0.8, "top_p": 0.9}
        base.update(config)
        return ReplyGenerator(llm_provider=self._StubLLM(base))

    @staticmethod
    def _request() -> ChatRequest:
        return ChatRequest(
            messages=[ChatMessage(role="system", content="人设"),
                      ChatMessage(role="user", content="崩铁现在up角色是谁")],
            temperature=0.8,
        )

    def test_temperature_is_clamped_down(self):
        generator = self._generator()
        request = self._request()

        used = generator._apply_server_search_guardrails(request)

        self.assertAlmostEqual(used, 0.3)
        self.assertAlmostEqual(request.temperature, 0.3)

    def test_temperature_never_raised(self):
        """配置本身更低时保留配置值——这一层只压不抬。"""
        generator = self._generator()
        request = self._request()
        request.temperature = 0.1

        generator._apply_server_search_guardrails(request)

        self.assertAlmostEqual(request.temperature, 0.1)

    def test_configurable_floor(self):
        generator = self._generator(search_temperature=0.0)
        request = self._request()
        generator._apply_server_search_guardrails(request)
        self.assertAlmostEqual(request.temperature, 0.0)

    def test_none_config_falls_back_to_builtin(self):
        """main._init_llm 在配置未填时会把该键显式置为 None，不能当成 0。"""
        generator = self._generator(search_temperature=None)
        request = self._request()
        generator._apply_server_search_guardrails(request)
        self.assertAlmostEqual(request.temperature, SERVER_SEARCH_TEMPERATURE)

    def test_guardrail_mentions_today_and_key_rules(self):
        generator = self._generator()
        request = self._request()
        before = len(request.messages)

        generator._apply_server_search_guardrails(request)

        self.assertEqual(len(request.messages), before + 1)
        added = request.messages[-1]
        self.assertEqual(added.role, "system")
        today = datetime.now().strftime("%Y年%m月%d日")
        self.assertIn(today, added.content)
        # 四条关键约束：时效取最新、不确定别编造、别串到别的作品
        self.assertIn("只回答【正在进行】的内容", added.content)
        self.assertIn("以日期最新的为准", added.content)
        self.assertIn("绝不要编造版本号", added.content)
        self.assertIn("别的游戏", added.content)


class _RecordingProvider:
    """记录每次请求，并按调用场景返回判断器结果或正文。"""

    model = "test"

    def __init__(self, decision: str, supports_server_search: bool = True,
                 config: dict | None = None):
        self.requests = []
        self._decision = decision
        self.supports_server_search = supports_server_search
        self.config = config if config is not None else {}

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

    async def test_guardrails_ride_along_with_declaration(self):
        """声明了服务端搜索 → 温度被压到下限，且带上了时效取舍约束。"""
        provider = await self._run(self._NEED)
        request = self._main_request(provider)

        self.assertIn(WEB_SEARCH, request.tools)
        self.assertLessEqual(request.temperature, SERVER_SEARCH_TEMPERATURE)
        guardrails = [
            m for m in request.messages
            if m.role == "system" and "本轮已开启联网检索" in str(m.content)
        ]
        self.assertEqual(len(guardrails), 1)

    async def test_no_guardrails_when_not_declared(self):
        """闲聊不声明搜索，也就不能压温度、不能塞约束（否则是全局行为改变）。"""
        provider = await self._run(self._SKIP)
        request = self._main_request(provider)

        self.assertTrue(self._judge_called(provider))
        self.assertNotIn(WEB_SEARCH, request.tools)
        self.assertGreater(request.temperature, SERVER_SEARCH_TEMPERATURE)
        self.assertFalse([
            m for m in request.messages
            if m.role == "system" and "本轮已开启联网检索" in str(m.content)
        ])

    async def test_no_declaration_when_provider_lacks_capability(self):
        provider = await self._run(self._NEED, supports=False)
        self.assertNotIn(WEB_SEARCH, self._main_request(provider).tools)


if __name__ == "__main__":
    unittest.main()

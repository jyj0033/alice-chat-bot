"""「服务端内置工具 + function 调用」同轮共存的三道防线。

背景（2026-11-13 线上事故）：
主模型 qianyemeng（gemini-3.8-flash-high，走 newapi 中转）**不允许**内置工具
（web_search）与 function calling（send_meme）出现在同一个请求里，混用必然
HTTP 400：

    Please enable tool_config.include_server_side_tool_invocations to use
    Built-in tools with Function calling.

客户端改不动：上游提示要加 tool_config.include_server_side_tool_invocations，
但中转不转发这个字段（snake_case / camelCase / 顶层三种写法实测全部仍 400）。

线上后果：11:28 一条 @ 消息被判「要回复」，检索判断判定「需要联网」→ 声明
web_search → 与 send_meme 混用 → 400 → 回退模型出了草稿但格式不合规 →
重写时**绕过了回退机制**直连主模型 → 二次 400 → 草稿为空 →
`[沉默] 放弃本轮回复` → 群里一条都没发出来。

三道防线：
  1. C′（源头）：声明 web_search 的那一轮**不注册** send_meme，根本不混用。
  2. A（兜底）：万一还有别的组合触发同一个 400，provider 识别报错后
     去掉服务端工具重发一次，保证回复发得出去（代价是该轮不联网）。
  3. 小修：格式重写路径改走 `_chat()`，让重写也能回退到备用 Provider。
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from modules.llm.base import ChatMessage, ChatRequest, ChatResponse
from modules.llm.responses_provider import ResponsesProvider
from modules.reply.generator import ReplyGenerator

WEB_SEARCH = {"type": "web_search"}
SEND_MEME = {
    "type": "function",
    "function": {
        "name": "send_meme",
        "description": "发一张表情包",
        "parameters": {"type": "object", "properties": {}},
    },
}
MIXED_TOOLS_ERROR = (
    '{"error":{"message":"Please enable tool_config.include_server_side_tool_invocations '
    'to use Built-in tools with Function calling.","type":"upstream_error","code":400}}'
)


class _FakeMemeManager:
    """最小可用的表情管理器桩：auto 开启、带工具定义和提示词。"""

    auto_send_enabled = True

    def build_prompt_guide(self) -> str:
        return "【表情包使用指南】分类：笑哭、无语……"

    def tool_definition(self) -> dict:
        return SEND_MEME


class MixedToolsDeclarationTests(unittest.TestCase):
    """C′：声明 web_search 的那一轮不能注册 send_meme。"""

    def _generator(self) -> ReplyGenerator:
        generator = ReplyGenerator(llm_provider=None)
        generator.meme_manager = _FakeMemeManager()
        return generator

    def test_search_round_registers_no_meme_tool(self):
        """联网轮：只有 web_search，没有 send_meme（混用就是那个 400）。

        注意 web_search 本身是 generate() 在 _build_request 之后追加的，
        这里断言的是「send_meme 没有被注册」——混用的另一半。
        """
        generator = self._generator()

        request = generator._build_request(
            "上下文", "今天北京天气如何", direction="to_bot",
            server_search_declared=True,
        )

        self.assertNotIn(SEND_MEME, request.tools or [])
        self.assertNotIn("send_meme", [str(t) for t in (request.tools or [])])

    def test_normal_round_still_registers_meme_tool(self):
        """不联网的轮次照常能发图——C′ 不能误伤日常发言。"""
        generator = self._generator()

        request = generator._build_request(
            "上下文", "在吗", direction="to_bot", server_search_declared=False,
        )

        self.assertNotIn(WEB_SEARCH, request.tools or [])
        self.assertIn(SEND_MEME, request.tools or [])

    def test_search_round_tells_model_not_to_send_meme(self):
        """工具没注册了，提示词也要改口，否则模型会硬凑表情包标记。"""
        generator = self._generator()

        request = generator._build_request(
            "上下文", "今天北京天气如何", direction="to_bot",
            server_search_declared=True,
        )

        blob = "\n".join(str(m.content) for m in request.messages)
        self.assertIn("本轮不支持发图", blob)
        self.assertNotIn("【表情包使用指南】", blob)

    def test_no_meme_manager_is_unaffected(self):
        """没有表情管理器时不能因为联网就报错。"""
        generator = ReplyGenerator(llm_provider=None)
        generator.meme_manager = None

        request = generator._build_request(
            "上下文", "今天北京天气如何", direction="to_bot",
            server_search_declared=True,
        )

        self.assertEqual(request.tools or [], [])


class MixedToolsErrorDetectionTests(unittest.TestCase):
    """A：识别「内置工具 + function 调用」共存被拒的那条 400。"""

    def test_detects_the_specific_400(self):
        self.assertTrue(
            ResponsesProvider._is_mixed_tools_error(400, MIXED_TOOLS_ERROR)
        )

    def test_ignores_other_errors(self):
        """只认这一条固定文案，别把别的 400 误判成要降级重发。"""
        self.assertFalse(ResponsesProvider._is_mixed_tools_error(500, MIXED_TOOLS_ERROR))
        self.assertFalse(ResponsesProvider._is_mixed_tools_error(400, '{"error":{"message":"rate limited"}}'))
        self.assertFalse(ResponsesProvider._is_mixed_tools_error(400, ""))
        self.assertFalse(ResponsesProvider._is_mixed_tools_error(400, None))

    def test_drop_server_tools_removes_only_server_tools(self):
        """降级重发只摘服务端工具，function 工具必须留着（还要靠它发图）。"""
        provider = ResponsesProvider({
            "api_key": "k", "base_url": "https://api.minimax.cn/v1",
            "model": "m", "web_search": True,
        })

        built = provider._build_tools(
            [SEND_MEME, WEB_SEARCH], drop_server_tools=True
        )

        self.assertNotIn(WEB_SEARCH, built)
        self.assertEqual([t.get("name") for t in built], ["send_meme"])

    def test_drop_server_tools_body_has_no_server_tool(self):
        provider = ResponsesProvider({
            "api_key": "k", "base_url": "https://api.minimax.cn/v1",
            "model": "m", "web_search": True,
        })
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="u")],
            tools=[SEND_MEME, WEB_SEARCH],
        )

        normal = provider._build_body(request)
        degraded = provider._build_body(request, drop_server_tools=True)

        self.assertIn(WEB_SEARCH, normal.get("tools", []))
        self.assertNotIn(WEB_SEARCH, degraded.get("tools", []))
        self.assertIn("send_meme", [t.get("name") for t in degraded.get("tools", [])])


class _FakeResponse:
    """模拟 aiohttp 响应对象。"""

    def __init__(self, status: int, text: str):
        self.status = status
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """记录每次 POST 的请求体，按脚本返回响应。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.bodies = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.bodies.append(json)
        return self._responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class MixedToolsFallbackRetryTests(unittest.IsolatedAsyncioTestCase):
    """A：真发一次请求，验证 400 后自动去掉服务端工具重发。"""

    async def test_retries_without_server_tools_after_400(self):
        provider = ResponsesProvider({
            "api_key": "k", "base_url": "https://api.minimax.cn/v1",
            "model": "m", "web_search": True,
        })
        ok_payload = (
            '{"status":"completed","output_text":"<say>好</say>",'
            '"output":[{"type":"message","content":[{"type":"output_text","text":"<say>好</say>"}]}]}'
        )
        session = _FakeSession([
            _FakeResponse(400, MIXED_TOOLS_ERROR),
            _FakeResponse(200, ok_payload),
        ])
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="u")],
            tools=[SEND_MEME, WEB_SEARCH],
        )

        with patch("aiohttp.ClientSession", return_value=session):
            resp = await provider.chat(request)

        self.assertEqual(resp.content, "<say>好</say>")
        self.assertEqual(len(session.bodies), 2)
        # 第一次带 web_search，第二次不带
        self.assertIn(WEB_SEARCH, session.bodies[0].get("tools", []))
        self.assertNotIn(WEB_SEARCH, session.bodies[1].get("tools", []))
        self.assertIn("send_meme", [t.get("name") for t in session.bodies[1].get("tools", [])])

    async def test_no_retry_when_error_is_unrelated(self):
        """别的 400 直接抛，不做无谓重试。"""
        provider = ResponsesProvider({
            "api_key": "k", "base_url": "https://api.minimax.cn/v1",
            "model": "m", "web_search": True,
        })
        session = _FakeSession([_FakeResponse(400, '{"error":{"message":"bad request"}}')])
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="u")],
            tools=[SEND_MEME, WEB_SEARCH],
        )

        with patch("aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError):
                await provider.chat(request)

        self.assertEqual(len(session.bodies), 1)

    async def test_no_retry_when_no_server_tools_present(self):
        """本来就没声明服务端工具时，不需要也不应该重发。"""
        provider = ResponsesProvider({
            "api_key": "k", "base_url": "https://api.minimax.cn/v1",
            "model": "m", "web_search": True,
        })
        session = _FakeSession([_FakeResponse(400, MIXED_TOOLS_ERROR)])
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="u")],
            tools=[SEND_MEME],
        )

        with patch("aiohttp.ClientSession", return_value=session):
            with self.assertRaises(RuntimeError):
                await provider.chat(request)

        self.assertEqual(len(session.bodies), 1)


class FormatRewriteFallbackTests(unittest.IsolatedAsyncioTestCase):
    """小修：格式重写必须走 _chat()，主模型挂了要能回退。"""

    async def test_rewrite_falls_back_to_backup_provider(self):
        """主模型在重写阶段抛错时，备用 Provider 顶上，重写不能直接失败。"""
        from modules.reply.generator import ReplyGenerator as RG

        class _MainLLM:
            model = "main-model"
            config = {}

            def __init__(self):
                self.calls = 0

            async def chat(self, request):
                self.calls += 1
                # 第一次（正文生成）成功但格式不合规；第二次（重写）抛错
                if self.calls == 1:
                    return ChatResponse(content="没有标签的草稿", model="main-model")
                raise RuntimeError("主模型挂了")

        class _BackupLLM:
            model = "backup-model"
            config = {}

            def __init__(self):
                self.calls = 0

            async def chat(self, request):
                self.calls += 1
                return ChatResponse(content="<say>重写成功</say>", model="backup-model")

        main = _MainLLM()
        backup = _BackupLLM()
        generator = RG(llm_provider=main)
        generator.fallback_llm = backup

        with patch("modules.reply.generator.asyncio.sleep", new_callable=AsyncMock):
            result = await generator.generate(
                context_prompt="[刚刚] 小明(对你说)：在吗",
                current_message="在吗",
                direction="to_bot",
            )

        self.assertIsNotNone(result, "重写失败后整轮回复被丢弃了")
        self.assertEqual(result.get("reply"), "重写成功")
        self.assertGreaterEqual(backup.calls, 1, "备用 Provider 没被用上")


class _RecordingProvider:
    """记录主请求的 tools，按调用场景返回判断器结果或正文。"""

    model = "test"
    # 主模型自带服务端搜索能力，否则检索判断器直接短路返回「不检索」，
    # 根本走不到「声明 web_search」这一步。
    supports_server_search = True

    def __init__(self, decision: str):
        self.requests = []
        self._decision = decision

    @staticmethod
    def _blob(request) -> str:
        return "\n".join(str(getattr(m, "content", "")) for m in request.messages)

    async def chat(self, request):
        self.requests.append(request)
        if "检索判断器" in self._blob(request):
            return ChatResponse(content=self._decision, model=self.model)
        return ChatResponse(content="<say>好嘞</say>", model=self.model)


class MixedToolsEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """走完整 generate()：声明 web_search 的那一轮绝不能同时带 send_meme。"""

    _NEED = (
        '<decision>{"need_search":true,"topic":"天气",'
        '"query":"北京 今日气温","freshness":"current"}</decision>'
    )

    async def test_search_round_has_no_send_meme_in_final_request(self):
        provider = _RecordingProvider(self._NEED)
        generator = ReplyGenerator(llm_provider=provider)
        generator.meme_manager = _FakeMemeManager()

        with patch("modules.reply.generator.asyncio.sleep", new_callable=AsyncMock):
            await generator.generate(
                context_prompt="[刚刚] 小明(对你说)：今天北京天气如何",
                current_message="今天北京天气如何",
                direction="to_bot",
            )

        main = None
        for request in provider.requests:
            if "检索判断器" not in _RecordingProvider._blob(request):
                main = request
                break
        self.assertIsNotNone(main, "没有捕获到主回复请求")

        types = [str(t.get("type")) for t in (main.tools or [])]
        self.assertIn("web_search", types, "该联网的消息没有声明 web_search")
        self.assertNotIn("function", types, "联网轮仍注册了 function 工具，会触发上游 400")

    async def test_non_search_round_keeps_send_meme(self):
        """不联网的轮次照常带 send_meme——不能为了避 400 把发图功能整体废掉。"""
        provider = _RecordingProvider(
            '<decision>{"need_search":false,"topic":"","query":"","freshness":"stable"}</decision>'
        )
        generator = ReplyGenerator(llm_provider=provider)
        generator.meme_manager = _FakeMemeManager()

        with patch("modules.reply.generator.asyncio.sleep", new_callable=AsyncMock):
            await generator.generate(
                context_prompt="[刚刚] 小明(对你说)：在吗",
                current_message="在吗",
                direction="to_bot",
            )

        main = None
        for request in provider.requests:
            if "检索判断器" not in _RecordingProvider._blob(request):
                main = request
                break
        self.assertIsNotNone(main)

        names = [
            (t.get("function") or {}).get("name")
            for t in (main.tools or [])
            if t.get("type") == "function"
        ]
        self.assertIn("send_meme", names)
        self.assertNotIn("web_search", [str(t.get("type")) for t in (main.tools or [])])


if __name__ == "__main__":
    unittest.main()

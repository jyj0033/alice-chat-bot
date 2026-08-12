"""LLM provider 响应解析的离线测试。"""

import unittest
from unittest.mock import AsyncMock

from modules.llm.base import ChatRequest
from modules.llm.openai_provider import OpenAIProvider


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


if __name__ == "__main__":
    unittest.main()

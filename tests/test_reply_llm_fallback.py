"""主模型调用失败时的跨 Provider 回退测试。

背景：第三方中转会偶发「HTTP 200 但 choices 为空」。generator 原先在主模型
抛异常时直接 `_give_up_reply`，一次中转抖动就是一条回复凭空消失、bot 沉默，
日志里只留一行 ERROR。这里锁定「换一家再问一次」这条兜底路径。
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from modules.llm.base import ChatResponse
from modules.reply.generator import ReplyGenerator


def _provider(*, content=None, error=None, model="test-model"):
    provider = MagicMock()
    provider.model = model
    provider.supports_server_search = False
    if error is not None:
        provider.chat = AsyncMock(side_effect=error)
    else:
        provider.chat = AsyncMock(
            return_value=ChatResponse(content=content, model=model)
        )
    return provider


class CrossProviderFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_falls_back_to_the_other_provider(self):
        """主模型调用失败时，应该换备用 Provider 重试并拿到可用回复。"""
        main = _provider(
            error=ValueError("LLM 返回了空 choices（HTTP 200 但没有候选结果）")
        )
        backup = _provider(content="<say>我还在，刚网卡了一下。</say>", model="MiniMax-M3.1")
        gen = ReplyGenerator(llm_provider=main, fallback_llm_provider=backup)

        result = await gen.generate(
            context_prompt="[刚刚] 小明：在吗",
            current_message="在吗",
            direction="to_bot",
        )

        self.assertIsNotNone(result)
        self.assertIn("我还在", result["reply"])
        backup.chat.assert_awaited_once()

    async def test_reraises_the_original_error_when_both_fail(self):
        """两家都失败时把原始异常交给调用方，而不是备用那家的异常。"""
        main = _provider(error=ValueError("主模型返回了空 choices"))
        backup = _provider(error=RuntimeError("备用也挂了"))
        gen = ReplyGenerator(llm_provider=main, fallback_llm_provider=backup)

        with self.assertRaises(ValueError) as ctx:
            await gen._chat(object(), "正文生成")

        self.assertIn("主模型返回了空 choices", str(ctx.exception))

    async def test_without_fallback_the_error_propagates(self):
        """只配了一家 Provider 时行为不变，仍由调用方决定是否沉默。"""
        main = _provider(error=ValueError("只有一家"))
        gen = ReplyGenerator(llm_provider=main)

        with self.assertRaises(ValueError):
            await gen._chat(object(), "正文生成")

    async def test_same_instance_is_not_retried(self):
        """备用就是主模型本身时不做无效重试，否则只是把同一个错误再犯一次。"""
        main = _provider(error=ValueError("同一个实例"))
        gen = ReplyGenerator(llm_provider=main, fallback_llm_provider=main)

        with self.assertRaises(ValueError):
            await gen._chat(object(), "正文生成")

        main.chat.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

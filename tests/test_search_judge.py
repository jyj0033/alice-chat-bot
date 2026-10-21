"""检索判断器测试：要不要查资料、查什么，以及带资料生成的兜底。

外部搜索后端（博查 / 豆包）已下线：联网统一走主模型的服务端 web_search，
这里只保留「要不要检索 + 查什么 + 本地资料库注入」这一层。
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from modules.llm.base import ChatResponse
from modules.reply.generator import ReplyGenerator


def _knowledge_store(summary="流萤", confidence="exact"):
    """一个命中「卡芙卡」的本地资料库 mock。"""
    store = MagicMock()
    store.knowledge_count = AsyncMock(return_value=1)
    store.search_knowledge = AsyncMock(return_value=[{
        "id": 1,
        "name": "卡芙卡",
        "aliases": [],
        "summary": summary,
        "subject": "崩坏：星穹铁道",
        "source": "米哈游wiki",
        "updated_at": "2026-03-01 12:00:00",
        "version": "2.7",
        "confidence": confidence,
    }])
    return store


class SearchJudgeTests(unittest.IsolatedAsyncioTestCase):
    def _make_generator(self, tool_llm, main_llm=None, knowledge_store=None):
        llm = main_llm or MagicMock()
        # 生产里回复模型带服务端 web_search；判断器据此认定「有联网能力」，
        # 不会因为外部搜索后端下线就永远判「不需要检索」。
        llm.supports_server_search = True
        gen = ReplyGenerator(
            llm_provider=llm,
            tool_llm_provider=tool_llm,
            knowledge_store=knowledge_store,
        )
        if not getattr(tool_llm, "model", None):
            tool_llm.model = "test"
        return gen

    def _needs_search(self, decision):
        return bool((decision or {}).get("need_search"))

    # === 判断器：要不要检索 ===

    async def test_judge_yes_when_llm_says_yes(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="YES", model="test"))
        gen = self._make_generator(tool_llm)
        decision = await gen._judge_need_search("绝区零现在开的谁的池子", "[刚刚] 小明：绝区零现在开的谁的池子")
        self.assertTrue(self._needs_search(decision))
        self.assertEqual(decision["query"], "绝区零现在开的谁的池子")
        tool_llm.chat.assert_awaited_once()

    async def test_judge_no_when_llm_says_no(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="NO", model="test"))
        gen = self._make_generator(tool_llm)
        decision = await gen._judge_need_search("晚饭吃啥", "[刚刚] 小明：晚饭吃啥")
        self.assertFalse(self._needs_search(decision))

    async def test_judge_negative_phrase_not_misparsed(self):
        """「不需要」这类否定不能被「需要」误判成要检索。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="不需要搜索", model="test"))
        gen = self._make_generator(tool_llm)
        decision = await gen._judge_need_search("随便聊聊", "[刚刚] 小明：随便聊聊")
        self.assertFalse(self._needs_search(decision))

    async def test_judge_media_description_never_searches(self):
        """表情包/图片识别描述不判断（富媒体内容，不是用户问句）。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="YES", model="test"))
        gen = self._make_generator(tool_llm)
        decision = await gen._judge_need_search(
            "[表情包，内容：白发红瞳的动漫角色闭眼咧嘴笑着…，回复：笑死]", "")
        self.assertFalse(self._needs_search(decision))
        tool_llm.chat.assert_not_awaited()

    async def test_group_short_reaction_skips_search_judge(self):
        """群聊短反应即使含有问句词，也不应先触发检索判断。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="YES", model="test"))
        gen = self._make_generator(tool_llm)
        decision = await gen._judge_need_search(
            "现在是什么", "[刚刚] 小明：现在是什么",
            direction="group",
            action_plan={"action": "react"},
        )
        self.assertFalse(self._needs_search(decision))
        tool_llm.chat.assert_not_awaited()

    async def test_judge_failure_conservative_no(self):
        """判断调用异常 → 按不需要检索处理，保证回复不中断。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=RuntimeError("boom"))
        gen = self._make_generator(tool_llm)
        decision = await gen._judge_need_search("今天天气", "[刚刚] 小明：今天天气")
        self.assertFalse(self._needs_search(decision))

    # === 判断器输出解析 ===

    def test_parse_structured_decision(self):
        raw = (
            "<think>先看上下文</think>"
            '<decision>{"need_search":true,"topic":"网络梗解释",'
            '"query":"xx梗 出处 意思","freshness":"stable"}</decision>'
        )
        decision = ReplyGenerator._parse_search_decision(raw, fallback_query="这是什么梗")
        self.assertTrue(decision["need_search"])
        self.assertEqual(decision["topic"], "网络梗解释")
        self.assertEqual(decision["query"], "xx梗 出处 意思")
        self.assertEqual(decision["freshness"], "stable")

    def test_parse_decision_defaults_gacha_to_current(self):
        decision = ReplyGenerator._parse_search_decision(
            '{"need_search": true, "topic": "游戏版本", "query": "鸣潮当前卡池"}',
            fallback_query="现在开谁",
        )
        self.assertEqual(decision["freshness"], "current")

    def test_expand_deictic_query_from_context(self):
        filled = ReplyGenerator._expand_search_query(
            "这是什么梗",
            "这是什么梗",
            "[刚刚] 小明：xx这个表情也太烂了\n[刚刚] 小红：这是什么梗",
        )
        self.assertIn("xx", filled)
        self.assertIn("这是什么梗", filled)

    async def test_judge_uses_structured_query_from_context(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(
            content='<decision>{"need_search":true,"topic":"游戏角色",'
                    '"query":"崩坏星穹铁道 卡芙卡","freshness":"stable"}</decision>',
            model="test",
        ))
        gen = self._make_generator(tool_llm)
        decision = await gen._judge_need_search(
            "那角色呢",
            "[刚刚] 小明：崩铁那个卡芙卡也太帅了\n[刚刚] 小红：那角色呢",
        )
        self.assertTrue(decision["need_search"])
        self.assertEqual(decision["query"], "崩坏星穹铁道 卡芙卡")
        self.assertEqual(decision["freshness"], "stable")

    # === generate() 里的分流 ===

    async def test_no_search_when_llm_judges_no(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="NO", model="test"))
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(content="<say>随便吃点</say>", model="test"))
        gen = self._make_generator(tool_llm, main_llm)
        gen_result = await gen.generate(
            context_prompt="[刚刚] 小明：晚饭吃啥",
            current_message="晚饭吃啥",
            direction="group",
        )
        reply = gen_result["reply"] if isinstance(gen_result, dict) else gen_result
        # LLM 判定不需要检索 → 走主 LLM，不进资料注入分支
        self.assertIsNotNone(reply)
        main_llm.chat.assert_awaited_once()

    async def test_no_knowledge_hits_falls_back_to_main_llm(self):
        """判断说要看、但本地资料库没命中 → 不进资料注入分支，主 LLM 正常回答。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(return_value=ChatResponse(content="YES", model="test"))
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(content="<say>没查到，正常回答</say>", model="test"))
        gen = self._make_generator(tool_llm, main_llm)
        gen_result = await gen.generate(
            context_prompt="[刚刚] 小明：今天有什么新闻？",
            current_message="今天有什么新闻",
            direction="group",
        )
        reply = gen_result["reply"] if isinstance(gen_result, dict) else gen_result
        self.assertIn("没查到", reply)
        # 判断调用一次；没有本地资料 → 不再带资料生成，主 LLM 兜底
        self.assertEqual(tool_llm.chat.call_count, 1)
        main_llm.chat.assert_awaited_once()

    # === 带资料生成失败 → 主 LLM 带同一份资料重答 ===

    async def test_tool_llm_think_only_retries_with_data_on_main_llm(self):
        """带资料生成的 LLM 只输出 <think> 思考块 → 主 LLM 带同一份资料重答。

        回归 2026-08-10 21:10：查到了资料但生成器返回纯思考块，
        _clean_thinking_process 清空后走了「回退无资料主 LLM」，凭记忆答错。
        """
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(content="YES", model="test"),  # 判断
            ChatResponse(content="<think>崩铁现在up的是流萤，记不太清了</think>", model="test"),
        ])
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(
            content="<say>崩铁现在up的是流萤。</say>", model="test"))
        gen = self._make_generator(
            tool_llm, main_llm, knowledge_store=_knowledge_store(summary="流萤"))
        gen_result = await gen.generate(
            context_prompt="[刚刚] 小明：崩铁现在up角色是谁？",
            current_message="崩铁现在up角色是谁",
            direction="group",
        )
        reply = gen_result["reply"] if isinstance(gen_result, dict) else gen_result
        self.assertIn("流萤", reply)
        # 主 LLM 带同一份资料重答
        main_llm.chat.assert_awaited_once()
        user_text = "\n".join(
            message.content
            for message in main_llm.chat.await_args.args[0].messages
            if message.role == "user"
        )
        self.assertIn("本地资料库", user_text)
        self.assertIn("流萤", user_text)
        self.assertIn("不符合发送格式", user_text)

    async def test_tool_llm_empty_content_retries_with_data(self):
        """带资料生成的 LLM 返回空 content（resp 非 None 但无内容）→ 同样带资料重答。"""
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(content="YES", model="test"),  # 判断
            ChatResponse(content="", model="test"),
        ])
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(
            content="<say>流萤是限定角色。</say>", model="test"))
        gen = self._make_generator(
            tool_llm, main_llm, knowledge_store=_knowledge_store(summary="流萤"))
        gen_result = await gen.generate(
            context_prompt="[刚刚] 小明：流萤是谁",
            current_message="流萤是谁",
            direction="group",
        )
        reply = gen_result["reply"] if isinstance(gen_result, dict) else gen_result
        self.assertIn("流萤", reply)
        main_llm.chat.assert_awaited_once()
        user_text = "\n".join(
            message.content
            for message in main_llm.chat.await_args.args[0].messages
            if message.role == "user"
        )
        self.assertIn("本地资料库", user_text)
        self.assertIn("不符合发送格式", user_text)


if __name__ == "__main__":
    unittest.main()

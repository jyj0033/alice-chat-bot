"""全局热梗/游戏资料库：与群黑话隔离，别名匹配优先。"""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from modules.llm.base import ChatResponse
from modules.memory.storage import MemoryStorage
from modules.reply.generator import ReplyGenerator


class KnowledgeStorageTests(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db_path = path
        self.store = MemoryStorage(db_path=path)

    def tearDown(self):
        self.store.close()
        try:
            os.remove(self.db_path)
        except OSError:
            pass

    def test_exact_alias_match_is_global(self):
        entry_id = self.store.upsert_knowledge(
            name="卡芙卡",
            aliases=["卡师傅", "Kafka"],
            summary="星核猎手，擅长控制",
            subject="崩坏：星穹铁道",
            source="米哈游wiki",
            version="2.7",
        )
        self.assertTrue(entry_id)
        hits = self.store.search_knowledge("卡师傅是谁")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["name"], "卡芙卡")
        self.assertEqual(hits[0]["confidence"], "exact")
        self.assertEqual(hits[0]["source"], "米哈游wiki")
        self.assertEqual(hits[0]["version"], "2.7")
        self.assertTrue(hits[0]["updated_at"])

    def test_same_name_different_subject_can_coexist(self):
        self.store.upsert_knowledge("爱丽丝", "崩铁角色", subject="崩坏：星穹铁道")
        self.store.upsert_knowledge("爱丽丝", "网络热梗解释", subject="网络梗")
        rows = self.store.list_knowledge()
        self.assertEqual(len(rows), 2)
        hits = self.store.search_knowledge("爱丽丝")
        self.assertGreaterEqual(len(hits), 2)
        self.assertTrue(all(item["confidence"] == "uncertain" for item in hits))

    def test_group_slang_is_not_knowledge(self):
        self.store.upsert_slang("卡芙卡", "群里对某个人的外号", session="group_1")
        self.store.upsert_knowledge(
            "卡芙卡", "星核猎手", subject="崩坏：星穹铁道", source="wiki"
        )
        slang = self.store.list_slang(session="group_1", enabled_only=True)
        self.assertEqual([row["meaning"] for row in slang], ["群里对某个人的外号"])
        self.assertEqual(slang[0]["session"], "group_1")
        hits = self.store.search_knowledge("卡芙卡", exclude_names=["卡芙卡"])
        self.assertEqual(hits, [])
        other_group = self.store.search_knowledge("卡芙卡")
        self.assertEqual(other_group[0]["summary"], "星核猎手")
        self.assertNotIn("session", other_group[0])

    def test_disabled_entry_is_not_searched(self):
        entry_id = self.store.upsert_knowledge("绝区零", "动作游戏")
        self.store.update_knowledge(entry_id, enabled=False)
        self.assertEqual(self.store.search_knowledge("绝区零"), [])
        self.assertEqual(self.store.knowledge_count(), 0)

    def test_upsert_updates_same_name_and_subject(self):
        first = self.store.upsert_knowledge("流萤", "旧解释", subject="崩铁")
        second = self.store.upsert_knowledge("流萤", "新解释", subject="崩铁", source="wiki")
        self.assertEqual(first, second)
        self.assertEqual(len(self.store.list_knowledge()), 1)
        self.assertEqual(self.store.get_knowledge(first)["summary"], "新解释")


class KnowledgeReplyTests(unittest.IsolatedAsyncioTestCase):
    def _make_generator(self, tool_llm, knowledge_store, main_llm=None):
        gen = ReplyGenerator(
            llm_provider=main_llm or MagicMock(),
            tool_llm_provider=tool_llm,
            knowledge_store=knowledge_store,
        )
        tool_llm.model = "test"
        return gen

    async def test_exact_knowledge_is_injected_into_reply(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(
                content='<decision>{"need_search":true,"topic":"游戏角色",'
                        '"query":"崩铁 卡芙卡","freshness":"stable"}</decision>',
                model="test",
            ),
            ChatResponse(content="<say>卡芙卡是星核猎手。</say>", model="test"),
        ])
        knowledge_store = MagicMock()
        knowledge_store.knowledge_count = AsyncMock(return_value=1)
        knowledge_store.search_knowledge = AsyncMock(return_value=[{
            "id": 1,
            "name": "卡芙卡",
            "aliases": ["卡师傅"],
            "summary": "星核猎手",
            "subject": "崩坏：星穹铁道",
            "source": "米哈游wiki",
            "updated_at": "2026-03-01 12:00:00",
            "version": "2.7",
            "confidence": "exact",
        }])
        gen = self._make_generator(tool_llm, knowledge_store)
        result = await gen.generate(
            context_prompt="[刚刚] 小明：崩铁那个卡芙卡\n[刚刚] 小红：那角色呢",
            current_message="那角色呢",
            direction="to_bot",
        )
        self.assertIn("星核猎手", result["reply"])
        last_content = tool_llm.chat.await_args.args[0].messages[-1].content
        self.assertIn("本地资料库", last_content)
        self.assertIn("米哈游wiki", last_content)
        self.assertIn("2026-03-01", last_content)
        self.assertIn("2.7", last_content)

    async def test_group_slang_is_excluded_from_knowledge_lookup(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(
                content='<decision>{"need_search":true,"topic":"游戏角色",'
                        '"query":"卡芙卡","freshness":"stable"}</decision>',
                model="test",
            ),
            ChatResponse(content="<say>哦那是她。</say>", model="test"),
        ])
        knowledge_store = MagicMock()
        knowledge_store.knowledge_count = AsyncMock(return_value=1)
        knowledge_store.search_knowledge = AsyncMock(return_value=[])
        main_llm = MagicMock()
        main_llm.chat = AsyncMock(return_value=ChatResponse(
            content="<say>哦那是她。</say>", model="test"
        ))
        gen = self._make_generator(tool_llm, knowledge_store, main_llm)
        await gen.generate(
            context_prompt="[刚刚] 小明：卡芙卡又来了",
            current_message="卡芙卡是谁",
            direction="to_bot",
            glossary=[{"term": "卡芙卡", "meaning": "群友外号"}],
        )
        kwargs = knowledge_store.search_knowledge.await_args.kwargs
        self.assertIn("卡芙卡", kwargs.get("exclude_names") or [])

    async def test_uncertain_knowledge_asks_model_to_be_cautious(self):
        tool_llm = MagicMock()
        tool_llm.chat = AsyncMock(side_effect=[
            ChatResponse(
                content='<decision>{"need_search":true,"topic":"网络梗解释",'
                        '"query":"xx梗 意思","freshness":"stable"}</decision>',
                model="test",
            ),
            ChatResponse(content="<say>我不太确定是不是这个。</say>", model="test"),
        ])
        knowledge_store = MagicMock()
        knowledge_store.knowledge_count = AsyncMock(return_value=1)
        knowledge_store.search_knowledge = AsyncMock(return_value=[{
            "id": 2,
            "name": "某个梗",
            "aliases": [],
            "summary": "可能相关",
            "subject": "网络梗",
            "source": "论坛",
            "updated_at": "2026-01-01",
            "version": "",
            "confidence": "uncertain",
        }])
        gen = self._make_generator(tool_llm, knowledge_store)
        await gen.generate(
            context_prompt="[刚刚] 小明：xx",
            current_message="这是什么梗",
            direction="to_bot",
        )
        last_content = tool_llm.chat.await_args.args[0].messages[-1].content
        self.assertIn("不确定", last_content)


if __name__ == "__main__":
    unittest.main()

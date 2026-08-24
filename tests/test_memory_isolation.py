"""会话隔离测试：私聊记忆不泄漏进群聊检索，各会话短时状态相互独立。"""

import os
import tempfile
import unittest
import asyncio
from datetime import datetime, timedelta

from modules.memory.storage import Memory, MemoryStorage
from modules.social.attention import AttentionManager


def _make_storage():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    storage = MemoryStorage(db_path=path)
    storage._tmp_path = path  # 便于测试后清理
    return storage


def _cleanup(storage):
    try:
        storage.conn.close()
    except Exception:
        pass
    try:
        os.remove(storage._tmp_path)
    except OSError:
        pass


class MemoryIsolationTests(unittest.TestCase):
    def setUp(self):
        self.storage = _make_storage()
        self.addCleanup(_cleanup, self.storage)
        for sid, content in [
            ("group_111", "群A：龍飛月喜欢催收"),
            ("private_222", "私密：我月底工资发不出来"),
            ("group_333", "群C：绝区零更新了"),
        ]:
            self.storage.store(Memory(
                content=content,
                memory_type="episodic",
                importance=0.9,
                source_session=sid,
            ))

    def test_group_retrieval_is_session_scoped_by_default(self):
        candidates = self.storage._get_candidates("group_111", top_k=200)
        sources = {m.source_session for m in candidates}
        self.assertEqual(sources, {"group_111"})

    def test_private_retrieval_is_session_scoped_by_default(self):
        candidates = self.storage._get_candidates("private_222", top_k=200)
        sources = {m.source_session for m in candidates}
        self.assertEqual(sources, {"private_222"})

    def test_cross_session_sharing_requires_explicit_opt_in(self):
        self.storage.share_across_sessions = True

        group_sources = {
            m.source_session
            for m in self.storage._get_candidates("group_111", top_k=200)
        }
        private_sources = {
            m.source_session
            for m in self.storage._get_candidates("private_222", top_k=200)
        }

        self.assertIn("group_333", group_sources)
        self.assertNotIn("private_222", group_sources)
        self.assertIn("group_111", private_sources)

    def test_time_decay_is_idempotent(self):
        memory = Memory(
            content="一条旧记忆",
            memory_type="episodic",
            importance=0.8,
            last_accessed=datetime.now() - timedelta(days=30),
        )
        self.storage.store(memory)

        self.storage.apply_time_decay(
            half_life_days=30, min_importance=0.0, max_age_days=999
        )
        first = self.storage._row_to_memory(
            self.storage.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory.id,)
            ).fetchone()
        ).importance
        self.storage.apply_time_decay(
            half_life_days=30, min_importance=0.0, max_age_days=999
        )
        second = self.storage._row_to_memory(
            self.storage.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory.id,)
            ).fetchone()
        ).importance

        self.assertAlmostEqual(first, 0.8, places=3)
        self.assertAlmostEqual(second, first, places=3)

    def test_unrelated_high_importance_memory_is_not_semantic_hit(self):
        self.storage.store(Memory(
            content="完全不相关的高重要性旧内容",
            memory_type="episodic",
            importance=0.99,
            source_session="group_111",
        ))
        results = self.storage.semantic_search(
            query="zzzxqv_unseen_query",
            session="group_111",
            limit=5,
        )
        self.assertEqual(results, [])

    def test_memory_page_filters_and_counts_before_pagination(self):
        for index in range(3):
            self.storage.store(Memory(
                content=f"分页测试消息 {index}",
                memory_type="episodic",
                importance=0.5 + index * 0.1,
                source_session="group_111",
            ))

        first, total = self.storage.get_memories_page(
            session="group_111",
            query="分页测试",
            memory_types=["episodic"],
            limit=2,
            offset=0,
        )
        second, second_total = self.storage.get_memories_page(
            session="group_111",
            query="分页测试",
            memory_types=["episodic"],
            limit=2,
            offset=2,
        )

        self.assertEqual(total, 3)
        self.assertEqual(second_total, 3)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)
        self.assertTrue({m.id for m in first}.isdisjoint({m.id for m in second}))
        self.assertTrue(all(m.source_session == "group_111" for m in first + second))

    def test_disabled_long_term_memory_skips_retrieval(self):
        from main import GroupChatBot

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.long_term_memory_enabled = False
        self.assertEqual(
            asyncio.run(bot._retrieve_memories("anything", "group_111")), []
        )

    def test_short_bot_ack_is_not_persisted(self):
        from main import GroupChatBot

        class _Storage:
            def __init__(self):
                self.stored = []

            async def find_similar(self, *args):
                return None

            async def store(self, memory):
                self.stored.append(memory)

        storage = _Storage()
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.long_term_memory_enabled = True
        bot.memory_storage = storage
        bot.config = {"qq": {"self_id": "bot"}}
        bot.personality = type("P", (), {"name": "爱丽丝"})()
        bot._memory_tasks = set()

        async def run():
            bot._store_bot_memory("group_111", "好呀")
            await asyncio.sleep(0)
            self.assertEqual(storage.stored, [])

            bot._store_bot_memory("group_111", "这个问题我会记住，之后继续接着聊")
            tasks = list(bot._memory_tasks)
            await asyncio.gather(*tasks)

        asyncio.run(run())
        self.assertEqual(len(storage.stored), 1)

    def test_recent_persistent_messages_restore_in_chronological_order(self):
        from main import GroupChatBot
        from modules.memory.context import ContextManager

        now = datetime.now()

        class _Storage:
            async def get_session_messages(self, session, limit=30, offset=0, before=None):
                return [
                    Memory(
                        content="小红：刚刚的新消息",
                        memory_type="episodic",
                        source_session=session,
                        created_at=now - timedelta(minutes=1),
                        metadata={"sender_id": "u2", "sender_name": "小红", "message_id": "m2"},
                    ),
                    Memory(
                        content="小明：前一条旧消息",
                        memory_type="episodic",
                        source_session=session,
                        created_at=now - timedelta(minutes=2),
                        metadata={"sender_id": "u1", "sender_name": "小明", "message_id": "m1"},
                    ),
                ]

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.long_term_memory_enabled = True
        bot.memory_storage = _Storage()
        bot.context_manager = ContextManager(max_messages=5, max_age_hours=2)
        bot.personality = type("P", (), {"name": "爱丽丝"})()

        asyncio.run(bot._restore_recent_context("group_111"))
        restored = bot.context_manager.get_window("group_111").get_recent(5)
        self.assertEqual([m.content for m in restored], ["前一条旧消息", "刚刚的新消息"])

    def test_global_retrieval_still_returns_everything(self):
        candidates = self.storage._get_candidates("", top_k=200)
        self.assertEqual(len(candidates), 3)


class AttentionIsolationTests(unittest.TestCase):
    def test_private_chats_have_separate_attention_buckets(self):
        mgr = AttentionManager()
        # 两个私聊用户分别来消息
        mgr.on_message_received("private_222", "u1")
        mgr.on_message_received("private_999", "u2")
        state_a = mgr.get_group_state("private_222")
        state_b = mgr.get_group_state("private_999")
        # 各自桶独立，互不影响
        self.assertNotEqual(id(state_a), id(state_b))
        self.assertEqual(len(mgr._group_states), 2)


if __name__ == "__main__":
    unittest.main()

"""会话隔离测试：私聊记忆不泄漏进群聊检索，各会话短时状态相互独立。"""

import os
import tempfile
import unittest
import asyncio
from datetime import datetime, timedelta

from modules.memory.storage import AsyncMemoryStorage, Memory, MemoryStorage
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

    def test_profile_materials_are_scoped_per_user_and_session(self):
        for index in range(3):
            self.storage.store(Memory(
                content=f"甲的画像素材 {index}",
                memory_type="episodic",
                source_session="group_profile_a",
                metadata={"sender_id": "u1", "sender_name": "甲"},
            ))
        self.storage.store(Memory(
            content="乙的画像素材",
            memory_type="episodic",
            source_session="group_profile_b",
            metadata={"sender_id": "u2", "sender_name": "乙"},
        ))
        self.storage.store(Memory(
            content="Bot：这是机器人自己的话",
            memory_type="episodic",
            source_session="group_profile_a",
            metadata={"sender_id": "bot", "sender_name": "爱丽丝", "is_bot": True},
        ))
        self.storage.store(Memory(
            content="甲：我也是",
            memory_type="episodic",
            source_session="group_profile_a",
            metadata={
                "sender_id": "u1",
                "sender_name": "甲",
                "profile_context_only": True,
            },
        ))

        scopes = set(self.storage.get_profile_scopes(limit=10))
        self.assertIn(("u1", "group_profile_a"), scopes)
        self.assertIn(("u2", "group_profile_b"), scopes)
        self.assertNotIn(("bot", "group_profile_a"), scopes)

        materials = self.storage.get_profile_materials(
            "u1", source_session="group_profile_a", limit=2
        )
        self.assertEqual(len(materials), 2)
        self.assertTrue(all(m.metadata.get("sender_id") == "u1" for m in materials))
        self.assertTrue(all(m.source_session == "group_profile_a" for m in materials))
        self.assertTrue(all(not m.metadata.get("profile_context_only") for m in materials))
        self.assertEqual(
            self.storage.get_profile_materials("bot", "group_profile_a"), []
        )

    def test_group_analysis_storage_is_isolated_and_report_is_idempotent(self):
        base = datetime.now()
        self.storage.store_group_analysis_message(Memory(
            content="今天开黑吗",
            memory_type="group_analysis",
            source_session="group_daily",
            created_at=base,
            metadata={
                "message_id": "daily-1",
                "sender_id": "u1",
                "sender_name": "甲",
            },
        ))
        self.storage.store_group_analysis_message(Memory(
            content="我晚上有空",
            memory_type="group_analysis",
            source_session="group_daily",
            created_at=base + timedelta(seconds=1),
            metadata={
                "message_id": "daily-2",
                "sender_id": "u2",
                "sender_name": "乙",
            },
        ))

        messages = self.storage.get_group_analysis_messages("group_daily")
        self.assertEqual([m.content for m in messages], ["今天开黑吗", "我晚上有空"])
        self.assertEqual(
            self.storage.get_group_analysis_sessions()[0]["message_count"], 2
        )
        self.assertEqual(
            self.storage.retrieve_session_recent("group_daily", limit=10), []
        )
        self.assertNotIn(
            "group_daily", {row["session"] for row in self.storage.list_sessions()}
        )

        report = Memory(
            content="第一版日报",
            memory_type="group_report",
            source_session="group_daily",
            metadata={"report_date": "2026-08-24", "days": 1},
        )
        self.storage.store_group_analysis_report(report)
        updated = Memory(
            content="更新版日报",
            memory_type="group_report",
            source_session="group_daily",
            metadata={"report_date": "2026-08-24", "days": 1},
        )
        self.storage.store_group_analysis_report(updated)
        self.assertEqual(
            len(self.storage.get_group_analysis_reports("group_daily")), 1
        )
        self.assertEqual(
            self.storage.get_group_analysis_report(
                "group_daily", "2026-08-24"
            ).content,
            "更新版日报",
        )

    def test_profile_context_returns_neighbors_and_reply_target(self):
        base = datetime.now()
        reply_target = Memory(
            content="乙：最近在学 Rust",
            memory_type="episodic",
            source_session="group_context",
            created_at=base - timedelta(minutes=10),
            metadata={
                "sender_id": "u2",
                "sender_name": "乙",
                "message_id": "reply-target",
            },
        )
        before = Memory(
            content="甲：你最近在学什么？",
            memory_type="episodic",
            source_session="group_context",
            created_at=base - timedelta(minutes=2),
            metadata={"sender_id": "u1", "sender_name": "甲"},
        )
        target = Memory(
            content="甲：我也是",
            memory_type="episodic",
            source_session="group_context",
            created_at=base - timedelta(minutes=1),
            metadata={
                "sender_id": "u1",
                "sender_name": "甲",
                "reply_to_id": "reply-target",
            },
        )
        after = Memory(
            content="丙：那还挺巧",
            memory_type="episodic",
            source_session="group_context",
            created_at=base,
            metadata={"sender_id": "u3", "sender_name": "丙"},
        )
        unrelated = Memory(
            content="其他会话不应混入",
            memory_type="episodic",
            source_session="group_other",
            created_at=base,
            metadata={"sender_id": "u9", "sender_name": "己"},
        )
        for memory in (reply_target, before, target, after, unrelated):
            self.storage.store(memory)

        context = self.storage.get_profile_context(target.id, before=1, after=1)
        self.assertEqual(
            [memory.content for memory in context],
            [reply_target.content, before.content, target.content, after.content],
        )
        self.assertNotIn(unrelated.content, [memory.content for memory in context])
        recent_context = self.storage.get_profile_context(
            target.id,
            before=1,
            after=1,
            since=base - timedelta(minutes=5),
        )
        self.assertNotIn(reply_target.content, [memory.content for memory in recent_context])

    def test_profile_pages_and_delete_suppression(self):
        now = datetime.now()
        profiles = []
        for index in range(3):
            profile = Memory(
                content=f"【用户画像 用户{index}】长期特征",
                memory_type="semantic",
                source_session="group_profile",
                created_at=now - timedelta(minutes=index),
                last_accessed=now - timedelta(minutes=index),
                metadata={
                    "profile": True,
                    "sender_id": f"profile-{index}",
                    "sender_name": f"用户{index}",
                },
            )
            self.storage.store(profile)
            profiles.append(profile)

        first, total = self.storage.get_profiles_page(limit=2, offset=0)
        second, second_total = self.storage.get_profiles_page(limit=2, offset=2)
        self.assertEqual(total, 3)
        self.assertEqual(second_total, 3)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1)

        filtered, filtered_total = self.storage.get_profiles_page(
            limit=10, offset=0, query="用户1"
        )
        self.assertEqual(filtered_total, 1)
        self.assertEqual([p.metadata["sender_id"] for p in filtered], ["profile-1"])

        old_material = Memory(
            content="旧画像素材",
            memory_type="episodic",
            source_session="group_profile",
            created_at=now - timedelta(minutes=1),
            metadata={"sender_id": "profile-0", "sender_name": "用户0"},
        )
        self.storage.store(old_material)
        self.assertTrue(self.storage.delete(profiles[0].id))
        suppressions = self.storage.get_profile_suppressions()
        suppression_time = suppressions[("profile-0", "group_profile")]
        self.assertEqual(
            self.storage.get_profile_materials(
                "profile-0", "group_profile", after=suppression_time
            ),
            [],
        )

        new_material = Memory(
            content="删除后新增的画像素材",
            memory_type="episodic",
            source_session="group_profile",
            created_at=suppression_time + timedelta(seconds=1),
            metadata={"sender_id": "profile-0", "sender_name": "用户0"},
        )
        self.storage.store(new_material)
        self.assertEqual(
            [m.content for m in self.storage.get_profile_materials(
                "profile-0", "group_profile", after=suppression_time
            )],
            ["删除后新增的画像素材"],
        )

    def test_profile_attempts_survive_round_trip(self):
        self.storage.set_profile_attempt("u1", "group_profile", 2, 8)
        self.assertEqual(
            self.storage.get_profile_attempts()[("u1", "group_profile")],
            (2, 8),
        )
        self.storage.clear_profile_attempt("u1", "group_profile")
        self.assertNotIn(("u1", "group_profile"), self.storage.get_profile_attempts())

    def test_async_profile_attempt_wrappers(self):
        async_storage = AsyncMemoryStorage(self.storage)

        async def run():
            await async_storage.set_profile_attempt("u2", "group_profile", 1, 3)
            attempts = await async_storage.get_profile_attempts()
            self.assertEqual(attempts[("u2", "group_profile")], (1, 3))
            await async_storage.clear_profile_attempt("u2", "group_profile")

        asyncio.run(run())
        self.assertNotIn(("u2", "group_profile"), self.storage.get_profile_attempts())

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

    def test_short_reply_is_persisted_as_context_only(self):
        from core.adapter.base import Message
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
        bot._memory_tasks = set()

        async def run():
            bot._store_long_term_memory(
                Message(
                    message_id="m2",
                    message_type="group",
                    sender_id="u1",
                    sender_name="甲",
                    content="我也是",
                    reply_to_id="m1",
                ),
                "group_111",
            )
            await asyncio.gather(*list(bot._memory_tasks))

        asyncio.run(run())
        self.assertEqual(len(storage.stored), 1)
        self.assertTrue(storage.stored[0].metadata["profile_context_only"])
        self.assertEqual(storage.stored[0].importance, 0.25)

    def test_recent_persistent_messages_restore_in_chronological_order(self):
        from main import GroupChatBot
        from modules.memory.context import ContextManager

        now = datetime.now()

        class _Storage:
            async def get_session_messages(self, session, limit=120, offset=0, before=None):
                return [
                    Memory(
                        content="小红：刚刚的新消息",
                        memory_type="episodic",
                        source_session=session,
                        created_at=now - timedelta(minutes=1),
                        metadata={"sender_id": "u2", "sender_name": "小红", "message_id": "m2",
                                  "meaningful": True},
                    ),
                    Memory(
                        content="小明：前一条旧消息",
                        memory_type="episodic",
                        source_session=session,
                        created_at=now - timedelta(minutes=2),
                        metadata={"sender_id": "u1", "sender_name": "小明", "message_id": "m1",
                                  "meaningful": True},
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

    def test_restore_skips_low_value_and_context_only_messages(self):
        """重启恢复窗口只取有实义内容的消息，跳过纯应声 / 仅上下文。"""
        from main import GroupChatBot
        from modules.memory.context import ContextManager

        now = datetime.now()

        class _Storage:
            async def get_session_messages(self, session, limit=120, offset=0, before=None):
                # 新→旧
                return [
                    Memory(
                        content="小红：嗯嗯",
                        memory_type="episodic",
                        source_session=session,
                        created_at=now - timedelta(minutes=1),
                        metadata={"sender_id": "u2", "sender_name": "小红",
                                  "message_id": "m5", "meaningful": False},
                    ),
                    Memory(
                        content="小明：我也是",
                        memory_type="episodic",
                        source_session=session,
                        created_at=now - timedelta(minutes=2),
                        metadata={"sender_id": "u1", "sender_name": "小明",
                                  "message_id": "m4", "meaningful": True,
                                  "profile_context_only": True},
                    ),
                    Memory(
                        content="小红：晚上开黑吗",
                        memory_type="episodic",
                        source_session=session,
                        created_at=now - timedelta(minutes=3),
                        metadata={"sender_id": "u2", "sender_name": "小红",
                                  "message_id": "m3", "meaningful": True},
                    ),
                ]

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.long_term_memory_enabled = True
        bot.memory_storage = _Storage()
        bot.context_manager = ContextManager(max_messages=5, max_age_hours=2)
        bot.personality = type("P", (), {"name": "爱丽丝"})()

        asyncio.run(bot._restore_recent_context("group_111"))
        restored = [
            m.content for m in bot.context_manager.get_window("group_111").get_recent(5)
        ]
        # 只有带 meaningful=True 且非仅上下文的 m3 被恢复
        self.assertEqual(restored, ["晚上开黑吗"])


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

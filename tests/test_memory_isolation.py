"""会话隔离测试：私聊记忆不泄漏进群聊检索，各会话短时状态相互独立。"""

import os
import tempfile
import unittest

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

    def test_group_retrieval_excludes_private_memories(self):
        candidates = self.storage._get_candidates("group_111", top_k=200)
        sources = {m.source_session for m in candidates}
        self.assertIn("group_111", sources)
        self.assertIn("group_333", sources)  # 跨群共享保持
        self.assertNotIn("private_222", sources)  # 私聊不外泄

    def test_private_retrieval_may_include_group_memories(self):
        candidates = self.storage._get_candidates("private_222", top_k=200)
        sources = {m.source_session for m in candidates}
        self.assertIn("private_222", sources)
        # 私聊仍可想起群聊记忆（更贴心），这是预期行为
        self.assertIn("group_111", sources)

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

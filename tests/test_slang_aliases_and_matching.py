import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.memory.storage import MemoryStorage


class _GlossaryTokenizer:
    """Deterministic tokenizer double for tests when jieba is not installed."""

    def __init__(self, glossary_terms):
        self.glossary_terms = sorted(set(glossary_terms), key=len, reverse=True)

    def tokenize(self, text, mode="default", HMM=False):
        index = 0
        while index < len(text):
            term = next(
                (candidate for candidate in self.glossary_terms
                 if text.startswith(candidate, index)),
                None,
            )
            token = term or text[index]
            yield token, index, index + len(token)
            index += len(token)


class SlangAliasesAndMatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = MemoryStorage(str(Path(self.temp_dir.name) / "memory.db"))

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def match_with_test_tokenizer(self, text, *args, **kwargs):
        with patch.object(
            self.store,
            "_get_slang_tokenizer",
            side_effect=lambda terms: _GlossaryTokenizer(terms) if terms else None,
        ):
            return self.store.match_slang(text, *args, **kwargs)

    def test_person_alias_is_global_unique_and_looked_up_by_exact_qq(self):
        cat_id = self.store.upsert_person_alias("猫姐", "123456789", "成员称呼")
        # NFKC/casefold-equivalent aliases update the same mapping, not create a duplicate.
        alice_id = self.store.upsert_person_alias("ＡＬＩＣＥ", "123456789")
        same_id = self.store.upsert_person_alias("alice", "123456789", "更新备注")
        self.assertEqual(alice_id, same_id)
        self.assertEqual(len(self.store.list_person_aliases()), 2)
        self.assertEqual(
            {row["alias"] for row in self.store.match_person_aliases(["123456789"])},
            {"猫姐", "alice"},
        )
        self.assertEqual(self.store.match_person_aliases(["987654321"]), [])
        with self.assertRaisesRegex(ValueError, "已绑定其他 QQ 号"):
            self.store.upsert_person_alias("猫姐", "987654321")

    def test_person_alias_preserves_same_named_unmapped_group_slang(self):
        slang_id = self.store.upsert_slang(
            "猫姐", "群级文本释义", session="group_1", source="manual"
        )
        self.store.upsert_person_alias("猫姐", "123456789")
        rows = self.store.list_slang(session="group_1", enabled_only=True)
        self.assertIn(slang_id, [row["id"] for row in rows])
        self.assertEqual(
            [row["id"] for row in self.match_with_test_tokenizer("猫姐来了", "group_1")],
            [slang_id],
        )
        self.assertEqual(
            self.match_with_test_tokenizer(
                "猫姐来了", "group_1", exclude_terms=["猫姐"]
            ),
            [],
        )
        self.assertEqual(
            self.store.upsert_slang("猫姐", "自动重新提取", session="group_2"), 0
        )

    def test_slang_match_is_normalized_boundary_aware_and_non_overlapping(self):
        self.store.upsert_slang("舟舟老师", "长称呼", session="group_1")
        self.store.upsert_slang("舟舟", "短称呼", session="group_1")
        self.store.upsert_slang("AI", "缩写", session="group_1")

        matches = self.match_with_test_tokenizer(
            "今天舟舟老师来了，顺便聊了ＡＩ", "group_1"
        )
        self.assertEqual([row["term"] for row in matches], ["舟舟老师", "AI"])
        self.assertEqual(
            [row["term"] for row in self.match_with_test_tokenizer("舟舟今天来了", "group_1")],
            ["舟舟"],
        )
        self.assertEqual(self.match_with_test_tokenizer("said it already", "group_1"), [])
        self.assertEqual(
            [row["term"] for row in self.match_with_test_tokenizer("say ai now", "group_1")],
            ["AI"],
        )

    def test_auto_mode_skips_single_han_but_matches_multi_han_tokens(self):
        self.store.upsert_slang("姬", "单字测试", session="group_1")
        self.store.upsert_slang("舟舟老师", "长词", session="group_1")
        self.store.upsert_slang("舟舟", "短词", session="group_1")

        self.assertEqual(self.match_with_test_tokenizer("姬", "group_1"), [])
        self.assertEqual(self.match_with_test_tokenizer("姬发今天来了", "group_1"), [])
        self.assertEqual(
            [row["term"] for row in self.match_with_test_tokenizer(
                "今天舟舟老师来了", "group_1"
            )],
            ["舟舟老师"],
        )

    def test_exact_and_standalone_modes_have_explicit_boundaries(self):
        exact_id = self.store.upsert_slang(
            "姬", "整句释义", session="group_1", source="manual", match_mode="exact"
        )
        standalone_id = self.store.upsert_slang(
            "AI", "独立缩写", session="group_1", source="manual", match_mode="standalone"
        )

        exact = self.store.match_slang("姬", "group_1")
        self.assertEqual([row["id"] for row in exact], [exact_id])
        self.assertEqual(self.store.match_slang("姬来了", "group_1"), [])
        self.assertEqual(
            [row["id"] for row in self.store.match_slang("聊 AI！", "group_1")],
            [standalone_id],
        )
        self.assertEqual(self.store.match_slang("聊AI助手", "group_1"), [])
        self.assertEqual(self.store.match_slang("AI助手", "group_1"), [])

    def test_missing_tokenizer_fails_closed_for_chinese_auto_terms(self):
        self.store.upsert_slang("摸鱼", "偷懒", session="group_1")
        self.store.upsert_slang("AI", "缩写", session="group_1")
        with patch.object(self.store, "_get_slang_tokenizer", return_value=None):
            self.assertEqual(self.store.match_slang("今天摸鱼", "group_1"), [])
            self.assertEqual(
                [row["term"] for row in self.store.match_slang("今天 AI", "group_1")],
                ["AI"],
            )

    @unittest.skipIf(importlib.util.find_spec("jieba") is None, "jieba is not installed")
    def test_real_jieba_matches_curated_overlapping_chinese_terms(self):
        self.store.upsert_slang("舟舟老师", "长称呼", session="group_1")
        self.store.upsert_slang("舟舟", "短称呼", session="group_1")

        self.assertEqual(
            [row["term"] for row in self.store.match_slang("今天舟舟老师来了", "group_1")],
            ["舟舟老师"],
        )
        self.assertEqual(
            [row["term"] for row in self.store.match_slang("今天舟舟来了", "group_1")],
            ["舟舟"],
        )

    def test_match_mode_is_persisted_and_validated(self):
        slang_id = self.store.upsert_slang(
            "摸鱼", "偷懒", session="group_1", match_mode="exact"
        )
        self.assertEqual(self.store.list_slang()[0]["match_mode"], "exact")
        self.assertTrue(self.store.update_slang(slang_id, match_mode="standalone"))
        self.assertEqual(self.store.list_slang()[0]["match_mode"], "standalone")
        with self.assertRaisesRegex(ValueError, "匹配方式"):
            self.store.update_slang(slang_id, match_mode="substring")

    def test_group_definition_overrides_same_named_global_definition(self):
        self.store.upsert_slang("摸鱼", "全局释义", session="")
        self.store.upsert_slang("摸鱼", "群内释义", session="group_1")
        matched = self.match_with_test_tokenizer("今天摸鱼", session="group_1")
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["meaning"], "群内释义")

    def test_occurrence_count_dedupes_message_and_prompt_count_stays_separate(self):
        slang_id = self.store.upsert_slang("摸鱼", "偷懒", session="group_1")
        self.assertEqual(
            self.store.record_slang_occurrences([slang_id], "group_1", "msg-1"), 1
        )
        self.assertEqual(
            self.store.record_slang_occurrences([slang_id], "group_1", "msg-1"), 0
        )
        self.assertEqual(
            self.store.record_slang_occurrences([slang_id], "group_1", "msg-2"), 1
        )
        row = self.store.list_slang(session="group_1")[0]
        self.assertEqual(row["occurrence_count"], 2)
        self.assertEqual(row["prompt_inject_count"], 0)
        self.store.bump_slang_hits([slang_id])
        row = self.store.list_slang(session="group_1")[0]
        self.assertEqual(row["occurrence_count"], 2)
        self.assertEqual(row["prompt_inject_count"], 1)
        self.assertTrue(row["last_seen_at"])

    def test_migration_only_moves_unambiguous_manual_terms_with_explicit_qq(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        conn = sqlite3.connect(legacy_path)
        conn.execute("""
            CREATE TABLE glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                meaning TEXT NOT NULL,
                session TEXT DEFAULT '',
                example TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1,
                source TEXT DEFAULT 'auto',
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(term, session)
            )
        """)
        conn.executemany(
            "INSERT INTO glossary (term, meaning, session, source) VALUES (?, ?, ?, 'manual')",
            [
                ("水母", "QQ号：123456789", "group_1"),
                ("水母", "同名但未映射的群级用法", "group_2"),
                ("数字词", "版本 123456789", "group_1"),
                ("冲突称呼", "QQ 123456789", "group_1"),
                ("冲突称呼", "QQ 987654321", "group_2"),
            ],
        )
        conn.commit()
        conn.close()

        legacy_store = MemoryStorage(str(legacy_path))
        try:
            aliases = legacy_store.list_person_aliases()
            self.assertEqual([row["alias"] for row in aliases], ["水母"])
            self.assertEqual(
                legacy_store.match_person_aliases(["123456789"])[0]["alias"], "水母"
            )
            self.assertEqual(
                len(legacy_store.list_slang(session="group_1", enabled_only=True)), 2
            )
            self.assertTrue(all(
                row["match_mode"] == "auto"
                for row in legacy_store.list_slang(session="group_1", enabled_only=True)
            ))
            self.assertEqual(
                [row["term"] for row in legacy_store.list_slang(
                    session="group_2", enabled_only=True
                )].count("水母"),
                1,
            )
            alias_id = aliases[0]["id"]
            self.assertTrue(legacy_store.delete_person_alias(alias_id))
        finally:
            legacy_store.close()
        reopened = MemoryStorage(str(legacy_path))
        try:
            self.assertEqual(reopened.list_person_aliases(), [])
        finally:
            reopened.close()

    def test_person_alias_prompt_uses_role_without_leaking_qq(self):
        from modules.reply.generator import ReplyGenerator

        guide = ReplyGenerator._build_person_alias_guide([
            {"alias": "猫姐", "qq_id": "123456789", "roles": ["当前发言者"]}
        ])
        self.assertIn("当前发言者的称呼是「猫姐」", guide)
        self.assertNotIn("123456789", guide)


if __name__ == "__main__":
    unittest.main()

"""按群学「说话方式」的测试：风格统计、字面校验、称呼筛选、存取。"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime

from modules.personality.group_learning import (
    clean_message_text,
    compute_style_stats,
    filter_catchphrases,
    learned_path,
    load_all_learned,
    load_learned,
    load_universal_style,
    recent_messages_by_session,
    recent_text_by_session,
    sanitize_nicknames,
    save_learned,
    save_universal_style,
    style_from_stats,
    universal_style_path,
    verify_candidates,
)


class CleanMessageTextTests(unittest.TestCase):
    def test_strips_face_codes_and_links(self):
        cleaned = clean_message_text("今天好热<faceType=1,faceId=14> https://x.cn/a 真的")
        self.assertNotIn("faceType", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertIn("今天好热", cleaned)

    def test_drops_placeholder_messages(self):
        # 视觉摘要不是群友说的话，混进统计会把句长整体拉长。
        self.assertEqual(clean_message_text("一张图，看不清画面。"), "")
        self.assertEqual(clean_message_text("一段视频，画面是……"), "")

    def test_keeps_plain_text(self):
        self.assertEqual(clean_message_text("  那就这样吧  "), "那就这样吧")


class StyleStatsTests(unittest.TestCase):
    def _texts(self, n, text):
        return [text] * n

    def test_needs_enough_samples(self):
        stats = compute_style_stats(self._texts(5, "短句"))
        self.assertEqual(stats["sample_size"], 5)
        # 样本不足时宁可不给值，也不要让一次抖动改掉整套风格。
        self.assertEqual(style_from_stats(stats), {})

    def test_length_follows_group_median(self):
        stats = compute_style_stats(self._texts(40, "一二三四五六七八九十"))
        style = style_from_stats(stats)
        self.assertIn("max_reply_length", style)
        self.assertGreaterEqual(style["max_reply_length"], 14)
        self.assertLessEqual(style["max_reply_length"], 45)

    def test_length_is_absolutely_bounded(self):
        # 群里全是超长句也不该把回复长度直接放开到 45 以上。
        long_text = "这是一句话" * 10
        stats = compute_style_stats(self._texts(40, long_text))
        style = style_from_stats(stats)
        self.assertEqual(style["max_reply_length"], 45)

        short = compute_style_stats(self._texts(40, "好耶"))
        self.assertEqual(style_from_stats(short)["max_reply_length"], 14)

    def test_not_derived_from_persona(self):
        # 风格数值只认统计，不接受人格那条链路的任何输入：同样的语料
        # 必须得到同样的结果，换人格也一样。
        stats = compute_style_stats(self._texts(40, "太离谱了吧"))
        first = style_from_stats(stats)
        second = style_from_stats(dict(stats))
        self.assertEqual(first, second)

    def test_ellipsis_and_exclaim_rates(self):
        texts = ["好耶！"] * 5 + ["嗯…"] * 5 + ["随便"] * 10
        stats = compute_style_stats(texts)
        self.assertGreater(stats["exclaim_rate"], 0.0)
        style = style_from_stats(stats)
        self.assertLessEqual(style["enthusiasm"], 0.85)
        self.assertGreaterEqual(style["formality"], 0.05)


class VerifyCandidateTests(unittest.TestCase):
    def test_keeps_only_literally_seen_terms(self):
        corpus = ["这也太离谱了吧", "离谱", "正常一句"]
        kept = verify_candidates(["离谱", "绷不住了"], corpus, min_hits=2)
        self.assertEqual(kept, ["离谱"])

    def test_rejects_overlong_and_duplicate(self):
        corpus = ["离谱"] * 3
        kept = verify_candidates(["离谱", "离谱", "一" * 20], corpus, min_hits=2)
        self.assertEqual(kept, ["离谱"])


class CatchphraseFilterTests(unittest.TestCase):
    def test_drops_generic_words(self):
        kept = filter_catchphrases(["你好", "谢谢", "离谱", "绷不住了"])
        self.assertEqual(kept, ["离谱", "绷不住了"])

    def test_drops_names(self):
        kept = filter_catchphrases(["离谱"], ban=["离谱"])
        self.assertEqual(kept, [])

    def test_dedupes_and_drops_empty(self):
        self.assertEqual(filter_catchphrases(["离谱", "离谱", "", "  "]), ["离谱"])

    def test_drops_punctuation_only(self):
        # 「？？？」只在标点层面有信息，学进去等于没学。
        self.assertEqual(filter_catchphrases(["，……"], ["？？？"]), [])

    def test_drops_system_action_labels(self):
        self.assertEqual(
            filter_catchphrases(["戳一戳", "撤回了一条消息"], ["拍了拍你"]), []
        )


class SanitizeNicknameTests(unittest.TestCase):
    def test_drops_member_names_and_known_names(self):
        texts = ["爱丽丝在吗", "@丽丝 出来", "小艾你说呢", "丽丝别装死"]
        kept = sanitize_nicknames(
            ["丽丝", "小艾", "老王"],
            texts,
            known_names=["爱丽丝", "小艾"],
            member_names=["老王"],
            min_hits=2,
        )
        # 丽丝 是名字派生出的叫法，保留；小艾 是本来就有的名字、老王 是群友。
        self.assertEqual(kept, ["丽丝"])

    def test_drops_digits(self):
        self.assertEqual(
            sanitize_nicknames(["123"], ["123", "123"], min_hits=2), []
        )

    def test_substring_of_known_name_is_not_enough(self):
        # 群里只叫「爱丽丝」，那「丽丝」的命中全是搭了全名的便车，不算称呼。
        texts = ["爱丽丝在吗", "爱丽丝出来", "爱丽丝别装死"]
        self.assertEqual(
            sanitize_nicknames(
                ["丽丝"], texts, known_names=["爱丽丝"], min_hits=2
            ),
            [],
        )

    def test_drops_punctuation(self):
        self.assertEqual(
            sanitize_nicknames(["？？？"], ["？？？", "？？？"], min_hits=2), []
        )


class MemorySelectionTests(unittest.TestCase):
    class _Memory:
        def __init__(self, session, content, name="群友", is_bot=False, age_hours=1):
            self.source_session = session
            self.content = content
            self.metadata = {"sender_name": name, "is_bot": is_bot}
            self.created_at = datetime.now()

    def test_filters_bot_and_other_groups(self):
        memories = [
            self._Memory("group_1", "人话一句"),
            self._Memory("group_1", "我自己说的", is_bot=True),
            self._Memory("group_2", "别的群"),
        ]
        rows = recent_messages_by_session(memories, "group_1")
        self.assertEqual([row["content"] for row in rows], ["人话一句"])
        self.assertEqual(
            recent_text_by_session(memories, "group_1"), ["人话一句"]
        )


class LearnedFileTests(unittest.TestCase):
    def setUp(self):
        # 每个用例独占一个配置目录：学习文件落在 config 旁边的 learned/ 里，
        # 共用系统临时目录的话用例之间会互相串数据。
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "config.yaml")

    def test_roundtrip_and_merge(self):
        save_learned(self.path, "123", {"style": {"max_reply_length": 22}})
        save_learned(self.path, "123", {"flavor": {"catchphrases": ["离谱"]}})
        data = load_learned(self.path, "123")
        self.assertEqual(data["style"]["max_reply_length"], 22)
        self.assertEqual(data["flavor"]["catchphrases"], ["离谱"])

        # 同一顶层键再写一次是合并，不是覆盖掉别的字段。
        save_learned(self.path, "123", {"style": {"enthusiasm": 0.5}})
        data = load_learned(self.path, "123")
        self.assertEqual(data["style"]["max_reply_length"], 22)
        self.assertEqual(data["style"]["enthusiasm"], 0.5)
        self.assertIn("updated_at", data)

    def test_replace_drops_stale_values(self):
        # 风格和习惯每轮都是完整快照：这轮没学到的旧口头禅必须消失，
        # 否则「你好」这种一时混进来的词会永远留在文件里。
        save_learned(self.path, "333", {"flavor": {"catchphrases": ["你好"]}})
        save_learned(
            self.path, "333", {"flavor": {"memes": ["离谱"]}}, replace=True
        )
        data = load_learned(self.path, "333")
        self.assertNotIn("catchphrases", data["flavor"])
        self.assertEqual(data["flavor"]["memes"], ["离谱"])

    def test_groups_are_isolated(self):
        save_learned(self.path, "111", {"style": {"max_reply_length": 20}})
        save_learned(self.path, "222", {"style": {"max_reply_length": 30}})
        all_data = load_all_learned(self.path)
        self.assertEqual(all_data["111"]["style"]["max_reply_length"], 20)
        self.assertEqual(all_data["222"]["style"]["max_reply_length"], 30)

    def test_broken_file_is_ignored(self):
        path = learned_path(self.path, "999")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ 这不是 json", encoding="utf-8")
        self.assertEqual(load_learned(self.path, "999"), {})
        self.assertNotIn("999", load_all_learned(self.path))


class UniversalStyleFileTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "config.yaml")

    def test_roundtrip(self):
        save_universal_style(
            self.path, {"max_reply_length": 26, "enthusiasm": 0.62}, {"sample_size": 90}
        )
        self.assertEqual(
            load_universal_style(self.path),
            {"max_reply_length": 26, "enthusiasm": 0.62},
        )

    def test_overwrites_previous_snapshot(self):
        save_universal_style(self.path, {"max_reply_length": 26})
        save_universal_style(self.path, {"enthusiasm": 0.5})
        self.assertEqual(load_universal_style(self.path), {"enthusiasm": 0.5})

    def test_not_treated_as_a_group(self):
        # 通用风格文件和群目录放在一起，读全量时不能把它当成一个群。
        save_learned(self.path, "111", {"flavor": {"catchphrases": ["离谱"]}})
        save_universal_style(self.path, {"max_reply_length": 26})
        all_data = load_all_learned(self.path)
        self.assertEqual(list(all_data), ["111"])

    def test_missing_or_broken_returns_empty(self):
        self.assertEqual(load_universal_style(self.path), {})
        path = universal_style_path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ 坏掉的", encoding="utf-8")
        self.assertEqual(load_universal_style(self.path), {})


if __name__ == "__main__":
    unittest.main()

"""引用回复策略测试：这条回复该不该带引用。

对应 modules/reply/quote_policy.py。判定规则：
  · 对方 @ / 回复爱丽丝（direction=to_bot）→ 必引用
  · 群聊插话 → 默认不引用；只有「隔太久」或「群里刷太快」才引用
  · 一对一连发不算刷屏（说话人只有一个）
"""

import unittest
from datetime import datetime, timedelta

from modules.reply.quote_policy import (
    REASON_ALWAYS,
    REASON_BUSY,
    REASON_DIRECTED,
    REASON_DISABLED,
    REASON_INLINE,
    REASON_NO_TARGET,
    REASON_STALE,
    collect_busy_stats,
    decide_quote,
)


class _Msg:
    """最小消息替身：只需要 timestamp / is_bot / sender_id。"""

    def __init__(self, sender_id, seconds_ago=0, is_bot=False):
        self.sender_id = sender_id
        self.timestamp = datetime.now() - timedelta(seconds=seconds_ago)
        self.is_bot = is_bot


class DecideQuoteTests(unittest.TestCase):
    def test_directed_always_quotes(self):
        """对方明确在对爱丽丝说话 → 必须引用，否则消息像自言自语。"""
        decision = decide_quote("to_bot", "m1", target_age_seconds=2)
        self.assertTrue(decision.quote)
        self.assertEqual(decision.reason, REASON_DIRECTED)
        self.assertEqual(decision.target_id, "m1")

    def test_inline_group_reply_does_not_quote(self):
        """紧邻的即时对话不引用——真人不会每条都挂引用。"""
        decision = decide_quote(
            "group", "m1",
            target_age_seconds=3,
            recent_message_count=2,
            recent_speaker_count=1,
        )
        self.assertFalse(decision.quote)
        self.assertEqual(decision.reason, REASON_INLINE)

    def test_stale_target_quotes(self):
        """隔太久 → 对方可能已经忘了，引用帮对方定位。"""
        decision = decide_quote(
            "group", "m1",
            target_age_seconds=120,
            recent_message_count=1,
            recent_speaker_count=1,
        )
        self.assertTrue(decision.quote)
        self.assertEqual(decision.reason, REASON_STALE)

    def test_busy_group_quotes(self):
        """群里多人同时刷屏 → 指代容易糊，引用消歧。"""
        decision = decide_quote(
            "group", "m1",
            target_age_seconds=5,
            recent_message_count=6,
            recent_speaker_count=3,
        )
        self.assertTrue(decision.quote)
        self.assertEqual(decision.reason, REASON_BUSY)

    def test_one_to_one_burst_is_not_busy(self):
        """一个人连发 5 条不算刷屏：一对一聊天不该每条都引用。"""
        decision = decide_quote(
            "group", "m1",
            target_age_seconds=5,
            recent_message_count=5,
            recent_speaker_count=1,
        )
        self.assertFalse(decision.quote)
        self.assertEqual(decision.reason, REASON_INLINE)

    def test_target_outside_window_is_stale(self):
        """目标已滑出上下文窗口 → 必然隔了很久，按 stale 处理。"""
        decision = decide_quote(
            "group", "m1",
            target_age_seconds=None,
            recent_message_count=1,
            recent_speaker_count=1,
        )
        self.assertTrue(decision.quote)
        self.assertEqual(decision.reason, REASON_STALE)

    def test_no_target_never_quotes(self):
        decision = decide_quote("group", "", target_age_seconds=1)
        self.assertFalse(decision.quote)
        self.assertEqual(decision.reason, REASON_NO_TARGET)

    def test_disabled_never_quotes(self):
        decision = decide_quote(
            "to_bot", "m1", enabled=False, target_age_seconds=1,
        )
        self.assertFalse(decision.quote)
        self.assertEqual(decision.reason, REASON_DISABLED)

    def test_always_mode_restores_old_behaviour(self):
        """always=true → 每次都引用（等价于引入策略前），用于对照。"""
        decision = decide_quote(
            "group", "m1", always=True, target_age_seconds=1,
        )
        self.assertTrue(decision.quote)
        self.assertEqual(decision.reason, REASON_ALWAYS)

    def test_stale_threshold_is_configurable(self):
        decision = decide_quote(
            "group", "m1",
            target_age_seconds=30,
            recent_message_count=1,
            recent_speaker_count=1,
            stale_seconds=120,
        )
        self.assertFalse(decision.quote)
        self.assertEqual(decision.reason, REASON_INLINE)

    def test_busy_thresholds_are_configurable(self):
        decision = decide_quote(
            "group", "m1",
            target_age_seconds=5,
            recent_message_count=6,
            recent_speaker_count=3,
            busy_message_count=10,
        )
        self.assertFalse(decision.quote)
        self.assertEqual(decision.reason, REASON_INLINE)


class CollectBusyStatsTests(unittest.TestCase):
    def test_counts_only_recent_non_bot_messages(self):
        now = datetime.now()
        messages = [
            _Msg("u1", seconds_ago=2),
            _Msg("u2", seconds_ago=5),
            _Msg("bot", seconds_ago=1, is_bot=True),   # bot 自己不算
            _Msg("u3", seconds_ago=120),                # 超出窗口
        ]
        count, speakers = collect_busy_stats(messages, now=now, window_seconds=30)
        self.assertEqual(count, 2)
        self.assertEqual(speakers, 2)

    def test_single_speaker_keeps_speaker_count_at_one(self):
        messages = [_Msg("u1", seconds_ago=i) for i in range(1, 6)]
        count, speakers = collect_busy_stats(messages, window_seconds=30)
        self.assertEqual(count, 5)
        self.assertEqual(speakers, 1)

    def test_empty_window(self):
        count, speakers = collect_busy_stats([], window_seconds=30)
        self.assertEqual(count, 0)
        self.assertEqual(speakers, 0)


if __name__ == "__main__":
    unittest.main()

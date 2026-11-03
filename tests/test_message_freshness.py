"""消息新鲜度门槛与群日报时间窗口的离线测试。

背景：QQ 掉线恢复后 NapCat 会把离线期间的消息一次性倒灌过来。逐条处理
既没有意义，也会造成短时间内密集收发（QQ 风控盯上的形态）。
"""

import time
import unittest
from datetime import datetime, timedelta

from core.adapter.base import Message
from main import GroupChatBot


def _bot(**freshness) -> GroupChatBot:
    bot = GroupChatBot.__new__(GroupChatBot)
    bot.config = {"qq": {"self_id": "2364385326"}}
    bot._freshness_config = {
        "enabled": True,
        "group_max_age_seconds": 300,
        "directed_max_age_seconds": 1800,
        "private_max_age_seconds": 3600,
    }
    bot._freshness_config.update(freshness)
    bot._stale_backlog_count = 0
    return bot


def _message(**kwargs) -> Message:
    values = {
        "message_id": "m1",
        "message_type": "group",
        "sender_id": "u2",
        "sender_name": "小红",
        "group_id": "g1",
        "content": "hello",
    }
    values.update(kwargs)
    return Message(**values)


class MessageFreshnessTests(unittest.TestCase):
    def test_fresh_group_message_is_kept(self):
        bot = _bot()
        self.assertFalse(bot._is_stale_backlog(_message(timestamp=time.time() - 5)))

    def test_old_group_message_is_dropped(self):
        bot = _bot()
        self.assertTrue(bot._is_stale_backlog(_message(timestamp=time.time() - 3600)))
        self.assertEqual(bot._stale_backlog_count, 1)

    def test_directed_message_has_wider_window(self):
        bot = _bot()
        # 10 分钟前明确 @ 机器人：普通群消息早过期了，定向消息还在窗口内
        self.assertFalse(
            bot._is_stale_backlog(
                _message(timestamp=time.time() - 600, mentioned_me=True)
            )
        )
        # 40 分钟前 @ 机器人：超出 30 分钟定向窗口
        self.assertTrue(
            bot._is_stale_backlog(
                _message(timestamp=time.time() - 2400, mentioned_me=True)
            )
        )

    def test_reply_to_bot_counts_as_directed(self):
        bot = _bot()
        self.assertFalse(
            bot._is_stale_backlog(
                _message(timestamp=time.time() - 600, reply_to_qq="2364385326")
            )
        )

    def test_private_message_uses_private_window(self):
        bot = _bot()
        self.assertFalse(
            bot._is_stale_backlog(
                _message(message_type="private", timestamp=time.time() - 600)
            )
        )
        self.assertTrue(
            bot._is_stale_backlog(
                _message(message_type="private", timestamp=time.time() - 7200)
            )
        )

    def test_message_without_timestamp_is_never_dropped(self):
        bot = _bot()
        self.assertFalse(bot._is_stale_backlog(_message(timestamp=None)))

    def test_future_timestamp_is_not_dropped(self):
        bot = _bot()
        self.assertFalse(
            bot._is_stale_backlog(_message(timestamp=time.time() + 30))
        )

    def test_disabled_config_keeps_everything(self):
        bot = _bot(enabled=False)
        self.assertFalse(
            bot._is_stale_backlog(_message(timestamp=time.time() - 86400))
        )


class AnalysisTimeWindowTests(unittest.TestCase):
    def test_fixed_time_still_matches(self):
        now = datetime(2026, 11, 3, 23, 50, 0)
        self.assertTrue(GroupChatBot._analysis_time_matches(now, "23:50"))
        self.assertFalse(GroupChatBot._analysis_time_matches(now, "12:00"))

    def test_window_matches_at_its_daily_target(self):
        """窗口语法：当天选出一个触发点，并在其后 6 分钟内持续匹配。"""
        day = datetime(2026, 11, 3)
        hits = [
            probe
            for probe in (
                day.replace(hour=22, minute=30) + timedelta(minutes=minute)
                for minute in range(0, 60 * 12)
            )
            if GroupChatBot._analysis_time_matches(probe, "22:30-23:30")
        ]
        self.assertTrue(hits)
        first = hits[0]
        self.assertTrue(
            GroupChatBot._analysis_time_matches(
                first + timedelta(minutes=6), "22:30-23:30"
            )
        )

    def test_window_fires_only_once_per_day(self):
        """同一天内只应命中一个很短的触发窗口，而不是整个时间窗。"""
        day = datetime(2026, 11, 3)
        hits = [
            probe
            for probe in (
                day.replace(hour=22, minute=30) + timedelta(minutes=minute)
                for minute in range(0, 60 * 12)
            )
            if GroupChatBot._analysis_time_matches(probe, "22:30-23:30")
        ]
        self.assertGreaterEqual(len(hits), 1)
        self.assertLessEqual(len(hits), 7)

    def test_window_target_differs_across_days(self):
        """不同日期的触发时刻应当不同，打散「每天固定同一时刻」的机器人特征。"""
        first_hits = {}
        for offset in range(7):
            day = datetime(2026, 11, 3) + timedelta(days=offset)
            hits = [
                probe
                for probe in (
                    day.replace(hour=22, minute=30) + timedelta(minutes=minute)
                    for minute in range(0, 60 * 12)
                )
                if GroupChatBot._analysis_time_matches(probe, "22:30-23:30")
            ]
            first_hits[day.date()] = hits[0] if hits else None
        self.assertGreaterEqual(len(set(first_hits.values())), 5)

    def test_invalid_value_never_matches(self):
        now = datetime(2026, 11, 3, 23, 50, 0)
        self.assertFalse(GroupChatBot._analysis_time_matches(now, "bad"))
        self.assertFalse(GroupChatBot._analysis_time_matches(now, "25:00"))
        self.assertFalse(GroupChatBot._analysis_time_matches(now, "22:xx-23:30"))


if __name__ == "__main__":
    unittest.main()

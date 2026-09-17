"""普通群聊目标判断批处理测试。"""

import unittest
from unittest.mock import patch

from main import GroupChatBot


class ConversationJudgeBatchingTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self) -> GroupChatBot:
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.config = {
            "conversation_judge": {
                "batch_window_seconds": 0.35,
                "max_batch_wait_seconds": 1.5,
            }
        }
        bot._session_message_versions = {"group_g1": 1}
        bot._conversation_judge_batch_started_at = {}
        return bot

    async def test_superseded_message_is_skipped_and_latest_message_flushes_batch(self):
        bot = self._bot()
        delays = []

        async def newer_message_arrives(delay):
            delays.append(delay)
            bot._session_message_versions["group_g1"] = 2

        with patch("main.asyncio.sleep", new=newer_message_arrives):
            first_ready = await bot._wait_for_conversation_judge_batch(
                "group_g1", 1
            )

        self.assertFalse(first_ready)
        self.assertAlmostEqual(delays[0], 0.35)
        self.assertIn("group_g1", bot._conversation_judge_batch_started_at)

        async def quiet_window(_delay):
            return None

        with patch("main.asyncio.sleep", new=quiet_window):
            latest_ready = await bot._wait_for_conversation_judge_batch(
                "group_g1", 2
            )

        self.assertTrue(latest_ready)
        self.assertNotIn("group_g1", bot._conversation_judge_batch_started_at)

    async def test_batch_wait_is_capped_by_maximum_wait(self):
        bot = self._bot()
        bot._conversation_judge_batch_started_at["group_g1"] = 10.0
        delays = []

        async def record_sleep(delay):
            delays.append(delay)

        with (
            patch("main.time.monotonic", return_value=11.4),
            patch("main.asyncio.sleep", new=record_sleep),
        ):
            ready = await bot._wait_for_conversation_judge_batch("group_g1", 1)

        self.assertTrue(ready)
        self.assertAlmostEqual(delays[0], 0.1)
        self.assertNotIn("group_g1", bot._conversation_judge_batch_started_at)

    async def test_zero_batch_window_disables_waiting(self):
        bot = self._bot()
        bot.config["conversation_judge"]["batch_window_seconds"] = 0
        delays = []

        async def record_sleep(delay):
            delays.append(delay)

        with patch("main.asyncio.sleep", new=record_sleep):
            ready = await bot._wait_for_conversation_judge_batch("group_g1", 1)

        self.assertTrue(ready)
        self.assertEqual(delays, [])


if __name__ == "__main__":
    unittest.main()

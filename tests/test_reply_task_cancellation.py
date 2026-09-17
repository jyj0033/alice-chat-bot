"""新群消息到达时撤销尚未发送的普通插话。"""

import asyncio
import unittest

from core.adapter.base import Message
from main import GroupChatBot


class ReplyTaskCancellationTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self) -> GroupChatBot:
        bot = GroupChatBot.__new__(GroupChatBot)
        bot._reply_tasks = {}
        bot._reply_task_decisions = {}
        return bot

    def _message(self, *, message_type: str = "group") -> Message:
        return Message(
            message_id="m2",
            message_type=message_type,
            sender_id="u2",
            sender_name="小红",
            group_id="g1" if message_type == "group" else None,
            content="独裁……",
        )

    async def _pending_task(self) -> asyncio.Task:
        async def wait_forever():
            await asyncio.Event().wait()

        task = asyncio.create_task(wait_forever())
        await asyncio.sleep(0)
        return task

    async def test_new_group_message_cancels_pending_ordinary_interjection(self):
        bot = self._bot()
        pending = await self._pending_task()
        bot._reply_tasks["group_g1"] = pending
        bot._reply_task_decisions["group_g1"] = {"explicit_directed": False}

        cancelled = bot._cancel_pending_group_interjection(self._message())

        self.assertTrue(cancelled)
        with self.assertRaises(asyncio.CancelledError):
            await pending

    async def test_new_group_message_keeps_explicitly_directed_reply(self):
        bot = self._bot()
        pending = await self._pending_task()
        bot._reply_tasks["group_g1"] = pending
        bot._reply_task_decisions["group_g1"] = {"explicit_directed": True}

        cancelled = bot._cancel_pending_group_interjection(self._message())

        self.assertFalse(cancelled)
        self.assertFalse(pending.cancelling())
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

    async def test_new_group_message_does_not_cut_off_reply_already_being_sent(self):
        bot = self._bot()
        pending = await self._pending_task()
        bot._reply_tasks["group_g1"] = pending
        bot._reply_task_decisions["group_g1"] = {
            "explicit_directed": False,
            "sending_started": True,
        }

        cancelled = bot._cancel_pending_group_interjection(self._message())

        self.assertFalse(cancelled)
        self.assertFalse(pending.cancelling())
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

    async def test_private_message_does_not_cancel_group_interjection(self):
        bot = self._bot()
        pending = await self._pending_task()
        bot._reply_tasks["group_g1"] = pending
        bot._reply_task_decisions["group_g1"] = {"explicit_directed": False}

        cancelled = bot._cancel_pending_group_interjection(
            self._message(message_type="private")
        )

        self.assertFalse(cancelled)
        self.assertFalse(pending.cancelling())
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending


if __name__ == "__main__":
    unittest.main()

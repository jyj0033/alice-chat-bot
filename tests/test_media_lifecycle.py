"""媒体证据、取消边界和关闭流程的离线回归测试。"""

import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from core.adapter.base import Message
from core.adapter.qq_adapter import QQAdapter
from main import GroupChatBot, _MessageFlowState
from modules.memory.context import ContextManager
from modules.social.conversation_judge import ConversationJudgeResult


class MediaLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self):
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.config = {}
        bot.personality = SimpleNamespace(name="爱丽丝", nickname="小艾")
        bot.meme_manager = None
        bot.context_manager = ContextManager()
        for name in (
            "_reply_tasks", "_reply_task_decisions", "_message_ingest_locks",
            "_conversation_judge_locks", "_session_message_versions",
            "_conversation_judge_batch_started_at", "_digest_tasks", "_group_analysis_tasks",
            "_media_tasks_by_message",
        ):
            setattr(bot, name, {})
        for name in ("_rich_media_tasks", "_meme_collect_tasks", "_memory_tasks"):
            setattr(bot, name, set())
        bot._tasks = []
        bot._start_time = time.time()
        bot._build_image_conversation_context = Mock(return_value="")
        bot._collect_group_image_urls = Mock(return_value=[])
        return bot

    def _message(self, **overrides):
        values = dict(
            message_id="image", message_type="group", group_id="g1",
            sender_id="u1", sender_name="小明", content="一张图，看不清画面。",
            rich_type="image", rich_only=True,
        )
        values.update(overrides)
        return Message(**values)

    def _record(self, bot, message):
        bot.context_manager.add_message(
            message.session_id, message.sender_id, message.sender_name, message.content,
            message_id=message.message_id,
        )

    def _state(self, message, **overrides):
        values = dict(
            session_id=message.session_id, is_reply_to_bot=False, continuing=False,
            rich_trigger={}, rich_reasons=[], rich_directed=False,
            recent_context_for_judgement=[], message_version=1,
        )
        values.update(overrides)
        return _MessageFlowState(**values)

    async def test_cancelling_reply_does_not_cancel_shared_image_enrichment(self):
        bot = self._bot()
        image = self._message()
        self._record(bot, image)
        entered, release = asyncio.Event(), asyncio.Event()
        description = "一张图，画面是鸣潮角色菲比的技能界面。"

        async def enrich(message, **kwargs):
            entered.set()
            await release.wait()
            message.content = description

        bot.qq_adapter = SimpleNamespace(enrich_message=enrich)
        enrichment = asyncio.create_task(bot._enrich_context_message(image, False))
        decision = dict(
            direction="group", probability=0.5, emotional_state=None,
            context=SimpleNamespace(), explicit_directed=False, enrichment_task=enrichment,
        )
        reply = asyncio.create_task(bot._compose_and_send(image, decision))
        bot._reply_tasks[image.session_id] = reply
        bot._reply_task_decisions[image.session_id] = decision
        try:
            await entered.wait()
            await asyncio.sleep(0)
            self.assertTrue(bot._cancel_pending_group_interjection(self._message(message_id="next")))
            with self.assertRaises(asyncio.CancelledError):
                await reply
            self.assertFalse(enrichment.cancelled())
            release.set()
            await enrichment
            self.assertEqual(bot.context_manager.get_window(image.session_id).get_recent(1)[0].content, description)
        finally:
            release.set()
            for task in (reply, enrichment):
                if not task.done():
                    task.cancel()
            await asyncio.gather(reply, enrichment, return_exceptions=True)

    async def test_target_judgement_receives_completed_image_description(self):
        bot = self._bot()
        image = self._message(mentioned_me=True)
        self._record(bot, image)
        bot._session_message_versions[image.session_id] = 1
        description = "一张图，画面是鸣潮角色菲比。"

        async def enrich(message, **kwargs):
            await asyncio.sleep(0)
            message.content = description

        # 用固定判断替身检查证据到达顺序，而不是测试真实模型质量。
        bot.qq_adapter = SimpleNamespace(enrich_message=enrich)
        bot.conversation_judge = SimpleNamespace(enabled=True, provider=object())

        async def checked_judge(message, **kwargs):
            self.assertEqual(message.content, description)
            self.assertEqual(kwargs["recent_messages"][-1].content, description)
            return ConversationJudgeResult.unavailable("offline")

        bot._judge_conversation_message = AsyncMock(side_effect=checked_judge)
        bot._decide_reply = Mock(return_value=None)
        await bot._decide_incoming_message(image, self._state(image))
        bot._judge_conversation_message.assert_awaited_once()
        self.assertFalse(bot._media_tasks_by_message)

    async def test_followup_waits_for_same_sender_pending_image(self):
        bot = self._bot()
        image = self._message()
        self._record(bot, image)
        entered, release = asyncio.Event(), asyncio.Event()

        async def enrich(message, **kwargs):
            entered.set()
            await release.wait()
            message.content = "一张图，画面是菲比的技能。"

        bot.qq_adapter = SimpleNamespace(enrich_message=enrich)
        image_waiter = asyncio.create_task(bot._prepare_media_for_judgement(image, self._state(image)))
        followup_waiter = None
        try:
            await entered.wait()
            followup = self._message(message_id="next", content="这个技能怎么用？", rich_type="", rich_only=False)
            self._record(bot, followup)
            followup_waiter = asyncio.create_task(bot._prepare_media_for_judgement(followup, self._state(followup)))
            await asyncio.sleep(0)
            self.assertFalse(followup_waiter.done())
            release.set()
            await asyncio.gather(image_waiter, followup_waiter)
            self.assertIn("菲比", bot.context_manager.get_window(image.session_id).get_recent(2)[0].content)
        finally:
            release.set()
            await asyncio.gather(*(task for task in (image_waiter, followup_waiter) if task), return_exceptions=True)

    async def test_explicit_reply_waits_for_older_pending_image(self):
        """明确引用的图即使已滑出最近 6 条，只要还在识别就必须等证据。"""
        bot = self._bot()
        image = self._message(message_id="old-image")
        self._record(bot, image)
        # 识别期间群里又来了一串闲聊，把那张图挤出「最近 6 条」的查找范围。
        for index in range(8):
            self._record(bot, self._message(
                message_id=f"chat-{index}", sender_id="u9", sender_name="路人",
                content=f"闲聊 {index}", rich_type="", rich_only=False,
            ))

        entered, release = asyncio.Event(), asyncio.Event()

        async def enrich(message, **kwargs):
            entered.set()
            await release.wait()
            message.content = "一张图，画面是旧截图里的声骸面板。"

        bot.qq_adapter = SimpleNamespace(enrich_message=enrich)
        # 模拟图片消息已入库、识别尚未完成。
        enrichment = asyncio.create_task(enrich(image))
        bot._media_tasks_by_message[(image.session_id, "old-image")] = enrichment
        bot._rich_media_tasks.add(enrichment)

        reply = self._message(
            message_id="ask", sender_id="u2", sender_name="小红",
            content="这张图里是什么？", rich_type="", rich_only=False,
            reply_to_id="old-image",
        )
        self._record(bot, reply)
        waiter = asyncio.create_task(bot._prepare_media_for_judgement(reply, self._state(reply)))
        try:
            await entered.wait()
            await asyncio.sleep(0)
            self.assertFalse(waiter.done(), "引用仍在识别的旧图时不能跳过等待")
            release.set()
            await asyncio.gather(waiter, enrichment)
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (waiter, enrichment) if task), return_exceptions=True
            )

    async def test_archive_update_waits_for_original_insert(self):
        bot = self._bot()
        image = self._message()
        self._record(bot, image)
        release = asyncio.Event()
        bot.memory_storage = SimpleNamespace(update_group_analysis_message_content=AsyncMock(return_value=True))

        async def enrich(message, **kwargs):
            message.content = "一张图，画面是声骸管理界面。"

        bot.qq_adapter = SimpleNamespace(enrich_message=enrich)
        archive = asyncio.create_task(release.wait())
        enhancement = asyncio.create_task(bot._enrich_context_message(image, False, archive_task=archive))
        try:
            await asyncio.sleep(0)
            bot.memory_storage.update_group_analysis_message_content.assert_not_awaited()
            release.set()
            await enhancement
            bot.memory_storage.update_group_analysis_message_content.assert_awaited_once_with(
                image.session_id, image.message_id, image.content + " [image]"
            )
        finally:
            release.set()
            await asyncio.gather(archive, enhancement, return_exceptions=True)

    async def test_stale_reply_does_not_retarget_itself_after_thinking(self):
        bot = self._bot()
        message = self._message(content="鸣潮，启动！", rich_type="", rich_only=False)
        self._record(bot, message)
        decision = dict(
            direction="group", probability=0.8, emotional_state=None,
            context=SimpleNamespace(topic_familiarity=0.5), explicit_directed=False,
            context_marker=bot._context_marker_for_message(message.session_id, message),
        )

        async def thinking(**kwargs):
            self._record(bot, self._message(message_id="new", content="晚饭吃什么？", rich_type="", rich_only=False))

        bot.thinking_delay = SimpleNamespace(wait=thinking)
        bot._wait_for_group_settle = AsyncMock()
        bot._later_messages_from_same_sender = Mock(return_value=[])
        bot._retrieve_memories = AsyncMock(return_value=[])
        bot._judge_context_message = AsyncMock()
        await bot._compose_and_send(message, decision)
        bot._retrieve_memories.assert_not_awaited()
        bot._judge_context_message.assert_not_awaited()

    async def test_shutdown_closes_storage_after_disconnect_exception_and_only_once(self):
        bot = self._bot()
        bot.memory_storage = SimpleNamespace(
            _storage=SimpleNamespace(_embedding_service=None), close=AsyncMock(),
        )
        bot.qq_adapter = SimpleNamespace(disconnect=AsyncMock(side_effect=RuntimeError("offline failure")))
        await asyncio.gather(bot.stop(), bot.stop())
        await bot.stop()
        bot.qq_adapter.disconnect.assert_awaited_once()
        bot.memory_storage.close.assert_awaited_once()

    async def test_review_receives_same_generation_context(self):
        captured = {}

        async def review(message, history, reply, *, direction, conversation_judgement=None, generation_context=""):
            captured.update(context=generation_context, judgement=conversation_judgement)
            return SimpleNamespace(available=True, should_reply=True)

        accepted, _ = await GroupChatBot._validate_retried_reply(
            SimpleNamespace(review_reply=review), self._message(), [], "开冲！", "group",
            {"target": "group"}, generation_context="[你的记忆] 已核实的游戏背景",
        )
        self.assertTrue(accepted)
        self.assertEqual(captured["context"], "[你的记忆] 已核实的游戏背景")
        self.assertEqual(captured["judgement"], {"target": "group"})

    def test_group_image_options_use_rich_media_config(self):
        bot = self._bot()
        bot.config = {"rich_media": {"image": {"group_max_images": 2}}, "image": {"vision": {"enabled": True}}}
        self.assertEqual(bot._image_group_config["group_max_images"], 2)


class AdapterLoopOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_future_is_created_and_resolved_on_qq_loop(self):
        adapter = QQAdapter({})
        owner = asyncio.get_running_loop()
        adapter._owner_loop = owner
        adapter._clients.add(object())
        seen = []

        async def broadcast(payload):
            seen.append(asyncio.get_running_loop())
            echo = json.loads(payload)["echo"]
            adapter._resolve_api_response(echo, {"status": "ok", "data": {"ok": True}})

        adapter._broadcast = broadcast
        result = await asyncio.to_thread(lambda: asyncio.run(adapter.call_api("offline", {}, timeout=0.5)))
        self.assertEqual(result, {"ok": True})
        self.assertEqual(seen, [owner])
        self.assertFalse(adapter._pending_api)

    async def test_disconnect_marshals_pending_tasks_to_their_owner_loop(self):
        adapter = QQAdapter({})
        owner = asyncio.get_running_loop()
        adapter._owner_loop = owner
        pending = asyncio.create_task(asyncio.Event().wait())
        adapter._message_tasks.add(pending)
        adapter._server = SimpleNamespace(close=Mock(), wait_closed=AsyncMock())
        await asyncio.to_thread(lambda: asyncio.run(adapter.disconnect()))
        self.assertTrue(pending.cancelled())
        adapter._server.close.assert_called_once()
        adapter._server.wait_closed.assert_awaited_once()

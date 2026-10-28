"""戳一戳处理的离线测试。"""

import asyncio
import json
import unittest
from types import SimpleNamespace

from core.adapter.base import Message
from core.adapter.qq_adapter import QQAdapter
from main import GroupChatBot
from modules.poke_handler import (
    PokeConfig,
    PokeResponder,
    clean_reply,
    parse_poke_event,
)

SELF_ID = "2364385326"


class _FakeProvider:
    """只提供 chat() 的假 provider。"""

    def __init__(self, content="在的在的，戳我干嘛", error=None):
        self.model = "fake-model"
        self.content = content
        self.error = error
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content, model=self.model, usage={})


def _group_poke(poker="10001", target=SELF_ID, group="724786753"):
    return {
        "post_type": "notice",
        "notice_type": "notify",
        "sub_type": "poke",
        "group_id": group,
        "user_id": poker,
        "target_id": target,
    }


def _private_poke(sender="10001", target=SELF_ID, peer="10001"):
    return {
        "post_type": "notice",
        "notice_type": "notify",
        "sub_type": "poke",
        "user_id": peer,
        "target_id": target,
        "sender_id": sender,
    }


class ParsePokeEventTests(unittest.TestCase):
    def test_group_poke_is_parsed(self):
        event = parse_poke_event(_group_poke(), SELF_ID)
        self.assertIsNotNone(event)
        self.assertTrue(event.is_group)
        self.assertEqual(event.session_id, "group_724786753")
        self.assertEqual(event.poker_id, "10001")
        self.assertEqual(event.target_id, SELF_ID)

    def test_private_poke_uses_sender_as_poker(self):
        event = parse_poke_event(_private_poke(), SELF_ID)
        self.assertIsNotNone(event)
        self.assertFalse(event.is_group)
        self.assertEqual(event.session_id, "private_10001")
        self.assertEqual(event.poker_id, "10001")

    def test_poke_at_someone_else_is_ignored(self):
        self.assertIsNone(parse_poke_event(_group_poke(target="99999"), SELF_ID))

    def test_self_initiated_poke_is_ignored(self):
        # 机器人自己戳别人：target 是对方，user_id 才是机器人自己。
        self.assertIsNone(parse_poke_event(_group_poke(poker=SELF_ID, target="10002"), SELF_ID))

    def test_non_poke_notice_is_ignored(self):
        self.assertIsNone(
            parse_poke_event(
                {"post_type": "notice", "notice_type": "group_ban", "sub_type": "ban"},
                SELF_ID,
            )
        )

    def test_missing_self_id_is_ignored(self):
        self.assertIsNone(parse_poke_event(_group_poke(), ""))


class PokeConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = PokeConfig.from_dict(None)
        self.assertTrue(config.enabled)
        self.assertTrue(config.reply)
        self.assertTrue(config.use_llm)
        self.assertTrue(config.poke_back)
        self.assertEqual(config.cooldown_seconds, 30.0)

    def test_values_are_applied(self):
        config = PokeConfig.from_dict(
            {"enabled": False, "reply": False, "use_llm": False, "cooldown_seconds": 5}
        )
        self.assertFalse(config.enabled)
        self.assertFalse(config.reply)
        self.assertFalse(config.use_llm)
        self.assertEqual(config.cooldown_seconds, 5.0)

    def test_invalid_values_fall_back_to_defaults(self):
        config = PokeConfig.from_dict({"cooldown_seconds": "abc", "enabled": "yes"})
        self.assertEqual(config.cooldown_seconds, 30.0)
        self.assertTrue(config.enabled)

    def test_custom_canned_replies(self):
        config = PokeConfig.from_dict({"canned": ["自定义一句", "  ", "再一句"]})
        self.assertEqual(config.canned, ("自定义一句", "再一句"))


class AdapterPokeActionTests(unittest.IsolatedAsyncioTestCase):
    """poke 动作的参数必须满足 NapCat 的参数表（user_id 必填）。"""

    def _adapter(self):
        adapter = QQAdapter({"self_id": SELF_ID, "ws_port": 3001})
        adapter.calls = []

        async def _fake_call_api(action, params, timeout=None):
            adapter.calls.append((action, params))
            return {"message_id": 1}

        adapter.call_api = _fake_call_api
        return adapter

    async def test_group_poke_includes_required_user_id(self):
        adapter = self._adapter()
        ok = await adapter.poke("group_333863057", "10001")
        self.assertTrue(ok)
        action, params = adapter.calls[0]
        self.assertEqual(action, "group_poke")
        self.assertEqual(params["group_id"], "333863057")
        self.assertEqual(params["user_id"], "10001")
        self.assertEqual(params["target_id"], "10001")

    async def test_friend_poke_targets_the_peer(self):
        adapter = self._adapter()
        ok = await adapter.poke("private_10001", "10001")
        self.assertTrue(ok)
        action, params = adapter.calls[0]
        self.assertEqual(action, "friend_poke")
        self.assertEqual(params["user_id"], "10001")
        self.assertNotIn("group_id", params)

    async def test_poke_without_target_is_skipped(self):
        adapter = self._adapter()
        ok = await adapter.poke("group_333863057", "")
        self.assertFalse(ok)
        self.assertEqual(adapter.calls, [])


class PokeResponderTests(unittest.IsolatedAsyncioTestCase):
    def _responder(self, config=None, provider=None, **kwargs):
        sent = []
        poked = []

        async def send_text(session_id, text):
            sent.append((session_id, text))
            return True

        async def send_poke(session_id, target_id):
            poked.append((session_id, target_id))
            return True

        responder = PokeResponder(
            config,
            self_id=SELF_ID,
            provider=provider if provider is not None else _FakeProvider(),
            send_text=send_text,
            send_poke=send_poke,
            **kwargs,
        )
        return responder, sent, poked

    async def test_group_poke_replies_and_pokes_back(self):
        responder, sent, poked = self._responder()
        result = await responder.handle(_group_poke())
        self.assertEqual(result["skipped"], None) if "skipped" in result else None
        self.assertTrue(result["reply_sent"])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], "group_724786753")
        self.assertEqual(sent[0][1], "在的在的，戳我干嘛")
        self.assertTrue(result["poked_back"])
        self.assertEqual(poked, [("group_724786753", "10001")])

    async def test_private_poke_replies_in_private_session(self):
        responder, sent, poked = self._responder()
        result = await responder.handle(_private_poke())
        self.assertTrue(result["reply_sent"])
        self.assertEqual(sent[0][0], "private_10001")
        self.assertEqual(poked, [("private_10001", "10001")])

    async def test_poke_at_other_is_skipped(self):
        responder, sent, poked = self._responder()
        result = await responder.handle(_group_poke(target="99999"))
        self.assertEqual(result["skipped"], "not_me")
        self.assertEqual(sent, [])
        self.assertEqual(poked, [])

    async def test_cooldown_blocks_repeat_poke(self):
        responder, sent, poked = self._responder()
        await responder.handle(_group_poke())
        result = await responder.handle(_group_poke())
        self.assertEqual(result["skipped"], "cooldown")
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(poked), 1)

    async def test_disabled_feature_is_skipped(self):
        responder, sent, poked = self._responder({"enabled": False})
        result = await responder.handle(_group_poke())
        self.assertEqual(result["skipped"], "disabled")
        self.assertEqual(sent, [])

    async def test_group_can_be_disabled(self):
        responder, sent, poked = self._responder({"group_enabled": False})
        result = await responder.handle(_group_poke())
        self.assertEqual(result["skipped"], "group_disabled")
        self.assertEqual(sent, [])

    async def test_private_can_be_disabled(self):
        responder, sent, poked = self._responder({"private_enabled": False})
        result = await responder.handle(_private_poke())
        self.assertEqual(result["skipped"], "private_disabled")
        self.assertEqual(sent, [])

    async def test_canned_reply_when_llm_disabled(self):
        provider = _FakeProvider()
        responder, sent, poked = self._responder(
            {"use_llm": False, "canned": ["固定话术"]}, provider=provider
        )
        await responder.handle(_group_poke())
        self.assertEqual(provider.calls, 0)
        self.assertEqual(sent[0][1], "固定话术")

    async def test_llm_failure_falls_back_to_canned(self):
        provider = _FakeProvider(error=RuntimeError("模型炸了"))
        responder, sent, poked = self._responder({"canned": ["兜底话术"]}, provider=provider)
        await responder.handle(_group_poke())
        self.assertEqual(provider.calls, 1)
        self.assertEqual(sent[0][1], "兜底话术")

    async def test_empty_llm_output_falls_back_to_canned(self):
        provider = _FakeProvider(content="   ")
        responder, sent, poked = self._responder({"canned": ["兜底话术"]}, provider=provider)
        await responder.handle(_group_poke())
        self.assertEqual(sent[0][1], "兜底话术")

    async def test_poke_back_cooldown(self):
        responder, sent, poked = self._responder(
            {"cooldown_seconds": 0, "poke_back_cooldown_seconds": 600}
        )
        await responder.handle(_group_poke())
        await responder.handle(_group_poke())
        self.assertEqual(len(poked), 1)
        self.assertEqual(len(sent), 2)

    async def test_self_initiated_poke_is_ignored(self):
        responder, sent, poked = self._responder()
        # target 和 poker 都是机器人自己：不该回话也不该回戳
        result = await responder.handle(_group_poke(poker=SELF_ID))
        self.assertEqual(result["skipped"], "self_poke")
        self.assertEqual(sent, [])
        self.assertEqual(poked, [])

    async def test_never_poke_back_own_self(self):
        responder, sent, poked = self._responder()
        # target 是自己、poker 也是自己：不该回戳
        await responder.handle(_group_poke(poker=SELF_ID))
        self.assertEqual(poked, [])

    async def test_reply_can_be_disabled_but_poke_back_kept(self):
        responder, sent, poked = self._responder({"reply": False})
        result = await responder.handle(_group_poke())
        self.assertEqual(sent, [])
        self.assertTrue(result["poked_back"])


class ProactivePokeTests(unittest.IsolatedAsyncioTestCase):
    """主动戳人：概率、冷却、护栏。"""

    def _responder(self, config=None):
        poked = []

        async def send_poke(session_id, target_id):
            poked.append((session_id, target_id))
            return True

        responder = PokeResponder(
            config,
            self_id=SELF_ID,
            provider=_FakeProvider(),
            send_text=None,
            send_poke=send_poke,
        )
        return responder, poked

    async def test_probability_one_always_pokes(self):
        responder, poked = self._responder({"proactive_probability": 1.0})
        for _ in range(5):
            responder.config.proactive_cooldown_seconds = 0
            await responder.maybe_poke("group_724786753", "10001")
        self.assertEqual(len(poked), 5)

    async def test_probability_zero_never_pokes(self):
        responder, poked = self._responder({"proactive_probability": 0.0})
        for _ in range(20):
            await responder.maybe_poke("group_724786753", "10001")
        self.assertEqual(poked, [])

    async def test_never_poke_self(self):
        responder, poked = self._responder({"proactive_probability": 1.0})
        await responder.maybe_poke("group_724786753", SELF_ID)
        self.assertEqual(poked, [])

    async def test_cooldown_blocks_repeat(self):
        responder, poked = self._responder(
            {"proactive_probability": 1.0, "proactive_cooldown_seconds": 600}
        )
        await responder.maybe_poke("group_724786753", "10001")
        await responder.maybe_poke("group_724786753", "10001")
        self.assertEqual(len(poked), 1)

    async def test_does_not_double_poke_right_after_poke_back(self):
        responder, poked = self._responder(
            {
                "proactive_probability": 1.0,
                "poke_back_cooldown_seconds": 600,
                "proactive_cooldown_seconds": 0,
            }
        )
        await responder.handle(_group_poke())          # 收到戳 -> 回戳
        await responder.maybe_poke("group_724786753", "10001")  # 紧接着主动戳
        self.assertEqual(len(poked), 1)


class WideProactivePokeTests(unittest.IsolatedAsyncioTestCase):
    """宽版主动戳：群里每条消息都掷骰子，不要求 bot 参与那条消息。"""

    class _FakeResponder:
        def __init__(self):
            self.calls = []

        async def maybe_poke(self, session_id, target_id):
            self.calls.append((session_id, target_id))
            return True

    def _bot(self, responder):
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.poke_responder = responder
        bot._poke_tasks = set()
        return bot

    def _message(self, **overrides):
        values = dict(
            message_id="m1", message_type="group", group_id="g1",
            sender_id="u1", sender_name="小明", content="随便说句话",
        )
        values.update(overrides)
        return Message(**values)

    async def test_group_message_schedules_poke(self):
        responder = self._FakeResponder()
        bot = self._bot(responder)
        bot._schedule_proactive_poke(self._message())
        for _ in range(3):
            await asyncio.sleep(0)
        self.assertEqual(responder.calls, [("group_g1", "u1")])

    async def test_missing_responder_does_not_raise(self):
        bot = GroupChatBot.__new__(GroupChatBot)
        # 测试用 __new__ 造对象，实例上没有 poke_responder 属性
        bot._schedule_proactive_poke(self._message())

    async def test_task_set_is_created_lazily(self):
        responder = self._FakeResponder()
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.poke_responder = responder
        bot._schedule_proactive_poke(self._message())
        for _ in range(3):
            await asyncio.sleep(0)
        self.assertEqual(responder.calls, [("group_g1", "u1")])
        self.assertIsInstance(bot._poke_tasks, set)

    async def test_empty_sender_is_skipped(self):
        responder = self._FakeResponder()
        bot = self._bot(responder)
        bot._schedule_proactive_poke(self._message(sender_id=""))
        for _ in range(3):
            await asyncio.sleep(0)
        self.assertEqual(responder.calls, [])


class CleanReplyTests(unittest.TestCase):
    def test_strips_quotes_and_newlines(self):
        self.assertEqual(clean_reply('"在的在的"\n'), "在的在的")

    def test_strips_prefix(self):
        self.assertEqual(clean_reply("好的，来了"), "来了")

    def test_truncates_long_text(self):
        self.assertEqual(len(clean_reply("啊" * 100)), 40)


class AdapterNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def test_notice_is_forwarded_to_callback(self):
        received = []

        async def on_notice(data):
            received.append(data)

        adapter = QQAdapter({"self_id": SELF_ID, "ws_port": 3001}, on_notice=on_notice)
        await adapter._handle_message(json.dumps(_group_poke()))
        for _ in range(3):
            await asyncio.sleep(0)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["sub_type"], "poke")

    async def test_message_events_do_not_trigger_notice_callback(self):
        received = []

        async def on_notice(data):
            received.append(data)

        adapter = QQAdapter({"self_id": SELF_ID, "ws_port": 3001}, on_notice=on_notice)
        await adapter._handle_message(
            json.dumps({"post_type": "meta_event", "meta_event_type": "heartbeat"})
        )
        for _ in range(3):
            await asyncio.sleep(0)
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()

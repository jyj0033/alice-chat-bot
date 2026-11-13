"""引用消息的正文不能混进说话人自己的话。

线上实例（2026-11-13 11:28:15，龍飛月 @爱丽丝）：

    自己说的：爱丽丝，来点幕后黑手的小说
    引用机器人的：灵魂原来还有分辨率的吗……？

存进记忆库却是黏在一起的：
    爱丽丝，来点幕后黑手的小说灵魂原来还有分辨率的吗……？

原因：官方适配器把被引用正文当成普通文本段拼在正文后面，而渲染层
`"".join(...)` 不加分隔符。引用的是机器人自己时更糟——机器人的话被记成了
群友说的话（下面「应该吧」那条就是）。

这里锁四件事：
1. 适配器解析出来的 content / outer_text 只有说话人自己的话；
2. 被引用原文单独放在 Message.quoted_content；
3. 记忆入库时引用原文只进 metadata，不进 content；
4. 上下文渲染和接话判断仍能看到引用的原话（信息不能丢）。
"""

import asyncio
import json
import unittest

from core.adapter.base import Message
from core.adapter.official_adapter import OP_DISPATCH, OfficialQQAdapter
from main import GroupChatBot
from modules.memory.context import ContextManager


APP_ID = "1020899999"
GROUP_OPENID = "B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5"
MEMBER_OPENID = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4"

# 线上 11:28 那条的真实两段
OWN_TEXT = "爱丽丝，来点幕后黑手的小说"
QUOTED_TEXT = "灵魂原来还有分辨率的吗……？"
# 11:28:05 那条：机器人的上一句被引用，群友回了个「应该吧」
SHORT_OWN_TEXT = "应该吧"


async def _noop():
    return None


def _adapter():
    config = {
        "self_id": APP_ID,
        "official": {"app_id": APP_ID, "app_secret": "secret-value"},
    }
    adapter = OfficialQQAdapter(config)

    async def _fake_request(method, path, **kwargs):
        return {"id": "ROBOT1.0_sent", "ext_info": {"ref_idx": "REFIDX_sent=="}}

    adapter._request_json = _fake_request
    adapter._connected = True
    return adapter


def _group_event(**overrides):
    payload = {
        "id": "ROBOT1.0_abc",
        "author": {
            "id": MEMBER_OPENID,
            "member_openid": MEMBER_OPENID,
            "member_role": "member",
            "username": "龍飛月",
            "bot": False,
        },
        "content": "今天天气不错",
        "group_openid": GROUP_OPENID,
        "message_type": 0,
        "timestamp": "2026-07-21T10:00:00+08:00",
        "message_scene": {
            "source": "default",
            "ext": ["msg_idx=REFIDX_own==", "auth_token=xxx"],
        },
    }
    payload.update(overrides)
    return payload


def _quoted_event(**overrides):
    """群友引用机器人消息的官方事件。"""
    return _group_event(
        message_type=103,
        content=OWN_TEXT,
        msg_elements=[
            {
                "msg_idx": "REFIDX_bot==",
                "message_type": 103,
                "content": QUOTED_TEXT,
                "author": {"id": "BOTMEMBER", "bot": True},
            }
        ],
        message_scene={
            "source": "default",
            "ext": ["ref_msg_idx=REFIDX_bot==", "msg_idx=REFIDX_own=="],
        },
        **overrides,
    )


class OfficialQuotedContentTests(unittest.IsolatedAsyncioTestCase):
    async def _receive(self, event):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()
        await adapter._handle_ws_payload(
            json.dumps({
                "op": OP_DISPATCH,
                "s": 21,
                "t": "GROUP_AT_MESSAGE_CREATE",
                "d": event,
            })
        )
        self.assertEqual(len(received), 1)
        return received[0]

    async def test_quoted_text_is_not_glued_into_sender_content(self):
        message = await self._receive(_quoted_event())
        self.assertEqual(message.content, OWN_TEXT)
        self.assertNotIn(QUOTED_TEXT, message.content)
        self.assertNotIn("小说灵魂原来还有分辨率", message.content)

    async def test_outer_text_also_excludes_the_quoted_text(self):
        """outer_text 才是写进长期记忆正文的那一份。"""
        message = await self._receive(_quoted_event())
        self.assertNotIn(QUOTED_TEXT, message.outer_text)
        self.assertEqual(message.outer_text, OWN_TEXT)

    async def test_quoted_content_is_kept_separately(self):
        message = await self._receive(_quoted_event())
        self.assertEqual(message.quoted_content, QUOTED_TEXT)

    async def test_quoted_bot_message_still_resolves_reply_target(self):
        """拆开正文不能把引用关系弄丢。"""
        message = await self._receive(_quoted_event())
        self.assertEqual(message.reply_to_qq, APP_ID)
        self.assertTrue(message.reply_to_id)

    async def test_message_without_quote_has_empty_quoted_content(self):
        message = await self._receive(_group_event())
        self.assertEqual(message.quoted_content, "")
        self.assertEqual(message.content, "今天天气不错")


class LongTermMemoryQuotedTests(unittest.TestCase):
    """记忆入库：引用原文只进 metadata，正文只放说话人自己的话。"""

    def _bot(self, storage):
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.long_term_memory_enabled = True
        bot.memory_storage = storage
        bot.config = {"qq": {"self_id": APP_ID}}
        bot.personality = type("P", (), {"name": "爱丽丝"})()
        bot._memory_tasks = set()
        return bot

    def test_quoted_content_goes_to_metadata_not_to_memory_content(self):
        class _Storage:
            def __init__(self):
                self.stored = []

            async def find_similar(self, *args):
                return None

            async def store(self, memory):
                self.stored.append(memory)

        storage = _Storage()
        bot = self._bot(storage)

        async def run():
            bot._store_long_term_memory(
                Message(
                    message_id="m1",
                    message_type="group",
                    sender_id=MEMBER_OPENID,
                    sender_name="龍飛月",
                    group_id=GROUP_OPENID,
                    content="爱丽丝，来点幕后黑手的小说，要那种幕后布局的",
                    quoted_content=QUOTED_TEXT,
                    mentioned_me=True,
                    mentioned_user_ids=[APP_ID],
                    reply_to_id="ROBOT1.0_bot",
                ),
                f"group_{GROUP_OPENID}",
            )
            await asyncio.gather(*list(bot._memory_tasks))

        asyncio.run(run())

        self.assertEqual(len(storage.stored), 1)
        memory = storage.stored[0]
        self.assertNotIn(QUOTED_TEXT, memory.content)
        self.assertNotIn("布局的灵魂原来", memory.content)
        self.assertEqual(memory.metadata["quoted_content"], QUOTED_TEXT)

    def test_messages_without_quote_store_empty_quoted_content(self):
        class _Storage:
            def __init__(self):
                self.stored = []

            async def find_similar(self, *args):
                return None

            async def store(self, memory):
                self.stored.append(memory)

        storage = _Storage()
        bot = self._bot(storage)

        async def run():
            bot._store_long_term_memory(
                Message(
                    message_id="m2",
                    message_type="group",
                    sender_id=MEMBER_OPENID,
                    sender_name="龍飛月",
                    group_id=GROUP_OPENID,
                    content="这条消息没有任何引用，就是普通的一句话而已",
                    mentioned_me=True,
                    mentioned_user_ids=[APP_ID],
                ),
                f"group_{GROUP_OPENID}",
            )
            await asyncio.gather(*list(bot._memory_tasks))

        asyncio.run(run())
        self.assertEqual(len(storage.stored), 1)
        self.assertEqual(storage.stored[0].metadata["quoted_content"], "")


class QuotedContextTests(unittest.TestCase):
    """上下文里仍要能看到引用的原话，否则接话会接错。"""

    def test_context_renders_quote_attribute(self):
        manager = ContextManager()
        manager.add_message(
            "group_g1",
            "u1",
            "龍飛月",
            OWN_TEXT,
            message_id="m1",
            reply_to_id="b1",
            quoted_content=QUOTED_TEXT,
        )
        text = manager.get_window("group_g1").build_conversation_text("爱丽丝", bot_id="bot")
        self.assertIn(QUOTED_TEXT, text)
        self.assertNotIn("小说灵魂原来还有分辨率", text)

    def test_reply_situation_falls_back_to_platform_quoted_content(self):
        """被引用的那条已经不在窗口里时，用平台给的原文兜底。"""
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.context_manager = ContextManager()
        message = Message(
            message_id="m1",
            message_type="group",
            sender_id="u1",
            sender_name="龍飛月",
            group_id="g1",
            content=SHORT_OWN_TEXT,
            reply_to_id="not-in-window",
            quoted_content=QUOTED_TEXT,
        )
        fields = bot._reply_situation_fields("group_g1", message)
        self.assertEqual(fields["quoted_content"], QUOTED_TEXT)

    def test_reply_situation_prefers_the_window_message(self):
        bot = GroupChatBot.__new__(GroupChatBot)
        bot.context_manager = ContextManager()
        bot.context_manager.add_message(
            "group_g1", "bot", "爱丽丝", QUOTED_TEXT, message_id="b1", is_bot=True
        )
        message = Message(
            message_id="m1",
            message_type="group",
            sender_id="u1",
            sender_name="龍飛月",
            group_id="g1",
            content=SHORT_OWN_TEXT,
            reply_to_id="b1",
            quoted_content="平台给的原文（应被窗口里的版本覆盖）",
        )
        fields = bot._reply_situation_fields("group_g1", message)
        self.assertEqual(fields["quoted_content"], QUOTED_TEXT)
        self.assertEqual(fields["quoted_sender"], "you")


if __name__ == "__main__":
    unittest.main()

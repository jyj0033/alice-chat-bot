"""官方通道的「主动消息开关」与 Markdown 发送测试。

覆盖三件新加的事：

* ``GROUP_MSG_REJECT`` / ``GROUP_MSG_RECEIVE`` —— 群管理员在机器人资料页
  关/开消息接收，关掉后机器人整个群静默；
* ``C2C_MSG_REJECT`` / ``C2C_MSG_RECEIVE`` —— 用户关/开主动消息推送；
* ``bot_state`` 查询与主动消息前置拦截，以及 Markdown（msg_type=2）发送。

不连真实网关：用假的 ``_request_json`` 顶掉 HTTP，用 ``_handle_ws_payload``
直接喂网关下行帧。
"""

import json
import unittest

from core.adapter.official_adapter import (
    OP_DISPATCH,
    OfficialQQAdapter,
    ProactiveBlockedError,
)


GROUP_OPENID = "B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5"
USER_OPENID = "C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6"
APP_ID = "1020899999"


def _noop():
    """on_message 回调要是同步可调用的，返回一个 None 就行。"""
    return None


def _adapter(**overrides):
    config = {
        "self_id": APP_ID,
        "official": {
            "app_id": APP_ID,
            "app_secret": "secret-value",
        },
    }
    config.update(overrides)
    adapter = OfficialQQAdapter(config)
    adapter.calls = []
    adapter.responses = {}

    async def _fake_request(method, path, **kwargs):
        adapter.calls.append((method, path, kwargs))
        key = (method, path)
        if key in adapter.responses:
            value = adapter.responses[key]
            if isinstance(value, Exception):
                raise value
            return value
        if path.endswith("/messages"):
            return {"id": "ROBOT1.0_sent", "ext_info": {"ref_idx": "REFIDX_sent=="}}
        return {}

    adapter._request_json = _fake_request
    adapter._connected = True
    return adapter


async def _dispatch(adapter, event_type, payload, seq=1):
    await adapter._handle_ws_payload(
        json.dumps({"op": OP_DISPATCH, "s": seq, "t": event_type, "d": payload})
    )


def _group_message_event(**overrides):
    payload = {
        "id": "ROBOT1.0_abc",
        "author": {
            "id": "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4",
            "member_openid": "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4",
            "member_role": "member",
            "username": "小明",
            "bot": False,
        },
        "content": "在吗",
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


class PushSwitchEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_msg_reject_mutes_the_group(self):
        adapter = _adapter()

        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1", "timestamp": 1},
        )

        self.assertTrue(adapter.is_group_muted(GROUP_OPENID))
        self.assertFalse(await adapter.proactive_allowed(f"group_{GROUP_OPENID}"))

    async def test_group_msg_receive_restores_the_group(self):
        adapter = _adapter()

        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )
        await _dispatch(
            adapter,
            "GROUP_MSG_RECEIVE",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )

        self.assertFalse(adapter.is_group_muted(GROUP_OPENID))
        self.assertTrue(await adapter.proactive_allowed(f"group_{GROUP_OPENID}"))

    async def test_c2c_msg_reject_blocks_proactive_to_that_user(self):
        adapter = _adapter()

        await _dispatch(adapter, "C2C_MSG_REJECT", {"openid": USER_OPENID, "timestamp": 1})

        self.assertFalse(await adapter.proactive_allowed(f"private_{USER_OPENID}"))
        # 不影响别的用户
        self.assertTrue(await adapter.proactive_allowed("private_OTHER_USER"))

    async def test_c2c_msg_receive_restores_proactive(self):
        adapter = _adapter()

        await _dispatch(adapter, "C2C_MSG_REJECT", {"openid": USER_OPENID})
        await _dispatch(adapter, "C2C_MSG_RECEIVE", {"openid": USER_OPENID})

        self.assertTrue(await adapter.proactive_allowed(f"private_{USER_OPENID}"))

    async def test_switch_event_is_not_treated_as_a_message(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message)

        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )

        self.assertEqual(received, [])
        self.assertEqual(adapter.messages_received, 0)

    async def test_event_overrides_stale_bot_state(self):
        """事件是实时推送，比 5 分钟缓存的 bot_state 新。"""
        adapter = _adapter()
        adapter.responses[("GET", f"/v2/groups/{GROUP_OPENID}/bot_state")] = {
            "allow_proactive_msg": True,
            "recv_msg_setting": "all",
            "member_role": "admin",
        }
        # 先查一次，缓存里是「允许」
        self.assertTrue(await adapter.proactive_allowed(f"group_{GROUP_OPENID}"))

        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )

        self.assertFalse(await adapter.proactive_allowed(f"group_{GROUP_OPENID}"))


class BotStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_state_fields_are_parsed(self):
        adapter = _adapter()
        adapter.responses[("GET", f"/v2/groups/{GROUP_OPENID}/bot_state")] = {
            "member_openid": "BOTOPENID",
            "joined_at": "2026-11-12T12:31:59+08:00",
            "allow_proactive_msg": True,
            "recv_msg_setting": "only_mention",
            "member_role": "admin",
        }

        state = await adapter.get_bot_state(GROUP_OPENID)

        self.assertTrue(state["allow_proactive_msg"])
        self.assertEqual(state["recv_msg_setting"], "only_mention")
        self.assertEqual(state["member_role"], "admin")

    async def test_bot_state_is_cached(self):
        adapter = _adapter()
        adapter.responses[("GET", f"/v2/groups/{GROUP_OPENID}/bot_state")] = {
            "allow_proactive_msg": True,
        }

        await adapter.get_bot_state(GROUP_OPENID)
        await adapter.get_bot_state(GROUP_OPENID)

        state_calls = [c for c in adapter.calls if c[1].endswith("/bot_state")]
        self.assertEqual(len(state_calls), 1)

    async def test_proactive_blocked_when_bot_state_says_no(self):
        adapter = _adapter()
        adapter.responses[("GET", f"/v2/groups/{GROUP_OPENID}/bot_state")] = {
            "allow_proactive_msg": False,
            "recv_msg_setting": "all",
        }

        self.assertFalse(await adapter.proactive_allowed(f"group_{GROUP_OPENID}"))

    async def test_proactive_allowed_when_state_is_unknown(self):
        """查不到状态时按「允许」处理 —— 宁可让官方拒一次，也不要误杀。"""
        adapter = _adapter()
        adapter.responses[("GET", f"/v2/groups/{GROUP_OPENID}/bot_state")] = {}

        self.assertTrue(await adapter.proactive_allowed(f"group_{GROUP_OPENID}"))

    async def test_proactive_allowed_when_query_fails(self):
        adapter = _adapter()
        adapter.responses[("GET", f"/v2/groups/{GROUP_OPENID}/bot_state")] = RuntimeError(
            "网络炸了"
        )

        self.assertTrue(await adapter.proactive_allowed(f"group_{GROUP_OPENID}"))


class ProactiveGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_proactive_message_is_blocked_and_not_sent(self):
        adapter = _adapter()
        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )

        ok = await adapter.send_message(f"group_{GROUP_OPENID}", "日报来啦")

        self.assertFalse(ok)
        # 根本没走到发消息那一步
        self.assertFalse([c for c in adapter.calls if c[1].endswith("/messages")])

    async def test_passive_reply_is_not_blocked(self):
        """群里 @ 机器人是有人等着回话的，被动回复不能被拦。"""
        adapter = _adapter()
        adapter.on_message_callback = lambda message: _noop()
        await _dispatch(adapter, "GROUP_AT_MESSAGE_CREATE", _group_message_event())
        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )
        adapter.responses[("GET", f"/v2/groups/{GROUP_OPENID}/bot_state")] = {
            "allow_proactive_msg": False,
        }

        ok, _ = await adapter.send_message_with_id(
            f"group_{GROUP_OPENID}", "好的", reply_to_id="ROBOT1.0_abc"
        )

        self.assertTrue(ok)

    async def test_blocked_send_raises_internally(self):
        adapter = _adapter()
        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )

        with self.assertRaises(ProactiveBlockedError):
            await adapter._post_message(
                f"group_{GROUP_OPENID}", {"msg_type": 0, "content": "x"}
            )

    async def test_blocked_send_does_not_consume_rate_limit(self):
        adapter = _adapter()
        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )

        for _ in range(5):
            with self.assertRaises(ProactiveBlockedError):
                await adapter._post_message(
                    f"group_{GROUP_OPENID}", {"msg_type": 0, "content": "x"}
                )

        # 频控额度没被这些失败的主动消息吃掉
        self.assertEqual(len(adapter.rate_limiter._bot_sends), 0)


class MarkdownSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_markdown_body_uses_nested_content(self):
        adapter = _adapter()

        ok, message_id = await adapter.send_markdown_with_id(
            f"group_{GROUP_OPENID}", "## 标题\n\n- 一条"
        )

        self.assertTrue(ok)
        self.assertTrue(message_id)
        method, path, kwargs = adapter.calls[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(path, f"/v2/groups/{GROUP_OPENID}/messages")
        body = kwargs["json_body"]
        self.assertEqual(body["msg_type"], 2)
        self.assertEqual(body["markdown"]["content"], "## 标题\n\n- 一条")
        # 填了 markdown 之后 content 必须为空，否则撞 22006
        self.assertNotIn("content", body)

    async def test_markdown_reply_keeps_msg_id(self):
        adapter = _adapter()
        adapter.on_message_callback = lambda message: _noop()
        await _dispatch(adapter, "GROUP_AT_MESSAGE_CREATE", _group_message_event())

        await adapter.send_markdown_with_id(
            f"group_{GROUP_OPENID}", "**加粗**", reply_to_id="ROBOT1.0_abc"
        )

        body = adapter.calls[-1][2]["json_body"]
        self.assertEqual(body["msg_id"], "ROBOT1.0_abc")
        self.assertEqual(body["msg_seq"], 1)

    async def test_empty_markdown_is_not_sent(self):
        adapter = _adapter()

        ok, _ = await adapter.send_markdown_with_id(f"group_{GROUP_OPENID}", "   ")

        self.assertFalse(ok)
        self.assertFalse([c for c in adapter.calls if c[1].endswith("/messages")])

    async def test_markdown_blocked_when_push_disabled(self):
        adapter = _adapter()
        await _dispatch(
            adapter,
            "GROUP_MSG_REJECT",
            {"group_openid": GROUP_OPENID, "op_member_openid": "OP1"},
        )

        ok = await adapter.send_markdown(f"group_{GROUP_OPENID}", "## 标题")

        self.assertFalse(ok)

    async def test_send_markdown_helper_returns_bool(self):
        adapter = _adapter()

        self.assertTrue(await adapter.send_markdown(f"group_{GROUP_OPENID}", "正文"))


if __name__ == "__main__":
    unittest.main()

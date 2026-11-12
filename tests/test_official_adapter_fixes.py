"""官方通道这轮补的四件事 + 主模型告警的测试。

1. 心跳 ACK 看门狗：半开连接（写得进去、收不到包）必须能被发现
2. 面板发图带 @：at_users 要渲染成 <@member_openid>，认不出来要报出来
3. 戳一戳降级：通道不支持就别白掷骰子；入站戳要能用普通文字回应
4. 群成员昵称：成员接口要白名单，但消息事件里本来就有 username
5. 主模型失败告警：ProviderHealth 要能记下失败并暴露给面板
"""

import json
import time
import unittest

from core.adapter.official_adapter import (
    OP_HEARTBEAT_ACK,
    OfficialQQAdapter,
)
from core.adapter.qq_adapter import QQAdapter
from modules.llm.health import HealthTrackedProvider, ProviderHealth


APP_ID = "1020899999"
GROUP_OPENID = "B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5"
MEMBER_OPENID = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4"
HEARTBEAT_INTERVAL = 41.2


def _adapter(**overrides):
    config = {
        "self_id": APP_ID,
        "official": {"app_id": APP_ID, "app_secret": "secret-value"},
    }
    config.update(overrides)
    adapter = OfficialQQAdapter(config)
    adapter.calls = []
    adapter.responses = {}
    adapter.messages = []
    adapter.notices = []

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
        if "/members/" in path:
            # 群成员接口是白名单能力，默认拿不到。
            raise RuntimeError("err_code=11253")
        return {}

    adapter._request_json = _fake_request
    adapter._connected = True

    async def _on_message(message):
        adapter.messages.append(message)

    async def _on_notice(notice):
        adapter.notices.append(notice)

    adapter.on_message_callback = _on_message
    adapter.on_notice_callback = _on_notice
    return adapter


def _group_event(**overrides):
    payload = {
        "id": "ROBOT1.0_abc",
        "author": {
            "id": MEMBER_OPENID,
            "member_openid": MEMBER_OPENID,
            "member_role": "member",
            "username": "小明",
            "bot": False,
        },
        "content": "今天天气不错",
        "group_openid": GROUP_OPENID,
        "message_type": 0,
        "timestamp": "2026-07-21T10:00:00+08:00",
        "message_scene": {"source": "default", "ext": ["msg_idx=REFIDX_own=="]},
    }
    payload.update(overrides)
    return payload


class HeartbeatWatchdogTests(unittest.IsolatedAsyncioTestCase):
    """P0：半开连接必须能被判定为已死。"""

    async def test_ack_frame_updates_watchdog_state(self):
        adapter = _adapter()
        self.assertFalse(adapter._ack_seen)

        await adapter._handle_ws_payload(json.dumps({"op": OP_HEARTBEAT_ACK}))

        self.assertTrue(adapter._ack_seen)
        self.assertGreater(adapter._last_ack_at, 0)

    async def test_stale_connection_is_detected(self):
        adapter = _adapter()
        adapter._ack_seen = True
        adapter._last_ack_at = time.time() - 200

        self.assertTrue(adapter._gateway_stale(HEARTBEAT_INTERVAL))

    async def test_fresh_connection_is_not_stale(self):
        adapter = _adapter()
        adapter._ack_seen = True
        adapter._last_ack_at = time.time()

        self.assertFalse(adapter._gateway_stale(HEARTBEAT_INTERVAL))

    async def test_never_acked_connection_is_not_killed(self):
        """平台没回过 ACK 就不判死 —— 误杀健康连接比漏判更糟。"""
        adapter = _adapter()
        adapter._ack_seen = False
        adapter._last_ack_at = time.time() - 1000

        self.assertFalse(adapter._gateway_stale(HEARTBEAT_INTERVAL))

    async def test_resumed_session_without_events_is_treated_as_dead(self):
        adapter = _adapter()
        adapter._resumed_connection = True
        adapter._connected_at = time.time() - 200
        adapter._last_event_at = adapter._connected_at

        self.assertTrue(adapter._resumed_session_is_dead(HEARTBEAT_INTERVAL))

    async def test_resumed_session_with_events_stays_alive(self):
        adapter = _adapter()
        adapter._resumed_connection = True
        adapter._connected_at = time.time() - 200
        adapter._last_event_at = time.time() - 10

        self.assertFalse(adapter._resumed_session_is_dead(HEARTBEAT_INTERVAL))

    async def test_fresh_identify_is_never_treated_as_dead(self):
        adapter = _adapter()
        adapter._resumed_connection = False
        adapter._connected_at = time.time() - 200
        adapter._last_event_at = adapter._connected_at

        self.assertFalse(adapter._resumed_session_is_dead(HEARTBEAT_INTERVAL))


class AtMentionTests(unittest.IsolatedAsyncioTestCase):
    """P1：面板发图带的 at_users 要真的渲染成 @。"""

    async def test_known_openid_renders_as_official_at(self):
        adapter = _adapter()
        adapter._remember_member_name(MEMBER_OPENID, "小明")

        prefix, problems = adapter._render_at_prefix([MEMBER_OPENID])

        self.assertEqual(prefix, f"<@{MEMBER_OPENID}> ")
        self.assertEqual(problems, [])

    async def test_unknown_id_is_reported_not_silently_dropped(self):
        adapter = _adapter()

        prefix, problems = adapter._render_at_prefix(["2335864693"])

        self.assertIn("<@2335864693>", prefix)
        self.assertEqual(len(problems), 1)
        self.assertIn("2335864693", problems[0])

    async def test_at_all_renders_as_official_all(self):
        adapter = _adapter()

        prefix, problems = adapter._render_at_prefix(["all"])

        self.assertEqual(prefix, "<@all> ")
        self.assertEqual(problems, [])

    async def test_send_images_puts_at_prefix_into_the_text_message(self):
        adapter = _adapter()
        adapter._remember_member_name(MEMBER_OPENID, "小明")
        adapter.responses[("POST", f"/v2/groups/{GROUP_OPENID}/files")] = {
            "file_info": "FILEINFO123"
        }

        result = await adapter.send_images(
            f"group_{GROUP_OPENID}", [b"\x89PNG1"], text="原图", at_users=[MEMBER_OPENID]
        )

        self.assertTrue(result["success"])
        bodies = [c[2]["json_body"] for c in adapter.calls if c[1].endswith("/messages")]
        self.assertEqual(bodies[0]["msg_type"], 0)
        self.assertEqual(bodies[0]["content"], f"<@{MEMBER_OPENID}> 原图")

    async def test_send_images_reports_unknown_at_in_errors(self):
        adapter = _adapter()
        adapter.responses[("POST", f"/v2/groups/{GROUP_OPENID}/files")] = {
            "file_info": "FILEINFO123"
        }

        result = await adapter.send_images(
            f"group_{GROUP_OPENID}", [b"\x89PNG1"], text="原图", at_users=["2335864693"]
        )

        self.assertTrue(result["success"])
        self.assertTrue(any("2335864693" in err for err in result["errors"]))


class PokeDegradationTests(unittest.IsolatedAsyncioTestCase):
    """P2：官方通道没有戳一戳接口，要降级得干净。"""

    async def test_capability_flags(self):
        self.assertFalse(OfficialQQAdapter.supports_poke)
        self.assertTrue(QQAdapter.supports_poke)

    async def test_poke_returns_false(self):
        adapter = _adapter()

        self.assertFalse(await adapter.poke(f"group_{GROUP_OPENID}", MEMBER_OPENID))

    async def test_poke_scene_becomes_a_notice_instead_of_a_message(self):
        adapter = _adapter()
        event = _group_event(
            content="",
            message_scene={"source": "default", "scene": "戳一戳", "ext": []},
        )

        await adapter._handle_event("GROUP_AT_MESSAGE_CREATE", event)

        self.assertEqual(len(adapter.notices), 1)
        notice = adapter.notices[0]
        self.assertEqual(notice["notice_type"], "notify")
        self.assertEqual(notice["sub_type"], "poke")
        self.assertEqual(notice["group_id"], GROUP_OPENID)
        self.assertEqual(notice["user_id"], MEMBER_OPENID)
        # 被戳的人填 self_id（AppID），和戳一戳响应器的判断口径一致。
        self.assertEqual(notice["target_id"], APP_ID)
        # 戳一戳不该再当成一条普通群消息进对话链路。
        self.assertEqual(adapter.messages, [])

    async def test_normal_group_event_is_not_treated_as_poke(self):
        adapter = _adapter()

        await adapter._handle_event("GROUP_AT_MESSAGE_CREATE", _group_event())

        self.assertEqual(adapter.notices, [])
        self.assertEqual(len(adapter.messages), 1)


class MemberNameFallbackTests(unittest.IsolatedAsyncioTestCase):
    """P3：昵称不用申请白名单也能拿到。"""

    async def test_member_name_is_remembered_from_event(self):
        adapter = _adapter()

        await adapter._handle_event("GROUP_AT_MESSAGE_CREATE", _group_event())

        self.assertEqual(adapter._member_names.get(MEMBER_OPENID), "小明")

    async def test_display_name_falls_back_to_event_username(self):
        adapter = _adapter()
        await adapter._handle_event("GROUP_AT_MESSAGE_CREATE", _group_event())

        name = await adapter.get_member_display_name(GROUP_OPENID, MEMBER_OPENID)

        self.assertEqual(name, "小明")

    async def test_display_name_empty_for_never_seen_member(self):
        adapter = _adapter()

        name = await adapter.get_member_display_name(GROUP_OPENID, MEMBER_OPENID)

        self.assertEqual(name, "")

    async def test_known_members_lists_openid_and_name(self):
        adapter = _adapter()
        await adapter._handle_event("GROUP_AT_MESSAGE_CREATE", _group_event())

        members = adapter.known_members()

        self.assertEqual(members, [{"id": MEMBER_OPENID, "name": "小明"}])


class _BoomProvider:
    model = "boom-model"

    async def chat(self, request):
        raise RuntimeError("HTTP 403 disabled in this account")


class _OkProvider:
    model = "ok-model"

    async def chat(self, request):
        return "ok"


class ProviderHealthTests(unittest.IsolatedAsyncioTestCase):
    """主模型失败告警。"""

    async def test_failure_is_recorded_and_reraised(self):
        health = ProviderHealth()
        provider = HealthTrackedProvider(_BoomProvider(), "primary", health)

        with self.assertRaises(RuntimeError):
            await provider.chat(object())

        snapshot = health.snapshot()["primary"]
        self.assertEqual(snapshot["failures"], 1)
        self.assertTrue(snapshot["degraded"])
        self.assertIn("403", snapshot["last_error"])
        self.assertEqual(health.degraded_providers(), ["primary"])

    async def test_success_clears_degraded(self):
        health = ProviderHealth()
        provider = HealthTrackedProvider(_BoomProvider(), "primary", health)
        with self.assertRaises(RuntimeError):
            await provider.chat(object())

        provider = HealthTrackedProvider(_OkProvider(), "primary", health)
        await provider.chat(object())

        snapshot = health.snapshot()["primary"]
        self.assertFalse(snapshot["degraded"])
        self.assertEqual(snapshot["successes"], 1)
        self.assertEqual(snapshot["model"], "ok-model")
        self.assertEqual(health.degraded_providers(), [])

    async def test_attributes_are_delegated(self):
        health = ProviderHealth()
        provider = HealthTrackedProvider(_OkProvider(), "primary", health)

        self.assertEqual(provider.model, "ok-model")


if __name__ == "__main__":
    unittest.main()

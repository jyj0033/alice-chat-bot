"""官方开放平台通道的离线测试。

不连真实网关：用假的 ``_request_json`` 顶掉 HTTP，用 ``_handle_ws_payload``
直接喂网关下行帧，覆盖事件解析、引用回复、富媒体上传和降级行为。
"""

import json
import unittest

from core.adapter.official_adapter import (
    INTENT_GROUP_AND_C2C,
    OP_DISPATCH,
    OP_HELLO,
    OfficialQQAdapter,
)


GROUP_OPENID = "B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5"
MEMBER_OPENID = "A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4"
USER_OPENID = "C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6"
APP_ID = "1020899999"


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
        "message_scene": {
            "source": "default",
            "ext": ["msg_idx=REFIDX_own==", "auth_token=xxx"],
        },
    }
    payload.update(overrides)
    return payload


def _c2c_event(**overrides):
    payload = {
        "id": "ROBOT1.0_c2c",
        "author": {
            "id": USER_OPENID,
            "user_openid": USER_OPENID,
            "union_openid": "",
            "username": "",
            "bot": False,
        },
        "content": "你好",
        "message_type": 0,
        "timestamp": "2026-07-21T10:00:00+08:00",
        "message_scene": {"source": "default", "ext": ["msg_idx=REFIDX_c2c=="]},
    }
    payload.update(overrides)
    return payload


class OfficialEventParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_at_message_sets_mentioned_me_and_canonical_bot_id(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {
                    "op": OP_DISPATCH,
                    "s": 12,
                    "t": "GROUP_AT_MESSAGE_CREATE",
                    "d": _group_event(),
                }
            )
        )

        self.assertEqual(len(received), 1)
        message = received[0]
        self.assertTrue(message.mentioned_me)
        self.assertEqual(message.message_type, "group")
        self.assertEqual(message.group_id, GROUP_OPENID)
        self.assertEqual(message.sender_id, MEMBER_OPENID)
        self.assertEqual(message.sender_name, "小明")
        self.assertIn(APP_ID, message.mentioned_user_ids)
        self.assertNotIn(APP_ID, message.mentioned_others)
        self.assertEqual(message.session_id, f"group_{GROUP_OPENID}")
        self.assertEqual(adapter.messages_received, 1)

    async def test_group_full_message_detects_at_by_mention_entry(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {
                    "op": OP_DISPATCH,
                    "s": 13,
                    "t": "GROUP_MESSAGE_CREATE",
                    "d": _group_event(
                        content="<@BOTID> 帮我看看这个",
                        mentions=[
                            {"id": "BOTMEMBER", "member_openid": "BOTMEMBER", "bot": True},
                        ],
                    ),
                }
            )
        )

        message = received[0]
        self.assertTrue(message.mentioned_me)
        self.assertIn(APP_ID, message.mentioned_user_ids)

    async def test_group_full_message_collects_other_mentions(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {
                    "op": OP_DISPATCH,
                    "s": 14,
                    "t": "GROUP_MESSAGE_CREATE",
                    "d": _group_event(
                        content="<@OTHER> 在吗",
                        mentions=[
                            {"id": "OTHER", "member_openid": "OTHER", "bot": False},
                        ],
                    ),
                }
            )
        )

        message = received[0]
        self.assertFalse(message.mentioned_me)
        self.assertIn("OTHER", message.mentioned_user_ids)
        self.assertIn("OTHER", message.mentioned_others)

    async def test_c2c_message_is_private_and_uses_user_openid(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {"op": OP_DISPATCH, "s": 15, "t": "C2C_MESSAGE_CREATE", "d": _c2c_event()}
            )
        )

        message = received[0]
        self.assertEqual(message.message_type, "private")
        self.assertEqual(message.sender_id, USER_OPENID)
        self.assertIsNone(message.group_id)
        self.assertEqual(message.session_id, f"private_{USER_OPENID}")

    async def test_duplicate_message_id_is_ignored(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()
        frame = json.dumps(
            {"op": OP_DISPATCH, "s": 16, "t": "GROUP_AT_MESSAGE_CREATE", "d": _group_event()}
        )

        await adapter._handle_ws_payload(frame)
        await adapter._handle_ws_payload(frame)

        self.assertEqual(len(received), 1)
        self.assertEqual(adapter.messages_received, 1)

    async def test_timestamp_is_parsed_to_unix_seconds(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {"op": OP_DISPATCH, "s": 17, "t": "GROUP_AT_MESSAGE_CREATE", "d": _group_event()}
            )
        )

        self.assertIsNotNone(received[0].timestamp)
        # 2026-07-21T10:00:00+08:00 == 2026-07-21T02:00:00Z
        self.assertAlmostEqual(received[0].timestamp, 1784599200.0, delta=1)

    async def test_attachment_becomes_image_segment_with_url(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {
                    "op": OP_DISPATCH,
                    "s": 18,
                    "t": "GROUP_AT_MESSAGE_CREATE",
                    "d": _group_event(
                        content="看图",
                        attachments=[
                            {
                                "content_type": "image/jpeg",
                                "filename": "photo.jpg",
                                "url": "https://multimedia.nt.qq.com.cn/download?x=1",
                                "width": 1920,
                                "height": 1080,
                                "size": 256000,
                            }
                        ],
                    ),
                }
            )
        )

        message = received[0]
        images = [seg for seg in message.segments if seg.type == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].url, "https://multimedia.nt.qq.com.cn/download?x=1")

    async def test_quoted_bot_message_sets_reply_to_bot(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {
                    "op": OP_DISPATCH,
                    "s": 19,
                    "t": "GROUP_AT_MESSAGE_CREATE",
                    "d": _group_event(
                        message_type=103,
                        content="同意",
                        msg_elements=[
                            {
                                "msg_idx": "REFIDX_bot==",
                                "message_type": 103,
                                "content": "机器人之前说的话",
                                "author": {"id": "BOTMEMBER", "bot": True},
                            }
                        ],
                        message_scene={
                            "source": "default",
                            "ext": [
                                "ref_msg_idx=REFIDX_bot==",
                                "msg_idx=REFIDX_own==",
                            ],
                        },
                    ),
                }
            )
        )

        message = received[0]
        self.assertEqual(message.reply_to_qq, APP_ID)
        self.assertTrue(message.reply_to_id)


class OfficialReplyTargetTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_to_inbound_event_uses_passive_reply_with_seq(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()
        await adapter._handle_ws_payload(
            json.dumps(
                {"op": OP_DISPATCH, "s": 20, "t": "GROUP_AT_MESSAGE_CREATE", "d": _group_event()}
            )
        )

        success, message_id = await adapter.send_message_with_id(
            f"group_{GROUP_OPENID}", "好的", reply_to_id="ROBOT1.0_abc"
        )

        self.assertTrue(success)
        self.assertEqual(message_id, "ROBOT1.0_sent")
        method, path, kwargs = adapter.calls[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(path, f"/v2/groups/{GROUP_OPENID}/messages")
        body = kwargs["json_body"]
        self.assertEqual(body["msg_type"], 0)
        self.assertEqual(body["content"], "好的")
        self.assertEqual(body["msg_id"], "ROBOT1.0_abc")
        self.assertEqual(body["msg_seq"], 1)

    async def test_second_reply_to_same_event_increments_seq(self):
        adapter = _adapter()
        received = []
        adapter.on_message_callback = lambda message: received.append(message) or _noop()
        await adapter._handle_ws_payload(
            json.dumps(
                {"op": OP_DISPATCH, "s": 21, "t": "GROUP_AT_MESSAGE_CREATE", "d": _group_event()}
            )
        )

        await adapter.send_message(f"group_{GROUP_OPENID}", "一", reply_to_id="ROBOT1.0_abc")
        await adapter.send_message(f"group_{GROUP_OPENID}", "二", reply_to_id="ROBOT1.0_abc")

        bodies = [call[2]["json_body"] for call in adapter.calls if call[1].endswith("/messages")]
        self.assertEqual([body["msg_seq"] for body in bodies], [1, 2])

    async def test_reply_to_unknown_id_falls_back_to_message_reference(self):
        adapter = _adapter()
        adapter._ref_idx_by_msg_id["ROBOT1.0_other"] = "REFIDX_other=="

        success, _ = await adapter.send_message_with_id(
            f"group_{GROUP_OPENID}", "好的", reply_to_id="ROBOT1.0_other"
        )

        self.assertTrue(success)
        body = adapter.calls[-1][2]["json_body"]
        self.assertNotIn("msg_id", body)
        self.assertEqual(body["message_reference"], {"message_id": "REFIDX_other=="})

    async def test_reply_to_ref_idx_directly_is_passed_through(self):
        adapter = _adapter()

        await adapter.send_message(
            f"group_{GROUP_OPENID}", "好的", reply_to_id="REFIDX_xxx=="
        )

        body = adapter.calls[-1][2]["json_body"]
        self.assertEqual(body["message_reference"], {"message_id": "REFIDX_xxx=="})

    async def test_no_reply_target_sends_proactive_message(self):
        adapter = _adapter()

        await adapter.send_message(f"group_{GROUP_OPENID}", "日报")

        body = adapter.calls[-1][2]["json_body"]
        self.assertNotIn("msg_id", body)
        self.assertNotIn("message_reference", body)

    async def test_private_session_uses_user_messages_endpoint(self):
        adapter = _adapter()

        await adapter.send_message(f"private_{USER_OPENID}", "在吗")

        method, path, _ = adapter.calls[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(path, f"/v2/users/{USER_OPENID}/messages")


class OfficialRichMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_image_uploads_then_sends_media_message(self):
        adapter = _adapter()
        adapter.responses[("POST", f"/v2/groups/{GROUP_OPENID}/files")] = {
            "file_info": "FILEINFO123",
            "ttl": 300,
        }

        success, message_id = await adapter.send_image_with_id(
            f"group_{GROUP_OPENID}", b"\x89PNG\r\n\x1a\n fake"
        )

        self.assertTrue(success)
        self.assertEqual(message_id, "ROBOT1.0_sent")
        upload_call = [c for c in adapter.calls if c[1].endswith("/files")][0]
        upload_body = upload_call[2]["json_body"]
        self.assertEqual(upload_body["file_type"], 1)
        self.assertFalse(upload_body["srv_send_msg"])
        self.assertIn("file_data", upload_body)

        send_body = adapter.calls[-1][2]["json_body"]
        self.assertEqual(send_body["msg_type"], 7)
        self.assertEqual(send_body["media"], {"file_info": "FILEINFO123"})

    async def test_large_file_uses_chunked_upload(self):
        adapter = _adapter()
        base = f"/v2/groups/{GROUP_OPENID}"
        adapter.responses[("POST", f"{base}/upload_prepare")] = {
            "upload_id": "upload_1",
            "block_size": "5",
            "parts": [{"index": 0, "presigned_url": "https://cos.example/put", "block_size": "5"}],
        }
        adapter.responses[("POST", f"{base}/upload_part_finish")] = {}
        adapter.responses[("POST", f"{base}/files")] = {"file_info": "FILEINFO_CHUNK"}

        put_calls = []

        async def _fake_put(url, blob, timeout=120.0):
            put_calls.append((url, blob))

        adapter._put_bytes = _fake_put
        blob = b"x" * (5 * 1024 * 1024 + 10)

        file_info = await adapter.upload_file(f"group_{GROUP_OPENID}", blob)

        self.assertEqual(file_info, "FILEINFO_CHUNK")
        self.assertEqual(len(put_calls), 1)
        prepare_body = [
            c for c in adapter.calls if c[1].endswith("/upload_prepare")
        ][0][2]["json_body"]
        self.assertEqual(prepare_body["file_size"], str(len(blob)))
        self.assertEqual(len(prepare_body["md5"]), 32)
        self.assertEqual(len(prepare_body["sha1"]), 40)

    async def test_send_images_sends_one_message_per_image(self):
        adapter = _adapter()
        adapter.responses[("POST", f"/v2/groups/{GROUP_OPENID}/files")] = {
            "file_info": "FILEINFO123"
        }

        result = await adapter.send_images(
            f"group_{GROUP_OPENID}", [b"\x89PNG1", b"\x89PNG2"], text="原图"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["sent"], 2)
        self.assertEqual(result["message_count"], 2)
        bodies = [c[2]["json_body"] for c in adapter.calls if c[1].endswith("/messages")]
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]["content"], "原图")
        self.assertNotIn("content", bodies[1])

    async def test_send_images_forward_mode_degrades_to_chunks(self):
        adapter = _adapter()
        adapter.responses[("POST", f"/v2/groups/{GROUP_OPENID}/files")] = {
            "file_info": "FILEINFO123"
        }

        result = await adapter.send_images(
            f"group_{GROUP_OPENID}", [b"\x89PNG1"], mode="forward"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "chunks")


class OfficialDegradationTests(unittest.IsolatedAsyncioTestCase):
    async def test_poke_is_not_supported(self):
        adapter = _adapter()
        self.assertFalse(await adapter.poke(f"group_{GROUP_OPENID}", MEMBER_OPENID))

    async def test_unsupported_onebot_action_returns_none(self):
        adapter = _adapter()
        self.assertIsNone(await adapter.call_api("get_forward_msg", {"message_id": "x"}))
        self.assertIsNone(await adapter.call_api("get_image", {"file": "x"}))

    async def test_get_login_info_returns_canonical_bot_id(self):
        adapter = _adapter()
        data = await adapter.call_api("get_login_info", {})
        self.assertEqual(data["user_id"], APP_ID)

    async def test_member_display_name_returns_empty_without_whitelist(self):
        adapter = _adapter()
        self.assertEqual(await adapter.get_member_display_name(GROUP_OPENID, MEMBER_OPENID), "")

    async def test_fetch_user_avatar_returns_none(self):
        adapter = _adapter()
        self.assertIsNone(await adapter.fetch_user_avatar(MEMBER_OPENID))

    async def test_empty_access_token_does_not_clear_cached_token(self):
        adapter = _adapter()
        adapter.access_token = "TOKEN"
        adapter.access_token = ""
        self.assertEqual(adapter.access_token, "TOKEN")


class OfficialAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_is_fetched_and_cached(self):
        adapter = _adapter()
        calls = []

        async def _fake_request(method, path, **kwargs):
            calls.append((method, path, kwargs))
            if path.endswith("getAppAccessToken"):
                return {"access_token": "TOKEN", "expires_in": 7200}
            return {}

        adapter._request_json = _fake_request

        first = await adapter._ensure_access_token()
        second = await adapter._ensure_access_token()

        self.assertEqual(first, "TOKEN")
        self.assertEqual(second, "TOKEN")
        self.assertEqual(len(calls), 1)
        body = calls[0][2]["json_body"]
        self.assertEqual(body["appId"], APP_ID)
        self.assertEqual(body["clientSecret"], "secret-value")

    async def test_missing_credentials_raise_clear_error(self):
        adapter = OfficialQQAdapter({"self_id": APP_ID, "official": {}})

        with self.assertRaises(RuntimeError) as ctx:
            await adapter._ensure_access_token()
        self.assertIn("app_id", str(ctx.exception))

    async def test_expired_token_is_refreshed(self):
        adapter = _adapter()
        adapter._access_token = "OLD"
        adapter._token_expires_at = 0.0
        calls = []

        async def _fake_request(method, path, **kwargs):
            calls.append(path)
            if path.endswith("getAppAccessToken"):
                return {"access_token": "NEW", "expires_in": 7200}
            return {}

        adapter._request_json = _fake_request
        token = await adapter._ensure_access_token()

        self.assertEqual(token, "NEW")
        self.assertEqual(len(calls), 1)


class OfficialGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_identify_payload_carries_intents_and_qqbot_token(self):
        adapter = _adapter()
        sent = []

        class _FakeWS:
            async def send(self, payload):
                sent.append(payload)

        await adapter._identify_or_resume(_FakeWS(), "TOKEN")

        payload = json.loads(sent[0])
        self.assertEqual(payload["op"], 2)
        self.assertEqual(payload["d"]["token"], "QQBot TOKEN")
        self.assertEqual(payload["d"]["intents"], INTENT_GROUP_AND_C2C)
        self.assertEqual(payload["d"]["shard"], [0, 1])

    async def test_resume_payload_used_when_session_known(self):
        adapter = _adapter()
        adapter._gateway_session_id = "sess-1"
        adapter._last_seq = 42
        sent = []

        class _FakeWS:
            async def send(self, payload):
                sent.append(payload)

        await adapter._identify_or_resume(_FakeWS(), "TOKEN")

        payload = json.loads(sent[0])
        self.assertEqual(payload["op"], 6)
        self.assertEqual(payload["d"]["session_id"], "sess-1")
        self.assertEqual(payload["d"]["seq"], 42)

    async def test_ready_event_records_session_and_bot_id(self):
        adapter = _adapter()

        await adapter._handle_ws_payload(
            json.dumps(
                {
                    "op": OP_DISPATCH,
                    "s": 1,
                    "t": "READY",
                    "d": {
                        "version": 1,
                        "session_id": "sess-1",
                        "user": {"id": "6158788878435714165", "username": "机器人", "bot": True},
                    },
                }
            )
        )

        self.assertEqual(adapter._gateway_session_id, "sess-1")
        self.assertIn("6158788878435714165", adapter._bot_ids)

    async def test_hello_opcode_is_ignored(self):
        adapter = _adapter()
        await adapter._handle_ws_payload(
            json.dumps({"op": OP_HELLO, "d": {"heartbeat_interval": 45000}})
        )
        self.assertEqual(adapter.messages_received, 0)

    async def test_last_seq_advances_only_after_dispatch(self):
        adapter = _adapter()
        adapter.on_message_callback = lambda message: _noop()

        await adapter._handle_ws_payload(
            json.dumps(
                {"op": OP_DISPATCH, "s": 77, "t": "GROUP_AT_MESSAGE_CREATE", "d": _group_event()}
            )
        )

        self.assertEqual(adapter._last_seq, 77)
        self.assertEqual(adapter._latest_s, 77)


async def _noop():
    return None


class OfficialPlatformSwitchTests(unittest.TestCase):
    def test_platform_official_selects_official_adapter(self):
        from main import GroupChatBot

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.config = {"qq": {"platform": "official", "self_id": APP_ID}}
        bot._get_rich_media_config = lambda: {}
        bot._init_vision_provider = lambda: None
        bot._init_poke_responder = lambda: None
        bot._handle_message = None
        bot._handle_notice = None

        bot._init_qq_adapter()

        self.assertIsInstance(bot.qq_adapter, OfficialQQAdapter)

    def test_platform_default_selects_napcat_adapter(self):
        from core.adapter.qq_adapter import QQAdapter
        from main import GroupChatBot

        bot = GroupChatBot.__new__(GroupChatBot)
        bot.config = {"qq": {"self_id": "1"}}
        bot._get_rich_media_config = lambda: {}
        bot._init_vision_provider = lambda: None
        bot._init_poke_responder = lambda: None
        bot._handle_message = None
        bot._handle_notice = None

        bot._init_qq_adapter()

        self.assertIsInstance(bot.qq_adapter, QQAdapter)


if __name__ == "__main__":
    unittest.main()

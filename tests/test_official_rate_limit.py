"""官方通道发送端频控的测试。

对应官方规则：Bot 维度 60/qpm（未认证 30/qpm）、单关系 20/qpm、
每群每天 1000 条。这里按更严的本地阈值验证行为。
"""

import asyncio
import json
import unittest

from core.adapter.official_adapter import (
    OfficialQQAdapter,
    OfficialAPIError,
    is_rate_limit_code,
)
from core.adapter.rate_limit import RateLimitedError, SendRateLimiter


GROUP_OPENID = "B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5"
APP_ID = "1020899999"
SESSION = f"group_{GROUP_OPENID}"


def _adapter(**overrides):
    config = {
        "self_id": APP_ID,
        "official": {"app_id": APP_ID, "app_secret": "secret-value"},
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
            return {"id": "ROBOT1.0_sent"}
        return {}

    adapter._request_json = _fake_request
    adapter._connected = True
    return adapter


class SendRateLimiterTests(unittest.IsolatedAsyncioTestCase):
    """限速器本身。"""

    async def test_allows_until_bot_limit(self):
        limiter = SendRateLimiter(bot_per_minute=3, session_per_minute=10)

        for _ in range(3):
            await limiter.acquire(SESSION, passive=True)

        self.assertEqual(limiter.stats()["bot_recent_per_minute"], 3)

    async def test_blocks_after_bot_limit(self):
        limiter = SendRateLimiter(bot_per_minute=2, session_per_minute=10)

        await limiter.acquire(SESSION, passive=True)
        await limiter.acquire(SESSION, passive=True)

        with self.assertRaises(RateLimitedError):
            await limiter.acquire(SESSION, passive=True)

        self.assertEqual(limiter.stats()["blocked"], 1)

    async def test_session_limit_is_separate_from_bot_limit(self):
        limiter = SendRateLimiter(bot_per_minute=100, session_per_minute=2)
        other = "group_OTHER"

        await limiter.acquire(SESSION, passive=True)
        await limiter.acquire(SESSION, passive=True)

        with self.assertRaises(RateLimitedError):
            await limiter.acquire(SESSION, passive=True)
        # 另一个会话不受影响
        await limiter.acquire(other, passive=True)

    async def test_active_message_is_skipped_instead_of_waiting(self):
        limiter = SendRateLimiter(bot_per_minute=1, session_per_minute=1)
        await limiter.acquire(SESSION, passive=True)

        with self.assertRaises(RateLimitedError) as ctx:
            await limiter.acquire(SESSION, passive=False)

        self.assertIn("不排队", str(ctx.exception))

    async def test_daily_fuse_trips(self):
        limiter = SendRateLimiter(
            bot_per_minute=1000, session_per_minute=1000, daily_per_session=2
        )
        await limiter.acquire(SESSION, passive=True)
        await limiter.acquire(SESSION, passive=True)

        with self.assertRaises(RateLimitedError) as ctx:
            await limiter.acquire(SESSION, passive=True)

        self.assertIn("上限", str(ctx.exception))

    async def test_official_limit_is_recorded(self):
        limiter = SendRateLimiter()
        limiter.note_official_limit(1100100, "消息被限频")

        stats = limiter.stats()
        self.assertEqual(stats["official_limited"], 1)
        self.assertEqual(stats["official_limit_codes"]["1100100"], 1)


class RateLimitErrorCodeTests(unittest.TestCase):
    """官方限频错误码的识别。"""

    def test_known_codes(self):
        self.assertTrue(is_rate_limit_code(1100100))
        self.assertTrue(is_rate_limit_code(40034100))
        self.assertTrue(is_rate_limit_code(40054013))
        self.assertTrue(is_rate_limit_code(504001))
        self.assertTrue(is_rate_limit_code("1100100"))

    def test_unrelated_codes(self):
        self.assertFalse(is_rate_limit_code(22006))
        self.assertFalse(is_rate_limit_code(11253))
        self.assertFalse(is_rate_limit_code(None))
        self.assertFalse(is_rate_limit_code("abc"))

    def test_api_error_carries_code(self):
        exc = OfficialAPIError("限频了", 1100100)
        self.assertTrue(exc.is_rate_limit)
        self.assertIsInstance(exc, RuntimeError)


class AdapterRateLimitTests(unittest.IsolatedAsyncioTestCase):
    """适配器接上限速器之后的行为。"""

    async def test_send_is_blocked_when_limit_reached(self):
        adapter = _adapter()
        adapter.rate_limiter = SendRateLimiter(bot_per_minute=1, session_per_minute=1)

        ok = await adapter.send_message(SESSION, "第一条")
        self.assertTrue(ok)

        ok = await adapter.send_message(SESSION, "第二条")
        self.assertFalse(ok)
        # 被拦下的这条根本没发出去
        self.assertEqual(len(adapter.calls), 1)

    async def test_official_rate_limit_error_is_logged_and_counted(self):
        adapter = _adapter()
        adapter.responses[("POST", f"/v2/groups/{GROUP_OPENID}/messages")] = OfficialAPIError(
            "官方接口 POST 失败：1100100 安全打击:消息被限频", 1100100
        )

        ok = await adapter.send_message(SESSION, "会被拒")

        self.assertFalse(ok)
        self.assertEqual(adapter.rate_limiter.stats()["official_limited"], 1)
        self.assertIn("1100100", adapter.rate_limiter.stats()["official_limit_codes"])

    async def test_stats_are_exposed_for_dashboard(self):
        adapter = _adapter()
        await adapter.send_message(SESSION, "hello")

        stats = adapter.rate_limiter.stats()
        self.assertEqual(stats["bot_recent_per_minute"], 1)
        self.assertIn(SESSION, stats["sessions"])
        self.assertEqual(stats["sessions"][SESSION]["daily"], 1)


if __name__ == "__main__":
    unittest.main()

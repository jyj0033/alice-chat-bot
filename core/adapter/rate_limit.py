"""官方通道的发送端频控。

官方规则（<https://bot.q.qq.com/wiki/develop/api-v2/>）：

* **Bot 维度（发送方）**：企业认证 / 个人身份证认证 60/qpm，未认证 30/qpm
* **单关系维度（接收方）**：20/qpm；每个群 1 天最多接收 1000 条
* **被动消息**：5 分钟有效，每条消息最多回 5 次（这条由适配器按 msg_id +
  msg_seq 处理，见 ``official_adapter._resolve_reply_target``）

这里按**未认证**口径再留两成余量，自己卡得更严：Bot 25/分钟、单关系
15/分钟、单会话每天 900 条。

**为什么要自己卡**：触顶之后官方直接拒收，那条消息就没了 —— 群里看着
就是「它不回话」，而日志里只有一句「发送消息失败」，很难联想到限频。
自己先卡住，至少能主动跳过、并留下能查的记录。

两种发送要区别对待：

* **被动回复**（有人在等它回话，带 msg_id）：额度不够就排队等一下，最多
  等 ``wait_seconds`` 秒；等不到就放弃这一条（宁可少回一句，也别把整条
  回复链路卡死）。
* **主动消息**（插话、群日报、表情包，不带 msg_id）：额度不够直接跳过，
  不排队 —— 主动发言晚几秒没人等，但卡住会拖慢主流程。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitedError(RuntimeError):
    """这条消息被频控拦下了（本地限速，或官方拒收）。"""


class SendRateLimiter:
    """按官方频控规则限制发送速度。

    阈值比官方更严一点，给「未认证」这个最坏口径留余量。
    """

    def __init__(
        self,
        bot_per_minute: int = 25,
        session_per_minute: int = 15,
        daily_per_session: int = 900,
        wait_seconds: float = 5.0,
    ):
        self.bot_per_minute = max(1, int(bot_per_minute))
        self.session_per_minute = max(1, int(session_per_minute))
        self.daily_per_session = max(1, int(daily_per_session))
        self.wait_seconds = max(0.0, float(wait_seconds))

        # Bot 维度：最近一分钟的发送时间戳
        self._bot_sends: deque[float] = deque()
        # 单关系维度：每个会话最近一分钟的发送时间戳
        self._session_sends: dict[str, deque[float]] = {}
        # 每日条数：会话 -> (日期, 条数)
        self._daily: dict[str, tuple[str, int]] = {}

        # 统计
        self.blocked_count = 0
        self.official_limit_count = 0
        self.official_limit_codes: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 内部计算
    # ------------------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    def _bot_wait(self, now: float) -> float:
        """Bot 维度还要等多久才能发。"""
        times = self._bot_sends
        while times and now - times[0] >= 60.0:
            times.popleft()
        if len(times) < self.bot_per_minute:
            return 0.0
        return max(0.0, (times[0] + 60.0) - now)

    def _session_wait(self, session_id: str, now: float) -> float:
        """单关系维度还要等多久才能发。"""
        times = self._session_sends.get(session_id)
        if not times:
            return 0.0
        while times and now - times[0] >= 60.0:
            times.popleft()
        if len(times) < self.session_per_minute:
            return 0.0
        return max(0.0, (times[0] + 60.0) - now)

    def _daily_count(self, session_id: str) -> int:
        day, count = self._daily.get(session_id, (self._today(), 0))
        if day != self._today():
            return 0
        return count

    def _record(self, session_id: str, now: float) -> None:
        self._bot_sends.append(now)
        self._session_sends.setdefault(session_id, deque()).append(now)
        today = self._today()
        day, count = self._daily.get(session_id, (today, 0))
        if day != today:
            day, count = today, 0
        self._daily[session_id] = (day, count + 1)

    # ------------------------------------------------------------------
    # 对外
    # ------------------------------------------------------------------

    async def acquire(self, session_id: str, *, passive: bool) -> None:
        """申请一次发送额度；不够就抛 :class:`RateLimitedError`。

        ``passive=True`` 表示这是被动回复（有人在等），额度不够时允许排队
        等一小会儿；``passive=False`` 是主动消息，不排队，直接跳过。
        """
        now = time.time()
        daily = self._daily_count(session_id)
        if daily >= self.daily_per_session:
            self.blocked_count += 1
            raise RateLimitedError(
                f"今日向该会话发送已达上限（{daily}/{self.daily_per_session} 条）"
            )

        wait = max(self._bot_wait(now), self._session_wait(session_id, now))
        if wait > 0:
            if not passive:
                self.blocked_count += 1
                raise RateLimitedError(
                    f"触发官方频控，需要等待 {wait:.1f} 秒（主动消息不排队，直接跳过）"
                )
            if wait > self.wait_seconds:
                self.blocked_count += 1
                raise RateLimitedError(
                    f"触发官方频控且等待过久（需 {wait:.1f} 秒 > {self.wait_seconds} 秒）"
                )
            await asyncio.sleep(wait)
            # 睡醒可能已经被别的任务抢走额度，再确认一次。
            now = time.time()
            wait = max(self._bot_wait(now), self._session_wait(session_id, now))
            if wait > 0.5:
                self.blocked_count += 1
                raise RateLimitedError(f"等待后额度仍不足（还需 {wait:.1f} 秒）")

        self._record(session_id, now)

    def note_official_limit(self, err_code: Any, message: str = "") -> None:
        """记一笔「被官方拒收」。"""
        self.official_limit_count += 1
        key = str(err_code)
        self.official_limit_codes[key] = self.official_limit_codes.get(key, 0) + 1

    def stats(self) -> dict[str, Any]:
        """给面板看的快照。"""
        now = time.time()
        bot_recent = sum(1 for ts in self._bot_sends if now - ts < 60.0)
        sessions = {}
        for session_id, times in self._session_sends.items():
            recent = sum(1 for ts in times if now - ts < 60.0)
            if recent or self._daily.get(session_id):
                sessions[session_id] = {
                    "recent_per_minute": recent,
                    "daily": self._daily_count(session_id),
                }
        return {
            "bot_recent_per_minute": bot_recent,
            "bot_limit_per_minute": self.bot_per_minute,
            "session_limit_per_minute": self.session_per_minute,
            "daily_limit_per_session": self.daily_per_session,
            "blocked": self.blocked_count,
            "official_limited": self.official_limit_count,
            "official_limit_codes": dict(self.official_limit_codes),
            "sessions": sessions,
        }

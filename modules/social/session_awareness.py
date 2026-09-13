"""同一人格跨会话的自觉：被赶闭嘴、同一句话不在两个群各回一遍。"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SILENCE_RE = re.compile(
    r"(出去|封印|闭嘴|别说话|别瞎说|一边玩|别回(?:了)?|别插嘴|别接话|安静)"
)
_WS_RE = re.compile(r"\s+")


@dataclass
class _Claim:
    session_id: str
    ts: float
    sent: bool = False


class SessionAwareness:
    """把多个群当成同一个自己。

    1. 群友明确赶人/封印后，本群先停掉插话，只保留明确 @/引用。
    2. 同一人把同一段话同步发到两个群时，只回其中一个。
    """

    def __init__(self, mute_seconds: float = 1800.0, utterance_window: float = 120.0):
        self.mute_seconds = mute_seconds
        self.utterance_window = utterance_window
        self._mute_until: dict[str, float] = {}
        self._claims: dict[str, _Claim] = {}

    @staticmethod
    def fingerprint(sender_id: str, text: str) -> str:
        compact = _WS_RE.sub("", (text or "").strip())[:80]
        return f"{sender_id}:{compact}"

    def mute(self, session_id: str, seconds: float | None = None) -> None:
        until = time.time() + (self.mute_seconds if seconds is None else seconds)
        self._mute_until[session_id] = until
        logger.info("[社交] %s 进入闭嘴，%.0f 秒", session_id, until - time.time())

    def is_muted(self, session_id: str) -> bool:
        until = self._mute_until.get(session_id, 0.0)
        if until <= time.time():
            self._mute_until.pop(session_id, None)
            return False
        return True

    def is_silence_request(
        self,
        text: str,
        *,
        bot_names: list[str],
        mentioned_me: bool = False,
        reply_to_me: bool = False,
    ) -> bool:
        content = text or ""
        if not _SILENCE_RE.search(content):
            return False
        if mentioned_me or reply_to_me:
            return True
        return any(name and name in content for name in bot_names if name)

    def claim_utterance(self, session_id: str, sender_id: str, text: str) -> bool:
        """跨群抢占同一句话。已被其他群占用则返回 False。"""
        compact = _WS_RE.sub("", (text or "").strip())
        if len(compact) < 8:
            return True
        key = self.fingerprint(sender_id, text)
        now = time.time()
        existing = self._claims.get(key)
        if existing and now - existing.ts <= self.utterance_window:
            if existing.session_id != session_id:
                return False
            existing.ts = now
            return True
        self._claims[key] = _Claim(session_id=session_id, ts=now)
        if len(self._claims) > 400:
            cutoff = now - self.utterance_window * 2
            self._claims = {
                k: v for k, v in self._claims.items() if v.ts >= cutoff
            }
        return True

    def mark_sent(self, session_id: str, sender_id: str, text: str) -> None:
        key = self.fingerprint(sender_id, text)
        claim = self._claims.get(key)
        if claim and claim.session_id == session_id:
            claim.sent = True
            claim.ts = time.time()
        else:
            self._claims[key] = _Claim(session_id=session_id, ts=time.time(), sent=True)

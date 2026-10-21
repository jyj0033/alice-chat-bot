"""引用回复策略：这条回复该不该带引用。

背景：以前发送路径只管「解析出引用目标」，解析出来就一定引用，于是爱丽丝在
群里每条回复都挂着引用，看着像机器人而不是群友。本模块把「要不要引用」从
发送路径里抽出来，按场景判断：

  1. 对方明确在跟爱丽丝说话（@ / 回复）→ 必须引用。否则对方不知道她在回哪
     一句，消息会被当成自言自语。私聊也走这条。
  2. 群聊插话 → 默认不引用。只有两种情况值得引用：
       · 隔太久（stale）：对方可能已经忘了、或话题已经跳走，引用能定位；
       · 群里刷太快（busy）：多个人同时在说，指代容易糊，引用能消歧。
  3. 紧邻的即时对话不引用 —— 真人不会每条都引用。

一对一连发不算「刷屏」：一个人连发 5 条和 5 个人各发 1 条，对「爱丽丝在回
谁」的指代难度完全不同，所以 busy 条件同时看条数和说话人数。

阈值全部走配置，不写死在代码里。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

# 决策原因（会写进日志和记忆 metadata，便于事后统计）
REASON_NO_TARGET = "no_target"    # 没有可引用的目标
REASON_DISABLED = "disabled"      # 引用总开关关闭
REASON_ALWAYS = "always"          # 强制每次都引用（旧行为）
REASON_DIRECTED = "directed"      # 对方明确在对爱丽丝说话
REASON_STALE = "stale"            # 隔太久，需要定位
REASON_BUSY = "busy"              # 群消息太密，需要消歧
REASON_INLINE = "inline"          # 紧邻的即时对话，不引用


@dataclass(frozen=True)
class QuoteDecision:
    """一次引用决策的结果。"""

    quote: bool
    reason: str
    target_id: str = ""


def collect_busy_stats(
    messages,
    *,
    now: datetime | None = None,
    window_seconds: int = 30,
) -> tuple[int, int]:
    """统计最近 window_seconds 内的非 bot 消息条数与不同说话人数。

    messages 只需要有 .timestamp / .is_bot / .sender_id 三个属性，
    这里用鸭子类型，避免和具体的消息类耦合。
    """
    now = now or datetime.now()
    cutoff = now - timedelta(seconds=max(0, int(window_seconds or 0)))
    count = 0
    speakers = set()
    for message in messages or []:
        if getattr(message, "is_bot", False):
            continue
        timestamp = getattr(message, "timestamp", None)
        if not isinstance(timestamp, datetime) or timestamp <= cutoff:
            continue
        count += 1
        speaker = str(getattr(message, "sender_id", "") or "")
        if speaker:
            speakers.add(speaker)
    return count, len(speakers)


def decide_quote(
    direction: str,
    target_id: str,
    *,
    enabled: bool = True,
    always: bool = False,
    target_age_seconds: float | None = None,
    recent_message_count: int = 0,
    recent_speaker_count: int = 0,
    stale_seconds: int = 60,
    busy_message_count: int = 5,
    busy_speaker_count: int = 2,
) -> QuoteDecision:
    """决定这条回复要不要带引用。

    Args:
        direction: "to_bot"（对方 @ 或回复了爱丽丝）/ "group"（群聊插话）
        target_id: 候选引用目标的消息 id，空串表示没有目标
        enabled: 引用总开关；False 表示完全不使用引用
        always: 强制每次都引用（等价于改动前的行为，用于对照）
        target_age_seconds: 目标消息距今多少秒；None 表示目标已经不在上下文
            窗口里（必然隔了很久，按 stale 处理）
        recent_message_count: 最近窗口内的非 bot 消息条数
        recent_speaker_count: 最近窗口内的不同说话人数
        stale_seconds: 超过多少秒算「隔太久」
        busy_message_count: 窗口内多少条消息算「刷屏」
        busy_speaker_count: 窗口内多少个不同说话人才算「刷屏」
    """
    target_id = str(target_id or "")
    if not target_id:
        return QuoteDecision(False, REASON_NO_TARGET, target_id)
    if not enabled:
        return QuoteDecision(False, REASON_DISABLED, target_id)
    if always:
        return QuoteDecision(True, REASON_ALWAYS, target_id)
    if direction == "to_bot":
        return QuoteDecision(True, REASON_DIRECTED, target_id)
    if target_age_seconds is None:
        # 目标已经滑出上下文窗口 → 必然隔了很久，引用帮对方定位
        return QuoteDecision(True, REASON_STALE, target_id)
    if target_age_seconds >= max(0, int(stale_seconds or 0)):
        return QuoteDecision(True, REASON_STALE, target_id)
    if (
        recent_message_count >= max(0, int(busy_message_count or 0))
        and recent_speaker_count >= max(0, int(busy_speaker_count or 0))
    ):
        return QuoteDecision(True, REASON_BUSY, target_id)
    return QuoteDecision(False, REASON_INLINE, target_id)

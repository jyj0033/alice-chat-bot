"""群聊学习：通用的说话风格数值 + 各群的称呼 / 口头禅 / 梗。

设计边界：
- **只学"怎么说"**，不碰人格自述（who you are）。
- 说话风格数值是**全群共用的一份**，和人格脱钩：它只由聊天统计决定，
  不从人格的 speaking_style 推导，也不因为人格怎么配而跟着变。
- 称呼、口头禅、梗天然是每个群自己的，按群存。
- 称呼和口头禅由模型提炼，但必须通过"至少出现两次"的字面校验，
  挡住模型编造出来的词。
- 风格存 ``config/learned/speaking_style.json``，群内习惯存
  ``config/learned/<group_id>.json``。
"""

from __future__ import annotations

import json
import os
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

LEARNED_DIRNAME = "learned"
UNIVERSAL_STYLE_FILENAME = "speaking_style.json"

# 样本少于这个数就不学：中位句长在几十条内抖得厉害。
MIN_STYLE_SAMPLES = 20

# 没有任何统计倾向时使用的通用锚点。刻意不是从人格里读的 —— 换了人格，
# 这套数值也不该跟着漂。
DEFAULT_ENTHUSIASM = 0.6
DEFAULT_FORMALITY = 0.3

_FACE_RE = re.compile(r"<faceType=[^>]*>")
_BRACKET_RE = re.compile(r"\[[^\]]{0,24}\]")
_URL_RE = re.compile(r"https?://\S+")
_PLACEHOLDER_PREFIXES = (
    "一张图，看不清画面",
    "一段视频，看不清画面",
    "一张图，",
    "一段视频，",
    "卡片消息",
    "[无法识别的消息]",
)


def learned_path(config_path: str | Path, group_id: str) -> Path:
    """按群取学习结果文件路径。"""
    base = Path(str(config_path)).resolve().parent
    return base / LEARNED_DIRNAME / f"{group_id}.json"


def universal_style_path(config_path: str | Path) -> Path:
    """通用的说话风格数值文件（全群共用一份）。"""
    base = Path(str(config_path)).resolve().parent
    return base / LEARNED_DIRNAME / UNIVERSAL_STYLE_FILENAME


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_universal_style(config_path: str | Path) -> dict[str, Any]:
    """读取通用风格数值；文件不存在或损坏都返回空 dict。"""
    path = universal_style_path(config_path)
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    style = data.get("style") if isinstance(data, dict) else None
    return style if isinstance(style, dict) else {}


def save_universal_style(
    config_path: str | Path, style: dict[str, Any], stats: dict[str, Any] | None = None
) -> dict[str, Any]:
    """整体覆盖写通用风格数值（每轮都是完整快照，不做合并）。"""
    payload = {
        "style": style,
        "stats": stats or {},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        _write_json(universal_style_path(config_path), payload)
    except Exception:
        pass
    return payload


def load_learned(config_path: str | Path, group_id: str) -> dict[str, Any]:
    """读取某个群已学到的内容；文件不存在或损坏都返回空 dict。"""
    path = learned_path(config_path, group_id)
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_all_learned(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """读取全部群的学习结果，键是 group_id。

    通用风格文件放在同一个目录里，但它不是群，必须排除，否则会被当成
    一个叫 ``speaking_style`` 的群。
    """
    base = Path(str(config_path)).resolve().parent / LEARNED_DIRNAME
    result: dict[str, dict[str, Any]] = {}
    if not base.is_dir():
        return result
    for item in base.glob("*.json"):
        if item.name == UNIVERSAL_STYLE_FILENAME:
            continue
        try:
            data = json.loads(item.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            result[item.stem] = data
    return result


def save_learned(
    config_path: str | Path,
    group_id: str,
    patch: dict[str, Any],
    replace: bool = False,
) -> dict[str, Any]:
    """把补丁按顶层键合并进该群的学习文件，返回合并后的内容。

    ``replace=True`` 时这些键整体替换而不是合并 —— 风格数值和群内习惯
    每一轮都是完整快照，用合并的话这一轮没学到的旧口头禅会永远留在文件
    里，越攒越多。
    """
    path = learned_path(config_path, group_id)
    current = load_learned(config_path, group_id)
    for key, value in (patch or {}).items():
        if (
            not replace
            and isinstance(value, dict)
            and isinstance(current.get(key), dict)
        ):
            current[key] = {**current[key], **value}
        else:
            current[key] = value
    current["updated_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        _write_json(path, current)
    except Exception:
        return current
    return current


def drop_learned_keys(
    config_path: str | Path, group_id: str, keys: Iterable[str]
) -> dict[str, Any]:
    """删掉某个群学习文件里的指定顶层键，返回删完的内容。

    风格数值已经改成全群共用一份了；老文件里按群存的 ``style`` / ``stats``
    留着只会误导后来看文件的人，学到一轮就顺手清掉。
    """
    path = learned_path(config_path, group_id)
    current = load_learned(config_path, group_id)
    if not current:
        return {}
    changed = False
    for key in keys:
        if key in current:
            current.pop(key, None)
            changed = True
    if changed:
        try:
            _write_json(path, current)
        except Exception:
            pass
    return current


def clean_message_text(raw: str) -> str:
    """去掉表情码、图片/卡片占位、链接，只留可用于统计的正文。"""
    text = str(raw or "")
    if not text.strip():
        return ""
    if text.strip().startswith(_PLACEHOLDER_PREFIXES):
        return ""
    text = _FACE_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _BRACKET_RE.sub(" ", text)
    return text.strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# 这些词在哪个群都高频出现，但学来没有任何区分度，塞进提示词只会让
# 回复变成"你好、哈哈、谢谢"的复读机。
GENERIC_PHRASES = {
    "你好", "您好", "大家好", "哈喽", "hello",
    "哈哈", "哈哈哈", "哈哈哈哈", "呵呵", "嘿嘿", "笑死",
    "谢谢", "多谢", "感谢", "辛苦了", "收到", "好的", "好滴",
    "在吗", "在不在", "有人吗", "晚安", "早上好", "早安", "拜拜",
    "嗯", "嗯嗯", "哦", "哦哦", "行", "行吧", "可以",
}


# 这些是 QQ 的系统动作提示，不是群友说的话，学进去毫无意义。
SYSTEM_PHRASES = (
    "戳一戳",
    "拍了拍",
    "撤回了一条消息",
    "加入了群聊",
    "邀请了",
    "群公告",
    "领取了",
    "红包",
    "转账",
    "语音通话",
)

# 至少要有一个汉字或字母，纯标点（"？？？"、"……"）不算表达。
_MEANINGFUL_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


def _has_meaningful_chars(term: str) -> bool:
    return bool(_MEANINGFUL_RE.search(term or ""))


def filter_catchphrases(
    candidates: Iterable[str], ban: Iterable[str] = ()
) -> list[str]:
    """去掉零区分度的通用词、纯标点和系统提示语。"""
    banned = {str(item).strip() for item in ban if str(item).strip()}
    kept: list[str] = []
    for candidate in candidates:
        term = str(candidate or "").strip()
        if not term or term in kept or term in banned:
            continue
        if term.casefold() in GENERIC_PHRASES or term in GENERIC_PHRASES:
            continue
        if not _has_meaningful_chars(term):
            continue
        if any(marker in term for marker in SYSTEM_PHRASES):
            continue
        kept.append(term)
    return kept


def compute_style_stats(texts: Iterable[str]) -> dict[str, Any]:
    """统计一批真人消息的长度与标点习惯。

    只统计纯文字消息：图片描述、卡片、链接、极短单字都不参与长度中位数，
    否则"一张图，画面是……"这类视觉摘要会把句长整体拉长。
    """
    rows: list[str] = []
    face_count = 0
    raw_count = 0
    for raw in texts:
        raw_count += 1
        if "<faceType=" in str(raw or ""):
            face_count += 1
        cleaned = clean_message_text(raw)
        if len(cleaned) >= 2:
            rows.append(cleaned)
    if not rows:
        return {"sample_size": 0}

    lengths = [len(row) for row in rows]
    total = len(rows)
    ellipsis = sum(1 for row in rows if "…" in row or "..." in row)
    exclaim = sum(1 for row in rows if "！" in row or "!" in row)
    question = sum(1 for row in rows if "？" in row or "?" in row)

    return {
        "sample_size": total,
        "raw_messages": raw_count,
        "median_len": int(statistics.median(lengths)),
        "mean_len": round(sum(lengths) / total, 1),
        "ellipsis_rate": round(ellipsis / total, 3),
        "exclaim_rate": round(exclaim / total, 3),
        "question_rate": round(question / total, 3),
        "face_rate": round(face_count / max(1, raw_count), 3),
    }


def style_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """把统计结果翻译成通用的说话风格字段。

    这套数值不跟人格挂钩：既不读人格的 ``speaking_style`` 当基线，也不
    反过来改人格。界限写死在函数里，人格怎么换都不影响结果的可复现性。

    取值做的是绝对限幅，不是"相对人格浮动多少"：宁可比真人稍微文静一点，
    也不要让机器人在一次统计抖动后突然变成话痨或冷脸。
    """
    if not stats or stats.get("sample_size", 0) < MIN_STYLE_SAMPLES:
        return {}

    median_len = int(stats.get("median_len") or 20)
    ellipsis_rate = float(stats.get("ellipsis_rate") or 0.0)
    exclaim_rate = float(stats.get("exclaim_rate") or 0.0)
    face_rate = float(stats.get("face_rate") or 0.0)

    # 回复长度跟着群里中位句长走，但落在 14-45 之间：低于 14 会碎成
    # 电报体，高于 45 在群里就是刷屏。
    reply_len = int(_clamp(median_len + 4, 14, 45))

    # 感叹号和表情用得越多，说明这群说话越外放；句号/正式度则相反。
    enthusiasm = DEFAULT_ENTHUSIASM + (exclaim_rate - 0.08) * 0.5 + (face_rate - 0.15) * 0.3
    enthusiasm = round(_clamp(enthusiasm, 0.3, 0.85), 2)

    formality = DEFAULT_FORMALITY - (face_rate - 0.15) * 0.4 - (exclaim_rate - 0.08) * 0.2
    formality = round(_clamp(formality, 0.05, 0.7), 2)

    return {
        "max_reply_length": reply_len,
        "use_ellipsis": ellipsis_rate >= 0.03,
        "ellipsis_frequency": round(_clamp(ellipsis_rate * 0.8, 0.02, 0.18), 3),
        "use_exclamation": exclaim_rate >= 0.10,
        "enthusiasm": enthusiasm,
        "formality": formality,
    }


def verify_candidates(candidates: Iterable[str], texts: Iterable[str], min_hits: int = 2) -> list[str]:
    """字面校验：候选词至少在 min_hits 条消息里真实出现过，才留下。

    模型整理口头禅时很容易顺手"加工"出没人说过的词，这一步把它挡掉。
    """
    corpus = [str(text or "") for text in texts if str(text or "").strip()]
    kept: list[str] = []
    for candidate in candidates:
        term = str(candidate or "").strip()
        if not term or len(term) > 12 or term in kept:
            continue
        hits = sum(1 for text in corpus if term in text)
        if hits >= min_hits:
            kept.append(term)
    return kept


def recent_messages_by_session(
    memories: Iterable[Any], session_id: str, hours: int = 168, limit: int = 800
) -> list[dict[str, str]]:
    """从记忆里筛出某群近期真人消息，返回 ``{"name","content"}`` 列表。

    排除 Bot 自己的发言和「一张图，看不清画面」这类占位：它们既不代表
    群友的说话习惯，也会把句长统计带偏。
    """
    since = datetime.now() - timedelta(hours=max(1, int(hours)))
    rows: list[tuple[Any, str, str]] = []
    for memory in memories:
        if getattr(memory, "source_session", "") != session_id:
            continue
        meta = getattr(memory, "metadata", None) or {}
        if meta.get("is_bot") or meta.get("profile_context_only"):
            continue
        created = getattr(memory, "created_at", None)
        if created is not None and created <= since:
            continue
        content = str(getattr(memory, "content", "") or "").strip()
        if not content:
            continue
        name = str(meta.get("sender_name") or "").strip()
        rows.append((created, name, content))
    rows.sort(key=lambda item: item[0] or datetime.min)
    return [
        {"name": name, "content": content}
        for _, name, content in rows[-limit:]
    ]


def recent_text_by_session(
    memories: Iterable[Any], session_id: str, hours: int = 168, limit: int = 800
) -> list[str]:
    """同 :func:`recent_messages_by_session`，只要正文。"""
    return [
        row["content"]
        for row in recent_messages_by_session(
            memories, session_id, hours=hours, limit=limit
        )
    ]


def sanitize_nicknames(
    candidates: Iterable[str],
    texts: Iterable[str],
    known_names: Iterable[str] = (),
    member_names: Iterable[str] = (),
    min_hits: int = 2,
) -> list[str]:
    """筛出「群友确实这么叫 Bot」的称呼。

    三道关：字面出现过至少 ``min_hits`` 次、不是群友自己的名字（否则会把
    某个人的外号当成 Bot 的称呼）、不是 Bot 本来就叫的名字（那一条没有
    任何信息量）。已知名字的派生形式（如「丽丝」「小艾姐」）会保留。

    校验前先把已知名字从语料里抹掉：否则满屏「爱丽丝」会让「丽丝」白捡
    几十次命中，看起来像群友单独这么叫，其实一次都没有。
    """
    known = [str(item).strip() for item in known_names if str(item).strip()]
    members = {str(item).strip().casefold() for item in member_names if str(item).strip()}
    corpus = [str(text or "") for text in texts if str(text or "").strip()]
    for name in sorted(known, key=len, reverse=True):
        corpus = [text.replace(name, " ") for text in corpus]

    kept: list[str] = []
    for candidate in verify_candidates(candidates, corpus, min_hits=min_hits):
        key = candidate.casefold()
        if key in members or candidate in known or candidate in kept:
            continue
        if candidate.isdigit() or not _has_meaningful_chars(candidate):
            continue
        if any(marker in candidate for marker in SYSTEM_PHRASES):
            continue
        kept.append(candidate)
        if len(kept) >= 6:
            break
    return kept

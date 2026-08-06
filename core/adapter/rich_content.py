"""OneBot 富媒体消息的结构化解析与安全语义渲染。

这里只做确定性的本地解析，不下载附件，也不执行卡片中的链接。消息的
``outer_text`` 只包含发送者在最外层输入的文字，可用于触发判断；卡片、转发
和链接的内容只进入 ``content``，避免引用材料误触发机器人。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
import json
import re
from typing import Any, Iterable
from urllib.parse import urlsplit


URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
CQ_RE = re.compile(r"\[CQ:([^,\]]+)((?:,[^\]]*)?)\]", re.IGNORECASE)


@dataclass
class MessageSegment:
    """跨模块使用的精简消息段。原始 data 不会直接放入 LLM 上下文。"""

    type: str
    text: str = ""
    summary: str = ""
    url: str = ""
    file: str = ""
    file_id: str = ""
    unique_id: str = ""
    data: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
            "summary": self.summary,
            "url": self.url,
            "file": self.file,
            "file_id": self.file_id,
            "unique_id": self.unique_id,
        }


def parse_message_segments(content: Any) -> list[MessageSegment]:
    """兼容 OneBot 数组消息和 CQ 字符串消息。"""
    if isinstance(content, list):
        segments: list[MessageSegment] = []
        for item in content:
            if isinstance(item, dict):
                segment_type = str(item.get("type", "text") or "text").lower()
                data = item.get("data", {}) or {}
                if not isinstance(data, dict):
                    data = {"value": data}
                segments.extend(_from_onebot_segment(segment_type, data))
            elif isinstance(item, str):
                segments.extend(_split_text_and_links(item))
        return segments

    if not isinstance(content, str):
        return [MessageSegment(type="text", text=str(content))]

    result: list[MessageSegment] = []
    cursor = 0
    for match in CQ_RE.finditer(content):
        if match.start() > cursor:
            result.extend(_split_text_and_links(content[cursor:match.start()]))
        segment_type = match.group(1).lower()
        data = _parse_cq_attributes(match.group(2))
        result.extend(_from_onebot_segment(segment_type, data))
        cursor = match.end()
    if cursor < len(content):
        result.extend(_split_text_and_links(content[cursor:]))
    return result


def render_segments(segments: Iterable[MessageSegment]) -> str:
    """生成给上下文使用的可读文本，不暴露临时 URL 或整块 JSON。"""
    parts: list[str] = []
    for segment in segments:
        if segment.type == "text":
            parts.append(segment.text)
        elif segment.type in ("at", "reply"):
            continue
        elif segment.summary:
            parts.append(segment.summary)
        else:
            parts.append(_fallback_label(segment.type))
    return _tidy("".join(parts))


def render_outer_text(segments: Iterable[MessageSegment]) -> str:
    """只渲染最外层文字；URL 仅保留为中性标记。"""
    parts: list[str] = []
    for segment in segments:
        if segment.type == "text":
            parts.append(segment.text)
        elif segment.type == "link":
            parts.append("[链接]")
    return _tidy("".join(parts))


def refresh_message_content(message: Any) -> None:
    """富媒体段被异步增强后重新生成 Message.content。"""
    message.content = render_segments(message.segments) or "[无法识别的消息]"


def is_rich_only(segments: Iterable[MessageSegment]) -> bool:
    """没有实际外层文字、只有附件/卡片/链接。"""
    return not any(
        segment.type == "text" and segment.text.strip()
        for segment in segments
    ) and any(segment.type not in ("at", "reply", "text") for segment in segments)


def primary_rich_type(segments: Iterable[MessageSegment]) -> str:
    priorities = (
        "forward", "video", "image", "mface", "face", "miniapp", "card",
        "share", "link", "file", "record", "music",
    )
    present = {segment.type for segment in segments}
    return next((kind for kind in priorities if kind in present), "")


def _from_onebot_segment(segment_type: str, data: dict[str, Any]) -> list[MessageSegment]:
    if segment_type == "text":
        return _split_text_and_links(str(data.get("text", "") or ""))

    if segment_type in ("at", "reply"):
        return [MessageSegment(type=segment_type, data=dict(data))]

    normalized = segment_type
    if segment_type == "image" and (
        str(data.get("file", "")).lower() == "marketface"
        or str(data.get("sub_type", "")) in ("1", "7")
    ):
        normalized = "mface"

    if segment_type in ("json", "xml", "lightapp", "share"):
        card = _parse_card(segment_type, data)
        if card:
            normalized, summary, url = card
            return [MessageSegment(
                type=normalized,
                summary=summary,
                url=url,
                file=str(data.get("file", "") or ""),
                file_id=str(data.get("file_id", data.get("id", "")) or ""),
                unique_id=str(data.get("file_unique", "") or ""),
                data=dict(data),
            )]
        fallback_kind = {
            "json": "card",
            "xml": "card",
            "lightapp": "miniapp",
            "share": "share",
        }[segment_type]
        return [MessageSegment(
            type=fallback_kind,
            summary=_fallback_label(fallback_kind),
            url=str(data.get("url", "") or ""),
            data=dict(data),
        )]

    summary = _segment_summary(normalized, data)
    return [MessageSegment(
        type=normalized,
        summary=summary,
        url=str(data.get("url", "") or ""),
        file=str(data.get("file", data.get("path", "")) or ""),
        file_id=str(data.get("file_id", data.get("id", "")) or ""),
        unique_id=str(data.get("file_unique", "") or ""),
        data=dict(data),
    )]


def _split_text_and_links(text: str) -> list[MessageSegment]:
    if not text:
        return []
    result: list[MessageSegment] = []
    cursor = 0
    for match in URL_RE.finditer(text):
        url = match.group(0).rstrip(".,，。!?！？;；:：)")
        end = match.start() + len(url)
        if match.start() > cursor:
            result.append(MessageSegment(type="text", text=text[cursor:match.start()]))
        result.append(MessageSegment(type="link", summary=_link_label(url), url=url))
        cursor = end
    if cursor < len(text):
        result.append(MessageSegment(type="text", text=text[cursor:]))
    return result


def _parse_cq_attributes(raw: str) -> dict[str, str]:
    raw = raw[1:] if raw.startswith(",") else raw
    data: dict[str, str] = {}
    if not raw:
        return data
    for item in raw.split(","):
        key, sep, value = item.partition("=")
        if sep:
            data[key.strip()] = _cq_unescape(value)
    return data


def _cq_unescape(value: str) -> str:
    return unescape(
        value.replace("&#44;", ",")
        .replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&amp;", "&")
    )


def _parse_card(segment_type: str, data: dict[str, Any]) -> tuple[str, str, str] | None:
    payload: Any = data
    for key in ("data", "content"):
        candidate = data.get(key)
        if isinstance(candidate, dict):
            payload = candidate
            break
        if isinstance(candidate, str) and candidate.strip():
            try:
                payload = json.loads(candidate)
                break
            except (json.JSONDecodeError, TypeError):
                continue

    if not isinstance(payload, dict):
        if segment_type == "share":
            title = _short(data.get("title"), 80)
            url = _first_url(data)
            return "share", _card_label("分享", title, url), url
        return None

    app = str(payload.get("app", "") or "").lower()
    view = str(payload.get("view", "") or "").lower()
    prompt = _short(payload.get("prompt"), 100)
    desc = _short(payload.get("desc"), 100)
    strings = list(_walk_strings(payload, max_depth=5))
    url = next((value for key, value in strings if key.lower() in {
        "url", "jumpurl", "jump_url", "qqdocurl", "weburl", "shareurl"
    } and value.startswith(("http://", "https://"))), "")
    title = next((value for key, value in strings if key.lower() in {
        "title", "name", "appname", "app_name", "source"
    } and value), "")
    subtitle = next((value for key, value in strings if key.lower() in {
        "desc", "description", "summary", "tag"
    } and value and value not in (title, prompt, desc)), "")

    is_forward = (
        "multimsg" in app
        or "forward" in view
        or "聊天记录" in prompt
        or "聊天记录" in desc
    )
    is_miniapp = segment_type == "lightapp" or "miniapp" in app or "lightapp" in app
    if is_forward:
        forward_id = next((value for key, value in strings if key.lower() in {"resid", "uniseq"}), "")
        copied = dict(data)
        if forward_id and not copied.get("id"):
            copied["id"] = forward_id
        data.clear()
        data.update(copied)
        return "forward", "[合并转发]", ""

    if is_miniapp:
        label = "小程序"
        kind = "miniapp"
    elif segment_type == "share":
        label = "分享"
        kind = "share"
    else:
        label = "卡片"
        kind = "card"

    detail = title or prompt or desc or subtitle
    return kind, _card_label(label, detail, url), url


def _walk_strings(value: Any, prefix: str = "", max_depth: int = 4):
    if max_depth < 0:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str):
                yield str(key), _short(child, 200)
            elif isinstance(child, (dict, list)):
                yield from _walk_strings(child, str(key), max_depth - 1)
    elif isinstance(value, list):
        for child in value[:20]:
            yield from _walk_strings(child, prefix, max_depth - 1)


def _first_url(data: dict[str, Any]) -> str:
    for key in ("url", "jumpUrl", "qqdocurl"):
        value = str(data.get(key, "") or "")
        if value.startswith(("http://", "https://")):
            return value
    return ""


def _segment_summary(segment_type: str, data: dict[str, Any]) -> str:
    labels = {
        "image": "图片", "mface": "表情包", "face": "QQ表情",
        "record": "语音", "video": "视频", "file": "文件",
        "forward": "合并转发", "node": "转发节点", "music": "音乐分享",
        "contact": "联系人分享", "location": "位置", "markdown": "Markdown消息",
        "miniapp": "小程序", "card": "卡片", "share": "分享",
    }
    label = labels.get(segment_type, segment_type or "未知消息")
    if segment_type == "file":
        name = _short(data.get("name") or data.get("file"), 60)
        return f"[文件：{name}]" if name else "[文件]"
    if segment_type == "video":
        name = _human_filename(data.get("name"))
        return f"[视频：{name}]" if name else "[视频]"
    if segment_type == "image":
        summary = _short(data.get("summary"), 60)
        if summary and summary not in ("[图片]", "图片"):
            return f"[图片：{summary}]"
    return f"[{label}]"


def _link_label(url: str) -> str:
    host = _display_host(url)
    path = urlsplit(url).path.lower()
    if path.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")):
        kind = "图片链接"
    elif path.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")):
        kind = "视频链接"
    else:
        kind = "链接"
    return f"[{kind}：{host}]" if host else f"[{kind}]"


def _card_label(kind: str, detail: str, url: str) -> str:
    detail = _clean_card_text(detail)
    if detail:
        return f"[{kind}：{detail}]"
    host = _display_host(url)
    return f"[{kind}：{host}]" if host else f"[{kind}]"


def _display_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()[:100]
    except ValueError:
        return ""


def _human_filename(value: Any) -> str:
    name = _short(value, 60)
    if not name or re.fullmatch(r"[a-fA-F0-9_-]{16,}(?:\.[a-zA-Z0-9]+)?", name):
        return ""
    return name


def _clean_card_text(value: Any) -> str:
    text = _short(value, 100)
    text = re.sub(r"^\[[^\]]{1,20}\]\s*", "", text)
    return text.strip()


def _short(value: Any, limit: int) -> str:
    if not isinstance(value, (str, int, float)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit]


def _fallback_label(segment_type: str) -> str:
    return _segment_summary(segment_type, {})


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

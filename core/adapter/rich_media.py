"""富媒体消息的可选增强。

转发消息通过 NapCat API 展开；链接预览使用带 SSRF 防护的短请求；图片 OCR
默认关闭，仅在明确对机器人说且管理员主动启用时调用。所有增强失败都只回退为
原占位符，不阻塞正常聊天。
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from html.parser import HTMLParser
import ipaddress
import logging
import re
import socket
import time
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin, urlsplit

try:
    import aiohttp
except ModuleNotFoundError:  # 允许只运行纯解析逻辑/精简测试环境
    aiohttp = None

from .base import Message
from .rich_content import (
    MessageSegment,
    parse_message_segments,
    refresh_message_content,
    render_segments,
)

logger = logging.getLogger(__name__)

ApiCaller = Callable[[str, dict[str, Any], float | None], Awaitable[Any]]


class RichMediaEnricher:
    """在协议解析之后补全安全、简短的语义信息。"""

    def __init__(self, config: dict[str, Any] | None, api_call: ApiCaller):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.api_call = api_call

        forward = config.get("forward", {}) or {}
        self.forward_enabled = bool(forward.get("enabled", True))
        self.forward_expand_undirected = bool(forward.get("expand_when_undirected", True))
        self.forward_max_nodes = max(1, int(forward.get("max_nodes", 12)))
        self.forward_max_chars = max(100, int(forward.get("max_chars", 600)))
        self.forward_timeout = max(0.5, float(forward.get("timeout", 5.0)))

        links = config.get("links", {}) or {}
        self.links_enabled = bool(links.get("enabled", True))
        self.links_directed_only = bool(links.get("directed_only", True))
        self.link_timeout = max(0.5, float(links.get("timeout", 3.0)))
        self.link_max_bytes = max(4096, int(links.get("max_bytes", 262144)))
        self.link_max_redirects = max(0, int(links.get("max_redirects", 3)))
        self.link_cache_ttl = max(60.0, float(links.get("cache_ttl", 1800)))

        image = config.get("image", {}) or {}
        self.image_ocr_enabled = bool(image.get("ocr_enabled", False))
        self.image_ocr_action = str(image.get("ocr_action", "ocr_image") or "ocr_image")
        self.image_ocr_timeout = max(0.5, float(image.get("ocr_timeout", 5.0)))

        self._preview_cache: OrderedDict[str, tuple[float, tuple[str, str]]] = OrderedDict()
        self._ocr_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._cache_size = max(16, int(config.get("cache_size", 256)))

    async def enrich(self, message: Message, *, directed: bool = False) -> Message:
        if not self.enabled or not message.segments:
            return message

        changed = False
        for segment in message.segments:
            try:
                if segment.type == "forward" and self.forward_enabled and (
                    directed or self.forward_expand_undirected
                ):
                    changed = await self._expand_forward(segment) or changed
                elif segment.type == "link" and self.links_enabled and (
                    directed or not self.links_directed_only
                ):
                    changed = await self._preview_link(segment) or changed
                elif segment.type == "image" and directed and self.image_ocr_enabled:
                    changed = await self._ocr_image(segment) or changed
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Rich media enrichment skipped (%s): %s", segment.type, exc)

        if changed:
            refresh_message_content(message)
        return message

    async def _expand_forward(self, segment: MessageSegment) -> bool:
        payload: Any = segment.data.get("content")
        forward_id = str(segment.data.get("id") or segment.file_id or "")
        if not payload and forward_id:
            try:
                payload = await self.api_call(
                    "get_forward_msg",
                    {"message_id": forward_id},
                    self.forward_timeout,
                )
            except Exception as exc:
                logger.debug("get_forward_msg failed for %s: %s", forward_id, exc)
                return False

        nodes = self._forward_nodes(payload)
        if not nodes:
            return False

        excerpts: list[str] = []
        for node in nodes[:self.forward_max_nodes]:
            sender, content = self._render_forward_node(node)
            if not content:
                continue
            excerpt = f"{sender}：{content}" if sender else content
            excerpts.append(excerpt[:120])

        if not excerpts:
            return False
        total = len(nodes)
        suffix = "；".join(excerpts)
        if total > len(excerpts):
            suffix += f"；另有{total - len(excerpts)}条"
        summary = f"[合并转发，共{total}条：{suffix}]"
        segment.summary = summary[:self.forward_max_chars].rstrip("；")
        if len(summary) > self.forward_max_chars and not segment.summary.endswith("]"):
            segment.summary = segment.summary.rstrip("，,。.;； ") + "…]"
        return True

    def _forward_nodes(self, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in ("messages", "message", "content", "nodes"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = self._forward_nodes(value)
                if nested:
                    return nested
        data = payload.get("data")
        if data is not payload:
            return self._forward_nodes(data)
        return []

    def _render_forward_node(self, node: Any) -> tuple[str, str]:
        if not isinstance(node, dict):
            return "", str(node)[:120]
        data = node.get("data", {}) if node.get("type") == "node" else node
        if not isinstance(data, dict):
            return "", ""
        sender_data = data.get("sender", {}) or {}
        sender = str(
            data.get("nickname")
            or data.get("name")
            or (sender_data.get("card") if isinstance(sender_data, dict) else "")
            or (sender_data.get("nickname") if isinstance(sender_data, dict) else "")
            or ""
        )[:40]
        content = data.get("content", data.get("message", ""))
        segments = parse_message_segments(content)
        # 嵌套转发不递归获取，防止深层展开和循环。
        text = render_segments(segments)
        return sender, text[:160]

    async def _preview_link(self, segment: MessageSegment) -> bool:
        url = segment.url
        if not url:
            return False
        cached = self._cache_get(self._preview_cache, url, self.link_cache_ttl)
        if cached is None:
            cached = await self._fetch_preview(url)
            self._cache_put(self._preview_cache, url, cached)
        title, description = cached
        if not title:
            return False
        host = (urlsplit(url).hostname or "").lower()
        detail = title[:100]
        if description and description.lower() not in detail.lower():
            detail += f"，{description[:120]}"
        segment.summary = f"[链接：{detail}（{host}）]" if host else f"[链接：{detail}]"
        return True

    async def _fetch_preview(self, url: str) -> tuple[str, str]:
        if aiohttp is None:
            logger.debug("aiohttp 未安装，跳过链接预览")
            return "", ""
        _validate_http_url(url)
        current = url
        resolver = _SafeResolver()
        connector = aiohttp.TCPConnector(resolver=resolver, ttl_dns_cache=0)
        timeout = aiohttp.ClientTimeout(total=self.link_timeout)
        headers = {
            "User-Agent": "AliceChatBot-LinkPreview/1.0",
            "Accept": "text/html,application/xhtml+xml;q=0.9",
        }
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
                for redirect_count in range(self.link_max_redirects + 1):
                    async with session.get(current, allow_redirects=False) as response:
                        if 300 <= response.status < 400 and response.headers.get("Location"):
                            if redirect_count >= self.link_max_redirects:
                                return "", ""
                            current = urljoin(current, response.headers["Location"])
                            _validate_http_url(current)
                            continue
                        if response.status < 200 or response.status >= 300:
                            return "", ""
                        content_type = response.headers.get("Content-Type", "").lower()
                        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                            return "", ""
                        body = bytearray()
                        async for chunk in response.content.iter_chunked(16384):
                            body.extend(chunk)
                            if len(body) > self.link_max_bytes:
                                return "", ""
                        charset = response.charset or "utf-8"
                        html = bytes(body).decode(charset, errors="replace")
                        parser = _PreviewHTMLParser()
                        parser.feed(html)
                        return parser.title, parser.description
        finally:
            await connector.close()
        return "", ""

    async def _ocr_image(self, segment: MessageSegment) -> bool:
        image_ref = segment.file or segment.file_id or segment.url
        if not image_ref:
            return False
        cache_key = segment.unique_id or image_ref
        cached = self._cache_get(self._ocr_cache, cache_key, 3600)
        if cached is None:
            result = await self.api_call(
                self.image_ocr_action,
                {"image": image_ref},
                self.image_ocr_timeout,
            )
            cached = self._extract_ocr_text(result)
            self._cache_put(self._ocr_cache, cache_key, cached)
        if not cached:
            return False
        segment.summary = f"[图片，识别到文字：{cached[:160]}]"
        return True

    @staticmethod
    def _extract_ocr_text(payload: Any) -> str:
        texts: list[str] = []

        def walk(value: Any, depth: int = 0) -> None:
            if depth > 5 or len(texts) >= 20:
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.lower() in ("text", "words") and isinstance(child, str):
                        cleaned = re.sub(r"\s+", " ", child).strip()
                        if cleaned:
                            texts.append(cleaned)
                    elif key.lower() in ("texts", "data", "result"):
                        walk(child, depth + 1)
            elif isinstance(value, list):
                for child in value[:30]:
                    walk(child, depth + 1)

        walk(payload)
        return " ".join(dict.fromkeys(texts))[:300]

    def _cache_get(self, cache: OrderedDict, key: str, ttl: float):
        entry = cache.get(key)
        if not entry:
            return None
        created_at, value = entry
        if time.monotonic() - created_at > ttl:
            cache.pop(key, None)
            return None
        cache.move_to_end(key)
        return value

    def _cache_put(self, cache: OrderedDict, key: str, value: Any) -> None:
        cache[key] = (time.monotonic(), value)
        cache.move_to_end(key)
        while len(cache) > self._cache_size:
            cache.popitem(last=False)


_ResolverBase = aiohttp.abc.AbstractResolver if aiohttp is not None else object


class _SafeResolver(_ResolverBase):
    """只返回公网地址，降低链接预览的 SSRF/DNS rebinding 风险。"""

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        _validate_hostname(host)
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        resolved = []
        seen: set[tuple[str, int]] = set()
        for address_family, _, proto, _, sockaddr in infos:
            address = sockaddr[0]
            _validate_ip(address)
            key = (address, address_family)
            if key in seen:
                continue
            seen.add(key)
            resolved.append({
                "hostname": host,
                "host": address,
                "port": port,
                "family": address_family,
                "proto": proto,
                "flags": socket.AI_NUMERICHOST,
            })
        if not resolved:
            raise OSError("链接域名没有可用的公网地址")
        return resolved

    async def close(self) -> None:
        return None


def _validate_http_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("只允许带域名的 HTTP(S) 链接")
    if parsed.username or parsed.password:
        raise ValueError("链接不能包含登录凭据")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("链接端口无效") from exc
    if port not in (None, 80, 443):
        raise ValueError("链接预览只允许标准 HTTP(S) 端口")
    _validate_hostname(parsed.hostname)


def _validate_hostname(host: str) -> None:
    normalized = host.rstrip(".").lower()
    if normalized in ("localhost", "localhost.localdomain") or normalized.endswith(".local"):
        raise ValueError("不允许访问本机或局域网域名")
    try:
        _validate_ip(normalized)
    except ValueError as exc:
        # 普通域名会在 resolver 中检查解析后的所有地址；非法/内网 IP 则直接拒绝。
        try:
            ipaddress.ip_address(normalized)
        except ValueError:
            return
        raise exc


def _validate_ip(address: str) -> None:
    ip = ipaddress.ip_address(address)
    if not ip.is_global:
        raise ValueError("不允许访问非公网地址")


class _PreviewHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._title_parts: list[str] = []
        self.description = ""

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._title_parts)).strip()[:150]

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() != "meta":
            return
        values = {str(key).lower(): str(value or "") for key, value in attrs}
        name = (values.get("name") or values.get("property") or "").lower()
        if name in ("description", "og:description") and not self.description:
            self.description = re.sub(r"\s+", " ", values.get("content", "")).strip()[:200]

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and len("".join(self._title_parts)) < 200:
            self._title_parts.append(data)

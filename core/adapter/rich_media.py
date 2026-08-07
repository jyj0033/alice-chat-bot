"""富媒体消息的可选增强。

转发消息通过 NapCat API 展开；链接预览使用带 SSRF 防护的短请求；图片 OCR
默认关闭，仅在明确对机器人说且管理员主动启用时调用。所有增强失败都只回退为
原占位符，不阻塞正常聊天。
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from datetime import datetime
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

    def __init__(
        self,
        config: dict[str, Any] | None,
        api_call: ApiCaller,
        vision_provider: Any | None = None,
    ):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.api_call = api_call
        # 视觉模型 Provider（可选）。启用后图片走 LLM 描述，OCR 只作无 vision 时的兜底。
        self.vision_provider = vision_provider

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

        # 图片→文字（视觉模型描述画面）
        self.image_to_text_enabled = bool(
            image.get("to_text_enabled", bool(self.vision_provider))
        )
        self.image_to_text_scope = str(image.get("to_text_scope", "mention_only")).lower()
        self.image_to_text_prompt = str(
            image.get(
                "to_text_prompt",
                "用一两句话（50字以内）描述图片内容，如果是表情包/梗图，重点说明它想表达的"
                "情绪和意图（如调侃、无语、赞同、嘲讽、自嘲），并结合前文对话判断含义。",
            )
        )
        self.image_to_text_timeout = max(1.0, float(image.get("to_text_timeout", 60)))
        self.image_to_text_context = bool(image.get("to_text_context", True))
        self.image_context_window = max(1, int(image.get("context_window", 6)))
        self.image_max_download_bytes = max(
            64 * 1024, int(image.get("max_download_bytes", 5 * 1024 * 1024))
        )
        self.image_cache_ttl = max(60.0, float(image.get("cache_ttl", 600)))
        self.image_max_images = max(1, int(image.get("max_images", 10)))

        # 连图整体识别：同一人短时间连发的纯图片，递增多图一起喂给视觉模型
        self.image_group_enabled = bool(image.get("group_enabled", True))
        self.image_group_interval_seconds = max(1, float(image.get("group_interval_seconds", 60)))
        self.image_group_max_images = max(2, int(image.get("group_max_images", 4)))

        self._preview_cache: OrderedDict[str, tuple[float, tuple[str, str]]] = OrderedDict()
        self._ocr_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._image_desc_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._cache_size = max(16, int(config.get("cache_size", 256)))
        # 最近图片识别结果（供 Web 面板复核识别是否正确），最旧自动丢弃
        self._recognition_log: deque[dict] = deque(maxlen=60)
        self._stats = {
            "messages_seen": 0,
            "forward_expanded": 0,
            "link_previewed": 0,
            "image_ocr": 0,
            "image_to_text": 0,
            "failures": 0,
        }

    async def enrich(
        self,
        message: Message,
        *,
        directed: bool = False,
        conversation_context: str = "",
        group_image_urls: list[str] | None = None,
    ) -> Message:
        if not self.enabled or not message.segments:
            return message

        self._stats["messages_seen"] += 1
        changed = False
        image_processed = 0
        for segment in message.segments:
            try:
                if segment.type == "forward" and self.forward_enabled and (
                    directed or self.forward_expand_undirected
                ):
                    enriched = await self._expand_forward(segment)
                    if enriched:
                        self._stats["forward_expanded"] += 1
                    changed = enriched or changed
                elif segment.type == "link" and self.links_enabled and (
                    directed or not self.links_directed_only
                ):
                    enriched = await self._preview_link(segment)
                    if enriched:
                        self._stats["link_previewed"] += 1
                    changed = enriched or changed
                elif segment.type in ("image", "mface"):
                    if self._image_describe_applicable(directed) and image_processed < self.image_max_images:
                        image_processed += 1
                        enriched = await self._describe_image(
                            segment, conversation_context, group_image_urls
                        )
                        if enriched:
                            self._stats["image_to_text"] += 1
                            self._record_recognition(message, segment)
                        changed = enriched or changed
                    elif self.image_ocr_enabled and directed:
                        enriched = await self._ocr_image(segment)
                        if enriched:
                            self._stats["image_ocr"] += 1
                        changed = enriched or changed
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["failures"] += 1
                logger.debug("Rich media enrichment skipped (%s): %s", segment.type, exc)

        if changed:
            refresh_message_content(message)
        return message

    def statistics(self) -> dict[str, int | bool]:
        return {"enabled": self.enabled, **self._stats}

    def recognition_history(self, limit: int = 20) -> list[dict]:
        """最近图片识别记录（新→旧），供 Web 面板复核识别结果。"""
        return list(self._recognition_log)[-limit:][::-1]

    def _record_recognition(self, message: Message, segment: MessageSegment) -> None:
        """记录一次成功的图片识别结果。"""
        try:
            self._recognition_log.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "session_id": message.session_id,
                "sender": message.sender_name,
                "rich_type": segment.type,
                "description": (segment.summary or "").strip(),
            })
        except Exception as exc:
            logger.debug("Recognition record failed: %s", exc)

    def _image_describe_applicable(self, directed: bool) -> bool:
        """判断当前图片是否需要走视觉描述。scope=all 时所有图片都识别。"""
        if not self.image_to_text_enabled or self.vision_provider is None:
            return False
        if self.image_to_text_scope == "all":
            return True
        return directed

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
                self._stats["failures"] += 1
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

    async def _describe_image(
        self,
        segment: MessageSegment,
        conversation_context: str = "",
        group_image_urls: list[str] | None = None,
    ) -> bool:
        """用视觉模型把图片转成文字描述（含意图解读），写入 segment.summary。

        group_image_urls：同一人此前连续发的图片 URL。非空时把整组一起喂给
        视觉模型判断整体含义，此时不读/不写缓存（组含义随上下文变化）。
        """
        image_ref = segment.unique_id or segment.url
        if not image_ref:
            return False

        is_group = bool(group_image_urls and self.image_group_enabled)
        if is_group:
            # 组识别结果依赖整组上下文，不能复用单图缓存，避免"旧含义"污染
            desc = await self._call_vision(segment, conversation_context, group_image_urls)
        else:
            cached = self._cache_get(self._image_desc_cache, image_ref, self.image_cache_ttl)
            if cached is None:
                cached = await self._call_vision(segment, conversation_context)
                if cached:
                    self._cache_put(self._image_desc_cache, image_ref, cached)
            desc = cached

        if not desc:
            return False
        prefix = "[表情包，内容：" if segment.type == "mface" else "[图片，内容："
        segment.summary = f"{prefix}{desc[:100]}]"
        return True

    async def _call_vision(
        self,
        segment: MessageSegment,
        conversation_context: str = "",
        group_image_urls: list[str] | None = None,
    ) -> str:
        """先直接传图床 URL；失败则下载→base64→data URL 重试一次。

        prompt 由「意图导向基础提示 + 消息类型提示 + 前文对话」组成，
        让视觉模型判断图/表情包在对话中想表达什么，而不只是描述画面。
        带 group_image_urls 时把整组图一起喂，判断组整体含义。
        """
        if self.vision_provider is None:
            return ""

        prompt = self._build_image_prompt(segment, conversation_context, group_image_urls)

        # 组内前图也走直传；若当前图 URL 直传失败，整体降级为「仅当前图」。
        group_urls = []
        if group_image_urls and self.image_group_enabled:
            cap = self.image_group_max_images - 1  # 除当前图外最多带几张
            group_urls = [u for u in group_image_urls[:cap] if u]

        # 1. 直传图床 URL（当前图 + 前图）
        url = segment.url
        if url:
            try:
                text = await self._vision_chat(prompt, [url, *group_urls])
                if text:
                    return text
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(
                    "Vision via URL failed (%s, %d group imgs), trying fallback: %s",
                    url, len(group_urls), exc,
                )

        # 2. 回退：下载当前图→base64→data URL（限大小，仅内存驻留，用完即弃）
        data_url = await self._download_image_data_url(segment)
        if not data_url:
            return ""
        try:
            # 多图调用失败时降级为仅当前图，避免前图 URL 拖垮整张识别
            if group_urls:
                try:
                    text = await self._vision_chat(prompt, [data_url, *group_urls])
                    if text:
                        return text
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Vision via group base64 failed, degrade to single: %s", exc)
            text = await self._vision_chat(prompt, [data_url])
            if text:
                return text
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats["failures"] += 1
            logger.debug("Vision via base64 failed: %s", exc)
        return ""

    def _build_image_prompt(
        self,
        segment: MessageSegment,
        conversation_context: str,
        group_image_urls: list[str] | None = None,
    ) -> str:
        """构造意图导向的视觉 prompt。"""
        parts = [self.image_to_text_prompt]
        if segment.type == "mface":
            parts.append("这是群友发的表情包/梗图，重点分析它想表达的情绪和意图。")
        if group_image_urls and self.image_group_enabled:
            parts.append(
                f"这些图是同一人连续发的（共{len(group_image_urls) + 1}张），"
                "请判断这组图合起来想表达什么、在回应什么，以及当前这张在组里的作用。"
            )
        if conversation_context and self.image_to_text_context:
            parts.append(f"前文对话：\n{conversation_context}\n结合前文判断这张图在说什么、在回应谁。")
        return "\n".join(parts)

    async def _vision_chat(self, prompt: str, image_urls: list[str]) -> str:
        from modules.llm.base import ChatMessage, ChatRequest

        # 必须显式指定视觉模型：ChatRequest.model 默认 "gpt-4o"，会覆盖 provider 配置的模型。
        # 用 getattr 防御：provider 若未暴露 model 属性，留空走 provider 自身默认。
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image_url in image_urls:
            if image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})

        request = ChatRequest(
            model=getattr(self.vision_provider, "model", "") or "",
            messages=[ChatMessage(role="user", content=content)],
            max_tokens=300,
            temperature=0.4,
        )
        response = await asyncio.wait_for(
            self.vision_provider.chat(request),
            timeout=self.image_to_text_timeout,
        )
        text = (response.content or "").strip()
        return text[:300]

    async def _download_image_data_url(self, segment: MessageSegment) -> str:
        """下载图片转 base64 data URL。仅内存驻留，返回后即可释放；超限返回空串。"""
        if aiohttp is None:
            return ""
        url = segment.url
        if not url:
            return ""
        _validate_http_url(url)
        resolver = _SafeResolver()
        connector = aiohttp.TCPConnector(resolver=resolver, ttl_dns_cache=0)
        timeout = aiohttp.ClientTimeout(total=max(10.0, self.image_to_text_timeout * 0.5))
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status < 200 or response.status >= 300:
                        return ""
                    content_type = response.headers.get("Content-Type", "").lower()
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        body.extend(chunk)
                        if len(body) > self.image_max_download_bytes:
                            logger.debug("Image download exceeds %d bytes, skipped", self.image_max_download_bytes)
                            return ""
            media_type = "image/jpeg"
            if "image/png" in content_type:
                media_type = "image/png"
            elif "image/gif" in content_type:
                media_type = "image/gif"
            elif "image/webp" in content_type:
                media_type = "image/webp"
            import base64
            encoded = base64.b64encode(bytes(body)).decode("ascii")
            return f"data:{media_type};base64,{encoded}"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Image download failed: %s", exc)
            return ""
        finally:
            await connector.close()

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

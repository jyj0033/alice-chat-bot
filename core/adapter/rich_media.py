"""富媒体消息的可选增强。

转发消息通过 NapCat API 展开；链接预览使用带 SSRF 防护的短请求；图片 OCR
默认关闭，仅在明确对机器人说且管理员主动启用时调用。所有增强失败都只回退为
原占位符，不阻塞正常聊天。
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
import io
import ipaddress
import json
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
    describe_media_in_words,
    parse_message_segments,
    refresh_message_content,
    render_segments,
)

logger = logging.getLogger(__name__)

# 视觉模型判定图片敏感/违规被拒时的占位描述：不暴露原图，只说明爱丽丝不想看。
# 作为摘要写入上下文后，LLM 会明白 bot 不愿讨论该内容，而不是当成"没有描述"。
SENSITIVE_IMAGE_NOTE = "爱丽丝不喜欢看这个内容"

# 图片识别的兜底提示词。
#
# 这里踩过一次大坑：旧提示词写的是「客观描述画面，不要推测人物关系/前因后果/
# 适用场景」，本意是防幻觉，实际效果是把模型最强的能力（认人、认作品、认梗）
# 一起禁掉了——群里发的角色图、游戏梗图，最后只变成「银发紫瞳的动漫少女」，
# 等于没认出来。现在的原则是「先认，认不出再描述，全程不许编造」。
DEFAULT_VISION_PROMPT = (
    "先认出这张图是什么，再用一两句话把最有用的信息说清楚。"
)


@dataclass(frozen=True)
class VisionResult:
    """视觉模型返回的最小、可审计结果。"""

    description: str
    confidence: float = 0.5
    uncertain: bool = True


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
            image.get("to_text_prompt", DEFAULT_VISION_PROMPT)
        )
        self.image_to_text_timeout = max(1.0, float(image.get("to_text_timeout", 60)))
        # get_image 是本地文件查找，NapCat 响应很快；单独设短超时避免失败路径拖垮识别
        self._get_image_timeout = min(10.0, max(2.0, self.image_to_text_timeout * 0.3))
        self.image_to_text_context = bool(image.get("to_text_context", True))
        self.image_context_window = max(1, int(image.get("context_window", 6)))
        # 下载上限是「能不能把图拿到手」，跟送进模型的体积无关；QQ 原图常见
        # 11~18MB，卡在 5MB 会让整批图片直接下载失败（线上实测日志）。
        self.image_max_download_bytes = max(
            64 * 1024, int(image.get("max_download_bytes", 20 * 1024 * 1024))
        )
        # 喂给视觉模型前的缩放参数。端点单张媒体上限 10MiB，留足余量压到 2MB。
        self.image_vision_max_side = max(320, int(image.get("vision_max_side", 1600)))
        self.image_vision_jpeg_quality = max(40, min(95, int(image.get("vision_jpeg_quality", 85))))
        self.image_vision_max_payload = max(
            200_000, int(image.get("vision_max_payload_bytes", 2_000_000))
        )
        # 端点会先输出 <think> 再给 JSON，而思维链同样计入 max_tokens。900 在
        # 「把整张梗图的文字照抄下来」这种长输出上会被截断成不闭合的 JSON
        # （线上实测出现过），给到 2000 留余量。只有生成侧计费，不算浪费。
        vision_section = image.get("vision", {}) or {}
        self.image_vision_max_tokens = max(
            300,
            int(
                vision_section.get("max_tokens")
                or image.get("to_text_max_tokens", 2000)
            ),
        )
        self.image_cache_ttl = max(60.0, float(image.get("cache_ttl", 600)))
        self.image_max_images = max(1, int(image.get("max_images", 10)))

        # 连图整体识别：同一人短时间连发的纯图片，递增多图一起喂给视觉模型
        self.image_group_enabled = bool(image.get("group_enabled", True))
        self.image_group_interval_seconds = max(1, float(image.get("group_interval_seconds", 60)))
        self.image_group_max_images = max(2, int(image.get("group_max_images", 4)))

        self._preview_cache: OrderedDict[str, tuple[float, tuple[str, str]]] = OrderedDict()
        self._ocr_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._image_desc_cache: OrderedDict[str, tuple[float, VisionResult]] = OrderedDict()
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
        group_image_urls: list[dict] | None = None,
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
                        else:
                            segment.summary = describe_media_in_words(segment.type)
                            enriched = True
                        changed = enriched or changed
                    elif self.image_ocr_enabled and directed:
                        enriched = await self._ocr_image(segment)
                        if enriched:
                            self._stats["image_ocr"] += 1
                        changed = enriched or changed
                    elif not (segment.summary and "画面是" in segment.summary):
                        segment.summary = describe_media_in_words(segment.type)
                        changed = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats["failures"] += 1
                if segment.type in ("image", "mface"):
                    logger.warning("图片富媒体增强失败（%s）：%s", segment.type, exc)
                else:
                    logger.debug("富媒体增强已跳过（%s）：%s", segment.type, exc)

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
            metadata = getattr(segment, "data", {}) or {}
            self._recognition_log.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "session_id": message.session_id,
                "message_id": str(message.message_id or ""),
                "sender": message.sender_name,
                "rich_type": segment.type,
                "description": (segment.summary or "").strip(),
                "confidence": round(float(metadata.get("vision_confidence", 0.0) or 0.0), 3),
                "uncertain": bool(metadata.get("vision_uncertain", True)),
            })
        except Exception as exc:
            logger.debug("保存识别记录失败：%s", exc)

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
                logger.debug("获取转发消息失败（%s）：%s", forward_id, exc)
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
        segment.summary = describe_media_in_words("image", f"上面写着：{cached[:160]}")
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
        group_image_urls: list[dict] | None = None,
    ) -> bool:
        """用视觉模型把图片转成客观文字描述，写入 segment.summary。

        group_image_urls：同一人此前连续发的图片（[{url, file}, ...]）。非空时把
        整组一起喂给视觉模型判断整体含义，此时不读/不写缓存（组含义随上下文变化）。
        """
        # 有些 OneBot 实现只给 file/file_id，不给 url 或 file_unique；不能让
        # 这些图片全部共用空字符串缓存键，否则后一张图会复用前一张的描述。
        image_ref = (
            segment.unique_id
            or segment.url
            or segment.file_id
            or segment.file
        )
        # 注意：哪怕 image_ref 为空也继续走，让 _call_vision 尝试用 segment.file
        # 兜底。原来的早 return 会让 NapCat 只发 file 没 url/unique_id 的图片
        # 完全没有识别机会，群里 bot 看到 "[图片]" 占位只能瞎回。

        is_group = bool(group_image_urls and self.image_group_enabled)
        if is_group:
            # 组识别结果依赖整组上下文，不能复用单图缓存，避免"旧含义"污染
            vision_result = await self._call_vision(
                segment, conversation_context, group_image_urls
            )
        else:
            cached = self._cache_get(self._image_desc_cache, image_ref, self.image_cache_ttl)
            if cached is None:
                cached = await self._call_vision(segment, conversation_context)
                if cached:
                    self._cache_put(self._image_desc_cache, image_ref, cached)
            vision_result = cached

        if not vision_result or not vision_result.description.strip():
            return False
        # 模板里已经写了「画面是」，模型再以「画面是…」开头会拼成「画面是画面是…」
        desc = _strip_scene_prefix(vision_result.description)
        metadata = getattr(segment, "data", None)
        if isinstance(metadata, dict):
            # 表情库使用这一份不带群聊语境的描述；segment.summary 仍保留给
            # 普通回复链路使用，避免把“图片是什么”和“当时在回应谁”混成一个字段。
            metadata["objective_summary"] = desc[:240]
            metadata["vision_confidence"] = vision_result.confidence
            metadata["vision_uncertain"] = vision_result.uncertain
        if desc == SENSITIVE_IMAGE_NOTE:
            segment.summary = describe_media_in_words(segment.type, "爱丽丝不想看")
            return True
        summary = describe_media_in_words(segment.type, desc[:160])
        if vision_result.uncertain:
            summary = summary.rstrip("。") + "（视觉识别不确定）。"
        segment.summary = summary
        return True

    async def _call_vision(
        self,
        segment: MessageSegment,
        conversation_context: str = "",
        group_image_urls: list[dict] | None = None,
    ) -> VisionResult | None:
        """识别图片：下载原图 → 缩放压体积 → 转 base64 喂视觉模型。

        为什么统一走 base64，而不是先把图片 URL 交给端点自己抓：
        线上实测端点对单张媒体有 10MiB 上限（400 `media exceeds size limit:
        max 10485760 bytes`），QQ 群里的原图普遍 11~18MB，直传 URL 必然被打回；
        而我们先下载再缩放，既绕开上限，也顺带修掉「动图 GIF 十几 MB」这类情况。
        代价是多一次下载，但 NapCat 通常已把文件缓存在本地，走 get_image 很快。
        """
        if self.vision_provider is None:
            logger.info("[图片] 未配置视觉模型，跳过识别")
            return None

        prompt = self._build_image_prompt(segment, conversation_context, group_image_urls)
        url = segment.url
        logger.info(
            "[图片] 进入视觉识别（类型=%s，含地址=%s，含文件=%s，"
            "含唯一标识=%s，组图数=%d）",
            segment.type, bool(url), bool(segment.file),
            bool(segment.unique_id), len(group_image_urls or []),
        )

        # 组内前图（list[dict]：{url, file}），cap 限制一次最多带几张（含当前图）
        group_items: list[dict] = []
        if group_image_urls and self.image_group_enabled:
            cap = self.image_group_max_images - 1  # 除当前图外最多带几张
            group_items = [g for g in group_image_urls[:cap] if g.get("url")]

        # 1. 当前图：下载 → 缩放 → base64（当前图拿不到就整次放弃）
        current_b64 = await self._download_image_data_url(segment)
        if not current_b64:
            logger.warning(
                "图片下载失败，无法识别：地址=%s，文件=%s", url, segment.file or segment.file_id
            )
            return None

        # 2. 组图尽力而为：抓不到的直接丢弃，绝不让一张坏图拖垮整组
        group_b64s: list[str] = []
        if group_items:
            raw = await asyncio.gather(
                *(self._download_group_image_base64(g) for g in group_items),
                return_exceptions=True,
            )
            group_b64s = [b for b in raw if isinstance(b, str) and b]

        try:
            if group_b64s:
                try:
                    result = await self._vision_chat(prompt, [current_b64, *group_b64s])
                    if result:
                        return result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if _is_sensitive_rejection(exc):
                        return VisionResult(SENSITIVE_IMAGE_NOTE, confidence=1.0, uncertain=False)
                    logger.warning("组图识别失败，降级为单图：%s", exc)
            result = await self._vision_chat(prompt, [current_b64])
            if result:
                return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if _is_sensitive_rejection(exc):
                return VisionResult(SENSITIVE_IMAGE_NOTE, confidence=1.0, uncertain=False)
            self._stats["failures"] += 1
            logger.warning("视觉识别失败：%s", exc)
        return None

    def _to_vision_data_url(self, data: bytes, media_type: str) -> str:
        """原始字节 → 压到端点吃得下的大小 → data URL。"""
        if not data:
            return ""
        shrunk, shrunk_type = _shrink_image_bytes(
            data,
            media_type,
            max_side=self.image_vision_max_side,
            quality=self.image_vision_jpeg_quality,
            max_bytes=self.image_vision_max_payload,
        )
        return _bytes_to_data_url(shrunk, shrunk_type)

    async def _download_group_image_base64(self, item: dict) -> str:
        """尽力把一张组图转成 base64 data URL；失败返回空串（由调用方丢弃）。

        先直连 URL；失败且有 file 时走 NapCat get_image 本地文件兜底。
        """
        url = str(item.get("url") or "")
        if not url:
            return ""
        data = await self._http_get_bytes(url)
        if data:
            return self._to_vision_data_url(data, _sniff_media_type(data, _guess_media_type(url, "")))
        file_ref = str(item.get("file") or "")
        if file_ref:
            try:
                result = await self.api_call("get_image", {"file": file_ref}, self._get_image_timeout)
                path = result.get("path") if isinstance(result, dict) else ""
                if path:
                    with open(path, "rb") as f:
                        raw = f.read()
                    return self._to_vision_data_url(
                        raw, _sniff_media_type(raw, _guess_media_type(path, ""))
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("组图调用获取图片接口回退失败（%s）：%s", file_ref[:40], exc)
        return ""

    def _build_image_prompt(
        self,
        segment: MessageSegment,
        conversation_context: str,
        group_image_urls: list[dict] | None = None,
    ) -> str:
        """构造「先认、认不出再描述、不许编造」的视觉 prompt。"""
        parts = [
            self.image_to_text_prompt,
            "识别要求：\n"
            "1. 只要你认得画面里的人、角色、作品、梗、名场面、地标、品牌、商品、"
            "动物品种或界面截图里的软件，就直接点名。这是最有用的信息——"
            "只写「一个长发女孩」等于没认出来。\n"
            "2. 认得出作品就写「作品名 + 角色名」，认得出真人就写名字或身份，"
            "认得出梗就写梗名（例如「熊猫头」「借口龙」）。\n"
            "3. 光凭印象认出来的**身份**要带保留词——写「疑似《崩坏3》的琪亚娜」"
            "「看起来像明日方舟的澄闪」，不要写成「这就是琪亚娜」。"
            "名字写对、语气留有余地，比写得肯定更重要。\n"
            "4. 画面里的文字照抄下来，它常常就是这张图的梗本身。\n"
            "5. 认不出来、或只是看着像但没有把握，就退回客观画面描述，"
            "并直说「看不出具体是谁」——不要硬给一个名字。\n"
            "6. 绝对不要编造角色名、作品名、出处或剧情。\n"
            "7. 不要结合群聊推断谁在说谁，不要替群友评价这张图、也不要写「适合用来……」；"
            "不要加引号复述群友说过的句子。\n"
            "关于 uncertain —— 它只表示「**这份描述本身可不可信**」，"
            "与认不认得出人物无关：\n"
            "  · 只有图看不清（太糊、太暗、被裁掉），或者你对自己写下的画面内容没把握时，"
            "才把 uncertain 置为 true；\n"
            "  · 认不出画面里的人是谁**不算**——那种情况按第 5 条在 description 里写"
            "「看不出具体是谁」就行，uncertain 仍然为 false；\n"
            "  · 拿不准时宁可少写一点、写保守一点，也不要因此把整条描述标成不可信。",
        ]
        if segment.type == "mface":
            parts.append("这是群友发的表情包/梗图，先看它是什么梗，再说画面。")
        if group_image_urls and self.image_group_enabled:
            parts.append(
                f"这些图是同一人连续发的（共{len(group_image_urls) + 1}张），"
                "请分别说明每张图是什么，不要推断这组图在回应谁或想表达什么。"
            )
        if conversation_context and self.image_to_text_context:
            parts.append(
                f"前文对话仅用于确认图片边界：\n{conversation_context}\n"
                "不要把前文人物、事件、评价或原话写进图片描述。"
            )
        parts.append(
            "只输出 JSON，不要输出解释，格式如："
            '{"description":"画面中可直接看到的事实",'
            '"confidence":0.85,"uncertain":false}。'
            "confidence 是你对「description 与画面是否相符」的把握；"
            "uncertain 只在描述本身不可信时才为 true（见上），认不出人物不算。"
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_vision_result(text: str) -> VisionResult | None:
        """解析视觉模型结果。

        端点会先输出 `<think>…</think>` 再给 JSON，而且历史上干过这类事：
        思考块把 token 预算吃光、正文被截断，此时如果把原文当描述兜底，就会把
        模型的英文推理过程（"The image shows an anime-style…"）当成图片描述
        写进群聊上下文。所以这里一律先剥思考块，再只从正文里找 JSON。
        """
        raw = str(text or "").strip()
        if not raw:
            return None
        body = _strip_thinking(raw)

        payload: dict[str, Any] | None = None
        for candidate in RichMediaEnricher._json_candidates(body):
            value = _loads_json_object(candidate)
            if value is not None:
                payload = value
                break

        if payload is None:
            # 1. 先试从被截断的 JSON 里捞字段——description 的值往往已经吐完了
            salvaged = _salvage_truncated_json(body)
            if salvaged is not None:
                return salvaged
            # 2. 正文本身就是 JSON 残片（连 description 的值都没吐出来）时，
            #    绝不能整段当描述：那会把 `{"description":"…` 写进群聊上下文。
            if body.lstrip().startswith("{") or '"description"' in body:
                return None
            # 3. 无结构化输出的旧模型：只能用正文兜底，且必须标记不确定。
            #    正文为空（整段都被思考块吃掉）时视为识别失败，不要硬造描述。
            if not body:
                return None
            return VisionResult(body[:240], confidence=0.45, uncertain=True)

        description = str(
            payload.get("description")
            or payload.get("summary")
            or payload.get("content")
            or ""
        ).strip()
        if not description:
            return None
        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        # uncertain 只表示「这份描述本身可不可信」，**不再**由 confidence 反推。
        # 旧代码有 `if confidence < 0.65: uncertain = True`，等于把「认不出这是谁」
        # （模型对身份没把握，confidence 常在 0.5~0.62）当成「描述不可用」，而下游
        # `_has_objective_media_evidence` 拿它当「没有客观摘要」→ 群聊里纯图片消息
        # 直接不插话。结果是 bot 对绝大多数图片闭嘴，代价却只是没认出名字。
        # 模型没给 uncertain 时按「描述可用」处理：它毕竟写出了内容。
        uncertain_value = payload.get("uncertain", False)
        if isinstance(uncertain_value, str):
            uncertain = uncertain_value.strip().lower() in {
                "1", "true", "yes", "是", "不确定"
            }
        else:
            uncertain = bool(uncertain_value)
        return VisionResult(description[:240], confidence=confidence, uncertain=uncertain)

    @staticmethod
    def _json_candidates(body: str) -> list[str]:
        """按「越靠后越可信」的顺序给出候选 JSON 串。

        模型常见两种写法：裸 JSON，以及 ```json … ``` 围栏。思考块里也常出现
        `{` 字符，所以不走贪心匹配，而是先用平铺正则取出所有对象再倒序试——
        真正的答案总是在最后。
        """
        text = body.strip()
        candidates: list[str] = []
        for match in reversed(list(_FLAT_JSON_RE.finditer(text))):
            candidates.append(match.group(0))
        # 兜底：也允许「没有嵌套、但被文本包围」的整段，交给 json.loads 判断
        if text not in candidates:
            candidates.append(text)
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        if stripped and stripped not in candidates:
            candidates.append(stripped)
        return candidates

    async def _vision_chat(self, prompt: str, image_urls: list[str]) -> VisionResult | None:
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
            # 预算太小会被 <think> 吃光、正文被截断（实测 300 时 4/10 完全没有输出），
            # 900 是实测足够且不夸张的档位。
            max_tokens=self.image_vision_max_tokens,
            temperature=0.4,
        )
        response = await asyncio.wait_for(
            self.vision_provider.chat(request),
            timeout=self.image_to_text_timeout,
        )
        result = self._parse_vision_result(response.content or "")
        if result is None:
            # 空响应/纯思考无正文 → 按识别失败处理，让上层走"看不清"而不是把
            # 思考过程当描述。
            logger.warning(
                "[图片] 视觉模型没有给出可用描述（原始输出 %d 字符）",
                len(response.content or ""),
            )
        return result

    async def _download_image_data_url(self, segment: MessageSegment) -> str:
        """下载当前图片，缩放压体积后转 base64 data URL。

        先直连图床 URL；失败/超限时走 NapCat get_image 取本地文件兜底
        （QQ 图床 URL 常带时效与防盗链，服务器直连失败的可靠替代）。
        仅内存驻留，返回后即可释放；失败返回空串。
        """
        url = segment.url
        if url and aiohttp is not None:
            data = await self._http_get_bytes(url)
            if data:
                return self._to_vision_data_url(
                    data, _sniff_media_type(data, _guess_media_type(url, ""))
                )
            logger.debug("图片地址下载失败（%s），正在改用获取图片接口", url[:80])
        # get_image 兜底：NapCat 已把图片下载到本地，取路径读文件
        return await self._download_image_data_url_via_get_image(segment)

    async def _download_image_data_url_via_get_image(self, segment: MessageSegment) -> str:
        """调用 NapCat get_image 取本地文件路径，读文件缩放后转 base64 data URL。"""
        file_ref = segment.file or segment.file_id or segment.unique_id
        if not file_ref:
            return ""
        try:
            result = await self.api_call("get_image", {"file": file_ref}, self._get_image_timeout)
            if not isinstance(result, dict):
                return ""
            path = result.get("path") or ""
            if path:
                with open(path, "rb") as f:
                    data = f.read()
                return self._to_vision_data_url(
                    data, _sniff_media_type(data, _guess_media_type(path, ""))
                )
            # 部分实现不返回本地路径，只给 url；再用直连试一次
            alt_url = result.get("url") or ""
            if alt_url and aiohttp is not None:
                data = await self._http_get_bytes(alt_url)
                if data:
                    return self._to_vision_data_url(
                        data, _sniff_media_type(data, _guess_media_type(alt_url, ""))
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("调用获取图片接口回退失败（%s）：%s", file_ref[:40], exc)
        return ""

    async def _http_get_bytes(self, url: str) -> bytes | None:
        """尽力下载 URL 内容（SSRF 安全解析），超限/失败返回 None。"""
        if aiohttp is None:
            return None
        try:
            _validate_http_url(url)
        except ValueError:
            return None
        resolver = _SafeResolver()
        connector = aiohttp.TCPConnector(resolver=resolver, ttl_dns_cache=0)
        timeout = aiohttp.ClientTimeout(total=max(10.0, self.image_to_text_timeout * 0.5))
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status < 200 or response.status >= 300:
                        return None
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        body.extend(chunk)
                        if len(body) > self.image_max_download_bytes:
                            logger.debug(
                                "图片下载超过 %d 字节，已跳过",
                                self.image_max_download_bytes,
                            )
                            return None
            return bytes(body)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("通过 HTTP 下载失败（%s）：%s", url[:80], exc)
            return None
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


def _is_sensitive_rejection(exc: Exception) -> bool:
    """判断视觉模型报错是否为「图片内容敏感/违规」被拒（MiniMax 等端点返回）。"""
    text = str(exc or "").lower()
    return any(token in text for token in ("sensitive", "inappropriate", "敏感", "违规"))


def _guess_media_type(name: str, default: str = "image/jpeg") -> str:
    """按文件名/URL 后缀猜图片 MIME 类型。"""
    n = (name or "").lower()
    if ".png" in n:
        return "image/png"
    if ".gif" in n:
        return "image/gif"
    if ".webp" in n:
        return "image/webp"
    if ".bmp" in n:
        return "image/bmp"
    return default or "image/jpeg"


def _sniff_media_type(data: bytes, fallback: str = "image/jpeg") -> str:
    """按文件魔数判断真实图片类型，防止后缀/URL 不可靠。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] in (b"GIF8"):
        return "image/gif"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    return fallback or "image/jpeg"


def _bytes_to_data_url(data: bytes, media_type: str) -> str:
    import base64
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


# 端点认得的静态图片格式；不在这张表里的（动图、BMP、AVIF…）一律转成 JPEG，
# 免得端点直接 400 掉整次识别。
_STATIC_VISION_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


def _shrink_image_bytes(
    data: bytes,
    media_type: str,
    *,
    max_side: int,
    quality: int,
    max_bytes: int,
) -> tuple[bytes, str]:
    """把图片缩到视觉端点吃得下的大小。

    为什么必须做：MiniMax 端点对单张媒体有 10MiB 硬上限（实测直接返回 400
    `media exceeds size limit: max 10485760 bytes`），而 QQ 群里的原图随手
    就是 11~18MB，所谓「动画表情」其实常是十几 MB 的 GIF——不缩就必然失败，
    bot 只能看到「一张图，看不清画面」。

    只动喂给视觉模型的这一份，原图照常入库/转发表情包，所以不影响画质。
    多帧动图只取第一帧：端点本来也不读动画，整段传上去只会白白撞体积上限，
    所以动图**一定**会被重编码成单帧 JPEG，哪怕重编码后反而大几十字节。
    任何失败都退回原图，绝不因为预处理把一次识别整死。
    """
    if not data:
        return data, media_type
    try:
        from PIL import Image
    except Exception:
        logger.debug("未安装 Pillow，跳过视觉图片缩放")
        return data, media_type

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.load()
            frame_count = int(getattr(probe, "n_frames", 1) or 1)
            if probe.mode in ("RGBA", "LA", "P"):
                rgba = probe.convert("RGBA")
                frame = Image.new("RGB", rgba.size, (255, 255, 255))
                frame.paste(rgba, mask=rgba.split()[-1])
            elif probe.mode != "RGB":
                frame = probe.convert("RGB")
            else:
                frame = probe.copy()

        width, height = frame.size
        if max(width, height) > max_side:
            ratio = max_side / float(max(width, height))
            frame = frame.resize(
                (max(1, int(width * ratio)), max(1, int(height * ratio))),
                Image.LANCZOS,
            )

        # 已是合规 JPEG 且体积够小 → 原样返回，避免无谓的重编码损失
        if media_type == "image/jpeg" and len(data) <= max_bytes:
            return data, media_type

        current_quality = max(40, min(95, int(quality)))
        encoded = b""
        for _ in range(6):
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=current_quality, optimize=True)
            encoded = buffer.getvalue()
            if len(encoded) <= max_bytes:
                break
            if current_quality > 60:
                current_quality -= 12
            else:
                frame = frame.resize(
                    (max(1, int(frame.size[0] * 0.75)), max(1, int(frame.size[1] * 0.75))),
                    Image.LANCZOS,
                )
        if not encoded:
            return data, media_type
        # 重编码反而变大（本来就压得很狠的小图）→ 静态图就直接用原图，没必要为了
        # 几 KB 走一次有损编码；但动图必须换成单帧 JPEG，否则「只取第一帧」形同虚设，
        # 端点收到的还是整段 GIF。也顺手挡掉端点未必支持的冷门静态格式。
        if (
            frame_count <= 1
            and media_type in _STATIC_VISION_MEDIA_TYPES
            and len(encoded) >= len(data)
            and len(data) <= max_bytes
        ):
            return data, media_type
        return encoded, "image/jpeg"
    except Exception as exc:
        logger.debug("视觉图片缩放失败，回退原图：%s", exc)
        return data, media_type


_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?(?:</think(?:ing)?>|$)", re.DOTALL | re.IGNORECASE
)
_FLAT_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# 被 max_tokens 截断时 JSON 对象不闭合，整块匹配必然失败，但 description 的值
# 往往已经完整吐出来了，所以单独再捞一次。
_DESCRIPTION_FIELD_RE = re.compile(
    r'"description"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"', re.DOTALL
)
_DESCRIPTION_FIELD_TRUNCATED_RE = re.compile(
    r'"description"\s*:\s*"(?P<value>.+)', re.DOTALL
)
_CONFIDENCE_FIELD_RE = re.compile(r'"confidence"\s*:\s*(?P<value>[0-9]*\.?[0-9]+)')
_UNCERTAIN_FIELD_RE = re.compile(r'"uncertain"\s*:\s*(?P<value>true|false)', re.IGNORECASE)
# 描述本身的开头，模板里已经写了「画面是」，重复一次会变成「画面是画面是…」。
_SCENE_NOUN = r"(?:这张)?(?:图|图片|画面|照片|表情包|视频|图里|图中)(?:中|里|上)?"
_SCENE_PREFIX_RE = re.compile(rf"^{_SCENE_NOUN}(?:是|为|显示|展示)[:：,，\s]*")
# 「画面疑似…」「图中好像…」：只能吃掉前面的名词，后面那个保留词必须留着——
# 它是「不许把疑似说成确证」这条规则赖以生效的东西，一起删掉就等于把保留词抹了。
_SCENE_HEDGE_PREFIX_RE = re.compile(
    rf"^{_SCENE_NOUN}(?=(?:疑似|好像|似乎|看起来|看着像))"
)


def _strip_thinking(text: str) -> str:
    """剥掉模型的思考块。

    端点会先吐 `<think>…</think>` 再给正文；截断时可能只有开标签没有闭合，
    所以这里连「未闭合的思考块」一起吃掉，避免把英文推理当成描述写进群聊。
    """
    return _THINK_BLOCK_RE.sub(" ", text or "").strip()


def _decode_json_string(value: str) -> str:
    """把 JSON 字符串字面量的内容解出来（处理 \\" \\n 这类转义）。"""
    try:
        return str(json.loads(f'"{value}"')).strip()
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(value or "").strip()


def _extract_description_field(text: str) -> str:
    """从（可能被截断的）JSON 正文里捞 description 的值，捞不到返回空串。

    为什么需要它：输出被 max_tokens 截断时 JSON 不闭合，`{...}` 整块匹配失败，
    于是落回「正文兜底」，把 `{"description":"一张 chibi 风格…","confidence":0.55`
    这串原始 JSON 当成图片描述写进了群聊上下文（线上实测出现过）。
    description 的值本身通常是完整的，所以按字段单独抽一次就能救回来。
    """
    body = text or ""
    complete = _DESCRIPTION_FIELD_RE.search(body)
    if complete:
        return _decode_json_string(complete.group("value"))
    truncated = _DESCRIPTION_FIELD_TRUNCATED_RE.search(body)
    if not truncated:
        return ""
    value = truncated.group("value")
    # 值后面可能还挂着已经完整的兄弟字段（"description":"…","confidence":0.5），切掉
    tail = value.rfind('","')
    if tail > 0:
        value = value[:tail]
    return _decode_json_string(value.strip().rstrip('"'))


def _strip_scene_prefix(text: str) -> str:
    """去掉描述开头的「画面是」这类引导语，避免和模板拼成「画面是画面是…」。

    模型既写「画面是一个动漫角色」，也写「画面疑似蔚蓝档案风格」。后者只能吃掉
    「画面」两字：把「疑似」一起删掉就等于把保留词抹了，下游复核就没法拦住
    草稿把疑似身份说成确证。
    """
    cleaned = str(text or "").strip()
    cleaned = _SCENE_PREFIX_RE.sub("", cleaned).strip()
    cleaned = _SCENE_HEDGE_PREFIX_RE.sub("", cleaned).strip()
    return cleaned or str(text or "").strip()


def _salvage_truncated_json(text: str) -> "VisionResult | None":
    """从被 max_tokens 截断的 JSON 正文里尽量捞回可用字段。

    能捞多少算多少：截断一般发生在 description 之后的字段上，所以优先救
    description。捞不到 uncertain 时按「描述可用」处理——description 的值既然
    已经完整吐出来了，就该拿去用；旧代码在这里一律标不确定，等于因为一句回答
    被截断就把整张图判成「看不清」。
    """
    body = text or ""
    description = _extract_description_field(body)
    if not description:
        return None
    confidence = 0.55
    match = _CONFIDENCE_FIELD_RE.search(body)
    if match:
        try:
            confidence = float(match.group("value"))
        except ValueError:
            pass
    uncertain = False
    flag = _UNCERTAIN_FIELD_RE.search(body)
    if flag:
        uncertain = flag.group("value").lower() == "true"
    return VisionResult(description[:240], confidence=confidence, uncertain=uncertain)


def _loads_json_object(candidate: str) -> dict[str, Any] | None:
    """把候选串解析成 dict；失败返回 None（不抛异常，调用方继续试下一个）。"""
    text = str(candidate or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(value, dict):
        return value
    return None


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

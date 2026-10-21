"""用大模型分析表情包并归类。

两步走：

1. **看图分析**：按固定格式产出【核心含义】【情绪倾向】【适用场景】【是否建议发送】。
   这一步必须看图——表情包库里的画面描述是"深蓝色短发、额头菱形印记、瞪着大眼"
   这类客观细节，里面一个情绪词都没有，光靠文字没法判断这张图表达的梗。
2. **归类**：把分析结果交给模型，从当前可用分类里挑一个最贴切的。

为什么不用关键词词表：词表只能匹配"无语""嘲讽"这类字面词，而模型写出来的
是"极度无语""阴阳怪气地嘲讽"这种更贴近梗的说法，词表命中率极低；
而且"委屈""困倦"这类分类压根没有词条，命中不了就永远躺在「待整理」。

分析结果里的【核心含义】会写回素材的 meaning 字段——选图时给模型看的
"画面描述"换成"阴阳怪气地嘲讽"，比"瞪着大眼"有用得多。
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from modules.llm.base import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)

# 表情包分析提示词：看图 → 固定四行结构。
ANALYZE_PROMPT = (
    "你是一个懂网络热梗的聊天助手。请分析我发给你的表情包，并严格按照以下格式输出，"
    "字数必须控制在50字以内，不要任何废话：\n"
    "\n"
    "【核心含义】（1句话，最多15字，如：阴阳怪气地嘲讽/极度无语/委屈求安慰）\n"
    "【情绪倾向】（正面/负面/中性）\n"
    "【适用场景】（1句话，如：对方说废话时/自嘲时/化解尴尬时）\n"
    "【是否建议发送】（是/否/谨慎）\n"
    "\n"
    "约束：只输出以上内容，绝不解释画面细节，不罗列废话。"
)

# 归类提示词：只看分析结果选分类，不再看图。
CATEGORIZE_PROMPT = (
    "你是表情包分类助手。根据下面的分析结果，从候选分类里挑一个最贴切的。\n"
    "只输出分类名本身，不要任何解释、不要标点、不要引号。\n"
    "\n"
    "候选分类：\n{categories}\n"
    "\n"
    "分析结果：\n{analysis}\n"
)

# 素材上表示"还没归类"的标记。只有带这个标记的才进自动归类队列，
# 这样功能上线前积压的旧素材不会被翻出来重整。
PENDING_TAG = "待归类"

# 分析结论里表示"这张图不适合自动发出去"的标记。
BLOCKED_TAG = "不建议发送"

_ANALYSIS_FIELD_RE = re.compile(
    r"^【\s*(核心含义|情绪倾向|适用场景|是否建议发送)\s*】\s*[:：]?\s*(.*)$"
)


@dataclass
class MemeAnalysis:
    """一张表情包的结构化分析结果。"""

    core_meaning: str = ""
    sentiment: str = "中性"
    scene: str = ""
    should_send: str = "谨慎"

    def to_text(self) -> str:
        return (
            f"【核心含义】{self.core_meaning}\n"
            f"【情绪倾向】{self.sentiment}\n"
            f"【适用场景】{self.scene}\n"
            f"【是否建议发送】{self.should_send}"
        )


def normalize_sentiment(value: str) -> str:
    text = str(value or "").strip()
    for key in ("正面", "负面", "中性"):
        if key in text:
            return key
    return "中性"


def normalize_send_advice(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "谨慎"
    if "谨慎" in text:
        return "谨慎"
    if text.startswith("否") or "不建议" in text or "不发送" in text:
        return "否"
    if text.startswith("是") or "建议" in text:
        return "是"
    return "谨慎"


def parse_analysis(text: str) -> Optional[MemeAnalysis]:
    """解析模型输出的四行结构。字段不全就返回 None（当失败处理）。"""
    fields: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().lstrip("*-• ").strip()
        matched = _ANALYSIS_FIELD_RE.match(line)
        if matched:
            fields[matched.group(1)] = matched.group(2).strip()
    if not fields.get("核心含义"):
        return None
    return MemeAnalysis(
        core_meaning=fields.get("核心含义", "")[:60],
        sentiment=normalize_sentiment(fields.get("情绪倾向", "")),
        scene=fields.get("适用场景", "")[:80],
        should_send=normalize_send_advice(fields.get("是否建议发送", "")),
    )


def parse_category(text: str, categories: list[str]) -> str:
    """从模型输出里取出分类名。取不到合法分类就返回空串。"""
    allowed = {str(name or "").strip() for name in categories if str(name or "").strip()}
    if not allowed:
        return ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().strip("「」\"'`:：,，。.、 \t")
        if not line:
            continue
        if line in allowed:
            return line
        # 模型偶尔会在分类名前后加解释，兜底做一次包含匹配。
        for name in allowed:
            if name in line and len(line) <= len(name) + 8:
                return name
    return ""


def sniff_media_type(data: bytes, filename: str = "") -> str:
    """按 magic bytes 猜图片类型，猜不到就看扩展名。"""
    head = bytes(data or b"")[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    lowered = str(filename or "").lower()
    for ext, media in (
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".gif", "image/gif"),
        (".webp", "image/webp"),
    ):
        if lowered.endswith(ext):
            return media
    return "image/jpeg"


class MemeCategorizer:
    """看图分析 + 归类。视觉模型看图，文本模型选分类。"""

    def __init__(
        self,
        vision_provider: Any = None,
        text_provider: Any = None,
        categories: Optional[list[str]] = None,
        *,
        category_descriptions: Optional[dict[str, str]] = None,
        analyze_timeout: float = 60.0,
        categorize_timeout: float = 30.0,
        analyze_max_tokens: int = 300,
        categorize_max_tokens: int = 60,
    ) -> None:
        self.vision_provider = vision_provider
        self.text_provider = text_provider
        self.categories = [str(name or "").strip() for name in (categories or []) if str(name or "").strip()]
        # 分类说明要一起给模型：光给名字的话，「困倦」和「委屈」在模型眼里
        # 分不开（实测半闭眼的初音被归成「委屈」）。带上"犯困、睡着或准备休息"
        # 这种说明，它才分得清哪个更贴切。
        self.category_descriptions = {
            str(key or "").strip(): str(value or "").strip()
            for key, value in (category_descriptions or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        self.analyze_timeout = max(5.0, float(analyze_timeout or 60.0))
        self.categorize_timeout = max(5.0, float(categorize_timeout or 30.0))
        self.analyze_max_tokens = max(64, int(analyze_max_tokens or 300))
        self.categorize_max_tokens = max(16, int(categorize_max_tokens or 60))

    def _render_categories(self) -> str:
        lines = []
        for name in self.categories:
            description = self.category_descriptions.get(name, "")
            lines.append(f"- {name}：{description}" if description else f"- {name}")
        return "\n".join(lines)

    @property
    def available(self) -> bool:
        return bool(self.vision_provider and self.text_provider and self.categories)

    # ---------------------------------------------------------------- 步骤一

    async def analyze(self, image_bytes: bytes, media_type: str = "image/jpeg") -> Optional[MemeAnalysis]:
        """看图分析。失败返回 None。"""
        if not self.vision_provider or not image_bytes:
            return None
        data_url = self._to_data_url(image_bytes, media_type)
        if not data_url:
            return None
        content: list[dict] = [
            {"type": "text", "text": ANALYZE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        request = ChatRequest(
            model=getattr(self.vision_provider, "model", "") or "",
            messages=[ChatMessage(role="user", content=content)],
            max_tokens=self.analyze_max_tokens,
            temperature=0.4,
        )
        try:
            response = await asyncio.wait_for(
                self.vision_provider.chat(request), timeout=self.analyze_timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("表情包分析失败：%s", exc)
            return None
        analysis = parse_analysis(getattr(response, "content", "") or "")
        if analysis is None:
            logger.warning(
                "表情包分析没有返回可用结构（原始输出 %d 字符）",
                len(getattr(response, "content", "") or ""),
            )
        return analysis

    def _to_data_url(self, data: bytes, media_type: str) -> str:
        """压到端点吃得下的大小再转 data URL。"""
        if not data:
            return ""
        try:
            from core.adapter.rich_media import prepare_image_for_vision

            return prepare_image_for_vision(data, media_type)
        except Exception as exc:
            logger.debug("表情包图片压缩失败：%s", exc)
            return ""

    # ---------------------------------------------------------------- 步骤二

    async def categorize(self, analysis: MemeAnalysis) -> str:
        """根据分析结果选分类。取不到合法分类就返回空串。"""
        if not self.text_provider or not self.categories:
            return ""
        request = ChatRequest(
            model=getattr(self.text_provider, "model", "") or "",
            messages=[
                ChatMessage(
                    role="user",
                    content=CATEGORIZE_PROMPT.format(
                        categories=self._render_categories(),
                        analysis=analysis.to_text(),
                    ),
                )
            ],
            max_tokens=self.categorize_max_tokens,
            temperature=0.1,
        )
        try:
            response = await asyncio.wait_for(
                self.text_provider.chat(request), timeout=self.categorize_timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("表情包归类失败：%s", exc)
            return ""
        category = parse_category(getattr(response, "content", "") or "", self.categories)
        if not category:
            logger.warning(
                "表情包归类没有返回合法分类（原始输出 %d 字符）",
                len(getattr(response, "content", "") or ""),
            )
        return category

    # ---------------------------------------------------------------- 合起来

    async def process(
        self, image_bytes: bytes, media_type: str = "image/jpeg"
    ) -> Optional[tuple[str, MemeAnalysis]]:
        """分析 + 归类。任一步失败返回 None（素材保持待整理，等下次重试）。"""
        if not self.available:
            return None
        analysis = await self.analyze(image_bytes, media_type)
        if analysis is None:
            return None
        category = await self.categorize(analysis)
        if not category:
            return None
        return category, analysis

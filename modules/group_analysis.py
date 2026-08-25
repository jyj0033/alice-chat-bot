"""轻量群聊日报。

这里只保留群日报真正需要的能力：本地统计 + 一次结构化 LLM 分析 + 文本报告。
消息采集和报告存储由 ``MemoryStorage`` 负责，避免把 AstrBot 插件的运行时、
图片渲染和多平台适配层带进主项目。
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import io
import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class _ScaledDraw:
    """在高分辨率画布上使用逻辑坐标绘图，最后缩小以获得抗锯齿效果。"""

    def __init__(self, draw, factor: int = 1):
        self._draw = draw
        self.factor = max(1, int(factor))

    def _coords(self, value):
        if isinstance(value, (tuple, list)):
            if all(isinstance(item, (int, float)) for item in value):
                scaled = tuple(round(float(item) * self.factor) for item in value)
                return list(scaled) if isinstance(value, list) else scaled
            values = [self._coords(item) for item in value]
            return values if isinstance(value, list) else tuple(values)
        return value

    def _kwargs(self, kwargs: dict) -> dict:
        scaled = dict(kwargs)
        for key in ("width", "stroke_width", "spacing"):
            if key in scaled and isinstance(scaled[key], (int, float)):
                scaled[key] = max(1, round(scaled[key] * self.factor))
        return scaled

    def line(self, xy, *args, **kwargs):
        return self._draw.line(self._coords(xy), *args, **self._kwargs(kwargs))

    def polygon(self, xy, *args, **kwargs):
        return self._draw.polygon(self._coords(xy), *args, **self._kwargs(kwargs))

    def rectangle(self, xy, *args, **kwargs):
        return self._draw.rectangle(self._coords(xy), *args, **self._kwargs(kwargs))

    def ellipse(self, xy, *args, **kwargs):
        return self._draw.ellipse(self._coords(xy), *args, **self._kwargs(kwargs))

    def rounded_rectangle(self, xy, radius=0, *args, **kwargs):
        return self._draw.rounded_rectangle(
            self._coords(xy),
            radius=max(0, round(radius * self.factor)),
            *args,
            **self._kwargs(kwargs),
        )

    def arc(self, xy, start, end, *args, **kwargs):
        return self._draw.arc(
            self._coords(xy), start, end, *args, **self._kwargs(kwargs)
        )

    def text(self, xy, text, *args, **kwargs):
        return self._draw.text(
            self._coords(xy), text, *args, **self._kwargs(kwargs)
        )

    def textlength(self, text, *args, **kwargs):
        return self._draw.textlength(text, *args, **self._kwargs(kwargs)) / self.factor

    def textbbox(self, xy, text, *args, **kwargs):
        box = self._draw.textbbox(
            self._coords(xy), text, *args, **self._kwargs(kwargs)
        )
        return tuple(round(value / self.factor) for value in box)

    def __getattr__(self, name):
        return getattr(self._draw, name)


def _draw_report_avatar(
    canvas,
    draw,
    avatar_paths: dict,
    avatar_cache: dict,
    sender_id: str,
    name: str,
    x: int,
    y: int,
    radius: int,
    fallback_fill: str,
    outline: str,
    font,
    text_fill: str,
    scale: int = 1,
) -> None:
    """绘制真实头像；文件不可用时稳定回退到首字占位头像。"""
    try:
        from PIL import Image, ImageDraw

        avatar_path = avatar_paths.get(str(sender_id or "").strip())
        if avatar_path:
            cache_key = str(avatar_path)
            if cache_key not in avatar_cache:
                try:
                    with Image.open(cache_key) as opened:
                        avatar_cache[cache_key] = opened.convert("RGB")
                except Exception:
                    avatar_cache[cache_key] = None
            source = avatar_cache.get(cache_key)
            if source is not None:
                render_scale = max(1, int(scale))
                diameter = max(2, int(radius) * 2 * render_scale)
                thumb = source.copy()
                resampling = getattr(Image, "Resampling", None)
                filter_mode = (
                    getattr(resampling, "LANCZOS", Image.LANCZOS)
                    if resampling
                    else Image.LANCZOS
                )
                thumb.thumbnail((diameter, diameter), filter_mode)
                tile = Image.new("RGB", (diameter, diameter), fallback_fill)
                tile.paste(
                    thumb,
                    ((diameter - thumb.width) // 2, (diameter - thumb.height) // 2),
                )
                mask = Image.new("L", (diameter, diameter), 0)
                ImageDraw.Draw(mask).ellipse(
                    (0, 0, diameter - 1, diameter - 1), fill=255
                )
                canvas.paste(
                    tile,
                    (
                        (x - radius) * render_scale,
                        (y - radius) * render_scale,
                    ),
                    mask,
                )
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline=outline,
                    width=max(2, radius // 12),
                )
                return
    except Exception:
        # 头像只是装饰素材，任何格式或 Pillow 异常都不应影响日报生成。
        pass

    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fallback_fill,
        outline=outline,
        width=max(2, radius // 12),
    )
    mark = str(name or "?").strip()[:1] or "?"
    box = draw.textbbox((0, 0), mark, font=font)
    draw.text(
        (x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 2),
        mark,
        font=font,
        fill=text_fill,
    )


class GroupDailyAnalysis:
    """从 ``Memory`` 消息生成可发送的群聊日报。"""

    EMOJI_RE = re.compile(
        r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF]"
    )
    _THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
    _SPACE_RE = re.compile(r"\s+")

    @staticmethod
    def _metadata(memory) -> dict:
        metadata = getattr(memory, "metadata", {}) or {}
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _content(cls, memory) -> str:
        return str(getattr(memory, "content", "") or "").strip()

    @classmethod
    def _is_command_memory(cls, memory) -> bool:
        """兼容旧版本已经写入的命令记录；新版本不再解析或执行命令。"""
        return bool(cls._metadata(memory).get("is_command"))

    @classmethod
    def _is_bot(cls, memory) -> bool:
        return bool(cls._metadata(memory).get("is_bot"))

    @classmethod
    def _source_messages(cls, messages: list) -> list:
        """保留日报原始素材；只丢弃空消息和旧版本命令记录。"""
        return [
            message
            for message in (messages or [])
            if cls._content(message) and not cls._is_command_memory(message)
        ]

    @classmethod
    def human_messages(cls, messages: list) -> list:
        """剔除 Bot 和旧命令记录，日报统计只看群友实际发言。"""
        return [
            message for message in cls._source_messages(messages) if not cls._is_bot(message)
        ]

    @classmethod
    def _sender(cls, memory) -> tuple[str, str]:
        metadata = cls._metadata(memory)
        sender_id = str(metadata.get("sender_id") or "").strip() or "unknown"
        sender_name = str(metadata.get("sender_name") or "").strip() or sender_id
        return sender_id, sender_name

    @classmethod
    def build_statistics(cls, messages: list) -> dict[str, Any]:
        """计算不需要 LLM 的基础统计，保证模型失败时仍能发出日报。"""
        messages = cls.human_messages(messages)
        user_counts: Counter[str] = Counter()
        user_chars: Counter[str] = Counter()
        user_replies: Counter[str] = Counter()
        user_emojis: Counter[str] = Counter()
        names: dict[str, str] = {}
        hours: Counter[int] = Counter()
        total_chars = 0
        reply_count = 0

        for message in messages:
            sender_id, sender_name = cls._sender(message)
            content = cls._content(message)
            names[sender_id] = sender_name
            user_counts[sender_id] += 1
            user_chars[sender_id] += len(content)
            emoji_count = len(cls.EMOJI_RE.findall(content))
            user_emojis[sender_id] += emoji_count
            total_chars += len(content)
            created_at = getattr(message, "created_at", None)
            if isinstance(created_at, datetime):
                hours[created_at.hour] += 1
            if cls._metadata(message).get("reply_to_id") or cls._metadata(message).get(
                "reply_to_qq"
            ):
                reply_count += 1
                user_replies[sender_id] += 1

        peak_hour = max(hours, key=hours.get) if hours else None
        top_users = []
        for sender_id, count in user_counts.most_common(8):
            top_users.append(
                {
                    "sender_id": sender_id,
                    "name": names.get(sender_id, sender_id),
                    "message_count": count,
                    "char_count": user_chars[sender_id],
                    "avg_chars": round(user_chars[sender_id] / max(1, count), 1),
                    "reply_count": user_replies[sender_id],
                    "emoji_count": user_emojis[sender_id],
                }
            )

        longest = sorted(
            (
                {
                    "sender_id": cls._sender(message)[0],
                    "sender": cls._sender(message)[1],
                    "content": cls._content(message),
                }
                for message in messages
            ),
            key=lambda item: len(item["content"]),
            reverse=True,
        )[:3]

        return {
            "message_count": len(messages),
            "participant_count": len(user_counts),
            "total_characters": total_chars,
            "reply_count": reply_count,
            "emoji_count": sum(user_emojis.values()),
            "hourly_activity": {str(hour): count for hour, count in sorted(hours.items())},
            "peak_hour": peak_hour,
            "top_users": top_users,
            "sender_names": names,
            "longest_messages": longest,
        }

    @classmethod
    def _message_lines(cls, messages: list, max_chars: int = 24000) -> str:
        source_messages = cls._source_messages(messages)
        by_message_id = {
            str(cls._metadata(message).get("message_id")): message
            for message in source_messages
            if cls._metadata(message).get("message_id")
        }
        lines = []
        for message in source_messages:
            sender_id, sender_name = cls._sender(message)
            created_at = getattr(message, "created_at", None)
            time_text = (
                created_at.strftime("%H:%M")
                if isinstance(created_at, datetime)
                else "??:??"
            )
            content = cls._content(message).replace("\n", " ")[:240]
            speaker = "我" if cls._is_bot(message) else sender_name
            metadata = cls._metadata(message)
            reply_note = ""
            reply_id = str(metadata.get("reply_to_id") or "").strip()
            target = by_message_id.get(reply_id) if reply_id else None
            if target:
                target_id, target_name = cls._sender(target)
                target_content = cls._content(target).replace("\n", " ")[:48]
                reply_note = f"（回复{target_name}：{target_content}）"
            elif metadata.get("reply_to_qq"):
                reply_note = f"（回复用户{str(metadata.get('reply_to_qq'))[-4:]}）"
            lines.append(f"[{time_text}] [{sender_id}] {speaker}: {content}{reply_note}")

        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        # 保留头尾，避免只看到一天开头或只看到刚刚发生的事。
        half = max(100, (max_chars - 80) // 2)
        return text[:half] + "\n…（中间消息已省略）…\n" + text[-half:]

    @classmethod
    def build_prompt(
        cls,
        messages: list,
        statistics: dict,
        max_chars: int = 24000,
        max_topics: int = 5,
        max_quotes: int = 3,
        max_titles: int = 5,
        bot_name: str = "爱丽丝",
        bot_persona: str = "",
    ) -> str:
        bot_name = cls._short_text(bot_name or "我", 30) or "我"
        participants = [
            f"{item['sender_id']}={item['name']}（发言{item['message_count']}条，"
            f"平均{item.get('avg_chars', 0)}字，回复{item.get('reply_count', 0)}条，"
            f"表情{item.get('emoji_count', 0)}个）"
            for item in statistics.get("top_users", [])
        ]
        persona_block = cls._short_text(bot_persona, 3000)
        schema = {
            "title": "今天群里最有画面的主题",
            "subtitle": "一句带具体情绪的副标题",
            "summary": "我以第一人称回顾今天发生的几件具体小事，120到180字",
            "atmosphere": "我对这段聊天的感觉，30到60字",
            "topics": [
                {
                    "name": "话题名称",
                    "detail": "包含谁提起、大家怎么接、最后有没有结论的具体描述，140到220字",
                    "sender_ids": ["用户ID"],
                }
            ],
            "profiles": [
                {
                    "sender_id": "用户ID",
                    "title": "贴合今天表现的称号",
                    "mbti": "可选的轻量性格标签",
                    "reason": "我为什么注意到这个人，80到140字",
                }
            ],
            "quotes": [
                {
                    "content": "必须来自原文的完整或近似原话",
                    "sender_id": "用户ID",
                    "reason": "我看到这里时的短点评，15到50字，像聊天，不像分析",
                }
            ],
            "unhinged_quotes": [
                {
                    "content": "必须来自原文的完整或近似原话",
                    "sender_id": "用户ID",
                    "score": 92,
                    "reason": "我对这句的口语化点评，15到50字，不写成报告",
                }
            ],
            "quality_review": {
                "title": "今天的群聊主题",
                "subtitle": "时间或情绪副标题",
                "summary": "我对今天群聊的总评，80到120字",
                "dimensions": [
                    {
                        "name": "抽象维度",
                        "percentage": 35,
                        "comment": "结合真实消息的具体、幽默或温柔点评",
                    }
                ],
            },
        }
        return (
            f"你是{bot_name}，是这个群里的普通成员。请用{bot_name}自己的第一人称和人格口吻写一份"
            "像日记一样的群聊日报，不是新闻编辑，也不是上帝视角的客观旁白。\n"
            "重点不是把统计数字换个说法，而是从真实消息里挑出几个具体瞬间：谁先提起了什么、"
            "谁接了话、话题怎么跑偏、最后留下了什么共识或笑点。能引用原话就引用，不能从原文推出的内容不要补。\n"
            "摘要、话题详情、群友理由和聊天锐评都要有具体内容，避免“大家热烈讨论”“气氛十分融洽”"
            "这类没有信息量的套话。可以吐槽、偏心或表达感受，但不要杜撰身份、关系、地点、游戏名和事件。\n"
            "如果原文里没有我的发言，不要虚构我参与过、说过或做过什么；我只能说我看到、听到或注意到。"
            "原文中标记为“我”的行才是 Bot 自己说过的话。\n"
            "标题可以有一点文学感或群聊梗，但必须能从当天消息得到依据；副标题要像给朋友看的手写批注。\n"
            f"topics 最多{max_topics}条，profiles 最多{max_titles}人，quotes 最多{max_quotes}句；"
            "unhinged_quotes 固定最多5句，按 score 从高到低排列；"
            "profiles 只能从参与者里挑有明显行为特征的人，MBTI 只是轻量玩笑标签，不要当成心理诊断。"
            "quality_review 如果素材不足可以返回空对象，但有素材时要给出3到5个具体维度和锐评。\n"
            "金句区要像我在群里看到后顺手记下来的东西：content 必须来自原文或只是删减标点，不能改写成鸡汤。"
            "reason 只写我当时的短反应和点评，15到50字，允许吐槽、偏心、接梗或补半句原因；不要解释‘这句话体现了什么’。"
            "可以参考‘好，话题又拐回来了’‘这句一出来我就知道今晚还早’‘这也能接上，服了’这种口气，但不能凭空补事实、关系或背景。"
            "避免使用‘具有代表性’‘可以看出’‘反映了’‘体现了’‘这说明’等报告腔，也不要把 reason 写成总结段落。"
            "点评风格：语言要接地气，多用互联网黑话；吐槽要精准、避重就轻，优先调侃具体场面，不上纲上线，不做人身攻击。"
            "unhinged_quotes 是独立的‘逆天语录’区：只挑今天最离谱、最反差、最让人接不上话的真实原话，"
            "不要因为单纯脏话、刷屏、普通问候或一般吐槽就入选；score 用0到100表示逆天程度，宁缺毋滥。"
            "sender_id、sender_ids 只能使用原文中的用户ID。"
            "输出必须是纯 JSON 对象，不要 Markdown 代码块，不要在 JSON 外解释。\n\n"
            f"【输出结构示例】\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"【我的人格背景】{persona_block or f'我是{bot_name}，说话简短，但会记住群里有趣的细节。'}\n"
            f"【参与者统计】{'、'.join(participants) or '无'}\n"
            f"【基础统计】消息{statistics.get('message_count', 0)}条，参与{statistics.get('participant_count', 0)}人，"
            f"文字{statistics.get('total_characters', 0)}字，回复{statistics.get('reply_count', 0)}条\n"
            f"【群聊原文】\n{cls._message_lines(messages, max_chars=max_chars)}"
        )

    @classmethod
    def _parse_json(cls, text: str) -> dict:
        text = cls._THINK_RE.sub("", text or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        start = text.find("{")
        if start < 0:
            return {}
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _short_text(cls, value: Any, limit: int) -> str:
        text = cls._SPACE_RE.sub(" ", str(value or "")).strip()
        return text[:limit]

    @classmethod
    def _avatar_sender_ids(cls, report: dict, max_count: int = 12) -> list[str]:
        """按日报实际展示顺序收集需要头像的群友 QQ 号。"""
        stats = report.get("statistics", {}) or {}
        seen: set[str] = set()
        result: list[str] = []

        def add(value: Any) -> None:
            sender_id = str(value or "").strip()
            if not sender_id or sender_id in seen or len(result) >= max_count:
                return
            seen.add(sender_id)
            result.append(sender_id)

        for item in stats.get("top_users", []) or []:
            if isinstance(item, dict):
                add(item.get("sender_id"))
        for section in (
            report.get("profiles"),
            report.get("titles"),
            report.get("quotes"),
            report.get("unhinged_quotes"),
        ):
            for item in section or []:
                if isinstance(item, dict):
                    add(item.get("sender_id"))
        for item in report.get("topics", []) or []:
            if isinstance(item, dict):
                for sender_id in item.get("sender_ids", []) or []:
                    add(sender_id)
        return result

    @staticmethod
    def _avatar_cache_path(cache_dir: Path, sender_id: str) -> Path:
        digest = hashlib.sha256(str(sender_id).encode("utf-8")).hexdigest()[:24]
        return cache_dir / f"{digest}.img"

    @staticmethod
    def _valid_avatar_bytes(content: bytes) -> bool:
        if not content or len(content) > 4 * 1024 * 1024:
            return False
        try:
            from PIL import Image

            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            return True
        except Exception:
            return False

    @classmethod
    async def fetch_avatars(
        cls,
        report: dict,
        fetcher: Any,
        cache_dir: str | Path,
        *,
        max_count: int = 12,
        cache_days: int = 7,
    ) -> dict[str, str]:
        """获取并缓存日报需要的 QQ 头像，失败时保留旧缓存或返回空映射。"""
        sender_ids = cls._avatar_sender_ids(report, max(1, int(max_count)))
        if not sender_ids:
            return {}

        root = Path(cache_dir)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.debug("群日报头像缓存目录创建失败: %s", exc)
            return {}

        max_age = max(1, int(cache_days)) * 86400
        now = time.time()
        avatars: dict[str, str] = {}
        stale: dict[str, Path] = {}
        pending: list[tuple[str, Path]] = []
        for sender_id in sender_ids:
            target = cls._avatar_cache_path(root, sender_id)
            try:
                if target.is_file() and target.stat().st_size > 0:
                    if now - target.stat().st_mtime <= max_age:
                        avatars[sender_id] = str(target)
                    else:
                        stale[sender_id] = target
                else:
                    pending.append((sender_id, target))
            except OSError:
                pending.append((sender_id, target))

        if not callable(fetcher):
            avatars.update({sender_id: str(path) for sender_id, path in stale.items()})
            return avatars

        pending.extend(stale.items())

        async def fetch_one(sender_id: str, target: Path):
            try:
                content = await fetcher(sender_id)
                if isinstance(content, (bytes, bytearray)) and cls._valid_avatar_bytes(content):
                    await asyncio.to_thread(target.write_bytes, bytes(content))
                    return sender_id, str(target)
            except Exception as exc:
                logger.debug("获取群友头像失败 %s: %s", sender_id, exc)
            if target.is_file() and target.stat().st_size > 0:
                return sender_id, str(target)
            return sender_id, ""

        results = await asyncio.gather(
            *(fetch_one(sender_id, target) for sender_id, target in pending),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, tuple) and len(result) == 2 and result[1]:
                avatars[result[0]] = result[1]
        return avatars

    @classmethod
    def _sender_id_from_item(cls, item: dict) -> str:
        value = item.get("sender_id") or item.get("user_id") or item.get("sender")
        return str(value or "").strip().strip("[]")

    @classmethod
    def _normalise_items(
        cls,
        raw: Any,
        known_ids: set[str],
        max_count: int,
        kind: str,
        source_messages: list,
    ) -> list[dict]:
        if not isinstance(raw, list):
            return []
        output = []
        for item in raw[:max_count * 2]:
            if not isinstance(item, dict):
                continue
            sender_id = cls._sender_id_from_item(item)
            if kind == "topic":
                name = cls._short_text(item.get("name") or item.get("topic"), 36)
                detail = cls._short_text(item.get("detail"), 220)
                sender_ids = item.get("sender_ids") or item.get("contributors") or []
                if isinstance(sender_ids, str):
                    sender_ids = re.split(r"[,，、\s]+", sender_ids)
                sender_ids = [
                    str(value).strip().strip("[]")
                    for value in sender_ids
                    if str(value).strip().strip("[]") in known_ids
                ][:8]
                if name and detail:
                    output.append(
                        {"name": name, "detail": detail, "sender_ids": sender_ids}
                    )
            elif kind in {"quote", "unhinged_quote"}:
                content = cls._short_text(item.get("content"), 220)
                reason = cls._short_text(item.get("reason"), 80)
                if not content or sender_id not in known_ids:
                    continue
                matched = cls._match_quote(content, sender_id, source_messages)
                if matched:
                    quote = {
                        "content": matched,
                        "sender_id": sender_id,
                        "reason": reason or "这句我得记一下",
                    }
                    if kind == "unhinged_quote":
                        try:
                            score = float(
                                item.get("score")
                                or item.get("unhinged_score")
                                or item.get("rank_score")
                                or 0
                            )
                        except (TypeError, ValueError):
                            score = 0
                        quote["score"] = max(0, min(100, int(round(score))))
                    output.append(quote)
            else:
                title = cls._short_text(item.get("title"), 24)
                reason = cls._short_text(item.get("reason"), 140)
                mbti = cls._short_text(
                    item.get("mbti") or item.get("profile") or item.get("profile_label"),
                    24,
                )
                if title and reason and sender_id in known_ids:
                    output.append(
                        {
                            "sender_id": sender_id,
                            "title": title,
                            "mbti": mbti,
                            "reason": reason,
                        }
                    )
            if len(output) >= max_count and kind != "unhinged_quote":
                break
        if kind == "unhinged_quote":
            output.sort(key=lambda item: item.get("score", 0), reverse=True)
        return output[:max_count]

    @classmethod
    def _normalise_quality_review(cls, raw: Any) -> dict[str, Any]:
        """清理聊天锐评，避免模型返回的百分比或空字段破坏渲染。"""
        if not isinstance(raw, dict):
            return {}
        dimensions = []
        raw_dimensions = raw.get("dimensions") or raw.get("items") or []
        if isinstance(raw_dimensions, list):
            for item in raw_dimensions[:6]:
                if not isinstance(item, dict):
                    continue
                name = cls._short_text(item.get("name") or item.get("title"), 20)
                comment = cls._short_text(item.get("comment") or item.get("detail"), 120)
                if not name or not comment:
                    continue
                try:
                    percentage = float(item.get("percentage", 0) or 0)
                except (TypeError, ValueError):
                    percentage = 0
                dimensions.append(
                    {
                        "name": name,
                        "percentage": max(0, min(100, round(percentage, 1))),
                        "comment": comment,
                    }
                )
        review = {
            "title": cls._short_text(raw.get("title"), 32),
            "subtitle": cls._short_text(raw.get("subtitle"), 48),
            "summary": cls._short_text(raw.get("summary"), 140),
            "dimensions": dimensions,
        }
        return review if any(review.values()) else {}

    @classmethod
    def _fallback_titles(cls, statistics: dict[str, Any], max_count: int) -> list[dict]:
        """LLM 不可用时，用可验证的统计特征生成轻量画像，避免图片只剩报表。"""
        fallback = []
        for item in (statistics.get("top_users") or [])[:max_count]:
            sender_id = str(item.get("sender_id") or "")
            name = str(item.get("name") or sender_id or "匿名")
            message_count = int(item.get("message_count") or 0)
            reply_count = int(item.get("reply_count") or 0)
            emoji_count = int(item.get("emoji_count") or 0)
            if reply_count >= 2:
                title = "接话担当"
                reason = f"我注意到{name}今天不只是发言，还接住了{reply_count}次话头，让几段聊天没有断掉。"
            elif emoji_count >= 2:
                title = "表情包供给者"
                reason = f"我看到{name}今天用了{emoji_count}个表情，聊天里的情绪基本都能在这里找到回应。"
            else:
                title = "今日发言担当"
                reason = f"我注意到{name}今天留下了{message_count}条消息，是我回看这段聊天时最常遇到的名字之一。"
            fallback.append(
                {
                    "sender_id": sender_id,
                    "title": title,
                    "mbti": "今日观察",
                    "reason": reason,
                }
            )
        return fallback

    @classmethod
    def _fallback_unhinged_quotes(cls, messages: list, max_count: int = 5) -> list[dict]:
        """LLM 不可用时，只从带明显离谱信号的原话里挑选逆天语录。"""
        signals = (
            ("逆天", 32),
            ("离谱", 28),
            ("抽象", 24),
            ("绷不住", 24),
            ("不是吧", 20),
            ("怎么会", 20),
            ("居然", 18),
            ("竟然", 18),
            ("救命", 18),
            ("什么鬼", 18),
            ("合理吗", 22),
            ("我服了", 22),
            ("？？", 16),
            ("??", 16),
        )
        reactions = (
            "这句的走向我是真没猜到，后面居然还能接上",
            "好家伙，原来还能这么说，群里又多了一种解法",
            "这句放在今天很难不被记住，实在太会拐了",
            "我看到这里停了一下，主要是没想到还能这样",
            "话题拐成这样也算本事，起点已经找不回来了",
        )
        candidates = []
        seen: set[str] = set()
        for order, message in enumerate(cls.human_messages(messages)):
            content = cls._content(message)
            if len(content) < 6 or content in seen:
                continue
            seen.add(content)
            score = sum(weight for signal, weight in signals if signal in content)
            if 10 <= len(content) <= 120:
                score += 6
            if any(mark in content for mark in ("？", "?", "！", "!", "……", "...")):
                score += 8
            metadata = cls._metadata(message)
            if metadata.get("reply_to_id") or metadata.get("reply_to_qq"):
                score += 6
            if score < 18:
                continue
            sender_id, _ = cls._sender(message)
            candidates.append((score, len(content), -order, content, sender_id))

        candidates.sort(reverse=True)
        return [
            {
                "content": content[:220],
                "sender_id": sender_id,
                "score": min(99, int(score)),
                "reason": reactions[index % len(reactions)],
            }
            for index, (score, _, _, content, sender_id) in enumerate(candidates[:max_count])
        ]

    @classmethod
    def _match_quote(cls, candidate: str, sender_id: str, messages: list) -> str:
        candidate_norm = cls._SPACE_RE.sub("", candidate)
        candidates = [
            cls._content(message)
            for message in cls.human_messages(messages)
            if cls._sender(message)[0] == sender_id and cls._content(message)
        ]
        if not candidates:
            return ""
        for source in candidates:
            source_norm = cls._SPACE_RE.sub("", source)
            if candidate_norm in source_norm or source_norm in candidate_norm:
                return source[:220]
        best = max(
            candidates,
            key=lambda source: difflib.SequenceMatcher(
                None, candidate_norm, cls._SPACE_RE.sub("", source)
            ).ratio(),
        )
        score = difflib.SequenceMatcher(
            None, candidate_norm, cls._SPACE_RE.sub("", best)
        ).ratio()
        return best[:220] if score >= 0.45 else ""

    @classmethod
    async def analyze(
        cls,
        messages: list,
        provider=None,
        max_chars: int = 24000,
        max_topics: int = 5,
        max_quotes: int = 3,
        max_titles: int = 5,
        max_tokens: int = 1800,
        bot_name: str = "爱丽丝",
        bot_persona: str = "",
    ) -> dict:
        """生成结构化日报；LLM 失败时返回可发送的纯统计结果。"""
        source_messages = cls._source_messages(messages)
        human_messages = cls.human_messages(source_messages)
        statistics = cls.build_statistics(human_messages)
        report: dict[str, Any] = {
            "statistics": statistics,
            "bot_name": cls._short_text(bot_name or "我", 30) or "我",
            "title": "",
            "subtitle": "",
            "summary": "",
            "topics": [],
            "quotes": [],
            "unhinged_quotes": [],
            "titles": [],
            "profiles": [],
            "quality_review": {},
            "atmosphere": "",
            "analysis_error": "",
        }
        if not human_messages:
            report["title"] = "今天群里还没有留下太多脚印"
            report["subtitle"] = "我先把这页空白留着"
            report["summary"] = "我这段时间没看到足够的群友发言，暂时没什么可总结的。"
            return report
        if provider is None:
            report["analysis_error"] = "未配置可用的 LLM"
            report["title"] = "我先把今天的脚印收好"
            report["subtitle"] = "等有空再慢慢补上故事"
            report["summary"] = "我先按自己看到的消息做了个统计，暂时没提炼出更具体的总结。"
            report["titles"] = cls._fallback_titles(statistics, max_titles)
            report["profiles"] = report["titles"]
            report["unhinged_quotes"] = cls._fallback_unhinged_quotes(
                human_messages, 5
            )
            return report

        from modules.llm.base import ChatRequest

        request = ChatRequest(temperature=0.45, max_tokens=max_tokens, top_p=0.92)
        request.add_system(
            "你只负责生成群聊日报 JSON，不要和群友对话或输出解释文字；报告必须保持 Bot 的第一人称视角。"
        )
        request.add_user(
            cls.build_prompt(
                source_messages,
                statistics,
                max_chars=max_chars,
                max_topics=max_topics,
                max_quotes=max_quotes,
                max_titles=max_titles,
                bot_name=bot_name,
                bot_persona=bot_persona,
            )
        )
        try:
            response = await provider.chat(request)
            parsed = cls._parse_json(getattr(response, "content", "") or "")
        except Exception as exc:
            report["analysis_error"] = str(exc)
            report["title"] = "我先把今天的脚印收好"
            report["subtitle"] = "AI 暂时没接上，我先替它看着"
            report["summary"] = "我先按自己看到的消息做了个统计，暂时没提炼出更具体的总结。"
            report["titles"] = cls._fallback_titles(statistics, max_titles)
            report["profiles"] = report["titles"]
            report["unhinged_quotes"] = cls._fallback_unhinged_quotes(
                human_messages, 5
            )
            return report

        known_ids = {cls._sender(message)[0] for message in human_messages}
        report["title"] = cls._short_text(parsed.get("title"), 32)
        report["subtitle"] = cls._short_text(parsed.get("subtitle"), 52)
        report["summary"] = cls._short_text(parsed.get("summary"), 220)
        report["atmosphere"] = cls._short_text(parsed.get("atmosphere"), 70)
        report["topics"] = cls._normalise_items(
            parsed.get("topics"), known_ids, max_topics, "topic", human_messages
        )
        report["quotes"] = cls._normalise_items(
            parsed.get("quotes"), known_ids, max_quotes, "quote", human_messages
        )
        report["unhinged_quotes"] = cls._normalise_items(
            parsed.get("unhinged_quotes"), known_ids, 5, "unhinged_quote", human_messages
        )
        report["titles"] = cls._normalise_items(
            parsed.get("profiles") or parsed.get("titles"),
            known_ids,
            max_titles,
            "title",
            human_messages,
        )
        report["profiles"] = report["titles"]
        report["quality_review"] = cls._normalise_quality_review(
            parsed.get("quality_review") or parsed.get("chat_quality_review")
        )
        if not report["summary"]:
            report["summary"] = "我看下来，今天主要是日常交流，暂时没提炼出更多东西。"
        if not report["title"]:
            report["title"] = "今天群里聊到了一些小事"
        if not report["subtitle"]:
            report["subtitle"] = report["atmosphere"] or "我把注意到的几处记了下来"
        if not report["titles"]:
            report["titles"] = cls._fallback_titles(statistics, max_titles)
            report["profiles"] = report["titles"]
        if not report["unhinged_quotes"]:
            report["unhinged_quotes"] = cls._fallback_unhinged_quotes(
                human_messages, 5
            )
        return report

    @classmethod
    def render_report(
        cls,
        report: dict,
        report_label: str = "今日",
        max_chars: int = 3200,
    ) -> str:
        stats = report.get("statistics", {}) or {}
        def ui_text(value: Any, limit: int) -> str:
            return cls.EMOJI_RE.sub("", cls._short_text(value, limit)).strip()

        bot_name = ui_text(report.get("bot_name"), 30) or "我"
        title = ui_text(report.get("title"), 40) or "今天群里聊到了一些小事"
        subtitle = ui_text(report.get("subtitle"), 60)
        lines = [
            f"{bot_name}的群聊日报 · {report_label}",
            f"{title}{f'｜{subtitle}' if subtitle else ''}",
            (
                f"消息 {stats.get('message_count', 0)} 条｜"
                f"参与 {stats.get('participant_count', 0)} 人｜"
                f"文字 {stats.get('total_characters', 0)} 字｜"
                f"回复 {stats.get('reply_count', 0)} 条"
            ),
        ]
        peak_hour = stats.get("peak_hour")
        if peak_hour is not None:
            lines.append(f"最活跃时段：{int(peak_hour):02d}:00-{(int(peak_hour) + 1) % 24:02d}:00")

        lines.append(
            f"\n我记下的：{ui_text(report.get('summary'), 220) or '暂时没提炼出总结。'}"
        )
        if report.get("atmosphere"):
            lines.append(f"我对这段聊天的感觉：{ui_text(report['atmosphere'], 40)}")

        top_users = stats.get("top_users", []) or []
        sender_names = stats.get("sender_names", {}) or {}
        profiles = report.get("profiles") or report.get("titles") or []
        if profiles:
            lines.append("\n我给几位群友留了个小标签：")
            for profile in profiles[:5]:
                sender = ui_text(
                    cls._name_for_id(profile.get("sender_id", ""), top_users, sender_names),
                    24,
                )
                badge = ui_text(profile.get("mbti"), 24)
                suffix = f"｜{badge}" if badge else ""
                lines.append(
                    f"- {sender}：{ui_text(profile.get('title'), 24)}{suffix}"
                    f"（{ui_text(profile.get('reason'), 140)}）"
                )

        topics = report.get("topics", []) or []
        if topics:
            lines.append("\n我注意到的话题：")
            for index, topic in enumerate(topics, 1):
                lines.append(
                    f"{index}. {ui_text(topic.get('name'), 40)}：{ui_text(topic.get('detail'), 220)}"
                )

        quotes = report.get("quotes", []) or []
        if quotes:
            lines.append("\n我忍不住记下的几句：")
            for quote in quotes:
                sender = ui_text(
                    cls._name_for_id(quote.get("sender_id", ""), top_users, sender_names),
                    24,
                )
                lines.append(
                    f"「{ui_text(quote.get('content'), 220)}」——{sender}"
                    f"\n  {ui_text(quote.get('reason'), 80)}"
                )

        unhinged_quotes = report.get("unhinged_quotes", []) or []
        if unhinged_quotes:
            lines.append("\n我挑出来的五句逆天现场：")
            for index, quote in enumerate(unhinged_quotes[:5], 1):
                sender = ui_text(
                    cls._name_for_id(quote.get("sender_id", ""), top_users, sender_names),
                    24,
                )
                score = quote.get("score")
                score_text = f"｜逆天度 {int(score)}" if isinstance(score, (int, float)) and score else ""
                lines.append(
                    f"{index}.「{ui_text(quote.get('content'), 220)}」——{sender}{score_text}"
                    f"\n  {ui_text(quote.get('reason'), 80)}"
                )

        quality = report.get("quality_review", {}) or {}
        if quality:
            quality_title = ui_text(quality.get("title"), 40) or "我对今天的群聊锐评"
            quality_subtitle = ui_text(quality.get("subtitle"), 60)
            lines.append(f"\n我对今天的群聊锐评：{quality_title}")
            if quality_subtitle:
                lines.append(quality_subtitle)
            if quality.get("summary"):
                lines.append(ui_text(quality["summary"], 140))
            for dimension in quality.get("dimensions", [])[:5]:
                lines.append(
                    f"{ui_text(dimension.get('name'), 20) or '未命名'}（{dimension.get('percentage', 0)}%）："
                    f"{ui_text(dimension.get('comment'), 120)}"
                )

        if report.get("analysis_error"):
            lines.append("\n（AI 分析暂不可用，以上是我按看到的消息做的本地统计）")
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 16].rstrip() + "\n…（已截断）"

    @classmethod
    def render_report_image(
        cls,
        report: dict,
        report_label: str = "今日",
        width: int = 1080,
    ) -> bytes | None:
        """渲染无表情符号、明亮梦幻手帐风格的群聊日报 PNG。"""
        image = cls._render_report_image_editorial(
            report, report_label=report_label, width=width
        )
        if image:
            return image
        # 新模板依赖更多排版细节；字体或 Pillow 版本不兼容时逐级回退。
        image = cls._render_report_image_rich(
            report, report_label=report_label, width=width
        )
        return image or cls._render_report_image_legacy(
            report, report_label=report_label, width=width
        )

    @classmethod
    def _render_report_image_editorial(
        cls,
        report: dict,
        report_label: str = "今日",
        width: int = 1080,
    ) -> bytes | None:
        """用 Lucide 线性图标和梦幻圆角卡片渲染稳定、可读的静态日报。"""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("Pillow 未安装，群聊日报将回退为文本")
            return None

        def font_path(bold: bool = False) -> str | None:
            configured = os.environ.get("ALICE_REPORT_FONT", "").strip()
            candidates = [
                configured,
                "C:/Windows/Fonts/Dengb.ttf" if bold else "C:/Windows/Fonts/Deng.ttf",
                "C:/Windows/Fonts/STKAITI.TTF" if bold else "C:/Windows/Fonts/STSONG.TTF",
                "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            ]
            for path in candidates:
                if path and os.path.exists(path):
                    return path
            return None

        regular_path = font_path(False)
        bold_path = font_path(True) or regular_path
        if not regular_path:
            logger.warning("未找到中文字体，群聊日报将回退为文本")
            return None

        render_scale = 2

        def load_font(size: int, bold: bool = False):
            path = bold_path if bold else regular_path
            try:
                return (
                    ImageFont.truetype(path, size=max(1, int(size * render_scale)))
                    if path
                    else ImageFont.load_default()
                )
            except OSError:
                return ImageFont.load_default()

        try:
            width = max(960, int(width))
            canvas_height = 12000
            margin = 76
            content_width = width - margin * 2

            # 这里故意避开深色编辑部、数据看板和学术报告的语气，改成轻盈的
            # 糖果色手帐：卡片有软阴影，背景像一张铺着云朵和气泡的纸。
            bg = "#f4f2ec"
            bg_2 = "#fffdfa"
            surface = "#fffefb"
            surface_alt = "#f8f1e8"
            ink = "#53494a"
            white = "#53494a"
            muted = "#887b79"
            dark_muted = "#776968"
            line = "#e5dbd3"
            acid = "#7fc8bb"
            cyan = "#8bcfe5"
            coral = "#ed806d"
            violet = "#b6a0d9"
            peach = "#f2b46f"
            butter = "#f7d77f"
            shadow = "#dbe5e5"
            palette = [coral, cyan, violet, acid, peach]
            avatar_paths = report.get("avatars", {}) or {}
            avatar_cache: dict[str, Any] = {}

            image = Image.new(
                "RGB",
                (width * render_scale, canvas_height * render_scale),
                bg,
            )
            draw = _ScaledDraw(ImageDraw.Draw(image), render_scale)

            def hex_rgb(value: str) -> tuple[int, int, int]:
                value = value.lstrip("#")
                return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))

            def blend(first: str, second: str, amount: float) -> str:
                left = hex_rgb(first)
                right = hex_rgb(second)
                amount = max(0.0, min(1.0, amount))
                return "#%02x%02x%02x" % tuple(
                    round(left[index] + (right[index] - left[index]) * amount)
                    for index in range(3)
                )

            # 先铺一层参考图里的米白纸张和细点底纹；正文随后全部收进纸面。
            for gradient_y in range(520):
                draw.line(
                    (0, gradient_y, width, gradient_y),
                    fill=blend("#eef5f3", "#fff1e8", gradient_y / 520),
                )
            for row, grid_y in enumerate(range(30, canvas_height, 60)):
                offset = 18 if row % 2 else 0
                for column, grid_x in enumerate(range(30 + offset, width, 60)):
                    if (row + column) % 5 == 0:
                        continue
                    radius = 2 if (row + column) % 4 == 0 else 1
                    draw.ellipse(
                        (grid_x - radius, grid_y - radius, grid_x + radius, grid_y + radius),
                        fill="#e8e3dc" if radius == 2 else "#eee8e2",
                    )
            paper_left, paper_top = 28, 24
            paper_right = width - 28
            draw.rounded_rectangle(
                (paper_left + 10, paper_top + 12, paper_right + 10, canvas_height - 12),
                radius=28,
                fill="#d8e2e2",
            )
            draw.rounded_rectangle(
                (paper_left, paper_top, paper_right, canvas_height - 24),
                radius=28,
                fill="#fffefa",
                outline="#d2c9c0",
                width=3,
            )
            draw.line(
                (paper_right - 10, paper_top + 22, paper_right - 10, canvas_height - 46),
                fill="#efb1a6",
                width=3,
            )
            draw.line(
                (paper_right - 4, paper_top + 22, paper_right - 4, canvas_height - 46),
                fill="#9bd7df",
                width=2,
            )

            hero_font = load_font(54, True)
            hero_small = load_font(16, True)
            section_font = load_font(26, True)
            item_font = load_font(22, True)
            body_font = load_font(20)
            body_bold = load_font(20, True)
            quote_font = load_font(25, True)
            small_font = load_font(16)
            small_bold = load_font(16, True)
            stat_font = load_font(42, True)

            def display_text(value: Any, limit: int | None = None) -> str:
                """界面文本去掉表情符号；原始消息仍只在分析链路中保留。"""
                text = cls._SPACE_RE.sub(" ", str(value or "")).strip()
                text = cls.EMOJI_RE.sub("", text).strip()
                return text[:limit] if limit else text

            def text_step(font, leading: int = 7) -> int:
                box = draw.textbbox((0, 0), "国Ag", font=font)
                return max(22, box[3] - box[1] + leading)

            def trim_line(text: str, font, max_width: int) -> str:
                text = str(text or "").strip()
                if draw.textlength(text, font=font) <= max_width:
                    return text
                suffix = "…"
                while text and draw.textlength(text + suffix, font=font) > max_width:
                    text = text[:-1]
                return text.rstrip() + suffix if text else suffix

            def wrap_text(
                value: Any,
                font,
                max_width: int,
                max_lines: int | None = None,
            ) -> list[str]:
                text = display_text(value)
                if not text or max_width <= 0:
                    return []
                lines: list[str] = []
                for paragraph in text.splitlines() or [""]:
                    current = ""
                    for char in paragraph:
                        candidate = current + char
                        if current and draw.textlength(candidate, font=font) > max_width:
                            lines.append(current.rstrip())
                            current = char.lstrip()
                        else:
                            current = candidate
                    if current:
                        lines.append(current.rstrip())
                if max_lines and len(lines) > max_lines:
                    lines = lines[:max_lines]
                    lines[-1] = trim_line(lines[-1], font, max_width)
                return lines

            def block_height(lines: list[str], font, leading: int = 7) -> int:
                return len(lines) * text_step(font, leading)

            def draw_block(
                lines: list[str],
                x: int,
                y: int,
                font,
                fill: str,
                leading: int = 7,
            ) -> int:
                current_y = y
                for text in lines:
                    draw.text((x, current_y), text, font=font, fill=fill)
                    current_y += text_step(font, leading)
                return current_y

            def draw_lucide(
                name: str,
                x: int,
                y: int,
                size: int,
                color: str,
                stroke: int = 2,
            ) -> None:
                """绘制 Lucide 同款 24x24 线性图标，避免用表情符号充当图标。"""
                scale = size / 24
                width_px = max(1, round(stroke * scale))

                def point(px: float, py: float) -> tuple[int, int]:
                    return round(x + px * scale), round(y + py * scale)

                def points(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
                    return [point(px, py) for px, py in values]

                def line_path(values: list[tuple[float, float]]) -> None:
                    draw.line(points(values), fill=color, width=width_px, joint="curve")

                if name == "activity":
                    line_path([(2, 12), (6, 12), (9, 3), (15, 21), (18, 12), (22, 12)])
                elif name == "message-square":
                    draw.rounded_rectangle(
                        (*point(3, 3), *point(21, 17)),
                        radius=max(2, round(2 * scale)),
                        outline=color,
                        width=width_px,
                    )
                    line_path([(7, 17), (4, 21), (7, 17)])
                elif name == "users":
                    draw.ellipse((*point(6, 4), *point(12, 10)), outline=color, width=width_px)
                    draw.ellipse((*point(15, 5), *point(20, 10)), outline=color, width=width_px)
                    draw.arc((*point(2, 12), *point(16, 24)), 180, 360, fill=color, width=width_px)
                    draw.arc((*point(12, 13), *point(23, 24)), 180, 360, fill=color, width=width_px)
                elif name == "type":
                    line_path([(4, 6), (20, 6), (4, 12), (17, 12), (4, 18), (12, 18)])
                elif name == "reply":
                    line_path([(9, 17), (4, 12), (9, 7)])
                    line_path([(4, 12), (15, 12), (20, 7), (20, 17)])
                elif name == "clock":
                    draw.ellipse((*point(3, 3), *point(21, 21)), outline=color, width=width_px)
                    line_path([(12, 7), (12, 12), (16, 14)])
                elif name == "hash":
                    line_path([(10, 3), (8, 21)])
                    line_path([(16, 3), (14, 21)])
                    line_path([(4, 9), (20, 9)])
                    line_path([(3, 15), (19, 15)])
                elif name == "quote":
                    line_path([(5, 7), (9, 7), (7, 12), (4, 12), (4, 9), (5, 7)])
                    line_path([(15, 7), (19, 7), (17, 12), (14, 12), (14, 9), (15, 7)])
                elif name == "bar-chart":
                    line_path([(4, 20), (4, 10)])
                    line_path([(10, 20), (10, 4)])
                    line_path([(16, 20), (16, 14)])
                    line_path([(22, 20), (22, 7)])
                elif name == "scan":
                    line_path([(8, 3), (5, 3), (3, 5), (3, 8)])
                    line_path([(16, 3), (19, 3), (21, 5), (21, 8)])
                    line_path([(3, 16), (3, 19), (5, 21), (8, 21)])
                    line_path([(21, 16), (21, 19), (19, 21), (16, 21)])
                elif name == "user-round":
                    draw.ellipse((*point(8, 3), *point(16, 11)), outline=color, width=width_px)
                    draw.arc((*point(4, 11), *point(20, 25)), 180, 360, fill=color, width=width_px)
                elif name == "arrow-up-right":
                    line_path([(5, 19), (19, 5)])
                    line_path([(9, 5), (19, 5), (19, 15)])
                elif name == "spark":
                    line_path([(12, 2), (14, 10), (22, 12), (14, 14), (12, 22), (10, 14), (2, 12), (10, 10), (12, 2)])

            # 小小的线性闪光是 Lucide 几何图标，不使用任何表情字符。
            draw_lucide("spark", 42, 28, 24, violet, stroke=1.4)
            draw_lucide("spark", width - 104, 78, 18, coral, stroke=1.3)
            draw.ellipse((width - 144, 34, width - 126, 52), outline="#f3d9e5", width=2)
            draw.ellipse((92, 448, 110, 466), outline="#dfeff4", width=2)

            def panel(
                x: int,
                y: int,
                panel_width: int,
                panel_height: int,
                fill: str,
                border: str = line,
                accent: str | None = None,
                shadow_color: str | None = None,
            ) -> None:
                radius = 28
                if shadow_color:
                    soft_shadow = blend(shadow_color, bg_2, 0.42)
                    draw.rounded_rectangle(
                        (x + 7, y + 8, x + panel_width + 7, y + panel_height + 8),
                        radius=radius,
                        fill=soft_shadow,
                    )
                draw.rounded_rectangle(
                    (x, y, x + panel_width, y + panel_height),
                    radius=radius,
                    fill=fill,
                    outline=border,
                    width=1,
                )
                if accent:
                    draw.rounded_rectangle(
                        (x + 26, y + 2, x + 116, y + 7),
                        radius=3,
                        fill=blend(accent, bg_2, 0.12),
                    )
                    draw.ellipse(
                        (x + panel_width - 22, y + 18, x + panel_width - 12, y + 28),
                        fill=blend(accent, bg_2, 0.08),
                    )

            def section_label(
                index: str,
                title: str,
                x: int,
                y: int,
                icon_name: str,
                fill: str = white,
                icon_color: str = acid,
            ) -> None:
                # 保留旧调用的 index 参数，但视觉上不再使用编号，避免像报告目录。
                badge_fill = blend(icon_color, bg_2, 0.84)
                draw.rounded_rectangle(
                    (x, y, x + 44, y + 44),
                    radius=15,
                    fill=badge_fill,
                    outline=line,
                    width=1,
                )
                draw_lucide(icon_name, x + 10, y + 10, 24, icon_color, stroke=1.8)
                draw.text((x + 60, y + 5), title, font=section_font, fill=fill)
                text_width = int(draw.textlength(title, font=section_font))
                decor_x = x + 76 + text_width
                if decor_x < width - margin - 52:
                    draw.ellipse((decor_x, y + 17, decor_x + 7, y + 24), fill=icon_color)
                    draw.ellipse((decor_x + 14, y + 14, decor_x + 20, y + 20), fill=blend(icon_color, bg_2, 0.38))
                    draw.ellipse((decor_x + 27, y + 19, decor_x + 32, y + 24), fill=blend(icon_color, bg_2, 0.62))

            def tag(
                x: int,
                y: int,
                text: str,
                fill: str,
                text_fill: str = ink,
                max_chars: int = 20,
            ) -> int:
                text = display_text(text, max_chars)
                if not text:
                    return x
                tag_width = int(draw.textlength(text, font=hero_small)) + 22
                draw.rounded_rectangle(
                    (x, y, x + tag_width, y + 30),
                    radius=15,
                    fill=fill,
                    outline=line,
                    width=1,
                )
                draw.text((x + 11, y + 6), text, font=hero_small, fill=text_fill)
                return x + tag_width + 8

            def display_name(sender_id: str) -> str:
                return display_text(
                    cls._name_for_id(sender_id, top_users, sender_names), 12
                ) or display_text(sender_id, 12) or "匿名"

            def avatar(
                x: int,
                y: int,
                name: str,
                radius: int,
                color: str,
                sender_id: str = "",
            ) -> None:
                _draw_report_avatar(
                    image,
                    draw,
                    avatar_paths,
                    avatar_cache,
                    sender_id,
                    name,
                    x,
                    y,
                    radius,
                    color,
                    ink,
                    small_bold,
                    ink,
                    render_scale,
                )

            stats = report.get("statistics", {}) or {}
            bot_name = display_text(report.get("bot_name"), 24) or "我"
            title = display_text(report.get("title"), 46) or "今天群里留下了一些片段"
            subtitle = display_text(report.get("subtitle"), 76)
            top_users = stats.get("top_users", []) or []
            sender_names = stats.get("sender_names", {}) or {}

            # 首页：像翻开一页属于 Bot 的群聊手帐。
            title_lines = wrap_text(title, hero_font, int(content_width * 0.68), max_lines=2)
            subtitle_lines = wrap_text(
                subtitle or f"{report_label}，我把今天注意到的几件小事收进这一页",
                body_font,
                int(content_width * 0.62),
                max_lines=2,
            )
            hero_height = max(
                286,
                172 + block_height(title_lines, hero_font, 3) + block_height(subtitle_lines, body_font, 3),
            )
            hero_y = 54
            panel(margin, hero_y, content_width, hero_height, bg_2, border=line, accent=acid, shadow_color=shadow)
            # 右侧留一块像小行星轨道一样的留白，让首页更像手帐插画而不是报表标题。
            art_cx = width - margin - 152
            art_cy = hero_y + 158
            draw.ellipse((art_cx - 74, art_cy - 74, art_cx + 74, art_cy + 74), fill="#f7efff")
            draw.ellipse((art_cx - 42, art_cy - 42, art_cx + 42, art_cy + 42), fill="#fff4d8", outline="#f2e2c0", width=2)
            draw.arc((art_cx - 92, art_cy - 42, art_cx + 92, art_cy + 54), 202, 344, fill="#dbc9f1", width=3)
            draw.arc((art_cx - 72, art_cy - 64, art_cx + 72, art_cy + 80), 20, 160, fill="#f5c7d3", width=2)
            draw.ellipse((art_cx + 58, art_cy - 58, art_cx + 72, art_cy - 44), fill="#a8dfee")
            draw.ellipse((art_cx - 76, art_cy + 44, art_cx - 62, art_cy + 58), fill="#b9e291")
            draw_lucide("spark", art_cx + 45, art_cy + 26, 22, coral, stroke=1.4)
            hero_label = f"{bot_name}的小小群聊日记"
            hero_label = trim_line(hero_label, hero_small, content_width - 330)
            draw.text((margin + 32, hero_y + 28), hero_label, font=hero_small, fill=violet)
            date_text = f"今天 · {datetime.now().strftime('%Y.%m.%d')}"
            date_width = int(draw.textlength(date_text, font=hero_small)) + 26
            date_x = width - margin - date_width - 26
            draw.rounded_rectangle(
                (date_x, hero_y + 22, date_x + date_width, hero_y + 50),
                radius=14,
                fill="#f6f0ff",
                outline=line,
                width=1,
            )
            draw.text((date_x + 13, hero_y + 28), date_text, font=hero_small, fill=dark_muted)
            draw_lucide("spark", width - margin - 80, hero_y + 78, 34, coral, stroke=1.5)
            draw.text((margin + 32, hero_y + 74), "今天的群里", font=small_bold, fill=dark_muted)
            draw_block(title_lines, margin + 32, hero_y + 110, hero_font, white, leading=3)
            subtitle_y = hero_y + 118 + block_height(title_lines, hero_font, 3)
            draw_block(subtitle_lines, margin + 34, subtitle_y, body_font, dark_muted, leading=3)
            today_badge = "今日记录"
            today_width = int(draw.textlength(today_badge, font=hero_small)) + 24
            draw.rounded_rectangle(
                (width - margin - today_width - 26, hero_y + hero_height - 78, width - margin - 26, hero_y + hero_height - 48),
                radius=15,
                fill="#fff0f5",
                outline=line,
                width=1,
            )
            draw.text(
                (width - margin - today_width - 14, hero_y + hero_height - 72),
                today_badge,
                font=hero_small,
                fill=coral,
            )
            y = hero_y + hero_height + 54

            # 今天的小数字：参考图的 2×2 统计格 + 一块醒目的高峰卡。
            section_label("", "今天留下的数字", margin, y, "activity")
            y += 56
            signal_height = 222
            left_width = int(content_width * 0.61)
            gap = 18
            highlight_width = content_width - left_width - gap
            panel(margin, y, left_width, signal_height, bg_2, border=line, accent=cyan, shadow_color=shadow)
            metrics = [
                ("消息总数", stats.get("message_count", 0), "message-square", acid, "条"),
                ("参与人数", stats.get("participant_count", 0), "users", cyan, "人"),
                ("文字数量", stats.get("total_characters", 0), "type", coral, "字"),
                ("回复次数", stats.get("reply_count", 0), "reply", violet, "次"),
            ]
            tile_gap = 12
            tile_width = (left_width - 48 - tile_gap) // 2
            tile_height = 84
            for index, (label, value, icon_name, color, unit) in enumerate(metrics):
                cell_x = margin + 20 + (index % 2) * (tile_width + tile_gap)
                cell_y = y + 20 + (index // 2) * (tile_height + tile_gap)
                draw.rounded_rectangle(
                    (cell_x, cell_y, cell_x + tile_width, cell_y + tile_height),
                    radius=16,
                    fill=["#fff4ef", "#eef9fb", "#f6f1ff", "#fff9df"][index],
                    outline=line,
                    width=1,
                )
                draw_lucide(icon_name, cell_x + 18, cell_y + 18, 22, color, stroke=1.7)
                value_text = trim_line(display_text(value, 12), stat_font, tile_width - 112)
                draw.text((cell_x + 54, cell_y + 12), value_text, font=stat_font, fill=ink)
                draw.text((cell_x + 56, cell_y + 57), f"{label} · {unit}", font=small_bold, fill=dark_muted)

            highlight_x = margin + left_width + gap
            panel(
                highlight_x,
                y,
                highlight_width,
                signal_height,
                "#fff5bd",
                border="#e7c976",
                accent=peach,
                shadow_color="#eadfc4",
            )
            draw_lucide("clock", highlight_x + 26, y + 28, 25, peach, stroke=1.8)
            draw.text((highlight_x + 66, y + 31), "最热闹的时段", font=small_bold, fill=dark_muted)
            peak_hour = stats.get("peak_hour")
            if peak_hour is not None:
                peak_text = f"{int(peak_hour):02d}:00-{(int(peak_hour) + 1) % 24:02d}:00"
                peak_hint = "我注意到大家在这里聊得最密"
            else:
                peak_text = "今天还没有高峰"
                peak_hint = "等下一阵热闹留下来"
            peak_lines = wrap_text(peak_text, section_font, highlight_width - 48, max_lines=2)
            draw_block(peak_lines, highlight_x + 26, y + 82, section_font, ink, leading=2)
            draw_block(
                wrap_text(peak_hint, small_font, highlight_width - 48, max_lines=3),
                highlight_x + 28,
                y + 150,
                small_font,
                dark_muted,
                leading=3,
            )
            y += signal_height + 30

            activity = stats.get("hourly_activity", {}) or {}
            counts = [int(activity.get(str(hour), activity.get(hour, 0)) or 0) for hour in range(24)]
            peak_hour = stats.get("peak_hour")
            chart_height = 286
            panel(margin, y, content_width, chart_height, surface, border=line, accent=coral, shadow_color=shadow)
            section_label("", "聊天的波纹", margin + 28, y + 24, "bar-chart", fill=ink, icon_color=coral)
            draw.text((width - margin - 104, y + 33), "从早到晚", font=hero_small, fill=muted)
            chart_left = margin + 40
            chart_right = width - margin - 40
            chart_top = y + 108
            chart_bottom = y + 226
            max_count = max(counts) if counts else 0
            draw.rounded_rectangle(
                (chart_left - 16, chart_top - 30, chart_right + 16, chart_bottom + 52),
                radius=24,
                fill="#fcfaff",
                outline=line,
                width=1,
            )
            draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="#dbcfe6", width=1)
            draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="#e8deef", width=1)
            points = []
            for hour, count in enumerate(counts):
                px = chart_left + round((chart_right - chart_left) * hour / 23)
                py = chart_bottom - round((count / max_count) * 104) if max_count else chart_bottom
                points.append((px, py))
            if points and max_count:
                draw.polygon(
                    [points[0], *points, (points[-1][0], chart_bottom), (points[0][0], chart_bottom)],
                    fill="#e9f6ef",
                )
                draw.line(points, fill=violet, width=3, joint="curve")
                for hour, point in enumerate(points):
                    color = coral if peak_hour is not None and hour == int(peak_hour) else cyan
                    radius = 7 if color == coral else 4
                    draw.ellipse(
                        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                        fill=color,
                        outline=ink,
                        width=2,
                    )
            for hour in (0, 6, 12, 18, 23):
                px = chart_left + round((chart_right - chart_left) * hour / 23)
                label = "24" if hour == 23 else f"{hour:02d}"
                draw.text((px - 10, chart_bottom + 18), label, font=small_font, fill=muted)
            peak_text = f"{int(peak_hour):02d}:00" if peak_hour is not None else "未形成高峰"
            peak_label = f"最热闹 · {peak_text}"
            peak_width = int(draw.textlength(peak_label, font=small_bold)) + 22
            draw.rounded_rectangle((margin + 40, y + 68, margin + 40 + peak_width, y + 96), radius=14, fill="#fff0f5", outline=line, width=1)
            draw.text((margin + 51, y + 73), peak_label, font=small_bold, fill=coral)
            y += chart_height + 48

            # 我看到的
            section_label("", "我看到的", margin, y, "scan")
            y += 56
            summary = display_text(report.get("summary"), 260) or "我暂时没有提炼出更具体的故事。"
            atmosphere = display_text(report.get("atmosphere"), 110)
            left_width = int(content_width * 0.63)
            # 左右两栏各留出 24px 内边距，避免“今天的空气”内卡片贴到外框边缘。
            right_width = content_width - left_width - 48
            summary_lines = wrap_text(summary, body_font, left_width - 68, max_lines=6)
            atmosphere_lines = wrap_text(atmosphere, small_font, right_width - 56, max_lines=5)
            field_height = max(
                272,
                154 + block_height(summary_lines, body_font, 8),
                180 + block_height(atmosphere_lines, small_font, 6),
            )
            panel(margin, y, content_width, field_height, surface, border=line, accent=acid, shadow_color=shadow)
            draw_lucide("message-square", margin + 30, y + 28, 25, coral, stroke=1.8)
            draw.text((margin + 72, y + 31), "我的小观察", font=small_bold, fill=coral)
            draw_block(summary_lines, margin + 34, y + 86, body_font, ink, leading=8)
            draw.rounded_rectangle((margin + left_width, y + 36, margin + left_width + 2, y + field_height - 36), radius=1, fill=line)
            inset_x = margin + left_width + 24
            inset_y = y + 24
            inset_h = field_height - 48
            draw.rounded_rectangle((inset_x, inset_y, inset_x + right_width, inset_y + inset_h), radius=22, fill=surface_alt, outline=line, width=1)
            draw_lucide("spark", inset_x + 26, inset_y + 28, 24, acid, stroke=1.5)
            draw.text((inset_x + 66, inset_y + 31), "今天的空气", font=hero_small, fill=acid)
            draw_block(atmosphere_lines or ["今天的聊天留下了自己的节奏。"], inset_x + 26, inset_y + 90, small_font, white, leading=6)
            draw.rounded_rectangle((inset_x + 26, inset_y + inset_h - 52, inset_x + right_width - 26, inset_y + inset_h - 50), radius=1, fill=line)
            draw.text(
                (inset_x + 26, inset_y + inset_h - 38),
                f"{bot_name}的观察",
                font=hero_small,
                fill=dark_muted,
            )
            y += field_height + 48

            # 04 / Topics
            topics = report.get("topics", []) or []
            if topics:
                section_label("", "今天聊到的小事", margin, y, "hash")
                y += 56
                topic_rows = []
                for item in topics[:6]:
                    name = display_text(item.get("name"), 44) or "未命名话题"
                    detail = display_text(item.get("detail"), 260)
                    ids = [str(value) for value in (item.get("sender_ids") or []) if str(value)]
                    names = [display_name(value) for value in ids[:5]]
                    participant = "一起聊到：" + ("、".join(names) if names else "今天的大家")
                    participant_lines = wrap_text(participant, small_bold, content_width - 220, max_lines=2)
                    detail_lines = wrap_text(detail, body_font, content_width - 220, max_lines=4)
                    row_height = max(
                        144,
                        74
                        + block_height(participant_lines, small_bold, 4)
                        + block_height(detail_lines, body_font, 7),
                    )
                    topic_rows.append((name, participant_lines, detail_lines, row_height))
                topics_height = 86 + sum(row[3] for row in topic_rows) + 20
                panel(margin, y, content_width, topics_height, surface, border=line, accent=coral, shadow_color=shadow)
                draw.text((margin + 32, y + 26), "今日话题  ·  Topics", font=hero_small, fill=coral)
                for ruled_y in range(y + 78, y + topics_height - 12, 42):
                    draw.line(
                        (margin + 26, ruled_y, width - margin - 26, ruled_y),
                        fill="#e8f0ee",
                        width=1,
                    )
                current_y = y + 76
                for index, (name, participant_lines, detail_lines, row_height) in enumerate(topic_rows, 1):
                    if index > 1:
                        draw.rounded_rectangle((margin + 32, current_y, width - margin - 32, current_y + 2), radius=1, fill=line)
                    row_y = current_y + 22
                    accent = palette[(index - 1) % len(palette)]
                    draw.rounded_rectangle(
                        (margin + 36, row_y + 10, margin + 70, row_y + 44),
                        radius=8,
                        fill=blend(accent, bg_2, 0.76),
                        outline=line,
                        width=1,
                    )
                    draw.line(
                        [(margin + 44, row_y + 27), (margin + 51, row_y + 34), (margin + 63, row_y + 19)],
                        fill=accent,
                        width=3,
                        joint="curve",
                    )
                    text_x = margin + 104
                    draw.text((text_x, row_y + 2), name, font=item_font, fill=ink)
                    participant_y = row_y + 42
                    draw_block(participant_lines, text_x, participant_y, small_bold, muted, leading=4)
                    detail_y = participant_y + block_height(participant_lines, small_bold, 4) + 8
                    draw_block(detail_lines, text_x, detail_y, body_font, ink, leading=7)
                    current_y += row_height
                y += topics_height + 48

            # 05 / People
            profiles = report.get("profiles") or report.get("titles") or []
            if profiles:
                section_label("", "群友画像  ·  Portraits", margin, y, "users")
                y += 56
                # 内卡片统一收进外框 32px，避免第二列在长内容或阴影下贴边溢出。
                profile_width = (content_width - 64 - 24) // 2
                profile_data = []
                for item in profiles[:8]:
                    sender_id = str(item.get("sender_id") or "")
                    name = display_name(sender_id)
                    profile_title = display_text(item.get("title"), 26) or "今日观察"
                    mbti = display_text(item.get("mbti"), 20)
                    reason_lines = wrap_text(display_text(item.get("reason"), 170), small_font, profile_width - 48, max_lines=5)
                    profile_height = max(214, 142 + block_height(reason_lines, small_font, 5))
                    profile_data.append(
                        (sender_id, name, profile_title, mbti, reason_lines, profile_height)
                    )
                people_height = 80
                for index in range(0, len(profile_data), 2):
                    people_height += max(row[5] for row in profile_data[index:index + 2]) + 18
                people_height += 22
                panel(margin, y, content_width, people_height, surface_alt, border=line, accent=violet, shadow_color=shadow)
                draw.text((margin + 32, y + 26), "我悄悄注意到的大家", font=hero_small, fill=violet)
                current_y = y + 78
                for index in range(0, len(profile_data), 2):
                    row_items = profile_data[index:index + 2]
                    row_height = max(row[5] for row in row_items)
                    for column, (sender_id, name, profile_title, mbti, reason_lines, _) in enumerate(row_items):
                        card_x = margin + 32 + column * (profile_width + 24)
                        card_y = current_y
                        colors = [acid, cyan, coral, violet]
                        accent = colors[(index // 2 + column) % len(colors)]
                        draw.rounded_rectangle(
                            (card_x, card_y, card_x + profile_width, card_y + row_height),
                            radius=20,
                            fill=surface,
                            outline=line,
                            width=1,
                        )
                        draw.rounded_rectangle(
                            (card_x + 18, card_y + 12, card_x + profile_width - 18, card_y + 18),
                            radius=3,
                            fill=accent,
                        )
                        avatar(card_x + 46, card_y + 50, name, 27, accent, sender_id)
                        draw.text((card_x + 86, card_y + 22), name, font=item_font, fill=ink)
                        next_x = tag(card_x + 86, card_y + 60, profile_title, accent, max_chars=14)
                        if mbti:
                            remaining = card_x + profile_width - 24 - next_x
                            if remaining >= 80:
                                tag(next_x, card_y + 60, mbti, surface_alt, ink, max_chars=10)
                        draw.rounded_rectangle((card_x + 24, card_y + 108, card_x + profile_width - 24, card_y + 110), radius=1, fill=line)
                        draw_block(reason_lines or ["我只根据今天看到的行为留下这个观察。"], card_x + 24, card_y + 130, small_font, muted, leading=5)
                    current_y += row_height + 18
                y += people_height + 48

            # 06 / Quotes
            quotes = report.get("quotes", []) or []
            if quotes:
                section_label("", "群聊金句  ·  Quotes", margin, y, "quote")
                y += 56
                quote_bubble_width = content_width - 190
                quote_data = []
                for item in quotes[:6]:
                    sender_id = str(item.get("sender_id") or "")
                    name = display_name(sender_id)
                    content_lines = wrap_text(
                        display_text(item.get("content"), 220),
                        quote_font,
                        quote_bubble_width - 82,
                        max_lines=3,
                    )
                    reason_lines = wrap_text(
                        display_text(item.get("reason"), 130),
                        small_font,
                        quote_bubble_width - 116,
                        max_lines=2,
                    )
                    quote_height = max(
                        132,
                        72
                        + block_height(content_lines, quote_font, 7)
                        + 30
                        + block_height(reason_lines, small_font, 5)
                        + 18,
                    )
                    quote_data.append((sender_id, name, content_lines, reason_lines, quote_height))
                quotes_height = 82 + sum(item[4] + 22 for item in quote_data) + 18
                panel(
                    margin,
                    y,
                    content_width,
                    quotes_height,
                    surface,
                    border=line,
                    accent=cyan,
                    shadow_color=shadow,
                )
                draw.text((margin + 32, y + 26), "这几句我记住了，像聊天一样留下来", font=hero_small, fill=cyan)
                current_y = y + 78
                for index, (sender_id, name, content_lines, reason_lines, row_height) in enumerate(quote_data):
                    right_aligned = index % 2 == 1
                    avatar_x = width - margin - 40 if right_aligned else margin + 40
                    avatar_radius = 27
                    avatar_gap = 14
                    bubble_width = quote_bubble_width
                    bubble_x = (
                        margin + 90
                        if not right_aligned
                        else avatar_x - avatar_radius - avatar_gap - bubble_width
                    )
                    bubble_y = current_y + 28
                    bubble_fill = "#fff2f5" if not right_aligned else "#fff7d8"
                    avatar(avatar_x, current_y + 48, name, avatar_radius, coral if not right_aligned else peach, sender_id)
                    name_x = bubble_x + 18 if not right_aligned else bubble_x + bubble_width - 18 - int(draw.textlength(name, font=small_bold))
                    draw.text((name_x, current_y + 3), name, font=small_bold, fill=dark_muted)
                    draw.rounded_rectangle(
                        (bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + row_height - 12),
                        radius=22,
                        fill=bubble_fill,
                        outline=line,
                        width=1,
                    )
                    tail = (
                        [(bubble_x, bubble_y + 24), (bubble_x - 16, bubble_y + 38), (bubble_x, bubble_y + 52)]
                        if not right_aligned
                        else [(bubble_x + bubble_width, bubble_y + 24), (bubble_x + bubble_width + 8, bubble_y + 38), (bubble_x + bubble_width, bubble_y + 52)]
                    )
                    draw.polygon(tail, fill=bubble_fill, outline=line)
                    draw_lucide("quote", bubble_x + 18, bubble_y + 16, 22, coral if not right_aligned else peach, stroke=1.5)
                    draw_block(content_lines or [""], bubble_x + 54, bubble_y + 16, quote_font, ink, leading=5)
                    reason_y = bubble_y + 22 + block_height(content_lines, quote_font, 7)
                    draw.rounded_rectangle(
                        (bubble_x + 22, reason_y - 8, bubble_x + bubble_width - 22, reason_y - 6),
                        radius=1,
                        fill="#eadfd5",
                    )
                    draw_block(reason_lines, bubble_x + 22, reason_y + 5, small_font, muted, leading=3)
                    current_y += row_height + 22
                y += quotes_height + 48

            # 07 / Unhinged quotes
            unhinged_quotes = report.get("unhinged_quotes", []) or []
            if unhinged_quotes:
                section_label(
                    "",
                    "逆天语录  ·  Wild Quotes",
                    margin,
                    y,
                    "spark",
                    fill=ink,
                    icon_color=coral,
                )
                y += 56
                wild_width = (content_width - 64 - 24) // 2
                wild_data = []
                for item in unhinged_quotes[:5]:
                    if not isinstance(item, dict):
                        continue
                    rank = len(wild_data) + 1
                    sender_id = str(item.get("sender_id") or "")
                    name = display_name(sender_id)
                    content_lines = wrap_text(
                        display_text(item.get("content"), 220),
                        quote_font,
                        wild_width - 48,
                        max_lines=3,
                    )
                    reason_lines = wrap_text(
                        display_text(item.get("reason"), 90),
                        small_font,
                        wild_width - 48,
                        max_lines=2,
                    ) or ["这句我得记一下"]
                    try:
                        score = max(0, min(100, int(round(float(item.get("score", 0) or 0)))))
                    except (TypeError, ValueError):
                        score = 0
                    content_height = block_height(content_lines or [""], quote_font, 4)
                    reason_height = block_height(reason_lines, small_font, 4)
                    card_height = max(244, 96 + content_height + 12 + 24 + reason_height + 20)
                    wild_data.append(
                        (
                            rank,
                            sender_id,
                            name,
                            content_lines,
                            reason_lines,
                            score,
                            card_height,
                        )
                    )
                if wild_data:
                    wild_height = 86 + sum(
                        max(item[6] for item in wild_data[index:index + 2]) + 18
                        for index in range(0, len(wild_data), 2)
                    ) + 20
                    panel(
                        margin,
                        y,
                        content_width,
                        wild_height,
                        "#fff7ec",
                        border="#ecd8c3",
                        accent=coral,
                        shadow_color=shadow,
                    )
                    draw.text(
                        (margin + 32, y + 24),
                        "今天最离谱的几句，宁缺毋滥",
                        font=hero_small,
                        fill=coral,
                    )
                    current_y = y + 78
                    for index in range(0, len(wild_data), 2):
                        row_items = wild_data[index:index + 2]
                        row_height = max(item[6] for item in row_items)
                        for column, item in enumerate(row_items):
                            rank, sender_id, name, content_lines, reason_lines, score, _ = item
                            card_x = margin + 32 + column * (wild_width + 24)
                            card_y = current_y
                            accent = [coral, peach, violet, acid][(rank - 1) % 4]
                            draw.rounded_rectangle(
                                (card_x + 6, card_y + 8, card_x + wild_width + 6, card_y + row_height + 8),
                                radius=20,
                                fill="#eadfd6",
                            )
                            draw.rounded_rectangle(
                                (card_x, card_y, card_x + wild_width, card_y + row_height),
                                radius=20,
                                fill=surface,
                                outline="#ead8cb",
                                width=1,
                            )
                            draw.rounded_rectangle(
                                (card_x + 20, card_y + 20, card_x + 62, card_y + 52),
                                radius=14,
                                fill=blend(accent, bg_2, 0.78),
                                outline=line,
                                width=1,
                            )
                            rank_text = f"{rank:02d}"
                            rank_width = int(draw.textlength(rank_text, font=hero_small))
                            draw.text(
                                (card_x + 41 - rank_width / 2, card_y + 26),
                                rank_text,
                                font=hero_small,
                                fill=accent,
                            )
                            avatar(card_x + 92, card_y + 46, name, 22, accent, sender_id)
                            score_label = f"逆天度 {score}" if score else "逆天现场"
                            score_width = int(draw.textlength(score_label, font=hero_small)) + 18
                            score_x = card_x + wild_width - 20 - score_width
                            draw.rounded_rectangle(
                                (score_x, card_y + 22, score_x + score_width, card_y + 50),
                                radius=14,
                                fill="#fff0e4",
                                outline="#f0cbb7",
                                width=1,
                            )
                            draw.text((score_x + 9, card_y + 27), score_label, font=hero_small, fill=coral)
                            name_x = card_x + 120
                            name_width = max(46, score_x - name_x - 10)
                            draw.text(
                                (name_x, card_y + 26),
                                trim_line(name, small_bold, name_width),
                                font=small_bold,
                                fill=ink,
                            )
                            content_y = card_y + 88
                            draw_block(content_lines or [""], card_x + 24, content_y, quote_font, ink, leading=4)
                            reason_y = content_y + block_height(content_lines or [""], quote_font, 4) + 12
                            draw.rounded_rectangle(
                                (card_x + 24, reason_y - 6, card_x + wild_width - 24, reason_y - 4),
                                radius=1,
                                fill="#eadfd5",
                            )
                            draw_block(reason_lines, card_x + 24, reason_y + 6, small_font, muted, leading=4)
                        current_y += row_height + 18
                    y += wild_height + 48

            # 08 / Quality review
            quality = report.get("quality_review", {}) or {}
            dimensions = quality.get("dimensions", []) if isinstance(quality, dict) else []
            if isinstance(quality, dict) and dimensions:
                section_label("", "群聊质量复盘  ·  Review", margin, y, "bar-chart")
                y += 56
                quality_title = display_text(quality.get("title"), 38) or "今天的群聊主题"
                quality_subtitle = display_text(quality.get("subtitle"), 62) or report_label
                quality_title = trim_line(quality_title, section_font, content_width - 330)
                quality_subtitle_lines = wrap_text(quality_subtitle, small_bold, 220, max_lines=2)
                quality_summary = display_text(quality.get("summary"), 170)
                dim_width = (content_width - 64 - 24) // 2
                dim_data = []
                for item in dimensions[:6]:
                    name = display_text(item.get("name"), 20) or "未命名"
                    comment = display_text(item.get("comment"), 130)
                    try:
                        percentage = max(0.0, min(100.0, float(item.get("percentage", 0) or 0)))
                    except (TypeError, ValueError):
                        percentage = 0.0
                    lines = wrap_text(comment, small_font, dim_width - 42, max_lines=4)
                    dim_data.append((name, percentage, lines))
                summary_lines = wrap_text(quality_summary, body_font, content_width - 112, max_lines=4)
                row_heights = [
                    max(
                        154,
                        *(112 + block_height(row[2], small_font, 5) for row in dim_data[index:index + 2]),
                    )
                    for index in range(0, len(dim_data), 2)
                ]
                dimensions_height = sum(row_height + 18 for row_height in row_heights)
                dimensions_bottom = y + 136 + max(0, dimensions_height - 18)
                summary_height = block_height(summary_lines, body_font, 7) if summary_lines else 0
                summary_y = dimensions_bottom + 34 if summary_lines else dimensions_bottom
                quality_bottom = max(
                    y + 190 + dimensions_height,
                    summary_y + summary_height + 28,
                )
                quality_height = quality_bottom - y
                panel(margin, y, content_width, quality_height, bg_2, border=line, accent=acid, shadow_color=shadow)
                draw.text((margin + 32, y + 28), quality_title, font=section_font, fill=ink)
                subtitle_y = y + 30
                for subtitle_line in quality_subtitle_lines:
                    subtitle_x = width - margin - 32 - int(draw.textlength(subtitle_line, font=small_bold))
                    draw.text((subtitle_x, subtitle_y), subtitle_line, font=small_bold, fill=dark_muted)
                    subtitle_y += text_step(small_bold, 4)
                bar_x = margin + 32
                bar_y = y + 92
                bar_width = content_width - 64
                draw.rounded_rectangle(
                    (bar_x, bar_y, bar_x + bar_width, bar_y + 24),
                    radius=12,
                    fill=surface_alt,
                )
                inner_left = bar_x + 2
                inner_right = bar_x + bar_width - 2
                inner_top = bar_y + 2
                inner_bottom = bar_y + 22
                total = sum(max(0.0, float(row[1] or 0)) for row in dim_data) or 1
                current_x = inner_left
                running = 0.0
                for index, (_, percentage, _) in enumerate(dim_data):
                    running += max(0.0, float(percentage or 0))
                    segment_end = (
                        inner_right
                        if index == len(dim_data) - 1
                        else inner_left + round((running / total) * (inner_right - inner_left))
                    )
                    if segment_end > current_x:
                        # 中间段使用直角矩形，颜色连续，不会因两端圆角留下白缝。
                        draw.rectangle(
                            (current_x, inner_top, segment_end, inner_bottom),
                            fill=palette[index % len(palette)],
                        )
                    current_x = segment_end
                cap_radius = 10
                draw.ellipse(
                    (inner_left, inner_top, inner_left + cap_radius * 2, inner_bottom),
                    fill=palette[0],
                )
                draw.ellipse(
                    (inner_right - cap_radius * 2, inner_top, inner_right, inner_bottom),
                    fill=palette[(len(dim_data) - 1) % len(palette)],
                )
                draw.rounded_rectangle(
                    (bar_x, bar_y, bar_x + bar_width, bar_y + 24),
                    radius=12,
                    outline=line,
                    width=1,
                )
                current_y = y + 136
                for row_index, index in enumerate(range(0, len(dim_data), 2)):
                    row_items = dim_data[index:index + 2]
                    row_height = row_heights[row_index]
                    for column, (name, percentage, lines) in enumerate(row_items):
                        card_x = margin + 32 + column * (dim_width + 24)
                        accent = palette[(index + column) % len(palette)]
                        draw.rounded_rectangle((card_x, current_y, card_x + dim_width, current_y + row_height), radius=20, fill=surface_alt, outline=line, width=1)
                        draw_lucide("activity", card_x + 18, current_y + 18, 20, accent, stroke=1.5)
                        draw.text((card_x + 52, current_y + 20), name, font=small_bold, fill=ink)
                        draw.text((card_x + dim_width - 74, current_y + 20), f"{percentage:.0f}%", font=small_bold, fill=accent)
                        draw.line((card_x + 18, current_y + 58, card_x + dim_width - 18, current_y + 58), fill=line, width=1)
                        draw_block(lines or ["今天没有足够素材形成更多判断。"], card_x + 18, current_y + 78, small_font, muted, leading=5)
                    current_y += row_height + 18
                if summary_lines:
                    draw_lucide("user-round", margin + 34, summary_y + 4, 28, acid, stroke=1.5)
                    draw_block(summary_lines, margin + 82, summary_y, body_font, ink, leading=7)
                y += quality_height + 48

            # 页脚留一点手帐式的收束，不再使用装饰性表情符号。
            footer_height = 176
            footer_y = y + 4
            panel(margin, footer_y, content_width, footer_height, bg_2, border=line, accent=acid, shadow_color=shadow)
            footer_label = trim_line(
                display_text(f"{bot_name}的今日小记 · {report_label}", 34),
                section_font,
                content_width - 150,
            )
            draw.text((margin + 32, footer_y + 28), footer_label, font=section_font, fill=ink)
            draw.text((margin + 34, footer_y + 76), "我把今天看到的热闹，轻轻收进这一页。", font=hero_small, fill=muted)
            footer_items = [
                ("记录范围", report_label, "clock", "#fff3d7"),
                ("素材来源", "群聊消息", "message-square", "#e8f6f4"),
                ("整理方式", f"{bot_name}的观察", "scan", "#f4edff"),
            ]
            footer_gap = 12
            footer_tile_width = (content_width - 64 - footer_gap * 2) // 3
            for index, (label, value, icon_name, fill) in enumerate(footer_items):
                tile_x = margin + 32 + index * (footer_tile_width + footer_gap)
                tile_y = footer_y + 112
                draw.rounded_rectangle(
                    (tile_x, tile_y, tile_x + footer_tile_width, tile_y + 48),
                    radius=14,
                    fill=fill,
                    outline=line,
                    width=1,
                )
                draw_lucide(icon_name, tile_x + 12, tile_y + 12, 22, acid, stroke=1.5)
                draw.text((tile_x + 44, tile_y + 8), label, font=small_font, fill=muted)
                draw.text((tile_x + 44, tile_y + 25), trim_line(value, small_bold, footer_tile_width - 58), font=small_bold, fill=ink)
            draw_lucide("arrow-up-right", width - margin - 78, footer_y + 35, 34, acid, stroke=1.6)

            final_height = min(canvas_height, max(footer_y + footer_height + margin, 520))
            # 裁切高度确定后再补纸张底边和两条彩色装订线，避免压到正文。
            draw.rounded_rectangle(
                (paper_left, paper_top, paper_right, final_height - 24),
                radius=28,
                outline="#d2c9c0",
                width=3,
            )
            draw.line(
                (paper_right - 10, paper_top + 22, paper_right - 10, final_height - 46),
                fill="#efb1a6",
                width=3,
            )
            draw.line(
                (paper_right - 4, paper_top + 22, paper_right - 4, final_height - 46),
                fill="#9bd7df",
                width=2,
            )
            output = io.BytesIO()
            cropped = image.crop(
                (
                    0,
                    0,
                    width * render_scale,
                    final_height * render_scale,
                )
            )
            resampling = getattr(Image, "Resampling", None)
            resize_filter = (
                getattr(resampling, "LANCZOS", Image.LANCZOS)
                if resampling
                else Image.LANCZOS
            )
            cropped.resize((width, final_height), resize_filter).save(
                output,
                format="PNG",
                optimize=True,
            )
            return output.getvalue()
        except Exception as exc:
            logger.warning("梦幻版群聊日报图片生成失败，将回退：%s", exc)
            return None

    @classmethod
    def _render_report_image_legacy(
        cls,
        report: dict,
        report_label: str = "今日",
        width: int = 1080,
    ) -> bytes | None:
        """把日报渲染成可直接通过 OneBot base64 段发送的 PNG。

        版式参考目标项目的 ``format`` 编辑风格和 ``scrapbook`` 手账风格，
        但全部使用本地 Pillow 绘制，避免引入浏览器或外部图片资源。
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("Pillow 未安装，群聊日报将回退为文本")
            return None

        def font_path(bold: bool = False) -> str | None:
            configured = os.environ.get("ALICE_REPORT_FONT", "").strip()
            candidates = [
                configured,
                "C:/Windows/Fonts/Dengb.ttf" if bold else "C:/Windows/Fonts/Deng.ttf",
                "C:/Windows/Fonts/STKAITI.TTF" if bold else "C:/Windows/Fonts/STSONG.TTF",
                "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            ]
            for path in candidates:
                if path and os.path.exists(path):
                    return path
            return None

        regular_path = font_path(False)
        bold_path = font_path(True) or regular_path
        if not regular_path:
            logger.warning("未找到中文字体，群聊日报将回退为文本")
            return None

        def load_font(size: int, bold: bool = False):
            path = bold_path if bold else regular_path
            if path:
                try:
                    return ImageFont.truetype(path, size=size)
                except OSError:
                    pass
            return ImageFont.load_default()

        try:
            width = max(720, int(width))
            canvas_height = 7000
            margin = 56
            content_width = width - margin * 2

            # 目标项目已有的模板分别强调编辑式留白和手账式卡片；这里取中间值，
            # 让图片有性格，但在手机上仍然能快速读完。
            paper = "#f6f1e7"
            ink = "#302d2a"
            muted = "#766f68"
            white = "#fffdf8"
            orange = "#ec7951"
            blue = "#b9dce8"
            pink = "#f2c8c8"
            yellow = "#f8e4a8"
            green = "#c9dfc7"
            image = Image.new("RGB", (width, canvas_height), paper)
            draw = ImageDraw.Draw(image)
            hero_font = load_font(48, True)
            label_font = load_font(18, True)
            date_font = load_font(20)
            section_font = load_font(27, True)
            item_title_font = load_font(23, True)
            body_font = load_font(23)
            small_font = load_font(19)
            stat_font = load_font(40, True)
            stat_label_font = load_font(18, True)
            bot_name = cls._short_text(report.get("bot_name"), 30) or "我"
            stats = report.get("statistics", {}) or {}

            # 纸张底纹只在卡片之间露出，避免大面积纯色背景显得单调。
            for dot_x in range(24, width, 32):
                for dot_y in range(24, canvas_height, 32):
                    draw.ellipse((dot_x, dot_y, dot_x + 2, dot_y + 2), fill="#e8dfd2")

            def line_height(font, extra: int = 8) -> int:
                box = draw.textbbox((0, 0), "国Ag", font=font)
                return max(30, box[3] - box[1] + extra)

            def wrap(text: Any, font, max_width: int) -> list[str]:
                text = str(text or "").strip()
                if not text:
                    return []
                result: list[str] = []
                for paragraph in text.splitlines() or [""]:
                    current = ""
                    for char in paragraph:
                        candidate = current + char
                        if current and draw.textlength(candidate, font=font) > max_width:
                            result.append(current)
                            current = char
                        else:
                            current = candidate
                    if current:
                        result.append(current)
                return result

            def draw_lines(
                lines: list[str],
                x: int,
                y: int,
                font,
                fill: str,
                gap: int = 7,
            ) -> int:
                step = line_height(font, gap)
                for line in lines:
                    draw.text((x, y), line, font=font, fill=fill)
                    y += step + gap
                return y

            def card(
                x: int,
                y: int,
                card_width: int,
                card_height: int,
                fill: str = white,
                outline: str = ink,
                shadow_color: str | None = None,
                shadow_offset: tuple[int, int] = (7, 8),
                radius: int = 22,
            ) -> None:
                if shadow_color:
                    sx, sy = shadow_offset
                    draw.rounded_rectangle(
                        (x + sx, y + sy, x + card_width + sx, y + card_height + sy),
                        radius=radius,
                        fill=shadow_color,
                    )
                draw.rounded_rectangle(
                    (x, y, x + card_width, y + card_height),
                    radius=radius,
                    fill=fill,
                    outline=outline,
                    width=3,
                )

            def section_heading(x: int, y: int, title: str, accent: str = orange) -> None:
                draw.rectangle((x, y + 5, x + 8, y + 37), fill=accent)
                draw.text((x + 24, y), title, font=section_font, fill=ink)

            def draw_list_card(
                x: int,
                y: int,
                card_width: int,
                title: str,
                rows: list[tuple[str, str]],
                accent: str,
                fill: str = white,
            ) -> int:
                if not rows:
                    return y
                row_data: list[tuple[list[str], list[str], int]] = []
                row_width = card_width - 92
                title_step = line_height(item_title_font, 4)
                body_step = line_height(body_font, 4)
                for row_title, row_body in rows[:6]:
                    title_lines = wrap(row_title, item_title_font, row_width)
                    body_lines = wrap(row_body, body_font, row_width)
                    row_height = max(1, len(title_lines)) * title_step
                    if body_lines:
                        row_height += 8 + len(body_lines) * body_step
                    row_data.append((title_lines, body_lines, row_height + 22))

                header_height = 72
                card_height = header_height + sum(item[2] for item in row_data) + 22
                card(x, y, card_width, card_height, fill=fill, shadow_color=blue, radius=20)
                section_heading(x + 24, y + 20, title, accent)
                current_y = y + header_height
                for index, (title_lines, body_lines, row_height) in enumerate(row_data, 1):
                    marker_y = current_y + 3
                    draw.rounded_rectangle(
                        (x + 24, marker_y, x + 53, marker_y + 29),
                        radius=9,
                        fill=accent,
                    )
                    marker = "“" if title in {"我挑出的几句", "我忍不住记下的几句"} else str(index).zfill(2)
                    marker_box = draw.textbbox((0, 0), marker, font=small_font)
                    marker_width = marker_box[2] - marker_box[0]
                    draw.text(
                        (x + 38 - marker_width // 2, marker_y + 4),
                        marker,
                        font=small_font,
                        fill=ink if marker != "“" else white,
                    )
                    text_x = x + 70
                    current_y = draw_lines(title_lines, text_x, current_y, item_title_font, ink, gap=2)
                    if body_lines:
                        current_y += 5
                        current_y = draw_lines(body_lines, text_x, current_y, body_font, muted, gap=2)
                    current_y = y + header_height + sum(item[2] for item in row_data[:index])
                    if index < len(row_data):
                        draw.line(
                            (x + 24, current_y - 10, x + card_width - 24, current_y - 10),
                            fill="#e5ddd2",
                            width=2,
                        )
                return y + card_height + 24

            # 顶部标题卡片：编辑式大标题 + 手账式彩色压纸阴影。
            header_y = 46
            title_lines = wrap(f"{bot_name}的群聊日报", hero_font, content_width - 320)
            header_h = max(196, 126 + len(title_lines) * line_height(hero_font, 4))
            card(
                margin,
                header_y,
                content_width,
                header_h,
                fill=white,
                shadow_color=blue,
                shadow_offset=(12, 12),
                radius=25,
            )
            draw.rounded_rectangle(
                (margin + 17, header_y + 24, margin + 31, header_y + header_h - 24),
                radius=7,
                fill=orange,
            )
            draw.text(
                (margin + 58, header_y + 28),
                "群聊日报  /  GROUP DAILY",
                font=label_font,
                fill=orange,
            )
            draw_lines(title_lines, margin + 58, header_y + 67, hero_font, ink, gap=1)
            subtitle = f"{report_label} · 我今天看到的群聊"
            peak_hour = stats.get("peak_hour")
            if peak_hour is not None:
                subtitle += f" · 最活跃 {int(peak_hour):02d}:00"
            subtitle_y = header_y + 72 + len(title_lines) * line_height(hero_font, 4)
            draw.text((margin + 60, subtitle_y), subtitle, font=date_font, fill=muted)
            badge_width = 168
            badge_x = width - margin - badge_width - 22
            badge_y = header_y + 28
            draw.rounded_rectangle(
                (badge_x, badge_y, badge_x + badge_width, badge_y + 48),
                radius=16,
                fill=yellow,
                outline=ink,
                width=2,
            )
            draw.text((badge_x + 20, badge_y + 12), datetime.now().strftime("%Y.%m.%d"), font=date_font, fill=ink)
            y = header_y + header_h + 30

            # 统计区：左侧四格数字，右侧突出最活跃时段。
            stats_h = 274
            stats_left_w = int(content_width * 0.57)
            stats_gap = 24
            stats_right_w = content_width - stats_left_w - stats_gap
            card(margin, y, stats_left_w, stats_h, fill=white, shadow_color=pink, radius=20)
            section_heading(margin + 24, y + 20, "今天的数字", orange)
            metrics = [
                ("消息", stats.get("message_count", 0)),
                ("参与", stats.get("participant_count", 0)),
                ("文字", stats.get("total_characters", 0)),
                ("回复", stats.get("reply_count", 0)),
            ]
            tile_gap = 14
            tile_width = (stats_left_w - 48 - tile_gap) // 2
            tile_height = 88
            for index, (label, value) in enumerate(metrics):
                tile_x = margin + 24 + (index % 2) * (tile_width + tile_gap)
                tile_y = y + 74 + (index // 2) * (tile_height + tile_gap)
                tile_fill = ["#fff4d6", "#e4f1f4", "#fbe5e2", "#e5f0dd"][index]
                draw.rounded_rectangle(
                    (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height),
                    radius=15,
                    fill=tile_fill,
                    outline=ink,
                    width=2,
                )
                draw.text((tile_x + 18, tile_y + 12), str(value), font=stat_font, fill=ink)
                draw.text((tile_x + 20, tile_y + 61), label, font=stat_label_font, fill=muted)

            peak_x = margin + stats_left_w + stats_gap
            card(peak_x, y, stats_right_w, stats_h, fill="#333840", shadow_color=yellow, radius=20)
            draw.text((peak_x + 28, y + 28), "我注意到的", font=label_font, fill="#f5c56a")
            draw.text((peak_x + 28, y + 62), "活跃时刻", font=section_font, fill="#fffdf8")
            if peak_hour is not None:
                peak_text = f"{int(peak_hour):02d}:00"
                peak_hint = "大家在这个时段聊得最密"
            else:
                peak_text = "—"
                peak_hint = "今天还没有明显的活跃高峰"
            draw.text((peak_x + 28, y + 122), peak_text, font=load_font(54, True), fill="#ffffff")
            peak_lines = wrap(peak_hint, body_font, stats_right_w - 56)
            draw_lines(peak_lines, peak_x + 30, y + 198, body_font, "#d9e0e4", gap=2)
            y += stats_h + 28

            summary = cls._short_text(report.get("summary"), 180) or "暂时没提炼出总结。"
            summary_lines = wrap(summary, body_font, content_width - 100)
            atmosphere = cls._short_text(report.get("atmosphere"), 60)
            atmosphere_lines = wrap(atmosphere, small_font, content_width - 100) if atmosphere else []
            summary_h = 92 + len(summary_lines) * line_height(body_font, 4)
            if atmosphere_lines:
                summary_h += 26 + len(atmosphere_lines) * line_height(small_font, 2)
            card(margin, y, content_width, summary_h, fill=white, shadow_color=green, radius=20)
            section_heading(margin + 24, y + 20, "我看到的", orange)
            draw_lines(summary_lines, margin + 50, y + 76, body_font, ink, gap=2)
            if atmosphere_lines:
                atmosphere_y = y + 78 + len(summary_lines) * line_height(body_font, 4)
                draw.text((margin + 50, atmosphere_y), "我的感觉：", font=small_font, fill=orange)
                draw_lines(atmosphere_lines, margin + 154, atmosphere_y, small_font, muted, gap=1)
            y += summary_h + 30

            # 两栏内容对应目标项目的编辑式网格；长内容在各自卡片内独立增长。
            column_gap = 24
            column_width = (content_width - column_gap) // 2
            left_y = y
            right_y = y

            top_users = stats.get("top_users", []) or []
            if top_users:
                member_rows = [
                    (
                        str(item.get("name") or item.get("sender_id") or "匿名"),
                        f"今天发了 {item.get('message_count', 0)} 条消息，"
                        f"共 {item.get('char_count', 0)} 字。",
                    )
                    for item in top_users[:5]
                ]
                left_y = draw_list_card(
                    margin,
                    left_y,
                    column_width,
                    "我注意到这些人说得比较多",
                    member_rows,
                    blue,
                    fill="#f8fcfa",
                )

            topics = report.get("topics", []) or []
            if topics:
                topic_rows = [
                    (
                        str(item.get("name") or "未命名话题"),
                        cls._short_text(item.get("detail"), 100),
                    )
                    for item in topics[:6]
                ]
                left_y = draw_list_card(
                    margin,
                    left_y,
                    column_width,
                    "我注意到的话题",
                    topic_rows,
                    orange,
                    fill="#fffaf2",
                )

            sender_names = stats.get("sender_names", {}) or {}
            quotes = report.get("quotes", []) or []
            if quotes:
                quote_rows = [
                    (
                        f"{cls._name_for_id(item.get('sender_id', ''), top_users, sender_names)} 说：",
                        f"「{cls._short_text(item.get('content'), 90)}」  "
                        f"{cls._short_text(item.get('reason'), 70)}",
                    )
                    for item in quotes[:5]
                ]
                right_y = draw_list_card(
                    margin + column_width + column_gap,
                    right_y,
                    column_width,
                    "我忍不住记下的几句",
                    quote_rows,
                    pink,
                    fill="#fff8f7",
                )

            titles = report.get("titles", []) or []
            if titles:
                title_rows = [
                    (
                        f"{cls._name_for_id(item.get('sender_id', ''), top_users, sender_names)} · "
                        f"{cls._short_text(item.get('title'), 35)}",
                        cls._short_text(item.get('reason'), 90),
                    )
                    for item in titles[:5]
                ]
                right_y = draw_list_card(
                    margin + column_width + column_gap,
                    right_y,
                    column_width,
                    "我想给的称号",
                    title_rows,
                    yellow,
                    fill="#fffdf2",
                )

            y = max(left_y, right_y)
            if report.get("analysis_error"):
                note_lines = wrap("AI 分析暂不可用，以上是我按看到的消息做的本地统计。", small_font, content_width - 70)
                note_h = 58 + len(note_lines) * line_height(small_font, 2)
                card(margin, y, content_width, note_h, fill="#fff4d6", shadow_color=pink, radius=18)
                draw.text((margin + 26, y + 18), "备注", font=label_font, fill=orange)
                draw_lines(note_lines, margin + 112, y + 18, small_font, muted, gap=1)
                y += note_h + 24

            # 收尾保留短信息，不加入目标项目的品牌或外部资源。
            footer_h = 104
            footer_y = y + 4
            card(margin, footer_y, content_width, footer_h, fill=ink, shadow_color=blue, radius=20)
            draw.text((margin + 28, footer_y + 24), f"{bot_name} · {report_label}", font=section_font, fill=white)
            draw.text(
                (margin + 30, footer_y + 68),
                "我只根据今天实际看到的群聊消息做这份记录。",
                font=small_font,
                fill="#d9d2c9",
            )
            stamp = "GROUP DAILY"
            stamp_box = draw.textbbox((0, 0), stamp, font=label_font)
            draw.text(
                (width - margin - 28 - (stamp_box[2] - stamp_box[0]), footer_y + 32),
                stamp,
                font=label_font,
                fill="#f5c56a",
            )

            final_height = min(canvas_height, max(footer_y + footer_h + margin, 420))
            output = io.BytesIO()
            image.crop((0, 0, width, final_height)).save(output, format="PNG", optimize=True)
            return output.getvalue()
        except Exception as exc:
            logger.warning("群聊日报图片生成失败，将回退为文本：%s", exc)
            return None

    @classmethod
    def _render_report_image_rich(
        cls,
        report: dict,
        report_label: str = "今日",
        width: int = 1080,
    ) -> bytes | None:
        """以参考项目的日记、画像、金句和锐评结构渲染日报。"""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("Pillow 未安装，群聊日报将回退为文本")
            return None

        def font_path(bold: bool = False) -> str | None:
            configured = os.environ.get("ALICE_REPORT_FONT", "").strip()
            candidates = [
                configured,
                "C:/Windows/Fonts/Dengb.ttf" if bold else "C:/Windows/Fonts/Deng.ttf",
                "C:/Windows/Fonts/STKAITI.TTF" if bold else "C:/Windows/Fonts/STSONG.TTF",
                "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            ]
            for path in candidates:
                if path and os.path.exists(path):
                    return path
            return None

        regular_path = font_path(False)
        bold_path = font_path(True) or regular_path
        if not regular_path:
            logger.warning("未找到中文字体，群聊日报将回退为文本")
            return None

        def load_font(size: int, bold: bool = False):
            path = bold_path if bold else regular_path
            try:
                return ImageFont.truetype(path, size=size) if path else ImageFont.load_default()
            except OSError:
                return ImageFont.load_default()

        try:
            width = max(720, int(width))
            canvas_height = 10000
            margin = 54
            content_width = width - margin * 2
            paper = "#f7f1e6"
            ink = "#302d2a"
            muted = "#716962"
            white = "#fffdf8"
            orange = "#ec7951"
            blue = "#b8dce9"
            pink = "#f2c8cf"
            yellow = "#f6dfa2"
            green = "#c8dfc7"
            purple = "#d8c9e9"
            image = Image.new("RGB", (width, canvas_height), paper)
            draw = ImageDraw.Draw(image)

            hero_font = load_font(48, True)
            section_font = load_font(28, True)
            item_title_font = load_font(24, True)
            body_font = load_font(22)
            quote_font = load_font(25, True)
            small_font = load_font(18)
            small_bold_font = load_font(18, True)
            stat_font = load_font(40, True)
            badge_font = load_font(17, True)

            for dot_x in range(22, width, 32):
                for dot_y in range(22, canvas_height, 32):
                    draw.ellipse((dot_x, dot_y, dot_x + 2, dot_y + 2), fill="#e8dfd2")

            stats = report.get("statistics", {}) or {}
            bot_name = cls._short_text(report.get("bot_name"), 30) or "我"
            report_title = cls._short_text(report.get("title"), 42) or "今天群里聊到了一些小事"
            report_subtitle = cls._short_text(report.get("subtitle"), 70)
            top_users = stats.get("top_users", []) or []
            sender_names = stats.get("sender_names", {}) or {}
            palette = [orange, blue, pink, green, yellow, purple]
            avatar_paths = report.get("avatars", {}) or {}
            avatar_cache: dict[str, Any] = {}

            def step(font, extra: int = 8) -> int:
                box = draw.textbbox((0, 0), "国Ag", font=font)
                return max(28, box[3] - box[1] + extra)

            def wrap(text: Any, font, max_width: int) -> list[str]:
                text = str(text or "").strip()
                if not text:
                    return []
                result: list[str] = []
                for paragraph in text.splitlines() or [""]:
                    current = ""
                    for char in paragraph:
                        candidate = current + char
                        if current and draw.textlength(candidate, font=font) > max_width:
                            result.append(current)
                            current = char
                        else:
                            current = candidate
                    if current:
                        result.append(current)
                return result

            def lines_height(lines: list[str], font, gap: int = 0) -> int:
                return len(lines) * (step(font, 6) + gap)

            def draw_lines(
                lines: list[str], x: int, y: int, font, fill: str, gap: int = 0
            ) -> int:
                line_step = step(font, 6) + gap
                for line in lines:
                    draw.text((x, y), line, font=font, fill=fill)
                    y += line_step
                return y

            def card(
                x: int,
                y: int,
                card_width: int,
                card_height: int,
                fill: str = white,
                outline: str = ink,
                shadow_color: str | None = None,
                shadow_offset: tuple[int, int] = (7, 8),
                radius: int = 20,
                outline_width: int = 3,
            ) -> None:
                if shadow_color:
                    sx, sy = shadow_offset
                    draw.rounded_rectangle(
                        (x + sx, y + sy, x + card_width + sx, y + card_height + sy),
                        radius=radius,
                        fill=shadow_color,
                    )
                draw.rounded_rectangle(
                    (x, y, x + card_width, y + card_height),
                    radius=radius,
                    fill=fill,
                    outline=outline,
                    width=outline_width,
                )

            def avatar(
                x: int,
                y: int,
                name: str,
                radius: int = 30,
                accent: str | None = None,
                sender_id: str = "",
            ) -> None:
                name = str(name or "?").strip() or "?"
                seed = sum(ord(char) for char in name)
                fill = accent or palette[seed % len(palette)]
                _draw_report_avatar(
                    image,
                    draw,
                    avatar_paths,
                    avatar_cache,
                    sender_id,
                    name,
                    x,
                    y,
                    radius,
                    fill,
                    ink,
                    small_bold_font,
                    ink,
                )

            def pill(x: int, y: int, text: str, fill: str, text_fill: str = ink) -> int:
                text = cls._short_text(text, 22)
                if not text:
                    return x
                text_width = int(draw.textlength(text, font=badge_font))
                pill_width = text_width + 24
                draw.rounded_rectangle(
                    (x, y, x + pill_width, y + 34), radius=10, fill=fill, outline=ink, width=2
                )
                draw.text((x + 12, y + 7), text, font=badge_font, fill=text_fill)
                return x + pill_width + 8

            def heading(x: int, y: int, title: str, accent: str = orange) -> None:
                draw.rectangle((x, y + 5, x + 8, y + 40), fill=accent)
                draw.text((x + 24, y), title, font=section_font, fill=ink)

            def display_name(sender_id: str) -> str:
                return cls._short_text(
                    cls._name_for_id(sender_id, top_users, sender_names), 12
                ) or str(sender_id)

            # 1. 标题：目标项目用的是“今天发生了什么”的日记标题，而不是纯报表标题。
            title_lines = wrap(report_title, hero_font, content_width - 320)
            subtitle_text = report_subtitle or f"{report_label} · 我把今天注意到的几件小事记下来"
            subtitle_lines = wrap(subtitle_text, body_font, content_width - 360)
            header_y = 44
            header_h = max(
                214,
                142
                + lines_height(title_lines, hero_font, 2)
                + lines_height(subtitle_lines, body_font, 1),
            )
            card(
                margin,
                header_y,
                content_width,
                header_h,
                fill=white,
                shadow_color=blue,
                shadow_offset=(12, 12),
                radius=25,
            )
            draw.rounded_rectangle(
                (margin + 18, header_y + 25, margin + 32, header_y + header_h - 25),
                radius=7,
                fill=orange,
            )
            draw.text(
                (margin + 60, header_y + 28),
                "群聊日报  /  GROUP DAILY",
                font=badge_font,
                fill=orange,
            )
            draw_lines(title_lines, margin + 60, header_y + 70, hero_font, ink, gap=2)
            subtitle_y = header_y + 76 + lines_height(title_lines, hero_font, 2)
            draw_lines(subtitle_lines, margin + 62, subtitle_y, body_font, muted, gap=1)
            date_text = datetime.now().strftime("%Y.%m.%d")
            badge_width = 168
            badge_x = width - margin - badge_width - 20
            badge_y = header_y + 28
            draw.rounded_rectangle(
                (badge_x, badge_y, badge_x + badge_width, badge_y + 48),
                radius=16,
                fill=yellow,
                outline=ink,
                width=2,
            )
            draw.text((badge_x + 20, badge_y + 12), date_text, font=body_font, fill=ink)
            y = header_y + header_h + 30

            # 2. 统计与 24 小时活跃轨迹。
            stats_h = 264
            left_width = int(content_width * 0.57)
            gap = 24
            right_width = content_width - left_width - gap
            card(margin, y, left_width, stats_h, fill=white, shadow_color=pink)
            heading(margin + 24, y + 20, "今天留下的数字", orange)
            metrics = [
                ("消息", stats.get("message_count", 0), "#fff1cf"),
                ("参与", stats.get("participant_count", 0), "#e3f0f4"),
                ("文字", stats.get("total_characters", 0), "#fbe3e0"),
                ("回复", stats.get("reply_count", 0), "#e3efd9"),
            ]
            tile_gap = 14
            tile_width = (left_width - 48 - tile_gap) // 2
            tile_height = 86
            for index, (label, value, fill) in enumerate(metrics):
                tile_x = margin + 24 + (index % 2) * (tile_width + tile_gap)
                tile_y = y + 76 + (index // 2) * (tile_height + tile_gap)
                draw.rounded_rectangle(
                    (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height),
                    radius=14,
                    fill=fill,
                    outline=ink,
                    width=2,
                )
                draw.text((tile_x + 18, tile_y + 11), str(value), font=stat_font, fill=ink)
                draw.text((tile_x + 20, tile_y + 60), label, font=small_bold_font, fill=muted)

            peak_hour = stats.get("peak_hour")
            peak_text = f"{int(peak_hour):02d}:00" if peak_hour is not None else "—"
            peak_hint = "我注意到大家在这个时段聊得最密" if peak_hour is not None else "今天还没有明显高峰"
            card(margin + left_width + gap, y, right_width, stats_h, fill="#333840", shadow_color=yellow)
            draw.text((margin + left_width + gap + 28, y + 28), "我注意到的", font=badge_font, fill="#f5c56a")
            draw.text((margin + left_width + gap + 28, y + 66), "活跃时刻", font=section_font, fill=white)
            draw.text((margin + left_width + gap + 28, y + 126), peak_text, font=load_font(54, True), fill=white)
            draw_lines(
                wrap(peak_hint, body_font, right_width - 56),
                margin + left_width + gap + 30,
                y + 204,
                body_font,
                "#d9e0e4",
                gap=2,
            )
            y += stats_h + 28

            chart_h = 224
            card(margin, y, content_width, chart_h, fill=white, shadow_color=blue)
            heading(margin + 24, y + 20, "24 小时活跃轨迹", blue)
            chart_left = margin + 38
            chart_right = width - margin - 32
            chart_top = y + 86
            chart_bottom = y + 170
            activity = stats.get("hourly_activity", {}) or {}
            counts = [int(activity.get(str(hour), 0) or 0) for hour in range(24)]
            max_count = max(counts) if counts else 0
            draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=ink, width=2)
            slot_width = max(16, (chart_right - chart_left) // 24)
            bar_width = max(8, slot_width - 12)
            for hour, count in enumerate(counts):
                bar_height = int((count / max_count) * 70) if max_count else 0
                bar_x = chart_left + hour * slot_width + (slot_width - bar_width) // 2
                if count:
                    bar_fill = orange if peak_hour is not None and hour == int(peak_hour) else blue
                    draw.rounded_rectangle(
                        (bar_x, chart_bottom - bar_height, bar_x + bar_width, chart_bottom),
                        radius=4,
                        fill=bar_fill,
                    )
                if hour % 3 == 0:
                    draw.text((bar_x - 4, chart_bottom + 12), f"{hour:02d}", font=small_font, fill=muted)
            y += chart_h + 30

            # 3. 日记摘要：用头像和气泡承载 Bot 的第一人称，而不是孤零零的一段说明。
            summary = cls._short_text(report.get("summary"), 240) or "我暂时没提炼出更具体的故事。"
            summary_lines = wrap(summary, body_font, content_width - 190)
            atmosphere = cls._short_text(report.get("atmosphere"), 80)
            atmosphere_lines = wrap(atmosphere, small_font, content_width - 220) if atmosphere else []
            bubble_height = 50 + lines_height(summary_lines, body_font, 3)
            if atmosphere_lines:
                bubble_height += 22 + lines_height(atmosphere_lines, small_font, 1)
            story_height = max(178, bubble_height + 80)
            card(margin, y, content_width, story_height, fill=white, shadow_color=green)
            heading(margin + 24, y + 20, "我今天记下的", orange)
            avatar(margin + 72, y + 112, bot_name, radius=32, accent=orange)
            bubble_x = margin + 132
            bubble_y = y + 70
            bubble_width = content_width - 172
            draw.rounded_rectangle(
                (bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height),
                radius=24,
                fill="#fffaf0",
                outline=ink,
                width=3,
            )
            draw.polygon(
                [(bubble_x, bubble_y + 32), (bubble_x - 18, bubble_y + 48), (bubble_x, bubble_y + 60)],
                fill="#fffaf0",
                outline=ink,
            )
            draw_lines(summary_lines, bubble_x + 26, bubble_y + 22, body_font, ink, gap=3)
            if atmosphere_lines:
                atmosphere_y = bubble_y + 28 + lines_height(summary_lines, body_font, 3)
                draw.text((bubble_x + 26, atmosphere_y), "我的感觉：", font=small_bold_font, fill=orange)
                draw_lines(atmosphere_lines, bubble_x + 128, atmosphere_y, small_font, muted, gap=1)
            y += story_height + 32

            # 4. 今日话题：每条话题都带参与者和过程，不再只有一行摘要。
            topics = report.get("topics", []) or []
            if topics:
                topic_header = 72
                topic_data = []
                for item in topics[:6]:
                    name = cls._short_text(item.get("name"), 40) or "未命名话题"
                    detail_lines = wrap(cls._short_text(item.get("detail"), 220), body_font, content_width - 158)
                    ids = [str(value) for value in (item.get("sender_ids") or []) if str(value)]
                    names = [display_name(value) for value in ids[:5]]
                    participant_text = "参与：" + ("、".join(names) if names else "我从消息里看到的几位群友")
                    row_height = 72 + lines_height(detail_lines, body_font, 3)
                    topic_data.append((name, participant_text, detail_lines, row_height))
                topic_height = topic_header + sum(row[3] + 16 for row in topic_data) + 18
                card(margin, y, content_width, topic_height, fill="#fffdf8", shadow_color=orange)
                heading(margin + 24, y + 20, "今日话题", orange)
                current_y = y + topic_header
                for index, (name, participant_text, detail_lines, row_height) in enumerate(topic_data, 1):
                    row_y = current_y
                    row_height -= 16
                    draw.rounded_rectangle(
                        (margin + 22, row_y, width - margin - 22, row_y + row_height),
                        radius=14,
                        fill=white,
                        outline="#d8d0c5",
                        width=2,
                    )
                    marker_x = margin + 48
                    marker_y = row_y + 22
                    draw.rounded_rectangle(
                        (marker_x - 18, marker_y - 4, marker_x + 18, marker_y + 32),
                        radius=10,
                        fill=[orange, blue, pink, green, yellow, purple][(index - 1) % 6],
                    )
                    marker = f"{index:02d}"
                    marker_box = draw.textbbox((0, 0), marker, font=small_bold_font)
                    draw.text(
                        (marker_x - (marker_box[2] - marker_box[0]) / 2, marker_y + 4),
                        marker,
                        font=small_bold_font,
                        fill=ink,
                    )
                    text_x = margin + 86
                    draw.text((text_x, row_y + 18), name, font=item_title_font, fill=ink)
                    draw.text((text_x, row_y + 55), participant_text, font=small_font, fill=muted)
                    draw_lines(detail_lines, text_x, row_y + 87, body_font, ink, gap=3)
                    current_y += row_height + 16
                y += topic_height + 32

            # 5. 群友画像：头像占位、称号、轻量标签和具体理由。
            profiles = report.get("profiles") or report.get("titles") or []
            if profiles:
                profile_header = 72
                profile_data = []
                profile_width = (content_width - 24) // 2
                for item in profiles[:8]:
                    sender_id = str(item.get("sender_id") or "")
                    name = display_name(sender_id)
                    profile_title = cls._short_text(item.get("title"), 28) or "今日观察"
                    profile_badge = cls._short_text(item.get("mbti"), 22)
                    reason_lines = wrap(cls._short_text(item.get("reason"), 140), small_font, profile_width - 48)
                    profile_height = max(162, 112 + lines_height(reason_lines, small_font, 2))
                    profile_data.append((sender_id, name, profile_title, profile_badge, reason_lines, profile_height))
                profile_height_total = profile_header
                for index in range(0, len(profile_data), 2):
                    profile_height_total += max(item[5] for item in profile_data[index:index + 2]) + 18
                profile_height_total += 18
                card(margin, y, content_width, profile_height_total, fill="#fffdf8", shadow_color=purple)
                heading(margin + 24, y + 20, "群友画像", purple)
                row_y = y + profile_header
                for index in range(0, len(profile_data), 2):
                    row_items = profile_data[index:index + 2]
                    row_height = max(item[5] for item in row_items)
                    for col, item in enumerate(row_items):
                        sender_id, name, profile_title, profile_badge, reason_lines, _ = item
                        card_x = margin + col * (profile_width + 24)
                        fill = ["#fff9e6", "#f4f0ff", "#eef8f4", "#fff1ef"][index // 2 % 4]
                        card(card_x, row_y, profile_width, row_height, fill=fill, shadow_color=None, radius=16, outline_width=2)
                        avatar(card_x + 43, row_y + 46, name, radius=27, sender_id=sender_id)
                        draw.text((card_x + 82, row_y + 22), name, font=item_title_font, fill=ink)
                        next_x = card_x + 82
                        next_x = pill(next_x, row_y + 58, profile_title, yellow)
                        if profile_badge:
                            pill(next_x, row_y + 58, profile_badge, blue)
                        draw.line(
                            (card_x + 24, row_y + 101, card_x + profile_width - 24, row_y + 101),
                            fill="#ded6cb",
                            width=2,
                        )
                        draw_lines(reason_lines, card_x + 24, row_y + 120, small_font, muted, gap=2)
                    row_y += row_height + 18
                y += profile_height_total + 32

            # 6. 金句：还原成一来一回的聊天气泡，并保留 AI 的具体锐评。
            quotes = report.get("quotes", []) or []
            if quotes:
                quote_header = 72
                quote_data = []
                bubble_width = int(content_width * 0.76)
                for item in quotes[:6]:
                    sender_id = str(item.get("sender_id") or "")
                    name = display_name(sender_id)
                    content_lines = wrap(cls._short_text(item.get("content"), 220), quote_font, bubble_width - 56)
                    reason_lines = wrap(cls._short_text(item.get("reason"), 110), small_font, bubble_width - 56)
                    bubble_height = 42 + lines_height(content_lines, quote_font, 2) + 20
                    bubble_height += 28 + lines_height(reason_lines, small_font, 1) + 20
                    quote_data.append((sender_id, name, content_lines, reason_lines, bubble_height))
                quote_height = quote_header + sum(max(112, item[4]) + 30 for item in quote_data) + 10
                card(margin, y, content_width, quote_height, fill="#fffdf8", shadow_color=pink)
                heading(margin + 24, y + 20, "群聊金句", pink)
                current_y = y + quote_header
                for index, (sender_id, name, content_lines, reason_lines, bubble_height) in enumerate(quote_data):
                    row_height = max(112, bubble_height)
                    right_aligned = index % 2 == 1
                    if right_aligned:
                        avatar_x = width - margin - 38
                        bubble_x = width - margin - 88 - bubble_width
                    else:
                        avatar_x = margin + 38
                        bubble_x = margin + 88
                    avatar_y = current_y + 42
                    avatar(avatar_x, avatar_y, name, radius=28, sender_id=sender_id)
                    name_x = bubble_x + 20 if not right_aligned else bubble_x + bubble_width - 20 - int(draw.textlength(name, font=small_bold_font))
                    draw.text((name_x, current_y + 4), name, font=small_bold_font, fill=ink)
                    bubble_y = current_y + 34
                    bubble_fill = "#fff1f5" if not right_aligned else "#fff6d9"
                    draw.rounded_rectangle(
                        (bubble_x, bubble_y, bubble_x + bubble_width, bubble_y + bubble_height),
                        radius=22,
                        fill=bubble_fill,
                        outline=ink,
                        width=3,
                    )
                    tail = (
                        [(bubble_x, bubble_y + 24), (bubble_x - 18, bubble_y + 38), (bubble_x, bubble_y + 52)]
                        if not right_aligned
                        else [(bubble_x + bubble_width, bubble_y + 24), (bubble_x + bubble_width + 18, bubble_y + 38), (bubble_x + bubble_width, bubble_y + 52)]
                    )
                    draw.polygon(tail, fill=bubble_fill, outline=ink)
                    quote_y = bubble_y + 22
                    quote_y = draw_lines(content_lines, bubble_x + 28, quote_y, quote_font, ink, gap=2)
                    quote_y += 8
                    draw_lines(reason_lines, bubble_x + 28, quote_y, small_font, muted, gap=1)
                    current_y += row_height + 30
                y += quote_height + 32

            # 7. 聊天质量锐评：参考项目里最有“人”的部分，保留具体吐槽和总评气泡。
            quality = report.get("quality_review", {}) or {}
            dimensions = quality.get("dimensions", []) if isinstance(quality, dict) else []
            if isinstance(quality, dict) and dimensions:
                quality_title = cls._short_text(quality.get("title"), 42) or "今天的群聊主题"
                quality_subtitle = cls._short_text(quality.get("subtitle"), 60) or report_label
                quality_summary = cls._short_text(quality.get("summary"), 150)
                dimension_data = []
                dim_width = (content_width - 42) // 2
                for item in dimensions[:6]:
                    dim_name = cls._short_text(item.get("name"), 20)
                    dim_comment = cls._short_text(item.get("comment"), 120)
                    dim_lines = wrap(dim_comment, small_font, dim_width - 34)
                    dimension_data.append((dim_name, item.get("percentage", 0), dim_lines))
                dimension_rows = 0
                for index in range(0, len(dimension_data), 2):
                    dimension_rows += max(100, *(92 + lines_height(item[2], small_font, 2) for item in dimension_data[index:index + 2])) + 16
                summary_lines = wrap(quality_summary, body_font, content_width - 190) if quality_summary else []
                quality_height = 142 + dimension_rows + (100 + lines_height(summary_lines, body_font, 3) if summary_lines else 0)
                card(margin, y, content_width, quality_height, fill="#fff9df", shadow_color=yellow)
                heading(margin + 24, y + 20, "群聊质量锐评", orange)
                title_x = margin + 28
                title_y = y + 76
                pill(title_x, title_y, quality_title, orange, white)
                draw.text((width - margin - 28 - int(draw.textlength(quality_subtitle, font=small_bold_font)), title_y + 8), quality_subtitle, font=small_bold_font, fill=muted)
                bar_x = margin + 28
                bar_y = y + 128
                bar_width = content_width - 56
                draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + 34), radius=12, fill=white, outline=ink, width=2)
                total_percentage = sum(max(0, float(item[1] or 0)) for item in dimension_data) or 1
                colors = [orange, pink, purple, blue, green, yellow]
                current_x = bar_x + 3
                for index, (_, percentage, _) in enumerate(dimension_data):
                    segment_width = int((max(0, float(percentage or 0)) / total_percentage) * (bar_width - 6))
                    if segment_width <= 0:
                        continue
                    draw.rectangle((current_x, bar_y + 3, current_x + segment_width, bar_y + 31), fill=colors[index % len(colors)])
                    current_x += segment_width
                row_y = y + 184
                for index in range(0, len(dimension_data), 2):
                    row_items = dimension_data[index:index + 2]
                    row_height = max(100, *(92 + lines_height(item[2], small_font, 2) for item in row_items))
                    for col, (dim_name, percentage, dim_lines) in enumerate(row_items):
                        dim_x = margin + 24 + col * (dim_width + 18)
                        fill = ["#fff1cf", "#ffe5e0", "#eee6fa", "#e1f0f4", "#e6f2df", "#fff5c8"][index // 2 % 6]
                        card(dim_x, row_y, dim_width, row_height, fill=fill, shadow_color=None, radius=12, outline_width=2)
                        draw.text((dim_x + 16, row_y + 14), dim_name or "未命名", font=small_bold_font, fill=orange)
                        draw.text((dim_x + dim_width - 70, row_y + 14), f"{float(percentage or 0):.0f}%", font=small_bold_font, fill=muted)
                        draw_lines(dim_lines, dim_x + 16, row_y + 52, small_font, ink, gap=2)
                    row_y += row_height + 16
                if summary_lines:
                    summary_y = y + quality_height - 34 - lines_height(summary_lines, body_font, 3)
                    avatar(margin + 70, summary_y + 30, bot_name, radius=28, accent=orange)
                    bubble_x = margin + 116
                    bubble_w = content_width - 148
                    bubble_h = 30 + lines_height(summary_lines, body_font, 3)
                    draw.rounded_rectangle((bubble_x, summary_y, bubble_x + bubble_w, summary_y + bubble_h), radius=20, fill=white, outline=ink, width=2)
                    draw_lines(summary_lines, bubble_x + 22, summary_y + 16, body_font, ink, gap=3)
                y += quality_height + 32

            footer_height = 112
            footer_y = y + 4
            card(margin, footer_y, content_width, footer_height, fill=ink, shadow_color=blue, radius=20)
            draw.text((margin + 28, footer_y + 24), f"{bot_name} · {report_label}", font=section_font, fill=white)
            draw.text(
                (margin + 30, footer_y + 72),
                "我只根据今天实际看到的群聊消息，留下一页自己的记录。",
                font=small_font,
                fill="#d9d2c9",
            )
            stamp = "GROUP DAILY"
            stamp_width = int(draw.textlength(stamp, font=badge_font))
            draw.text((width - margin - 28 - stamp_width, footer_y + 32), stamp, font=badge_font, fill="#f5c56a")

            final_height = min(canvas_height, max(footer_y + footer_height + margin, 520))
            output = io.BytesIO()
            image.crop((0, 0, width, final_height)).save(output, format="PNG", optimize=True)
            return output.getvalue()
        except Exception as exc:
            logger.warning("丰富版群聊日报图片生成失败，将回退为文本：%s", exc)
            return None

    @staticmethod
    def _name_for_id(
        sender_id: str, top_users: list[dict], sender_names: dict[str, str] | None = None
    ) -> str:
        for user in top_users:
            if str(user.get("sender_id")) == str(sender_id):
                return str(user.get("name") or sender_id)
        if sender_names and str(sender_id) in sender_names:
            return str(sender_names[str(sender_id)] or sender_id)
        return str(sender_id)

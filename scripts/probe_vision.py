"""视觉识别质量探针。

背景：生产里图片描述几乎全是「一张图，画面是[动画表情]」或「看不清画面」，
用户反馈"认不出图里的人物是谁，只描述了画面长什么样"。

两种模式：

  pipeline（默认，最有说服力）
      用生产同一份配置造一个真的 RichMediaEnricher，直接调 _describe_image
      跑完整链路：真实 prompt → 下载 → 缩放压体积 → base64 → 端点 → 解析
      → segment.summary。打印的就是群里会看到的那句话。
      实测这条链路是唯一能同时暴露「10MiB 上限」「5MB 下载上限」「prompt 禁
      识别」三个问题的测法。

  prompts（A/B）
      同一张图跑两套 prompt 对比，用来回答"是 prompt 的锅还是模型的锅"。
      注意 production 那一路**直接调用 RichMediaEnricher._build_image_prompt**，
      不再在这里抄一份规则——抄一份就会漂移，而漂移过的探针会把结论带偏。

用法（容器内 /app）：
    python3 /app/scripts/probe_vision.py --dirs /tmp/vsamples
    python3 /app/scripts/probe_vision.py --mode prompts --prompts production,identify
    python3 /app/scripts/probe_vision.py --models MiniMax-M3,MiniMax-M3.1-Flash-Preview
    python3 /app/scripts/probe_vision.py --save /app/logs/vision_probe.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.environ.get("ALICE_CONFIG", "config/config.yaml")

# 候选新 prompt：把「认人/认作品」从禁止改成鼓励，只保留「不许编造」这条底线。
# 仅用于 A/B 对照；生产那一路永远取代码里的真实 prompt。
IDENTIFY_BASE_PROMPT = (
    "先认出这张图是什么，再用一两句话把最有用的信息说清楚。"
)

IDENTIFY_EXTRA = (
    "识别要求：\n"
    "1. 只要你认得画面里的人、角色、作品、梗、名场面、地标、品牌、商品、"
    "动物品种或界面截图里的软件，就直接点名。这是最有用的信息——"
    "只写「一个长发女孩」等于没认出来。\n"
    "2. 认得出作品就写全「作品名 + 角色名」，认得出真人就写名字或身份，"
    "认得出梗就写梗名（例如「熊猫头」「借口龙」）。\n"
    "3. 画面里的文字照抄下来，它常常就是这张图的梗本身。\n"
    "4. 认不出来、或只是看着像但没有把握，就退回客观画面描述，"
    "并直说「看不出具体是谁」——不要硬给一个名字。\n"
    "5. 绝对不要编造角色名、作品名、出处或剧情；拿不准就把 uncertain 置为 true。\n"
    "6. 不要结合群聊推断谁在说谁，不要替群友评价这张图、也不要写「适合用来……」；"
    "不要加引号复述群友说过的句子。"
)

JSON_RULE = (
    "只输出 JSON，不要输出解释，格式如："
    '{"description":"画面中可直接看到的事实",'
    '"confidence":0.85,"uncertain":false}。'
    "无法确认的内容不要猜，confidence 低于0.65时 uncertain 必须为 true。"
)


def load_image_config() -> dict:
    """复刻 main._get_rich_media_config 的合并逻辑，拿到 enricher 真正看到的那份配置。"""
    from core.config_store import load_config

    cfg = load_config(CONFIG_PATH)
    top_image = dict(cfg.get("image", {}) or {})
    rich_image = dict((cfg.get("rich_media", {}) or {}).get("image", {}) or {})
    image_config = {**rich_image, **top_image}
    if isinstance(rich_image.get("vision"), dict) and isinstance(top_image.get("vision"), dict):
        image_config["vision"] = {**rich_image["vision"], **top_image["vision"]}
    return cfg, image_config


def build_vision_provider(model_override: str | None = None, image_config: dict | None = None):
    """复刻 main._init_vision_provider 的合并逻辑，保证读到的是生产同一份配置。"""
    from modules.llm.openai_provider import create_provider

    cfg, merged = (None, None)
    if image_config is None:
        cfg, merged = load_image_config()
    else:
        merged = image_config

    vc = dict(merged.get("vision", {}) or {})
    primary: dict = {}
    if cfg is None:
        from core.config_store import load_config

        primary = (load_config(CONFIG_PATH).get("llm") or {}).get("primary") or {}
    else:
        primary = (cfg.get("llm") or {}).get("primary") or {}
    if not str(vc.get("api_key") or "").strip():
        vc["api_key"] = primary.get("api_key", "")
    if not str(vc.get("base_url") or "").strip():
        vc["base_url"] = primary.get("base_url") or "https://api.minimax.cn/v1"
    if not str(vc.get("model") or "").strip():
        vc["model"] = primary.get("model") or "MiniMax-M3"
    if not str(vc.get("provider_type") or "").strip():
        vc["provider_type"] = primary.get("provider_type") or "openai_compatible"
    if model_override:
        vc["model"] = model_override

    provider = create_provider(
        vc.get("provider_type", "openai_compatible"),
        {
            "provider_type": vc.get("provider_type", "openai_compatible"),
            "api_key": vc.get("api_key", ""),
            "base_url": vc.get("base_url", "https://api.minimax.cn/v1"),
            "model": vc.get("model", "MiniMax-M3"),
            "timeout": vc.get("timeout", 60),
            "temperature": vc.get("temperature", 0.4),
            "max_tokens": merged.get("to_text_max_tokens", 900),
        },
    )
    return provider, {
        "provider_type": vc.get("provider_type"),
        "base_url": vc.get("base_url"),
        "model": vc.get("model"),
        "has_key": bool(str(vc.get("api_key") or "").strip()),
        "to_text_prompt": merged.get("to_text_prompt"),
        "to_text_max_tokens": merged.get("to_text_max_tokens"),
        "max_download_bytes": merged.get("max_download_bytes"),
        "vision_max_side": merged.get("vision_max_side"),
        "vision_jpeg_quality": merged.get("vision_jpeg_quality"),
        "vision_max_payload_bytes": merged.get("vision_max_payload_bytes"),
    }


def build_enricher(model_override: str | None = None):
    """造一个真的 RichMediaEnricher（不接 QQ，只跑识别链路）。"""
    from core.adapter.rich_media import RichMediaEnricher

    _, image_config = load_image_config()
    provider, effective = build_vision_provider(model_override, image_config)
    enricher = RichMediaEnricher(
        {"image": image_config}, lambda *_: None, vision_provider=provider
    )
    return enricher, effective


def collect_images(root: str, limit: int) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    found: list[str] = []
    if os.path.isfile(root):
        return [root]
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in exts:
                found.append(os.path.join(dirpath, name))
    found.sort()
    return found[:limit]


def to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as handle:
        raw = handle.read()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


async def ask(
    provider, model: str, prompt: str, data_url: str, timeout: float, max_tokens: int
):
    from modules.llm.base import ChatMessage, ChatRequest

    request = ChatRequest(
        model=model or "",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            )
        ],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    started = time.time()
    try:
        response = await asyncio.wait_for(provider.chat(request), timeout=timeout)
    except Exception as exc:  # 单次失败不能毁掉整轮对照
        return {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - started, 1)}
    text = (response.content or "").strip()
    return {"raw": text, "seconds": round(time.time() - started, 1)}


def parse_description(raw: str) -> str:
    """复用生产解析：拿 description 字段，拿不到就用原文。"""
    from core.adapter.rich_media import RichMediaEnricher

    result = RichMediaEnricher._parse_vision_result(raw)
    if result is None:
        return "（空）"
    flag = "" if not result.uncertain else "  [uncertain]"
    return f"{result.description}  (conf={result.confidence:.2f}{flag})"


_THINK_RE = None


def split_think(raw: str) -> tuple[str, str]:
    """把 <think>…</think> 拆出来，方便人眼看清模型到底想到了什么。"""
    global _THINK_RE
    import re

    if _THINK_RE is None:
        _THINK_RE = re.compile(r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)", re.DOTALL | re.I)
    match = _THINK_RE.search(raw or "")
    if not match:
        return "", raw or ""
    think = match.group(1).strip()
    return think, (raw[: match.start()] + raw[match.end() :]).strip()


async def run_pipeline(images: list[str], args, records: list[dict]) -> None:
    """跑生产的真实识别链路，打印群里实际会看到的那句描述。"""
    from core.adapter.rich_content import MessageSegment
    from core.adapter.rich_media import RichMediaEnricher

    enricher, effective = build_enricher()
    print(f"视觉 provider：{effective['base_url']} / {effective['model']}")
    print(f"缩放参数：max_side={enricher.image_vision_max_side} "
          f"quality={enricher.image_vision_jpeg_quality} "
          f"payload≤{enricher.image_vision_max_payload}B "
          f"max_tokens={enricher.image_vision_max_tokens} "
          f"下载上限={enricher.image_max_download_bytes}B")
    print("=" * 78)

    # 只把「下载」换成读本地文件，其余全部是生产代码（缩放/编码/解析/入库格式）
    async def read_local(url: str) -> bytes | None:
        try:
            with open(url, "rb") as handle:
                return handle.read()
        except OSError:
            return None

    enricher._http_get_bytes = read_local  # type: ignore[assignment]

    # 诊断钩子：包装（不是替换）生产函数，行为完全不变，只是顺手记下
    # 「真正发出去的字节数」和「端点原始返回」，否则出了 uncertain 只能瞎猜。
    captured: dict[str, object] = {}

    real_parse = RichMediaEnricher._parse_vision_result  # staticmethod → 普通函数

    def record_parse(text: str):
        captured["raw"] = text
        return real_parse(text)

    real_to_url = enricher._to_vision_data_url

    def record_url(data: bytes, media_type: str) -> str:
        url = real_to_url(data, media_type)
        payload = url.split(",", 1)[1] if "," in url else ""
        captured["original_bytes"] = len(data)
        captured["sent_bytes"] = len(payload) // 4 * 3
        return url

    enricher._parse_vision_result = record_parse  # type: ignore[assignment]
    enricher._to_vision_data_url = record_url  # type: ignore[assignment]

    for path in images:
        size = os.path.getsize(path)
        print(f"\n>>> {os.path.basename(path)}  ({size / 1024 / 1024:.1f}MB)")
        captured.clear()
        started = time.time()
        segment = MessageSegment(type="image", url=path)
        try:
            ok = await asyncio.wait_for(
                enricher._describe_image(segment, ""), timeout=args.timeout
            )
        except Exception as exc:
            print(f"    错误: {type(exc).__name__}: {exc}")
            records.append(
                {"image": os.path.basename(path), "size": size,
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        elapsed = round(time.time() - started, 1)
        sent = captured.get("sent_bytes")
        original_bytes = captured.get("original_bytes")
        if isinstance(sent, int) and isinstance(original_bytes, int):
            print(f"    喂给端点：{original_bytes / 1024:.0f}KB → {sent / 1024:.0f}KB")
        if not ok:
            print(f"    识别失败（{elapsed}s）→ 群里只会看到「[图片]」占位")
            print(f"    端点原始返回：{str(captured.get('raw'))[:400]}")
            records.append({"image": os.path.basename(path), "size": size,
                            "ok": False, "seconds": elapsed,
                            "raw": captured.get("raw")})
            continue
        print(f"    summary: {segment.summary}")
        print(f"    端点原始返回：{str(captured.get('raw'))[:400]}")
        records.append({
            "image": os.path.basename(path),
            "size": size,
            "ok": True,
            "seconds": elapsed,
            "original_bytes": original_bytes,
            "sent_bytes": sent,
            "summary": segment.summary,
            "raw": captured.get("raw"),
            "objective_summary": (segment.data or {}).get("objective_summary"),
            "confidence": (segment.data or {}).get("vision_confidence"),
            "uncertain": (segment.data or {}).get("vision_uncertain"),
        })

    ok_count = sum(1 for r in records if r.get("ok"))
    print("\n" + "=" * 78)
    print(f"汇总：样本={len(records)} 识别成功={ok_count} 失败={len(records) - ok_count}")


async def run_prompts(images: list[str], args, records: list[dict]) -> None:
    """同一张图跑多套 prompt 对照。"""
    from core.adapter.rich_content import MessageSegment

    enricher, effective = build_enricher()
    provider, _ = build_vision_provider()
    max_tokens = args.max_tokens or enricher.image_vision_max_tokens

    print(f"视觉 provider：{effective['base_url']} / {effective['model']}")
    print(f"max_tokens={max_tokens}（生产配置值）")
    print("=" * 78)

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [""]
    wanted = [p.strip() for p in args.prompts.split(",") if p.strip()]

    prompts: dict[str, str] = {}
    for name in wanted:
        if name in ("production", "current"):
            # 直接调生产函数：探针不抄规则，就不会与代码漂移
            prompts["production"] = enricher._build_image_prompt(
                MessageSegment(type="image"), ""
            )
        elif name == "identify":
            prompts["identify"] = "\n".join(
                [IDENTIFY_BASE_PROMPT, IDENTIFY_EXTRA, JSON_RULE]
            )

    print("production prompt（生产真实值）:")
    for line in prompts.get("production", "").splitlines():
        print("   ", line)
    print("=" * 78)

    for model in models:
        run_provider = provider
        if model:
            run_provider, _ = build_vision_provider(model)
        for path in images:
            try:
                data_url = to_data_url(path)
            except OSError as exc:
                print(f"[跳过] {path} 读取失败：{exc}")
                continue
            for name, prompt in prompts.items():
                label = f"{model or effective['model']} / {name}"
                print(f"\n>>> {os.path.basename(path)}  [{label}]")
                outcome = await ask(
                    run_provider, model, prompt, data_url, args.timeout, max_tokens
                )
                record = {
                    "image": os.path.basename(path),
                    "model": model or effective["model"],
                    "prompt": name,
                    "seconds": outcome.get("seconds"),
                }
                if outcome.get("error"):
                    print("    错误:", outcome["error"])
                    record["error"] = outcome["error"]
                else:
                    think, body = split_think(outcome["raw"])
                    if args.show_think and think:
                        print("    思考:", think[:500].replace("\n", " "))
                        print("    正文:", body[:300].replace("\n", " ") or "（无正文）")
                    else:
                        print("    原始:", outcome["raw"][:400].replace("\n", " ") or "（空）")
                    print("    解析:", parse_description(outcome["raw"]))
                    record["raw"] = outcome["raw"]
                    record["think"] = think
                    record["has_json"] = "{" in body
                    record["description"] = parse_description(outcome["raw"])
                records.append(record)

    print("\n" + "=" * 78)
    print("汇总")
    for name in prompts:
        rows = [r for r in records if r.get("prompt") == name]
        errors = sum(1 for r in rows if r.get("error"))
        no_json = sum(1 for r in rows if r.get("has_json") is False)
        print(
            f"  {name:12s} 样本={len(rows)} 失败={errors} 无JSON(被截断)={no_json}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", default="/tmp/vsamples", help="图片文件或目录")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--mode", default="pipeline", choices=["pipeline", "prompts"])
    parser.add_argument("--models", default="", help="逗号分隔；留空用生产配置的模型")
    parser.add_argument("--prompts", default="production,identify", help="production / identify")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="prompts 模式用；留空取生产配置的 to_text_max_tokens",
    )
    parser.add_argument(
        "--show-think", action="store_true", help="单独打印 <think> 内容，看模型认没认出来"
    )
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    images = collect_images(args.dirs, args.limit)
    if not images:
        print(f"没有找到图片：{args.dirs}")
        return 2
    print(f"样本 {len(images)} 张：")
    for path in images:
        print("  -", path)
    print("=" * 78)

    records: list[dict] = []
    if args.mode == "pipeline":
        await run_pipeline(images, args, records)
    else:
        await run_prompts(images, args, records)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(
                {"mode": args.mode, "records": records},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print("已保存:", args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

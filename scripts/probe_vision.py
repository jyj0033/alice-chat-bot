"""视觉识别质量探针：同一张图，对比「当前限制型 prompt」与「允许点名的 prompt」。

背景：生产里图片描述几乎全是「一张图，画面是[动画表情]」或「看不清画面」，
用户反馈"认不出图里的人物是谁，只描述了画面长什么样"。

这个脚本只回答两个问题：
1. 视觉端点到底能不能认人/认作品？（换 prompt 前后对比）
2. 是 prompt 的锅还是模型的锅？（可以换模型名重跑）

用法（容器内 /app）：
    python3 /app/scripts/probe_vision.py
    python3 /app/scripts/probe_vision.py --dirs /tmp/vsamples
    python3 /app/scripts/probe_vision.py --models MiniMax-M3,MiniMax-M3.1-Flash-Preview
    python3 /app/scripts/probe_vision.py --prompts current,identify
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

# 生产当前实际使用的兜底 prompt（config.yaml 的 image.to_text_prompt 同名同义）
DEFAULT_BASE_PROMPT = (
    "用一两句话（50字以内）客观描述图片中能直接看到的内容：主体、动作或表情、"
    "画面文字、明显颜色和构图；不要推测人物关系、前因后果、情绪意图或适用场景。"
)

# 当前代码在 _build_image_prompt 里追加的规则（原样复制，保证 baseline 一致）
CURRENT_EXTRA = (
    "输出规则：只写图片中直接可观察的事实，不要结合群聊推断谁在说谁、"
    "不要解释梗的前因后果，不要写“适合用来……”或替群友评价这张图。"
)

# 候选新 prompt：把「认人/认作品」从禁止改成鼓励，只保留「不许编造」这条底线
IDENTIFY_BASE_PROMPT = (
    "先认出这张图是什么，再用一两句话把最有用的信息说清楚。"
)

IDENTIFY_EXTRA = (
    "识别要求：\n"
    "1. 只要你认得画面里的人、角色、作品、梗、名场面、地标、品牌、商品、"
    "动物品种或界面截图里的软件，就直接点名——这是最有用的信息，"
    "不要因为怕说错就跳过，只写「一个长发女孩」等于没认出来。\n"
    "2. 认得出作品名就说全：作品名 + 角色名（例如「《崩坏：星穹铁道》的流萤」），"
    "认得出真人就说名字或身份，认得出梗名就说梗名（例如「熊猫头」「借口龙」）。\n"
    "3. 画面里的文字照抄下来，它常常就是这张图的梗本身。\n"
    "4. 认不出来、或只是看着像但没有把握，就退回客观画面描述，"
    "并直说「看不出具体是谁」——不要硬给一个名字。\n"
    "5. 绝对不要编造角色名、作品名、出处或剧情；拿不准就把 uncertain 置为 true。"
)

JSON_RULE = (
    "只输出 JSON，不要输出解释，格式如："
    '{"description":"画面中可直接看到的事实",'
    '"confidence":0.85,"uncertain":false}。'
    "无法确认的内容不要猜，confidence 低于0.65时 uncertain 必须为 true。"
)


def build_prompts(base_prompt: str, extra: str, with_context: bool = False) -> str:
    parts = [base_prompt, extra]
    if with_context:
        parts.append(
            "前文对话仅用于确认图片边界：\n（探针不注入前文）\n"
            "不要把前文人物、事件、评价或原话写进图片描述。"
        )
    parts.append(JSON_RULE)
    return "\n".join(parts)


def build_vision_provider(model_override: str | None = None):
    """复刻 main._init_vision_provider 的合并逻辑，保证读到的是生产同一份配置。"""
    from core.config_store import load_config
    from modules.llm.openai_provider import create_provider

    cfg = load_config(CONFIG_PATH)
    image_config = dict(cfg.get("image", {}) or {})
    rich_image = (cfg.get("rich_media", {}) or {}).get("image", {}) or {}
    image_config = {**rich_image, **image_config}
    if isinstance(rich_image.get("vision"), dict) and isinstance(
        image_config.get("vision"), dict
    ):
        image_config["vision"] = {
            **rich_image["vision"],
            **image_config["vision"],
        }
    vc = dict(image_config.get("vision", {}) or {})
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
            "max_tokens": 300,
        },
    )
    effective = {
        "provider_type": vc.get("provider_type"),
        "base_url": vc.get("base_url"),
        "model": vc.get("model"),
        "has_key": bool(str(vc.get("api_key") or "").strip()),
        "to_text_prompt": image_config.get("to_text_prompt") or DEFAULT_BASE_PROMPT,
    }
    return provider, effective


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
    provider, model: str, prompt: str, data_url: str, timeout: float, max_tokens: int = 300
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
    elapsed = round(time.time() - started, 1)
    text = (response.content or "").strip()
    return {"raw": text, "seconds": elapsed}


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


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", default="/tmp/vsamples", help="图片文件或目录")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--models", default="", help="逗号分隔；留空用生产配置的模型")
    parser.add_argument(
        "--prompts", default="current,identify", help="current / identify"
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="与生产一致默认为 300；调大可看出截断到底吃掉了什么",
    )
    parser.add_argument(
        "--show-think", action="store_true", help="单独打印 <think> 内容，看模型认没认出来"
    )
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    provider, effective = build_vision_provider()
    print("=" * 78)
    print("视觉 provider（复刻生产合并逻辑）")
    for key, value in effective.items():
        if key == "to_text_prompt":
            value = str(value)[:70] + "…"
        print(f"  {key:16s} = {value}")
    print("=" * 78)

    images = collect_images(args.dirs, args.limit)
    if not images:
        print(f"没有找到图片：{args.dirs}")
        return 2
    print(f"样本 {len(images)} 张：")
    for path in images:
        print("  -", path)
    print("=" * 78)

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [""]
    wanted = [p.strip() for p in args.prompts.split(",") if p.strip()]

    prompts: dict[str, str] = {}
    if "current" in wanted:
        prompts["current"] = build_prompts(effective["to_text_prompt"], CURRENT_EXTRA)
    if "identify" in wanted:
        prompts["identify"] = build_prompts(IDENTIFY_BASE_PROMPT, IDENTIFY_EXTRA)

    records: list[dict] = []
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
                    run_provider, model, prompt, data_url, args.timeout, args.max_tokens
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
    for model in models or [effective["model"]]:
        for name in prompts:
            rows = [
                r
                for r in records
                if r["model"] == (model or effective["model"]) and r["prompt"] == name
            ]
            errors = sum(1 for r in rows if r.get("error"))
            no_json = sum(1 for r in rows if r.get("has_json") is False)
            print(
                f"  {model or effective['model']:32s} {name:9s} "
                f"样本={len(rows)} 失败={errors} 无JSON(被截断)={no_json}"
            )

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(
                {"effective": effective, "records": records},
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print("已保存:", args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

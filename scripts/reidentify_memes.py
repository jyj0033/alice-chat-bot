#!/usr/bin/env python3
"""全量重识别表情包：重新跑视觉识别，修正 meaning/description 与 category。

- 对 catalog 中每张图调用 vision provider（config.yaml image.vision，MiniMax-M3），
  引导模型输出结构化 JSON：是否为独立表情包 + 50字客观描述 + 分类。
- 非表情包（照片/截图/随手拍/对话框flag等）写入待删除清单（--dry-run 时只打印）。
- 支持 --dry-run / --apply / --id <hash> 三种模式。
用法（容器内）：
    python scripts/reidentify_memes.py --dry-run
    python scripts/reidentify_memes.py --apply

注意：本脚本的视觉调用必须与线上识别保持一致，否则它会被自己过时的参数坑死。
2026-09-28 对齐过一次，三处都曾经是坏的：
  1. max_tokens=300 —— M3 先吐 <think> 再给 JSON，思维链计入预算，300 会把输出
     截断成不闭合的 JSON，`_parse_json` 解析不出来 → 这张直接跳过。
     现在从 to_text_max_tokens 读（生产 2000）。
  2. 直接把原图字节塞进请求 —— 端点单张媒体上限 10MiB，超了 400。
     现在走 `_shrink_image_bytes` 压到 ≤vision_max_side/≤vision_max_payload_bytes。
  3. `_parse_json` 不剥思考块、且正序试 JSON —— 思考块里的 `{` 会抢先命中。
     现在复用 `_strip_thinking` + `RichMediaEnricher._json_candidates`（倒序）。
  4. 分类词表只认 `DEFAULT_CATEGORIES` 那 8 个 —— 图库里由面板手工整理的分类
     （委屈/愤怒/困倦/拒绝…）会被当非法值打回「待整理」，一次平掉二十多张。
     现在词表 = 默认 8 个 ∪ 图库里实际存在的分类，并把当前分类告诉模型。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reidentify")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from modules.llm.base import ChatMessage, ChatRequest  # noqa: E402
from modules.llm.openai_provider import create_provider  # noqa: E402
from modules.meme_manager import DEFAULT_CATEGORIES  # noqa: E402
from core.config_store import load_config as load_config_file  # noqa: E402
from core.adapter.rich_media import (  # noqa: E402
    RichMediaEnricher,
    _bytes_to_data_url,
    _extract_description_field,
    _guess_media_type,
    _loads_json_object,
    _shrink_image_bytes,
    _sniff_media_type,
    _strip_thinking,
)

ALLOWED_CATEGORIES = set(DEFAULT_CATEGORIES)
MINIMAX_OPENAI_BASE_URL = "https://api.minimax.cn/v1"

# 与线上识别一致的下限/上限，配置缺项时的兜底
DEFAULT_VISION_MAX_SIDE = 1600
DEFAULT_VISION_JPEG_QUALITY = 85
DEFAULT_VISION_MAX_PAYLOAD = 2_000_000
DEFAULT_VISION_MAX_TOKENS = 2000

# --apply 时只删除白名单里的 id（人眼/明确理由确认过的非表情包），
# 其余 is_meme=false 只当作分类/表述变更保留，避免模型误杀动漫表情包。
DELETE_ALLOWLIST = {
    "236bde4e76cacaee104c6c940a9971dbde7a60bd35ec3bcf9a47555ed136fd41",  # 论坛梗图文截图
    "1b139823477038b29f77d7a036e7b650d58490f0eebc251a6336f48ef7641698",  # 微博六宫格截图
    "0882d68021243136cedf5c55cd716c1769443ba3719cf8ea614fdccd8b9c98e5",  # 游戏升级弹窗
    "d1a2a3145a37bda2b2958628e096932799541c54a324ce0335ee476b72a28e72",  # 游戏截图
}

PROMPT_TEMPLATE = """请识别这张图片，严格按下面的 JSON 输出，不要输出任何多余文字：

{
  "is_meme": true 或 false,
  "description": "用一句话（50字以内）客观描述画面：主体是什么、人物表情/动作、图上有哪些文字，直接可观察，不要推断前因后果、不要解释梗、不要写『适合用来…』",
  "category": "从 ⟨CATEGORIES⟩ 里选一个，不要自己造新词"
}

注意：description 里要用「」引用画面文字，绝对不要用英文双引号"，否则 JSON 会解析失败。

判断 is_meme 的规则：只有真正的表情包/梗图/表情贴图才是 true（有夸张表情、网络梗、动物表情、动漫颜艺等）。以下都算 false：普通生活照片、风景、游戏截图、聊天记录截图、网页弹窗、二维码、色情擦边图、以及『xxx不喜欢看这个』这类功能标记图。

这张图当前归在「⟨CURRENT⟩」。如果新描述放到这个分类里仍然合适，就继续选它；只有明显不符时才换。"""


def _allowed_categories(catalog: dict[str, Any]) -> list[str]:
    """可用分类 = 代码里的默认词表 ∪ 图库里实际存在过的分类。

    不能只用 `DEFAULT_CATEGORIES`：面板里手工整理出来的分类（委屈/愤怒/困倦/拒绝…）
    不在默认词表里，只认默认值会把它们当非法值打回「待整理」，一次平掉二十多张。
    """
    existing = {
        str(v.get("category") or "").strip()
        for v in (catalog.get("memes") or {}).values()
    }
    vocab = set(ALLOWED_CATEGORIES) | {c for c in existing if c}
    # 默认词表在前、其余按字典序，保证输出稳定
    return list(DEFAULT_CATEGORIES) + sorted(vocab - set(DEFAULT_CATEGORIES))


def _build_prompt(categories: list[str], current: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("⟨CATEGORIES⟩", " / ".join(categories))
        .replace("⟨CURRENT⟩", current or "待整理")
    )


def _load_config() -> dict[str, Any]:
    cfg_path = REPO / "config" / "config.yaml"
    return load_config_file(cfg_path)


def _image_tuning(cfg: dict[str, Any]) -> dict[str, int]:
    """读和线上识别同一份「缩放 / 预算」参数。

    合并规则与 main._get_rich_media_config 一致：rich_media.image 打底，
    顶层 image 覆盖（顶层只放 vision，但保持一致免得以后踩坑）。
    """
    rich_image = dict((cfg.get("rich_media", {}) or {}).get("image", {}) or {})
    top_image = dict(cfg.get("image", {}) or {})
    merged = {**rich_image, **top_image}
    return {
        "max_side": int(merged.get("vision_max_side", DEFAULT_VISION_MAX_SIDE)),
        "quality": int(merged.get("vision_jpeg_quality", DEFAULT_VISION_JPEG_QUALITY)),
        "max_payload": int(
            merged.get("vision_max_payload_bytes", DEFAULT_VISION_MAX_PAYLOAD)
        ),
        "max_tokens": int(merged.get("to_text_max_tokens", DEFAULT_VISION_MAX_TOKENS)),
    }


def _init_vision(cfg: dict[str, Any], max_tokens: int) -> Any:
    vision = dict((cfg.get("image", {}) or {}).get("vision", {}) or {})
    provider_type = str(vision.get("provider_type") or "openai_compatible").lower()
    base_url = str(vision.get("base_url") or "").lower()
    if (
        provider_type == "minimax"
        or "api.minimaxi.com" in base_url
        or "api.minimax.cn" in base_url
        or ("minimax" in str(vision.get("model") or "").lower()
            and provider_type in {"anthropic", "claude"})
    ):
        vision["provider_type"] = "openai_compatible"
        vision["base_url"] = MINIMAX_OPENAI_BASE_URL
    if not vision.get("enabled", False):
        logger.critical("image.vision.enabled 为 false")
        raise SystemExit(1)
    return create_provider(
        vision.get("provider_type", "openai_compatible"),
        {
            "provider_type": vision.get("provider_type", "openai_compatible"),
            "api_key": vision.get("api_key", ""),
            "base_url": vision.get("base_url", "https://api.openai.com/v1"),
            "model": vision.get("model", "gpt-4o-mini"),
            "timeout": vision.get("timeout", 60),
            "temperature": vision.get("temperature", 0.4),
            "max_tokens": max_tokens,
        },
    )


def _data_url(path: Path, tuning: dict[str, int]) -> str:
    """把本地图片压成端点吃得下的 data URL。

    不能直接塞原图：端点单张媒体上限 10MiB，超了直接 400。
    走生产同一套 `_shrink_image_bytes`（顺带把动图拍平成单帧 JPEG）。
    """
    data = path.read_bytes()
    media_type = _sniff_media_type(data, _guess_media_type(path.name, "image/jpeg"))
    shrunk, media_type = _shrink_image_bytes(
        data,
        media_type,
        max_side=tuning["max_side"],
        quality=tuning["quality"],
        max_bytes=tuning["max_payload"],
    )
    return _bytes_to_data_url(shrunk, media_type)


async def _vision(provider: Any, data_url: str, prompt: str, max_tokens: int) -> str:
    request = ChatRequest(
        model=getattr(provider, "model", "") or "",
        messages=[ChatMessage(role="user", content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    last_exc: Exception | None = None
    for attempt in (1, 2, 3):
        try:
            resp = await provider.chat(request)
            return (resp.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 3:
                logger.warning("  视觉调用失败（第%d次）：%s，正在重试…", attempt, exc)
                await asyncio.sleep(2.0 * attempt)
    raise last_exc  # type: ignore[misc]


def _parse_json(text: str) -> dict | None:
    """解析模型输出，复用线上识别那套（不再自己造一套）。

    旧实现有两个会被 max_tokens 放大的毛病：不剥 `<think>`（思考块里的 `{` 会
    抢先命中），以及正序取 JSON（真正的答案总在最后）。现在都走线上同一套：
    `_strip_thinking` → `RichMediaEnricher._json_candidates`（倒序）→ `_loads_json_object`。
    """
    body = _strip_thinking(text)
    for cand in RichMediaEnricher._json_candidates(body):
        obj = _loads_json_object(cand)
        if obj and ({"is_meme", "category"} & set(obj)):
            return obj
        repaired = _repair_unescaped_quotes(cand)
        if repaired:
            obj = _loads_json_object(repaired)
            if obj and ({"is_meme", "category"} & set(obj)):
                return obj
    # 输出被截断、对象不闭合时，至少把 description 捞回来（is_meme/category 交给调用方保守处理）
    salvaged = _extract_description_field(body)
    if salvaged:
        return {"description": salvaged, "_salvaged": True}
    return None


def _repair_unescaped_quotes(text: str) -> str | None:
    # 只有 description 的字符串值里可能出现未转义的“”（category 取值固定、is_meme 是布尔）。
    # 思路：找 "description": 后字符串值的起始引号 O，然后依次把其后每个 " 当作值结束引号 E 试验：
    #   [O+1, E) 内的引号当作内文引号替换成「，tail 原样保留，能整体解析成功即为正确切分。
    key = '"description"'
    ki = text.find(key)
    if ki < 0:
        return None
    o = text.find('"', ki + len(key))
    if o < 0:
        return None
    # 收集 O 之后所有引号位置作为候选结束点
    candidates = [i for i in range(o + 1, len(text)) if text[i] == '"']
    for e in candidates:
        inner = text[o + 1 : e].replace('"', "「")
        cand = text[: o + 1] + inner + text[e:]
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "is_meme" in obj:
                return cand
        except json.JSONDecodeError:
            continue
    return None


def _load_catalog() -> dict[str, Any]:
    path = REPO / "data" / "memes" / "catalog.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _path_for(idx: str, item: dict) -> Path:
    return REPO / "data" / "memes" / str(item.get("category", "待整理")) / str(item.get("filename"))


def _relocate(item: dict, new_cat: str) -> bool:
    """把条目挪到新分类目录，返回 category 是否真的变了。

    只有「文件确实落在新目录」才改写 `item['category']`。旧代码是
    `if src.exists() and not dst.exists(): src.rename(dst)` 之后**无条件**改 category：
    目标已有同名文件时文件没搬、catalog 却指向新目录 → 之后 `_path_for` 全部取不到文件。
    """
    old_cat = str(item.get("category") or "待整理")
    if new_cat == old_cat:
        return False
    filename = str(item.get("filename"))
    base = REPO / "data" / "memes"
    src, dst = base / old_cat / filename, base / new_cat / filename
    if src.exists():
        if dst.exists():
            logger.warning(
                "分类未迁移（目标已有同名文件）：%s %s -> %s", filename, old_cat, new_cat
            )
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    elif not dst.exists():
        logger.warning("分类未迁移（源文件缺失）：%s，保留原分类 %s", src, old_cat)
        return False
    item["category"] = new_cat
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写回 catalog.json 并删除非表情包")
    ap.add_argument("--dry-run", action="store_true", help="(默认行为) 只打印结果不写回")
    ap.add_argument("--id", help="只处理指定 hash（完整或前12位）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 张")
    ap.add_argument("--fail", action="store_true", help="失败也继续（默认遇错中断以便修）")
    ap.add_argument("--json", dest="json_out", default="", help="把逐张结果写到这个文件，便于复盘")
    args = ap.parse_args()

    cfg = _load_config()
    tuning = _image_tuning(cfg)
    logger.info(
        "视觉参数：max_tokens=%d 缩放≤%dpx q%d 载荷≤%dB（与线上识别同源）",
        tuning["max_tokens"], tuning["max_side"], tuning["quality"], tuning["max_payload"],
    )
    provider = _init_vision(cfg, tuning["max_tokens"])
    catalog = _load_catalog()
    categories = _allowed_categories(catalog)
    logger.info("可用分类（默认 %d + 图库自定义 %d）：%s",
                len(DEFAULT_CATEGORIES), len(categories) - len(DEFAULT_CATEGORIES),
                " / ".join(categories))

    raw_items = list(catalog["memes"].items())
    if args.id:
        raw_items = [kv for kv in raw_items if str(kv[0]).startswith(args.id.lower())]
    if args.limit > 0:
        raw_items = raw_items[: args.limit]

    results: list[dict] = []

    async def run() -> None:
        errors = 0
        for idx, (meme_id, item) in enumerate(raw_items, 1):
            path = _path_for(meme_id, item)
            if not path.exists():
                logger.warning("[%d/%d] %s 文件缺失 %s", idx, len(raw_items), meme_id[:12], path)
                continue
            old_cat = str(item.get("category", "")).strip()
            logger.info("[%d/%d] 正在识别 %s（分类：%s）", idx, len(raw_items), meme_id[:12], old_cat)
            prompt = _build_prompt(categories, old_cat)
            try:
                text = await _vision(
                    provider, _data_url(path, tuning), prompt, tuning["max_tokens"]
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.error("[%d/%d] %s 识别失败：%s", idx, len(raw_items), meme_id[:12], exc)
                if not args.fail:
                    raise
                continue
            obj = _parse_json(text)
            if not obj:
                errors += 1
                logger.error("[%d/%d] %s 输出无法解析：%r", idx, len(raw_items), meme_id[:12], text[:160])
                continue
            # 输出被截断时只捞回 description：is_meme / category 无据可依，
            # 保守当作「是表情包」保留，并沿用旧分类，绝不因此进待删清单。
            salvaged = bool(obj.pop("_salvaged", False))
            desc = str(obj.get("description", "")).strip().replace("\n", " ")[:240]
            if salvaged:
                is_meme = True
                cat = old_cat
            else:
                is_meme = bool(obj.get("is_meme", True))
                cat = str(obj.get("category", "")).strip()
            if cat not in categories:
                # 模型造了新词：沿用原分类，不去动用户的整理结果
                cat = old_cat if old_cat in categories else "待整理"
            changed = (
                (str(item.get("meaning", "")) != desc)
                or (str(item.get("category", "")) != cat)
            )
            results.append({
                "id": meme_id, "path": str(path), "is_meme": is_meme, "salvaged": salvaged,
                "old_meaning": item.get("meaning", ""), "new_meaning": desc,
                "old_category": item.get("category", ""), "new_category": cat, "changed": changed, "raw": text,
            })
            flag = "非表情包-待删" if not is_meme else ("已变更" if changed else "不变")
            if salvaged:
                flag += "（截断-仅捞回描述）"
            logger.info(
                "  %s｜旧分类[%s] %s → 新分类[%s] %s",
                flag, results[-1]["old_category"], results[-1]["old_meaning"][:20],
                cat, desc[:40],
            )
        logger.info("处理完成，失败或无法解析：%d 项", errors)

    asyncio.run(run())

    deletes = [r for r in results if not r["is_meme"]]
    changes = [r for r in results if r["is_meme"] and r["changed"]]
    salvaged = [r for r in results if r.get("salvaged")]

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "tuning": tuning,
                    "applied": bool(args.apply),
                    "processed": len(results),
                    "deletes": len(deletes),
                    "changes": len(changes),
                    "salvaged": len(salvaged),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info("逐张结果已写入 %s", out_path)

    print("\n===== 汇总 =====")
    print(f"处理 {len(results)}，非表情包候选删除 {len(deletes)}，表述/分类变更 {len(changes)}")
    if salvaged:
        print(f"（其中 {len(salvaged)} 项输出被截断，仅捞回描述、保守保留原分类）")
    if not args.apply:
        print("\n--dry-run 未写回。要实际生效请加 --apply。")
        if deletes:
            print("\n【待删除候选】（--apply 时执行）")
            for r in deletes:
                print(f"  {r['id'][:12]} | {r['old_category']} | {r['old_meaning'][:40]} | -> {r['new_meaning'][:40]}")
        if changes:
            print("\n【待变更】")
            for r in changes[:50]:
                print(f"  {r['id'][:12]} | {r['old_category']}->{r['new_category']} | {r['old_meaning'][:24]} -> {r['new_meaning'][:40]}")
        return

    # ---- apply ----
    changed_any = False
    for r in changes:
        item = catalog["memes"].get(r["id"])
        if not item:
            continue
        if _relocate(item, r["new_category"]):
            changed_any = True
        item["meaning"] = r["new_meaning"]
        item["description"] = r["new_meaning"]

    # 白名单硬删：无论本次模型怎么判，4 个确认非表情包必删
    for hid in DELETE_ALLOWLIST:
        item = catalog["memes"].pop(hid, None)
        if item:
            f = REPO / "data" / "memes" / str(item.get("category", "待整理")) / str(item.get("filename"))
            f.unlink(missing_ok=True)
            changed_any = True
            print(f"已删除(白名单): {hid[:12]} | {item.get('meaning', '')[:40]}")

    # 非白名单的 false 当作保留：写回新表述，但不删文件
    for r in deletes:
        if r["id"] in DELETE_ALLOWLIST:
            continue  # 白名单已在上面硬删，跳过
        item = catalog["memes"].get(r["id"])
        if item:
            item["meaning"] = r["new_meaning"]
            item["description"] = r["new_meaning"]
            _relocate(item, r["new_category"])
            changed_any = True
            print(f"保留(非白名单): {r['id'][:12]} | 新[{(item.get('category', ''))}] {r['new_meaning'][:40]}")

    if changed_any:
        out = REPO / "data" / "memes" / "catalog.json"
        out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("catalog.json 已写回。")
    else:
        print("无变更，catalog.json 未改动。")


if __name__ == "__main__":
    main()

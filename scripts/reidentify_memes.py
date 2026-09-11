#!/usr/bin/env python3
"""全量重识别表情包：重新跑视觉识别，修正 meaning/description 与 category。

- 对 catalog 中每张图调用 vision provider（config.yaml image.vision，MiniMax-M3），
  引导模型输出结构化 JSON：是否为独立表情包 + 50字客观描述 + 分类。
- 非表情包（照片/截图/随手拍/对话框flag等）写入待删除清单（--dry-run 时只打印）。
- 支持 --dry-run / --apply / --id <hash> 三种模式。
用法（容器内）：
    python scripts/reidentify_memes.py --dry-run
    python scripts/reidentify_memes.py --apply
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
from core.config_store import load_config as load_config_file  # noqa: E402

ALLOWED_CATEGORIES = {"待整理", "开心", "无语", "吐槽", "鼓励", "卖萌", "震惊", "其他"}

# --apply 时只删除白名单里的 id（人眼/明确理由确认过的非表情包），
# 其余 is_meme=false 只当作分类/表述变更保留，避免模型误杀动漫表情包。
DELETE_ALLOWLIST = {
    "236bde4e76cacaee104c6c940a9971dbde7a60bd35ec3bcf9a47555ed136fd41",  # 论坛梗图文截图
    "1b139823477038b29f77d7a036e7b650d58490f0eebc251a6336f48ef7641698",  # 微博六宫格截图
    "0882d68021243136cedf5c55cd716c1769443ba3719cf8ea614fdccd8b9c98e5",  # 游戏升级弹窗
    "d1a2a3145a37bda2b2958628e096932799541c54a324ce0335ee476b72a28e72",  # 游戏截图
}

PROMPT = """请识别这张图片，严格按下面的 JSON 输出，不要输出任何多余文字：

{
  "is_meme": true 或 false,
  "description": "用一句话（50字以内）客观描述画面：主体是什么、人物表情/动作、图上有哪些文字，直接可观察，不要推断前因后果、不要解释梗、不要写『适合用来…』",
  "category": "从 待整理/开心/无语/吐槽/鼓励/卖萌/震惊/其他 里选一个"
}

注意：description 里要用「」引用画面文字，绝对不要用英文双引号"，否则 JSON 会解析失败。

判断 is_meme 的规则：只有真正的表情包/梗图/表情贴图才是 true（有夸张表情、网络梗、动物表情、动漫颜艺等）。以下都算 false：普通生活照片、风景、游戏截图、聊天记录截图、网页弹窗、二维码、色情擦边图、以及『xxx不喜欢看这个』这类功能标记图。"""


def _load_config() -> dict[str, Any]:
    cfg_path = REPO / "config" / "config.yaml"
    return load_config_file(cfg_path)


def _init_vision(cfg: dict[str, Any]) -> Any:
    vision = (cfg.get("image", {}) or {}).get("vision", {}) or {}
    if not vision.get("enabled", False):
        logger.critical("image.vision.enabled 为 false")
        raise SystemExit(1)
    return create_provider(
        vision.get("provider_type", "openai"),
        {
            "provider_type": vision.get("provider_type", "openai"),
            "api_key": vision.get("api_key", ""),
            "base_url": vision.get("base_url", "https://api.openai.com/v1"),
            "model": vision.get("model", "gpt-4o-mini"),
            "timeout": vision.get("timeout", 60),
            "temperature": vision.get("temperature", 0.4),
            "max_tokens": 300,
        },
    )


def _data_url(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/" + (path.suffix.lstrip(".") or "png") + ";base64," + base64.b64encode(data).decode()


async def _vision(provider: Any, data_url: str) -> str:
    request = ChatRequest(
        model=getattr(provider, "model", "") or "",
        messages=[ChatMessage(role="user", content=[
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])],
        max_tokens=300,
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
    # 剥掉 ```json ... ``` 代码块围栏
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.S)
    for cand in re.findall(r"\{[^{}]*\}", text, re.S):
        obj = _try_parse(cand)
        if obj:
            return obj
    # 兜底：直接整体尝试
    obj = _try_parse(text)
    if obj:
        return obj
    return None


def _try_parse(cand: str) -> dict | None:
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict) and "is_meme" in obj and "category" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # 仅当确有 is_meme 时尝试修复未转义引号
    if "is_meme" not in cand:
        return None
    repaired = _repair_unescaped_quotes(cand)
    if repaired is not None:
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict) and "is_meme" in obj:
                return obj
        except json.JSONDecodeError:
            pass
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写回 catalog.json 并删除非表情包")
    ap.add_argument("--dry-run", action="store_true", help="(默认行为) 只打印结果不写回")
    ap.add_argument("--id", help="只处理指定 hash（完整或前12位）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 张")
    ap.add_argument("--fail", action="store_true", help="失败也继续（默认遇错中断以便修）")
    args = ap.parse_args()

    cfg = _load_config()
    provider = _init_vision(cfg)
    catalog = _load_catalog()

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
            logger.info("[%d/%d] 正在识别 %s（分类：%s）", idx, len(raw_items), meme_id[:12], item.get("category"))
            try:
                text = await _vision(provider, _data_url(path))
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.error("[%d/%d] %s 识别失败：%s", idx, len(raw_items), meme_id[:12], exc)
                if not args.fail:
                    raise
                continue
            obj = _parse_json(text)
            if not obj:
                logger.error("[%d/%d] %s 输出无法解析：%r", idx, len(raw_items), meme_id[:12], text[:160])
                continue
            is_meme = bool(obj.get("is_meme", True))
            desc = str(obj.get("description", "")).strip().replace("\n", " ")[:240]
            cat = str(obj.get("category", "")).strip()
            if cat not in ALLOWED_CATEGORIES:
                cat = "待整理"
            changed = (
                (str(item.get("meaning", "")) != desc)
                or (str(item.get("category", "")) != cat)
            )
            results.append({
                "id": meme_id, "path": str(path), "is_meme": is_meme,
                "old_meaning": item.get("meaning", ""), "new_meaning": desc,
                "old_category": item.get("category", ""), "new_category": cat, "changed": changed, "raw": text,
            })
            flag = "非表情包-待删" if not is_meme else ("已变更" if changed else "不变")
            logger.info(
                "  %s｜旧分类[%s] %s → 新分类[%s] %s",
                flag, results[-1]["old_category"], results[-1]["old_meaning"][:20],
                cat, desc[:40],
            )
        logger.info("处理完成，失败或无法解析：%d 项", errors)

    asyncio.run(run())

    deletes = [r for r in results if not r["is_meme"]]
    changes = [r for r in results if r["is_meme"] and r["changed"]]

    print("\n===== 汇总 =====")
    print(f"处理 {len(results)}，非表情包候选删除 {len(deletes)}，表述/分类变更 {len(changes)}")
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
        old_cat = item.get("category", "待整理")
        new_cat = r["new_category"]
        if new_cat != old_cat:
            src = REPO / "data" / "memes" / old_cat / str(item.get("filename"))
            dst_dir = REPO / "data" / "memes" / new_cat
            dst = dst_dir / str(item.get("filename"))
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                src.rename(dst)
            item["category"] = new_cat
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
            if r["new_category"] != item.get("category", "待整理"):
                old_cat = item.get("category", "待整理")
                new_cat = r["new_category"]
                src = REPO / "data" / "memes" / old_cat / str(item.get("filename"))
                dst_dir = REPO / "data" / "memes" / new_cat
                dst = dst_dir / str(item.get("filename"))
                dst_dir.mkdir(parents=True, exist_ok=True)
                if src.exists() and not dst.exists():
                    src.rename(dst)
                item["category"] = new_cat
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

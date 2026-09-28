"""验证两件事（决定生产参数取值）：

1. output_text 解析健壮性
   —— 上层 _parse 优先读顶层 output_text；若缺失才回退拼 message 片段。
      需要确认 MiniMax 是否真的返回顶层 output_text，以及 message 片段里
      到底装了什么（前一轮观测到 message item 体积达 4.7 万字符，
      而最终 <say> 很短，若不确认可能把整段搜索内容当成回复正文）。

2. 输出预算下限该设多少
   —— 观测到一次无搜索的请求把整 1200 输出预算全用于 reasoning，
      status=incomplete、无 message 产出。上层 MIN_OUTPUT_TOKENS=1200 偏低。
      对比 1200 / 2500 两档的完成率与 reasoning 占用。

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_output_budget.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from core.config_store import load_config  # noqa: E402

INSTRUCTIONS = (
    "你是群里的一员，直接用自己的口吻说话。"
    "要发给群友的话必须放进 <say>...</say>；标签以外的任何内容不会发送。"
)
QUESTION = "崩铁现在up角色是谁"


async def call(session, url, key, model, tools, max_out) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": QUESTION}],
        "max_output_tokens": max_out,
    }
    if tools:
        body["tools"] = tools
    async with session.post(
        url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        raw = await resp.text()
        if resp.status != 200:
            return {"status": resp.status, "error": raw[:600]}
        return {"status": 200, "payload": json.loads(raw)}


def inspect(tag: str, max_out: int, res: dict) -> dict:
    print(f"\n{'=' * 70}\n{tag}（max_output_tokens={max_out}）\n{'=' * 70}")
    if res.get("error"):
        print(f"  ✗ HTTP {res['status']}：{res['error']}")
        return {"ok": False}

    p = res["payload"]
    items = [i for i in (p.get("output") or []) if isinstance(i, dict)]
    types = [i.get("type") for i in items]
    usage = p.get("usage") or {}
    top_output_text = p.get("output_text")

    print(f"  status            : {p.get('status')}")
    print(f"  incomplete_details: {p.get('incomplete_details')}")
    print(f"  output 类型       : {types}")
    print(f"  usage             : {json.dumps(usage, ensure_ascii=False)}")

    # ---- 1. output_text 解析路径 ----
    print("\n  --- output_text 解析路径 ---")
    print(f"  顶层 output_text 是否存在: {top_output_text is not None}")
    print(f"  顶层 output_text 长度    : {len(top_output_text) if isinstance(top_output_text, str) else 'N/A'}")

    fallback = ""
    msg_parts_info = []
    for item in items:
        if item.get("type") != "message":
            continue
        for part in (item.get("content") or []):
            if not isinstance(part, dict):
                continue
            piece = part.get("text", "") if part.get("type") == "output_text" else ""
            fallback += piece
            msg_parts_info.append({
                "part_type": part.get("type"),
                "text_len": len(piece),
                "keys": sorted(part.keys()),
                "annotations": len(part.get("annotations") or []),
            })
    print(f"  回退拼接（message 内 output_text）长度: {len(fallback)}")
    for info in msg_parts_info:
        print(f"    part 详情: {info}")

    if isinstance(top_output_text, str) and top_output_text:
        print(f"  顶层 output_text 前 120 字: {top_output_text[:120]!r}")
    if fallback:
        print(f"  回退拼接 前 120 字       : {fallback[:120]!r}")

    return {
        "ok": True,
        "status": p.get("status"),
        "prompt": usage.get("input_tokens", 0),
        "completion": usage.get("output_tokens", 0),
        "has_top_output_text": isinstance(top_output_text, str) and bool(top_output_text),
        "top_len": len(top_output_text) if isinstance(top_output_text, str) else 0,
        "fallback_len": len(fallback),
        "n_reasoning": sum(1 for t in types if t == "reasoning"),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/config/config.yaml")
    ap.add_argument("--base", default="https://api.minimax.cn")
    args = ap.parse_args()

    cfg = load_config(args.config)
    primary = (cfg.get("llm") or {}).get("primary") or {}
    key = str(primary.get("api_key") or "")
    model = str(primary.get("model") or "MiniMax-M3.1-Flash-Preview")
    url = args.base.rstrip("/") + "/v1/responses"
    if not key:
        raise SystemExit("配置里没有 llm.primary.api_key")

    print("输出预算与 output_text 解析健壮性验证")
    print(f"端点 : {url}\n模型 : {model}\n问题 : {QUESTION}")

    results = {}
    async with aiohttp.ClientSession() as session:
        r = await call(session, url, key, model, [{"type": "web_search"}], 1200)
        results["搜索/预算1200"] = inspect("A 带搜索、预算 1200", 1200, r)

        r = await call(session, url, key, model, [{"type": "web_search"}], 2500)
        results["搜索/预算2500"] = inspect("B 带搜索、预算 2500", 2500, r)

        r = await call(session, url, key, model, None, 1200)
        results["无搜索/预算1200"] = inspect("C 不搜索、预算 1200", 1200, r)

    print(f"\n{'=' * 70}\n汇总\n{'=' * 70}")
    print(f"  {'场景':<18}{'status':>12}{'input':>10}{'output':>9}{'thinking块':>10}"
          f"{'顶层文本':>10}{'回退文本':>10}")
    for tag, r in results.items():
        if not r.get("ok"):
            print(f"  {tag:<18}{'请求失败':>12}")
            continue
        print(f"  {tag:<18}{str(r['status']):>12}{r['prompt']:>10,}{r['completion']:>9,}"
              f"{r['n_reasoning']:>10}{r['top_len']:>10,}{r['fallback_len']:>10,}")

    vals = [r for r in results.values() if r.get("ok")]
    if vals:
        all_top = all(r["has_top_output_text"] for r in vals)
        print(f"\n  → 顶层 output_text 是否每次都有: {all_top}")
        print(f"    （为 True 说明 _parse 走主路径，回退分支只在异常时启用）")
        inc = [t for t, r in results.items() if r.get("ok") and r["status"] != "completed"]
        print(f"  → 未完成的场景: {inc or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

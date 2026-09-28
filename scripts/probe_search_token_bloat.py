"""诊断：Responses 端点上「服务端 web_search」为什么把 input token 顶到几十万。

现象（test_responses_provider.py，用例 B「崩铁现在up角色是谁」）：
  - 不联网：prompt_tokens ≈ 926
  - 联网  ：prompt_tokens ≈ 686004，completion_tokens ≈ 2268（> max_output_tokens 1200）

要弄清的问题：
  1. usage 里到底有哪些字段？服务端搜索的开销记在 input 还是 output？
  2. output 数组里 web_search_call 有几条？它们的体积多大？是否内嵌了抓取到的网页正文？
  3. 同一问题关掉 web_search 时 usage 是多少（对照组）？
  4. 给 web_search 加 max_uses 之类的约束端点是否接受？能否压住膨胀？

只打印结构与体积，不打印正文，避免刷屏。
用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_search_token_bloat.py
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


def tree_summary(node, depth: int = 0, max_depth: int = 3) -> list[str]:
    """输出 JSON 结构骨架 + 各字段字符串长度，定位体积来源。"""
    lines: list[str] = []
    pad = "    " * depth
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                size = len(json.dumps(v, ensure_ascii=False))
                lines.append(f"{pad}{k}: {type(v).__name__}({len(v)}) 约{size}字符")
                if depth < max_depth:
                    lines.extend(tree_summary(v, depth + 1, max_depth))
            elif isinstance(v, str):
                lines.append(f"{pad}{k}: str(len={len(v)})")
            else:
                lines.append(f"{pad}{k}: {v!r}")
    elif isinstance(node, list):
        for idx, v in enumerate(node[:6]):
            if isinstance(v, (dict, list)):
                size = len(json.dumps(v, ensure_ascii=False))
                lines.append(f"{pad}[{idx}]: {type(v).__name__} 约{size}字符")
                if depth < max_depth:
                    lines.extend(tree_summary(v, depth + 1, max_depth))
            else:
                lines.append(f"{pad}[{idx}]: {str(v)[:60]!r}")
        if len(node) > 6:
            lines.append(f"{pad}… 其余 {len(node) - 6} 项省略")
    return lines


async def call(session, url, key, model, text, tools, max_out=1200) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    body = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": text}],
        "max_output_tokens": max_out,
    }
    if tools is not None:
        body["tools"] = tools
    async with session.post(
        url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        raw = await resp.text()
        if resp.status != 200:
            return {"status": resp.status, "error": raw[:800]}
        return {"status": 200, "payload": json.loads(raw), "raw_len": len(raw)}


def report(tag: str, res: dict) -> dict:
    print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}")
    if res.get("error"):
        print(f"  ✗ HTTP {res['status']}：{res['error']}")
        return {"ok": False}

    payload = res["payload"]
    items = [i for i in (payload.get("output") or []) if isinstance(i, dict)]
    types = [i.get("type") for i in items]
    usage = payload.get("usage") or {}
    n_search = sum(1 for t in types if t == "web_search_call")

    print(f"  原始响应体积  : {res.get('raw_len', 0):,} 字符")
    print(f"  status        : {payload.get('status')}")
    print(f"  output 类型   : {types}")
    print(f"  服务端搜索次数: {n_search}")
    print(f"  usage         : {json.dumps(usage, ensure_ascii=False)}")

    # usage 明细里若含 input_tokens_details，往往能区分「提示」与「缓存/工具」
    for key in ("input_tokens_details", "output_tokens_details", "completion_tokens_details"):
        if key in usage:
            print(f"    {key}: {json.dumps(usage[key], ensure_ascii=False)}")

    print("\n  --- output 结构骨架（定位体积来源）---")
    for line in tree_summary(items, max_depth=2):
        print(f"  {line}")

    # 单独看 web_search_call 的完整结构（它是否内嵌网页正文）
    for item in items:
        if item.get("type") == "web_search_call":
            print(f"\n  --- web_search_call 完整结构（约{len(json.dumps(item, ensure_ascii=False)):,}字符）---")
            for line in tree_summary(item, max_depth=3):
                print(f"  {line}")

    return {
        "ok": True,
        "prompt": usage.get("input_tokens", 0),
        "completion": usage.get("output_tokens", 0),
        "total": usage.get("total_tokens", 0),
        "n_search": n_search,
        "types": types,
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

    print("服务端 web_search 的 token 膨胀诊断")
    print(f"端点  : {url}")
    print(f"模型  : {model}")
    print(f"问题  : {QUESTION}")

    results: dict[str, dict] = {}
    async with aiohttp.ClientSession() as session:
        r_off = await call(session, url, key, model, QUESTION, tools=None)
        results["关闭搜索"] = report("【对照组】不带 web_search", r_off)

        r_on = await call(session, url, key, model, QUESTION, tools=[{"type": "web_search"}])
        results["开放搜索"] = report("【实验组】带 web_search（无约束）", r_on)

        r_lim = await call(
            session, url, key, model, QUESTION,
            tools=[{"type": "web_search", "max_uses": 1}],
        )
        results["max_uses=1"] = report("【约束实验】web_search 加 max_uses=1", r_lim)

    print(f"\n{'=' * 70}\n汇总\n{'=' * 70}")
    print(f"  {'场景':<16}{'input':>12}{'output':>10}{'合计':>12}{'搜索次数':>10}")
    for tag, r in results.items():
        if not r.get("ok"):
            print(f"  {tag:<16}{'请求失败':>12}")
            continue
        print(f"  {tag:<16}{r['prompt']:>12,}{r['completion']:>10,}{r['total']:>12,}{r['n_search']:>10}")

    off = results.get("关闭搜索") or {}
    on = results.get("开放搜索") or {}
    lim = results.get("max_uses=1") or {}
    if off.get("ok") and on.get("ok"):
        multiple = on["prompt"] / max(off["prompt"], 1)
        print(f"\n  → 开搜索后 input token 放大 {multiple:,.1f} 倍"
              f"（{off['prompt']:,} → {on['prompt']:,}）")
    if lim.get("ok") and on.get("ok"):
        delta = on["prompt"] - lim["prompt"]
        print(f"  → max_uses=1 时 input = {lim['prompt']:,}"
              f"（相对无约束{'减少' if delta > 0 else '增加'} {abs(delta):,}），"
              f"搜索次数 {lim['n_search']}")
    elif not lim.get("ok"):
        print("  → max_uses 约束被端点拒绝，需另寻限流方式")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

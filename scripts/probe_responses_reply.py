"""探测：Responses 端点作为「正文回复」时的形态与截断风险。

全切方案的关键未知数：`llm.primary.max_tokens` 生产值是 500，而 Responses 的
`max_output_tokens` 把 reasoning token 计入。若推理吃掉大半预算，`<say>` 会截断或为空。

本脚本用真实的回复契约（<say> 占位 + web_search 常驻声明）跑四组：
  A 闲聊（不该搜）        max_output_tokens=500   ← 生产值
  B 时效提问（该搜）      max_output_tokens=500   ← 生产值
  C 触发发表情            max_output_tokens=500
  D 闲聊对照              max_output_tokens=1200  ← 放大后是否就正常了

记录：status / incomplete_details / 输出项类型 / output_text / <say> 是否闭合 /
usage（尤其 reasoning token）/ function_call 结构。

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_responses_reply.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from core.config_store import load_config  # noqa: E402

INSTRUCTIONS = (
    "你是群里的一员，直接用自己的口吻说话。"
    "要发给群友的话必须放进 <say>...</say>；标签以外的任何内容都不会发送，"
    "包括分析、判断、整理上下文、系统规则和给自己的提醒。"
    "<say> 里只写你会对群友说的那句话，不要写步骤。"
    "没有自然想说的内容就输出 <silent>，不要写空的 <say>。"
    "只发图时可以没有 <say>，调用 send_meme 即可。"
    "群聊里通常简短些，通常不超过 20 个中文字符。"
)

SEND_MEME_TOOL = {
    "type": "function",
    "name": "send_meme",
    "description": "选一张表情包发到当前对话。文字与表情可共存；觉得一张图比纯文字更贴切时调用。",
    "parameters": {
        "type": "object",
        "properties": {"category": {"type": "string", "description": "表情包分类"}},
    },
}
WEB_SEARCH_TOOL = {"type": "web_search"}

CASES = [
    ("A 闲聊（不该搜）", "晚饭吃啥", 500),
    ("B 时效提问（该搜）", "崩铁现在up角色是谁", 500),
    ("C 触发发表情", "笑死我了哈哈哈哈", 500),
    ("D 闲聊对照（放大预算）", "晚饭吃啥", 1200),
]


async def run(session, url, key, model, label, text, budget) -> dict:
    body = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": text}],
        "tools": [SEND_MEME_TOOL, WEB_SEARCH_TOOL],
        "max_output_tokens": budget,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    print(f"\n{'=' * 66}\n[{label}]  输入={text!r}  max_output_tokens={budget}\n{'=' * 66}")
    async with session.post(
        url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=180),
    ) as resp:
        raw = await resp.text()
        if resp.status != 200:
            print(f"    ✗ HTTP {resp.status}：{raw[:300]}")
            return {"label": label, "ok": False, "error": raw[:300]}
        payload = json.loads(raw)

    items = [i for i in (payload.get("output") or []) if isinstance(i, dict)]
    types = [i.get("type") for i in items]
    n_reasoning = sum(1 for t in types if t == "reasoning")
    n_msg = sum(1 for t in types if t == "message")
    func_calls = [i for i in items if i.get("type") == "function_call"]
    text_out = payload.get("output_text") or ""

    # <say> 是否闭合（截断检测）
    has_say = "<say>" in text_out
    closed = bool(re.search(r"<say>[\s\S]*?</say>", text_out))
    truncated_text = has_say and not closed

    usage = payload.get("usage") or {}
    details = usage.get("output_tokens_details") or {}

    print(f"    status             = {payload.get('status')}")
    print(f"    incomplete_details = {json.dumps(payload.get('incomplete_details') or {}, ensure_ascii=False)}")
    print(f"    输出项类型         = {types}")
    print(f"    reasoning 项数     = {n_reasoning} | message 项数 = {n_msg}")
    print(f"    usage              = output={usage.get('output_tokens')} "
          f"reasoning={details.get('reasoning_tokens')} "
          f"total={usage.get('total_tokens')}")
    print(f"    function_call      = {json.dumps([{k: v for k, v in f.items() if k in ('name', 'call_id', 'arguments', 'status')} for f in func_calls], ensure_ascii=False)[:400]}")
    print(f"    output_text        = {text_out[:260].replace(chr(10), ' ')!r}")
    print(f"    <say> 出现={has_say} 闭合={closed} 疑似截断={truncated_text}")
    if payload.get("error"):
        print(f"    error              = {json.dumps(payload['error'], ensure_ascii=False)[:200]}")

    return {
        "label": label, "ok": True, "status": payload.get("status"),
        "incomplete": payload.get("incomplete_details"),
        "types": types, "n_reasoning": n_reasoning,
        "reasoning_tokens": details.get("reasoning_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "func_calls": [f.get("name") for f in func_calls],
        "has_say": has_say, "closed": closed, "truncated": truncated_text,
        "text": text_out[:300],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/config/config.yaml")
    ap.add_argument("--base", default="https://api.minimax.cn")
    args = ap.parse_args()

    cfg = load_config(args.config)
    primary = (cfg.get("llm") or {}).get("primary") or {}
    key = str(primary.get("api_key") or "")
    model = str(primary.get("model") or "MiniMax-M3")
    url = args.base.rstrip("/") + "/v1/responses"

    print("Responses 端点作为正文回复：形态与截断风险探测")
    print(f"端点 : {url}")
    print(f"模型 : {model}")
    print(f"密钥 : ***{key[-4:]}")
    print(f"生产配置 llm.primary.max_tokens = 500（本次 A/B/C 用它）")

    results = []
    async with aiohttp.ClientSession() as session:
        for label, text, budget in CASES:
            results.append(await run(session, url, key, model, label, text, budget))

    print(f"\n{'=' * 66}\n结论\n{'=' * 66}")
    print(f"{'用例':<22}{'status':<12}{'reasoning':<11}{'out':<6}{'<say>闭合':<11}{'函数调用'}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['label']:<22}请求失败")
            continue
        print(f"{r['label']:<22}{str(r['status']):<12}{str(r['reasoning_tokens']):<11}"
              f"{str(r['output_tokens']):<6}{str(r['closed']):<11}{r['func_calls'] or '-'}")
    print()
    trunc = [r for r in results if r.get("truncated")]
    incomplete = [r for r in results if r.get("status") == "incomplete"]
    if trunc or incomplete:
        print("  ⚠ 存在截断/未完成：小 max_output_tokens 确实会被 reasoning 挤占，")
        print("     全切时不能沿用 500，需要按 reasoning 开销上调预算。")
    else:
        print("  ✓ 500 预算下未观察到截断，但需结合 reasoning token 量决定安全余量。")
    low = [r for r in results if r.get("ok") and r.get("reasoning_tokens")]
    if low:
        print(f"     实测 reasoning token 量：{[r['reasoning_tokens'] for r in low]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

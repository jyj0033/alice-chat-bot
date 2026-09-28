"""实测：MiniMax Server Tools（服务端 web_search）在当前 key 上是否可用。

背景：MiniMax 提供 Server Tools —— 搜索在 MiniMax 服务端执行，**单次请求内**
完成「模型决定搜 → 服务端搜 → 基于结果作答」，不需要我们做 tool_use / tool_result
多轮回传，也就不存在 role="tool" 配对、协议泄漏这些问题。

但官方文档写明它**只支持 Anthropic Messages API 与 OpenAI Responses API**，
而本项目走的是 OpenAI 兼容的 /v1/chat/completions。本脚本实测到底通不通。

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_server_tools.py
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

QUESTION = "帮我搜一下今天上海的天气"

# 待测组合：(标签, base_url, 端点路径, 请求体构造器)
def _anthropic_body(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": QUESTION}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }


def _responses_body(model: str) -> dict:
    return {
        "model": model,
        "input": QUESTION,
        "tools": [{"type": "web_search"}],
    }


TARGETS = [
    ("Anthropic Messages", "https://api.minimax.cn", "/anthropic/v1/messages", _anthropic_body, "anthropic"),
    ("OpenAI Responses", "https://api.minimax.cn", "/v1/responses", _responses_body, "bearer"),
    ("Anthropic Messages", "https://api.minimaxi.com", "/anthropic/v1/messages", _anthropic_body, "anthropic"),
    ("OpenAI Responses", "https://api.minimaxi.com", "/v1/responses", _responses_body, "bearer"),
]


async def probe(session: aiohttp.ClientSession, label, base, path, body_fn, auth, key, model) -> dict:
    url = base + path
    headers = {"Content-Type": "application/json"}
    if auth == "anthropic":
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {key}"

    print(f"\n{'-' * 66}")
    print(f"[{label}] POST {url}")
    try:
        async with session.post(
            url, headers=headers, json=body_fn(model),
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            text = await resp.text()
            print(f"    HTTP {resp.status}")
            if resp.status != 200:
                print(f"    响应：{text[:300]}")
                return {"label": label, "url": url, "status": resp.status, "ok": False}
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                print(f"    非 JSON：{text[:200]}")
                return {"label": label, "url": url, "status": resp.status, "ok": False}
    except Exception as exc:
        print(f"    ✗ 请求异常：{type(exc).__name__}: {exc}")
        return {"label": label, "url": url, "status": None, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}

    # 解析两种格式的响应
    blocks = [b for b in (payload.get("content") or payload.get("output") or [])
              if isinstance(b, dict)]
    types = [b.get("type") for b in blocks]

    # 两种接口的「搜索证据」块名不同，别混用：
    #   Anthropic Messages → server_tool_use + web_search_tool_result
    #   OpenAI Responses   → web_search_call（结果内联，不单独暴露）
    if any(t == "server_tool_use" for t in types):
        n_searches = sum(1 for t in types if t == "server_tool_use")
        results_exposed = any(t == "web_search_tool_result" for t in types)
    else:
        n_searches = sum(1 for t in types if t == "web_search_call")
        results_exposed = n_searches > 0  # 结果内联，由 web_search_call 代表
    used_search = n_searches > 0

    if payload.get("output_text"):
        answer = payload["output_text"]
    else:
        answer = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    print(f"    内容块类型：{types}")
    print(f"    服务端搜索次数：{n_searches} | 结果可见：{'是' if results_exposed else '否'}")
    print(f"    最终文本长度：{len(answer)}")
    print(f"    文本预览：{answer[:200].replace(chr(10), ' ')}")

    ok = used_search
    print(f"    → {'✓ 服务端搜索已生效' if ok else '⚠ 通了但没触发搜索'}")
    return {"label": label, "url": url, "status": resp.status, "ok": ok,
            "block_types": types, "search_used": used_search,
            "n_searches": n_searches, "answer_len": len(answer)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    primary = (cfg.get("llm") or {}).get("primary") or {}
    key = str(primary.get("api_key") or "")
    model = str(primary.get("model") or "MiniMax-M3")
    if not key:
        raise SystemExit("配置里没有 llm.primary.api_key")

    print("MiniMax Server Tools（服务端 web_search）可用性实测")
    print(f"模型 : {model}")
    print(f"密钥 : ***{key[-4:]}")
    print(f"问题 : {QUESTION}")

    results = []
    async with aiohttp.ClientSession() as session:
        for label, base, path, body_fn, auth in TARGETS:
            results.append(await probe(session, label, base, path, body_fn, auth, key, model))

    print(f"\n{'=' * 66}\n结论\n{'=' * 66}")
    hit = [r for r in results if r.get("ok")]
    for r in results:
        mark = "✓" if r.get("ok") else "✗"
        print(f"  {mark} {r['label']:22s} {r['url']}")
    print()
    if hit:
        print(f"  → {hit[0]['label']} 可用（{hit[0]['url']}），可以直接用服务端搜索，")
        print("     不需要自己写工具循环。")
    else:
        print("  → 当前 key / 区域都没有打通 Server Tools：")
        print("     可能是区域端点不匹配、或该模型/套餐未开通。")
        print("     那就继续用「外部搜索后端 + 注入」，或走已验证的自建工具循环。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""实测：Anthropic Messages 端点上，客户端工具（send_meme）能否与服务端 web_search 共存。

为什么要测：项目注释（main.py:826-827）写着「MiniMax 的 function calling 在 OpenAI
端点上可用，Anthropic 端点上不支持」。而服务端 web_search 只在 Anthropic / Responses
端点可用。若客户端工具在 Anthropic 端点真的不可用，就没法用一个端点同时覆盖
「回复 + send_meme + 服务端搜索」，必须拆成两个 provider。

本脚本测三组：
  1. 只带客户端工具 send_meme                → 是否返回 tool_use？
  2. 客户端 send_meme + 服务端 web_search 同时 → 两者能否共存？
  3. 只带服务端 web_search（对照）            → 复现可用性

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_anthropic_tools.py
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

SEND_MEME_TOOL = {
    "name": "send_meme",
    "description": (
        "选一张表情包发到当前对话。文字与表情可共存；如果图片本身已经完整表达回应，"
        "优先只发图、不重复生成同义文字；觉得一张图比纯文字更贴切时调用。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "表情包分类，如 无语/开心/得意"},
        },
    },
}

WEB_SEARCH_SERVER_TOOL = {"type": "web_search_20250305", "name": "web_search"}

SYSTEM = (
    "你是群里的一员，直接用自己的口吻说话。"
    "要发给群友的话必须放进 <say>...</say>；标签以外的任何内容都不会发送。"
    "只发图时可以没有 <say>，调用 send_meme 即可。"
)


async def call(session, url, key, model, user_text, tools, max_tokens=800) -> dict:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user_text}],
        "tools": tools,
    }
    async with session.post(
        url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            return {"status": resp.status, "error": text[:400]}
        return {"status": 200, "payload": json.loads(text)}


def summarize(tag: str, res: dict) -> dict:
    print(f"\n{'-' * 66}\n{tag}")
    if res.get("error"):
        print(f"    ✗ HTTP {res['status']}：{res['error']}")
        return {"ok": False, "status": res["status"], "error": res["error"]}
    payload = res["payload"]
    blocks = [b for b in (payload.get("content") or []) if isinstance(b, dict)]
    types = [b.get("type") for b in blocks]
    client_tools = [b.get("name") for b in blocks if b.get("type") == "tool_use"]
    server_tools = [b.get("name") for b in blocks if b.get("type") == "server_tool_use"]
    answer = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    print(f"    stop_reason = {payload.get('stop_reason')}")
    print(f"    块类型      = {types}")
    print(f"    客户端 tool_use = {client_tools}")
    print(f"    服务端 server_tool_use = {server_tools}")
    print(f"    文本        = {answer[:160].replace(chr(10), ' ') or '(空)'}")
    return {
        "ok": True, "status": 200, "types": types,
        "client_tools": client_tools, "server_tools": server_tools,
        "text_len": len(answer),
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
    url = args.base.rstrip("/") + "/anthropic/v1/messages"

    print("Anthropic Messages 端点：客户端工具 vs 服务端搜索 共存性实测")
    print(f"端点 : {url}")
    print(f"模型 : {model}")
    print(f"密钥 : ***{key[-4:]}")

    results = {}
    async with aiohttp.ClientSession() as session:
        r1 = await call(session, url, key, model,
                        "笑死我了哈哈哈", [SEND_MEME_TOOL])
        results["client_only"] = summarize("实验 1：只带客户端工具 send_meme", r1)

        r2 = await call(session, url, key, model,
                        "崩铁现在up角色是谁？顺便发个表情",
                        [SEND_MEME_TOOL, WEB_SEARCH_SERVER_TOOL], max_tokens=1200)
        results["both"] = summarize("实验 2：客户端 send_meme + 服务端 web_search 同时", r2)

        r3 = await call(session, url, key, model,
                        "帮我搜一下今天上海的天气", [WEB_SEARCH_SERVER_TOOL], max_tokens=1200)
        results["server_only"] = summarize("实验 3（对照）：只带服务端 web_search", r3)

    print(f"\n{'=' * 66}\n结论\n{'=' * 66}")
    c, b, s = results["client_only"], results["both"], results["server_only"]
    print(f"  客户端 tool_use（send_meme）单独      : {c.get('client_tools') or '无'}")
    print(f"  客户端 tool_use 与服务端搜索共存      : {b.get('client_tools') or '无'}"
          f" + server={b.get('server_tools') or '无'}")
    print(f"  服务端搜索                            : {s.get('server_tools') or '无'}")
    print()
    if c.get("client_tools"):
        print("  → 客户端工具在 Anthropic 端点是可用的（与代码注释相反）。")
        if b.get("client_tools") and b.get("server_tools"):
            print("     且能与服务端 web_search 共存于同一请求 ——")
            print("     单端点即可覆盖「回复 + 表情 + 搜索」，无需拆 provider。")
        else:
            print("     但两者在同一请求里叠加时表现异常，需要分开调用。")
    else:
        print("  → 客户端工具在 Anthropic 端点确实不可用（符合代码注释）。")
        print("     结论：搜索与回复要走 Anthropic 端点，send_meme 必须留在")
        print("     /v1/chat/completions，需按用途拆分两个 provider。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

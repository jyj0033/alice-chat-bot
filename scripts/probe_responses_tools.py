"""实测：OpenAI Responses 端点（/v1/responses）能否同时支撑客户端工具 send_meme 与服务端 web_search。

为什么测：服务端 web_search 支持两个端点 —— Anthropic Messages 与 OpenAI Responses。
Anthropic 那条路已实测可用（客户端 send_meme 与服务端搜索可共存）。
本脚本补测 Responses 这条路，判断能否不走 Anthropic、留在 OpenAI 风格端点上。

Responses 的请求/响应结构与 Chat Completions 完全不同：
  - 请求：input（而非 messages）、tools 为扁平结构 {"type":"function","name":...}
  - 响应：output 数组（含 function_call / web_search_call / message）、output_text 汇总

三组实验：
  1. 只带客户端工具 send_meme        → 是否返回 function_call？
  2. 客户端 send_meme + 服务端搜索    → 能否共存？
  3. 只带服务端搜索（对照）

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_responses_tools.py
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
    "type": "function",
    "name": "send_meme",
    "description": (
        "选一张表情包发到当前对话。文字与表情可共存；如果图片本身已经完整表达回应，"
        "优先只发图、不重复生成同义文字；觉得一张图比纯文字更贴切时调用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "表情包分类，如 无语/开心/得意"},
        },
    },
}

WEB_SEARCH_SERVER_TOOL = {"type": "web_search"}

INSTRUCTIONS = (
    "你是群里的一员，直接用自己的口吻说话。"
    "要发给群友的话必须放进 <say>...</say>；标签以外的任何内容不会发送。"
    "只发图时可以没有 <say>，调用 send_meme 即可。"
)


async def call(session, url, key, model, user_text, tools) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    body = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": user_text}],
        "tools": tools,
        "max_output_tokens": 1200,
    }
    async with session.post(
        url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=150),
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            return {"status": resp.status, "error": text[:500]}
        return {"status": 200, "payload": json.loads(text)}


def summarize(tag: str, res: dict) -> dict:
    print(f"\n{'-' * 66}\n{tag}")
    if res.get("error"):
        print(f"    ✗ HTTP {res['status']}：{res['error']}")
        return {"ok": False, "status": res["status"], "error": res["error"]}

    payload = res["payload"]
    items = [i for i in (payload.get("output") or []) if isinstance(i, dict)]
    types = [i.get("type") for i in items]
    func_calls = [i.get("name") for i in items if i.get("type") == "function_call"]
    n_search = sum(1 for t in types if t == "web_search_call")
    final = payload.get("output_text") or ""
    if not final:
        for i in items:
            if i.get("type") == "message":
                for c in (i.get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        final += c.get("text", "")
    print(f"    status      = {payload.get('status')}")
    print(f"    output 类型 = {types}")
    print(f"    客户端 function_call = {func_calls}")
    print(f"    服务端搜索次数       = {n_search}")
    print(f"    最终文本   = {final[:160].replace(chr(10), ' ') or '(空)'}")
    err = payload.get("error")
    if err:
        print(f"    error 字段  = {json.dumps(err, ensure_ascii=False)[:200]}")
    return {
        "ok": True, "status": payload.get("status"), "types": types,
        "function_calls": func_calls, "n_search": n_search, "text_len": len(final),
        "error": err,
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

    print("OpenAI Responses 端点：客户端工具 vs 服务端搜索 共存性实测")
    print(f"端点 : {url}")
    print(f"模型 : {model}")
    print(f"密钥 : ***{key[-4:]}")

    async with aiohttp.ClientSession() as session:
        r1 = await call(session, url, key, model,
                        "笑死我了哈哈哈", [SEND_MEME_TOOL])
        s1 = summarize("实验 1：只带客户端工具 send_meme", r1)

        r2 = await call(session, url, key, model,
                        "崩铁现在up角色是谁？顺便发个表情",
                        [SEND_MEME_TOOL, WEB_SEARCH_SERVER_TOOL])
        s2 = summarize("实验 2：客户端 send_meme + 服务端 web_search 同时", r2)

        r3 = await call(session, url, key, model,
                        "帮我搜一下今天上海的天气", [WEB_SEARCH_SERVER_TOOL])
        s3 = summarize("实验 3（对照）：只带服务端 web_search", r3)

    print(f"\n{'=' * 66}\n结论\n{'=' * 66}")
    print(f"  send_meme 单独        : {s1.get('function_calls') or '未触发（模型可能只是不想发图，非能力问题）'}"
          f"  status={s1.get('status')}")
    print(f"  send_meme + 搜索 共存 : {s2.get('function_calls') or '无'}"
          f"  搜索{s2.get('n_search')}次  status={s2.get('status')}")
    print(f"  服务端搜索            : {s3.get('n_search')}次  status={s3.get('status')}")
    print()
    # 以「实验 2 共存」为判据，而不是实验 1：模型对某句话不发图属正常随机行为，
    # 不能据此判定工具调用不可用（曾误判一次）。
    coexist = bool(s2.get("function_calls")) and bool(s2.get("n_search"))
    server_ok = bool(s3.get("n_search"))
    if coexist:
        print(f"  → Responses 端点可同时覆盖「表情 + 搜索」：客户端 function_call 与")
        print(f"     服务端 web_search 在同一请求内共存，且服务端搜索共 {s2.get('n_search')} 次。")
    elif server_ok:
        print("  → Responses 端点服务端搜索可用，但客户端工具调用未验证通过。")
    else:
        print("  → Responses 这条路的服务端搜索也没打通，建议走 Anthropic Messages。")
    print()
    print(f"  提示：本组实验服务端搜索次数偏高（实验3={s3.get('n_search')} 次），")
    print("       比 Anthropic 端点（2~4 次）更费 token 与时间，选型时需权衡。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

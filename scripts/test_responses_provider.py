"""端到端验证新的 ResponsesProvider（走真实 LLMProvider 接口 + 真实端点）。

验证点（对应全切方案必须处理的几件事）：
  1. 多条 system 消息能正确合并进 instructions，且不影响输出
  2. web_search 走「按需声明」：由调用方塞进 request.tools 才放行；
     能力开关关闭时必须被剔除（防配置与调用不一致误开联网）
  3. ChatResponse.tool_calls 的结构与 generator._extract_send_meme_call 期望一致
     （{"name": "send_meme", "arguments": {dict}}）
  4. 传入 stop（Chat Completions 的协议边界序列）时不会报错——被正确丢弃
  5. max_output_tokens 下限生效（传 500 时实际用 >= MIN_OUTPUT_TOKENS）
  6. 输出无协议泄漏（<invoke / <]minimax / <tool_call / <parameter）

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/test_responses_provider.py
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_store import load_config  # noqa: E402
from modules.llm.base import ChatMessage, ChatRequest  # noqa: E402
from modules.llm.responses_provider import (  # noqa: E402
    MIN_OUTPUT_TOKENS,
    ResponsesProvider,
)

# 与 generator 注入的协议边界一致，用来验证「Responses 无 stop 参数」的处理
PROTOCOL_STOPS = ["]<]minimax", "<]minimax"]
LEAK_MARKERS = ("<invoke", "<tool_call", "<]minimax", "<parameter")

SEND_MEME_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "send_meme",
        "description": "选一张表情包发到当前对话。觉得一张图比纯文字更贴切时调用。",
        "parameters": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "表情包分类"}},
        },
    },
}

SYSTEM_A = (
    "你是群里的成员，直接用自己的口吻说话。"
    "要发给群友的话必须放进 <say>...</say>；标签以外的任何内容不会发送。"
)
SYSTEM_B = "只发图时可以没有 <say>，调用 send_meme 即可。群聊里通常不超过 20 个中文字符。"

CASES = [
    ("A 闲聊", "晚饭吃啥"),
    ("B 需要联网", "崩铁现在up角色是谁"),
    ("C 触发发图", "笑死我了哈哈哈哈哈"),
]


def check_leak(text: str) -> bool:
    return any(m in (text or "") for m in LEAK_MARKERS)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    primary = dict((cfg.get("llm") or {}).get("primary") or {})
    if not primary.get("api_key"):
        raise SystemExit("配置里没有 llm.primary.api_key")

    provider = ResponsesProvider({
        "api_key": primary.get("api_key"),
        "base_url": primary.get("base_url"),
        "model": primary.get("model"),
        "timeout": 180,
        "max_tokens": primary.get("max_tokens", 500),   # 故意沿用生产的 500
        "web_search": True,
    })

    print("ResponsesProvider 端到端验证")
    print(f"端点            : {provider.url}")
    print(f"模型            : {provider.model}")
    print(f"服务端搜索      : {provider.supports_server_search}")
    print(f"输出上限        : {provider.max_output_tokens}"
          f"（传入 max_tokens={primary.get('max_tokens')}，下限 {MIN_OUTPUT_TOKENS}）")
    assert provider.max_output_tokens >= MIN_OUTPUT_TOKENS, "输出上限下限未生效"
    print("  ✓ 断言 5 通过：输出上限下限生效（防 reasoning 挤占导致截断）")

    # 断言 2：工具转换（web_search 按需声明）
    declared = [SEND_MEME_TOOL_OPENAI, {"type": "web_search"}]
    converted = provider._build_tools(declared)
    names = [t.get("name") for t in converted]
    assert {"type": "web_search"} in converted, "按需声明的 web_search 未被放行"
    assert "send_meme" in names, f"send_meme 未转换：{converted}"
    meme_tool = next(t for t in converted if t.get("name") == "send_meme")
    assert meme_tool["type"] == "function" and "parameters" in meme_tool, "转换结构不对"
    # 未声明时不得凭空追加
    assert {"type": "web_search"} not in provider._build_tools([SEND_MEME_TOOL_OPENAI]), \
        "未声明 web_search 却被自动追加（按需声明被破坏）"
    # 能力关闭时必须剔除
    off_provider = ResponsesProvider({
        "api_key": "x", "base_url": primary.get("base_url"),
        "model": primary.get("model"), "web_search": False,
    })
    assert {"type": "web_search"} not in off_provider._build_tools(declared), \
        "web_search 能力关闭时未剔除服务端工具"
    print(f"  ✓ 断言 2 通过：web_search 按需放行/关闭即剔除 → "
          f"{[t.get('type') + ':' + str(t.get('name', '-')) for t in converted]}")

    ok = True
    for label, text in CASES:
        print(f"\n{'-' * 66}\n[{label}] {text}\n{'-' * 66}")
        req = ChatRequest(
            messages=[
                ChatMessage(role="system", content=SYSTEM_A),
                ChatMessage(role="system", content=SYSTEM_B),   # 断言 1：多条 system
                ChatMessage(role="user", content=text),
            ],
            model=provider.model,
            temperature=0.8,
            max_tokens=500,
            stop=PROTOCOL_STOPS,                                  # 断言 4：无 stop 参数
            # 模拟上层「按需声明」：判断器认为需要查资料时才会带上 web_search
            tools=[SEND_MEME_TOOL_OPENAI, {"type": "web_search"}],
        )
        resp = await provider.chat(req)
        content = resp.content or ""
        calls = list(resp.tool_calls or [])
        say = re.findall(r"<say>[\s\S]*?</say>", content)
        leak = check_leak(content)
        print(f"    finish_reason = {resp.finish_reason}")
        print(f"    content       = {content[:220].replace(chr(10), ' ')!r}")
        print(f"    tool_calls    = {calls}")
        print(f"    usage         = {resp.usage}")
        print(f"    <say> 闭合={bool(say)}  泄漏={leak}")

        if resp.finish_reason == "incomplete":
            print("    ✗ 未完成（被截断）")
            ok = False
        if leak:
            print("    ✗ 存在协议泄漏")
            ok = False
        if not say and not calls:
            print("    ✗ 既无 <say> 也无工具调用")
            ok = False

        # 断言 3：tool_calls 结构符合 _extract_send_meme_call 期望
        for tc in calls:
            if not isinstance(tc, dict) or "name" not in tc:
                print(f"    ✗ tool_calls 结构不符合期望：{tc}")
                ok = False
            elif not isinstance(tc.get("arguments"), dict):
                print(f"    ✗ arguments 不是 dict：{tc.get('arguments')!r}")
                ok = False
            else:
                print(f"    ✓ tool_call 结构合格：name={tc['name']} args={tc['arguments']}")

    print(f"\n{'=' * 66}\n结论\n{'=' * 66}")
    print(f"  {'✓ 全部断言通过，可投入配置切换' if ok else '✗ 存在未通过项，先修再切'}")
    await provider.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""量化正文回复路径在 Responses 端点上的完成率。

背景：切换后后台辅助任务（表达习惯/黑话清理/群日报）出现约 71% 的
status=incomplete（推理吃满 max_output_tokens、无 message 产出）。
本脚本用真实 generator 跑多样本，测量回复路径本身的失败率。

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_reply_reliability.py
"""
from __future__ import annotations

import asyncio
import logging
import sys

sys.path.insert(0, "/app")
logging.basicConfig(level=logging.WARNING)

from core.config_store import load_config  # noqa: E402
from main import _normalize_llm_provider_config  # noqa: E402
from modules.llm.openai_provider import create_provider  # noqa: E402
from modules.reply.generator import ReplyGenerator  # noqa: E402

CASES = [
    ("闲聊", "晚饭吃啥"),
    ("闲聊", "今天好累啊"),
    ("接梗", "笑死我了哈哈哈哈哈"),
    ("需联网", "崩铁现在up角色是谁"),
    ("问梗", "这个梗什么意思"),
    ("观点", "你觉得考研还是工作好"),
]


async def main() -> int:
    cfg = load_config("/app/config/config.yaml")
    primary = dict((cfg.get("llm") or {}).get("primary") or {})
    pc = _normalize_llm_provider_config(primary)
    provider = create_provider(
        pc.get("provider_type", "openai_compatible"),
        {
            "provider_type": pc.get("provider_type"),
            "api_key": pc.get("api_key", ""),
            "base_url": pc.get("base_url", ""),
            "model": pc.get("model", ""),
            "timeout": pc.get("timeout", 120),
            "temperature": pc.get("temperature", 0.8),
            "max_tokens": pc.get("max_tokens", 2000),
            "top_p": pc.get("top_p", 0.9),
            "web_search": pc.get("web_search", False),
            "max_output_tokens": pc.get("max_output_tokens"),
        },
    )
    print(f"provider={type(provider).__name__} 上限={provider.max_output_tokens}")

    rows = []
    for label, text in CASES:
        generator = ReplyGenerator(llm_provider=provider)
        seen = []
        original = provider.chat

        async def spy(request):
            response = await original(request)
            seen.append((request, response))
            return response

        provider.chat = spy
        try:
            reply = await generator.generate(
                context_prompt=f"[刚刚] 小明(对你说)：{text}",
                current_message=text,
                direction="to_bot",
            )
        finally:
            provider.chat = original

        main_reqs = [
            (r, resp) for r, resp in seen
            if not any("检索判断器" in str(getattr(m, "content", "")) for m in r.messages)
        ]
        statuses = [resp.finish_reason for _, resp in main_reqs]
        searches = [
            sum(1 for i in (resp.raw_response or {}).get("output", [])
                if isinstance(i, dict) and i.get("type") == "web_search_call")
            for _, resp in main_reqs
        ]
        usage = [resp.usage.get("completion_tokens", 0) for _, resp in main_reqs]
        text_out = (reply or {}).get("reply") if isinstance(reply, dict) else reply
        rows.append((label, text, statuses, searches, usage, text_out))
        print(f"\n[{label}] {text}")
        print(f"  status={statuses} 搜索={searches} 输出token={usage}")
        print(f"  回复={text_out!r}")

    total = [s for _, _, st, _, _, _ in rows for s in st]
    bad = [s for s in total if s != "completed"]
    print("\n" + "=" * 66)
    print(f"完成率：{len(total) - len(bad)}/{len(total)}   未完成={bad}")
    empty = [t for *_, t in rows if not t]
    print(f"空回复数：{len(empty)}/{len(rows)}")
    await provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

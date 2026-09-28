"""生产配置下的端到端冒烟：确认「按需联网」在真实链路里生效。

不构造整只 bot，只做 main._init_llm 对 primary 所做的同样事情：
  读配置 → _normalize_llm_provider_config → create_provider → ReplyGenerator。

判据：
  1. provider_type 归一化后仍是 responses（没被 MiniMax 归一化改写）
  2. 收到的 provider 是 ResponsesProvider 且 supports_server_search=True
  3. 闲聊用例：请求里**没有** web_search，prompt_tokens 在几百量级
  4. 需联网用例：请求里**有** web_search，且模型确实搜了

用法（容器内，读的是挂载进来的生产配置）：
    docker cp scripts/verify_prod_server_search.py alice-chat-bot:/app/scripts/
    docker exec alice-chat-bot python3 /app/scripts/verify_prod_server_search.py
"""
from __future__ import annotations

import asyncio
import logging
import sys

sys.path.insert(0, "/app")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")

from core.config_store import load_config  # noqa: E402
from main import _normalize_llm_provider_config  # noqa: E402
from modules.llm.openai_provider import create_provider  # noqa: E402
from modules.reply.generator import ReplyGenerator  # noqa: E402

CASES = [("闲聊", "晚饭吃啥"), ("需联网", "崩铁现在up角色是谁")]


def build_provider():
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
    return pc, provider


def has_web_search(request) -> bool:
    return any(
        isinstance(t, dict) and t.get("type") == "web_search"
        for t in (getattr(request, "tools", None) or [])
    )


async def run_case(provider, label: str, text: str) -> dict:
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

    main_reqs = [(r, resp) for r, resp in seen if not _is_judge(r)]
    declared = [has_web_search(r) for r, _ in main_reqs]
    searches = [
        sum(1 for i in (resp.raw_response or {}).get("output", [])
            if isinstance(i, dict) and i.get("type") == "web_search_call")
        for _, resp in main_reqs
    ]
    usage = [resp.usage for _, resp in main_reqs]

    print(f"\n[{label}] {text}")
    print(f"  请求数={len(seen)}  主请求数={len(main_reqs)}")
    print(f"  主请求声明 web_search = {declared}")
    print(f"  服务端搜索次数        = {searches}")
    print(f"  usage                = {usage}")
    print(f"  回复 = {reply!r}")
    return {"declared": any(declared), "searches": sum(searches), "reply": reply}


def _is_judge(request) -> bool:
    return any("检索判断器" in str(getattr(m, "content", "")) for m in request.messages)


async def main() -> int:
    pc, provider = build_provider()
    print("=" * 70)
    print(f"归一化后 provider_type = {pc.get('provider_type')}  web_search = {pc.get('web_search')}")
    print(f"provider = {type(provider).__name__}  supports_server_search = "
          f"{getattr(provider, 'supports_server_search', None)}")
    print("=" * 70)

    results = {}
    for label, text in CASES:
        results[label] = await run_case(provider, label, text)

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    ok = True
    if pc.get("provider_type") != "responses":
        print(f"  ✗ provider_type 被改写为 {pc.get('provider_type')}")
        ok = False
    if getattr(provider, "supports_server_search", None) is not True:
        print("  ✗ provider 未开启服务端搜索")
        ok = False
    if results["闲聊"]["declared"]:
        print("  ✗ 闲聊也声明了 web_search（按需声明失效）")
        ok = False
    else:
        print("  ✓ 闲聊未声明 web_search")
    if not results["需联网"]["declared"]:
        print("  ✗ 需联网的消息没有声明 web_search")
        ok = False
    else:
        print("  ✓ 需联网的消息声明了 web_search，"
              f"服务端搜索 {results['需联网']['searches']} 次")
    print(f"\n  {'✓ 生产链路按需联网生效' if ok else '✗ 链路不符合预期'}")
    await provider.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

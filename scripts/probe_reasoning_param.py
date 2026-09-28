"""探测：Responses 端点是否支持 reasoning 参数（能否压低推理占用）。

动机：切换后后台辅助任务出现 status=incomplete、输出项只有 reasoning、
文本长度=0（推理吃满 max_output_tokens，一条 message 都没产出）。
若端点支持 reasoning.effort 之类的降档参数，就能从源头解决，而不必
为辅助任务单独挂一个 chat/completions Provider。

用一个「从词表里挑出不合语境的条目、只输出 JSON」的辅助任务式提示词，
逐档测试，观察 HTTP 状态、output_tokens 与是否产出 message。

用法（容器内）：
    docker exec alice-chat-bot python3 /app/scripts/probe_reasoning_param.py
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
    "你是群聊黑话词表的清理助手。判断每个词条是否明显不属于该群语境、"
    "或是格式错误的碎片（如单个语气词、半截句子）。只输出一行 JSON："
    '{"drop": [id, ...]}，不要解释、不要复述词表。'
)

# 模拟真实失败场景：64 条自动词条
ENTRIES = "\n".join(
    f"{1000 + i}. {w}"
    for i, w in enumerate([
        "yyds", "蚌埠住了", "典", "急了", "我服了", "笑死", "6", "啊这",
        "绝了", "润了", "破防了", "上大分", "稳", "好家伙", "离谱",
        "????????", "嗯", "哦", "额", "……", "haha", "2333", "乐",
        "麻了", "裂开", "寄", "冲", "开摆", "绷不住", "有内味了",
        "你礼貌吗", "格局小了", "绷", "我裂开了", "地狱笑话", "活久见",
        "离谱他妈给离谱开门", "属于是", "拷打", "上强度", "画饼",
        "顶级理解", "抽象", "缝合怪", "整活", "岁月静好", "退！退！退！",
        "真下头", "好耶", "栓Q", "666", "哈哈哈", "。" , "??",
        "1", "2", "3", "a", "b", "test", "test2", "无", "空", "……",
    ])
)
QUESTION = f"词表如下：\n{ENTRIES}\n\n请挑出应删除的条目 id。"

VARIANTS = [
    ("基线（不带 reasoning）", None),
    ("reasoning.effort=low", {"reasoning": {"effort": "low"}}),
    ("reasoning.effort=minimal", {"reasoning": {"effort": "minimal"}}),
    ("reasoning.effort=none", {"reasoning": {"effort": "none"}}),
    ("reasoning.enabled=false", {"reasoning": {"enabled": False}}),
]
MAX_OUT = 2000


async def call(session, url, key, model, extra) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": QUESTION}],
        "max_output_tokens": MAX_OUT,
    }
    if extra:
        body.update(extra)
    async with session.post(
        url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        raw = await resp.text()
        if resp.status != 200:
            return {"status": resp.status, "error": raw[:300]}
        return {"status": 200, "payload": json.loads(raw)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    primary = (cfg.get("llm") or {}).get("primary") or {}
    key = str(primary.get("api_key") or "")
    model = str(primary.get("model") or "MiniMax-M3.1-Flash-Preview")
    url = str(primary.get("base_url") or "https://api.minimax.cn/v1").rstrip("/") + "/responses"

    print("Responses 端点 reasoning 参数探测")
    print(f"端点={url}\n模型={model}\n输出上限={MAX_OUT}\n词条数=64\n")

    rows = []
    async with aiohttp.ClientSession() as session:
        for name, extra in VARIANTS:
            res = await call(session, url, key, model, extra)
            if res.get("error"):
                print(f"[{name}] ✗ HTTP {res['status']}：{res['error'][:200]}")
                rows.append((name, f"HTTP {res['status']}", "-", "-", "参数被拒"))
                continue
            p = res["payload"]
            items = [i for i in (p.get("output") or []) if isinstance(i, dict)]
            types = [i.get("type") for i in items]
            usage = p.get("usage") or {}
            has_msg = "message" in types
            print(f"[{name}] status={p.get('status')} 输出项={types}")
            print(f"    input={usage.get('input_tokens')} output={usage.get('output_tokens')}"
                  f" 有message={has_msg} incomplete={p.get('incomplete_details')}")
            if has_msg:
                txt = p.get("output_text") or ""
                print(f"    文本前 100 字: {txt[:100]!r}")
            rows.append((
                name, p.get("status"), usage.get("output_tokens"),
                "有" if has_msg else "无",
                json.dumps(p.get("incomplete_details"), ensure_ascii=False),
            ))

    print("\n" + "=" * 74)
    print(f"{'变体':<26}{'status':<12}{'output':<9}{'message':<9}{'incomplete'}")
    for name, status, out, msg, inc in rows:
        print(f"{name:<26}{str(status):<12}{str(out):<9}{msg:<9}{inc}")
    print("=" * 74)

    # 关键回归：走 Provider 默认档位（不显式传 reasoning），确认生产路径已修好
    from modules.llm.base import ChatMessage, ChatRequest
    from modules.llm.responses_provider import ResponsesProvider

    provider = ResponsesProvider({
        "api_key": key,
        "base_url": str(primary.get("base_url") or "https://api.minimax.cn/v1"),
        "model": model,
        "timeout": 180,
        "max_tokens": 500,
        "web_search": False,
    })
    resp = await provider.chat(ChatRequest(messages=[
        ChatMessage(role="system", content=INSTRUCTIONS),
        ChatMessage(role="user", content=QUESTION),
    ]))
    ok = resp.finish_reason == "completed" and bool((resp.content or "").strip())
    print(f"\n[Provider 默认档位 effort={provider.reasoning_effort}] "
          f"status={resp.finish_reason} output={resp.usage.get('completion_tokens')} "
          f"有文本={bool((resp.content or '').strip())}")
    print(f"    文本前 100 字: {(resp.content or '')[:100]!r}")
    print(f"  → {'✓ 同等任务已能正常产出' if ok else '✗ 仍然空回复'}")

    print("\n判读：若某档 status=completed 且 有message，说明该参数可用且能解决推理吃空预算。")
    await provider.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

"""验证：本项目的模型端点能否支撑「搜索工具循环」（第二轮 role="tool" 回灌）。

背景：搜索功能曾用 function calling 工具循环实现，后因「内容泄漏 / 空回复 /
原生 <invoke> 标记 / 轮数超限」被撤掉，改成「预判断 + 结果注入」。
send_meme 至今仍走原生 tool_use，说明**第一轮** tool call 可用；
未验证的是**第二轮**：把工具结果以 role="tool" 回灌后，模型能否正常收尾。

本脚本用项目自己的 OpenAIProvider（即生产同一路径）跑三个探针：

  探针 A：带 tools 请求 → 模型是否发出结构化 tool_calls？还是泄漏 <invoke>？
  探针 B：回灌 role="tool" 结果 → 模型能否输出合格 <say>？还是空/泄漏？
  探针 C：不带 tools 请求（对照组）→ 模型是否直接凭记忆回答。

用法（在容器内执行，依赖与配置齐备）：
    docker exec alice-chat-bot python3 /app/scripts/verify_tool_roundtrip.py
可选：
    --config /app/config/config.yaml   指定配置
    --provider minimax                 指定 llm.primary / llm.providers.<name>
    --question "崩铁现在up角色是谁"     换一个测试问题
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_store import load_config  # noqa: E402
from modules.llm.base import ChatMessage, ChatRequest  # noqa: E402
from modules.llm.openai_provider import OpenAIProvider  # noqa: E402

# 与 generator._build_request 里的发送契约保持一致，保证测的是真实场景。
SYSTEM_PROMPT = (
    "你是群里的一员，直接用自己的口吻说话。"
    "要发给群友的话必须放进 <say>...</say>；标签以外的任何内容都不会发送，"
    "包括分析、判断、整理上下文、系统规则和给自己的提醒。"
    "<say> 里只写你会对群友说的那句话，不要写步骤。"
    "没有自然想说的内容就输出 <silent>，不要写空的 <say>。"
    "不要说自己是机器人、AI、bot 或程序。"
)

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "联网搜索实时信息。当问题涉及当前版本、卡池、角色、天气、新闻、"
            "价格、赛程等会随时间变化的内容，或你需要确认某个事实时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
}

# 假装搜到的结果，用来测「回灌」这一步，不依赖真实搜索后端。
FAKE_RESULTS = (
    "搜索到的资料：\n"
    "1. [2026-09-20] 崩坏星穹铁道当前卡池一览\n"
    "来源：米游社\n"
    "本期跃迁卡池 UP 五星角色为「流萤」，四星陪跑为「桂乃芬／娜塔莎」。\n"
    "https://example.com/hsr-banner\n"
    "2. [2026-09-18] 星穹铁道 9 月卡池时间表\n"
    "来源：游侠网\n"
    "9 月上半为流萤复刻，下半切换为「知更鸟」。\n"
)

# 只认「工具协议标记」泄漏；<say>/<silent> 是合法发送契约，不算泄漏。
LEAK_MARKERS = ("<invoke", "<tool_call", "<]minimax", "<parameter")

# 工具循环最多跑几轮（防止模型一直搜下去）
MAX_ROUNDS = 3


def has_leak(text: str) -> bool:
    """是否残留工具协议标记（会脏到群聊里）。"""
    return any(p in (text or "") for p in LEAK_MARKERS)


def show(title: str, text: str) -> None:
    body = (text or "").replace("\n", "\\n")
    print(f"    {title}: {body[:400] if body else '(空)'}")


async def run_probe(llm: OpenAIProvider, question: str, label: str) -> dict:
    print(f"\n{'=' * 68}\n{label}\n{'=' * 68}")
    base_messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=question),
    ]

    # ---------- 探针 A：带 tools ----------
    req_a = ChatRequest(
        messages=list(base_messages),
        model=llm.model,
        temperature=0.7,
        max_tokens=600,
        tools=[WEB_SEARCH_TOOL],
    )
    print("\n[探针 A] 带 web_search 工具发起请求")
    resp_a = await llm.chat(req_a)
    calls = list(resp_a.tool_calls or [])
    names = [c.get("name") for c in calls]
    args = [c.get("arguments") for c in calls]
    print(f"    finish_reason = {resp_a.finish_reason}")
    show("content", resp_a.content)
    print(f"    tool_calls    = {json.dumps(names, ensure_ascii=False)}  args={json.dumps(args, ensure_ascii=False)[:200]}")
    leak_a = has_leak(resp_a.content)

    called = bool(calls) and "web_search" in names
    say_a = re.findall(r"<say>(.*?)</say>", resp_a.content or "", flags=re.DOTALL)
    if called:
        note = "✓ 发出结构化 tool_calls"
    elif say_a:
        note = "△ 没调工具，直接给了 <say> 回答"
    else:
        note = "✗ 既没调工具也没给出 <say>"
    print(f"    → {note}{'（content 里泄漏了协议标记！）' if leak_a else ''}")

    round1 = {
        "called": called,
        "leaked": leak_a,
        "empty": not (resp_a.content or "").strip(),
        "say_count": len(say_a),
        "tool_calls": calls,
    }

    # ---------- 探针 B：回灌 role="tool"（必须应答每一个 tool_call） ----------
    print("\n[探针 B] 把结果以 role=\"tool\" 回灌，让模型收尾")
    print(f"    第一轮 tool_calls 明细：{json.dumps(calls, ensure_ascii=False)[:400]}")
    if not called:
        print("    跳过：第一轮没发出 tool_calls，无法构造回灌消息")
        round1["round2"] = {"skipped": True}
        return {"round1": round1}

    messages = list(base_messages)
    rounds = []
    final = None
    for rnd in range(1, MAX_ROUNDS + 1):
        # 上一轮的调用 → assistant 消息 + 每个 call 一条 tool 响应
        assistant_calls = []
        tool_msgs = []
        for idx, c in enumerate(calls):
            # MiniMax 有时不回 id；缺失时补一个，否则协议无法配对
            call_id = str(c.get("id") or f"call_{rnd}_{idx}")
            assistant_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": c.get("name") or "web_search",
                    "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
                },
            })
            tool_msgs.append(ChatMessage(
                role="tool", content=FAKE_RESULTS, tool_call_id=call_id,
            ))
        # 纯工具调用消息的 content 用 None（OpenAI 规范形态），不用空串
        messages.append(ChatMessage(
            role="assistant", content=None, tool_calls=assistant_calls,
        ))
        messages.extend(tool_msgs)
        print(f"    · 第 {rnd} 轮：已回灌 {len(tool_msgs)} 条 tool 结果"
              f"（call_id={[m.tool_call_id for m in tool_msgs]}）")

        req_b = ChatRequest(
            messages=messages, model=llm.model, temperature=0.7,
            max_tokens=600, tools=[WEB_SEARCH_TOOL],
        )
        try:
            resp_b = await llm.chat(req_b)
        except Exception as exc:  # 端点不接受 role="tool" 时会在这里炸
            print(f"    ✗ 请求失败（端点可能不支持 role=\"tool\"）：{type(exc).__name__}: {exc}")
            round1["round2"] = {"error": f"{type(exc).__name__}: {exc}"}
            return {"round1": round1}

        content_b = resp_b.content or ""
        calls = list(resp_b.tool_calls or [])
        say = re.findall(r"<say>(.*?)</say>", content_b, flags=re.DOTALL)
        leak_b = has_leak(content_b)
        print(f"      finish_reason={resp_b.finish_reason} "
              f"tool_calls={[c.get('name') for c in calls]} <say>={len(say)} 泄漏={leak_b}")
        show("      content", content_b)
        rounds.append({
            "round": rnd, "finish_reason": resp_b.finish_reason,
            "tool_calls": [c.get("name") for c in calls],
            "say_count": len(say), "leaked": leak_b, "content": content_b[:300],
        })
        if not calls:  # 不再调工具 → 本轮就是最终回复
            final = {"content": content_b, "say": say, "leaked": leak_b}
            break

    if final is None:
        # 一直循环调工具、收不了尾
        verdict = f"✗ {MAX_ROUNDS} 轮内始终在调工具，收不了尾"
        ok = False
    else:
        say, empty_b, leak_b = final["say"], not final["content"].strip(), final["leaked"]
        if say and not empty_b and not leak_b:
            verdict = "✓ 通过：能基于工具结果正常收尾"
        elif empty_b:
            verdict = "✗ 空回复：回灌后吐不出内容"
        elif leak_b:
            verdict = "✗ 协议泄漏：tool 标记混进了可见文本"
        else:
            verdict = "⚠ 有内容但没 <say>：格式不合发送契约"
        ok = bool(say) and not empty_b and not leak_b
    print(f"    → {verdict}")

    round1["round2"] = {
        "ok": ok, "verdict": verdict, "rounds": rounds,
        "error": None,
    }
    return {"round1": round1}


async def run_control(llm: OpenAIProvider, question: str) -> dict:
    """对照组：不带 tools，看模型是否直接凭记忆回答。"""
    print(f"\n{'=' * 68}\n[探针 C] 对照组：不带 tools\n{'=' * 68}")
    req = ChatRequest(
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=question),
        ],
        model=llm.model,
        temperature=0.7,
        max_tokens=600,
    )
    try:
        resp = await llm.chat(req)
    except Exception as exc:
        print(f"    ✗ 失败：{type(exc).__name__}: {exc}")
        return {"error": str(exc)}
    content = resp.content or ""
    show("content", content)
    say = re.findall(r"<say>(.*?)</say>", content, flags=re.DOTALL)
    print(f"    → <say> {len(say)} 段（对照组只是看它是否会硬答）")
    return {"say_count": len(say), "content": content[:300]}


def build_provider(cfg: dict, name: str) -> OpenAIProvider:
    llm_cfg = cfg.get("llm") or {}
    if name == "primary" or not llm_cfg.get("providers"):
        raw = dict(llm_cfg.get("primary") or {})
    else:
        raw = dict((llm_cfg.get("providers") or {}).get(name) or {})
    if not raw:
        raise SystemExit(f"配置里找不到 provider：{name}")
    if not raw.get("api_key"):
        raise SystemExit(f"provider「{name}」没有 api_key")
    raw.setdefault("provider_type", "openai_compatible")
    provider = OpenAIProvider(raw)
    key = str(raw.get("api_key") or "")
    print(f"端点 : {raw.get('base_url')}")
    print(f"模型 : {provider.model}")
    print(f"密钥 : ***{key[-4:] if key else ''}")
    return provider


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/config/config.yaml")
    ap.add_argument("--provider", default="primary")
    ap.add_argument("--question", default="崩铁现在up角色是谁")
    args = ap.parse_args()

    print("工具循环验证 —— 搜索第二轮 role=\"tool\" 回灌")
    print(f"配置文件: {args.config}")
    cfg = load_config(args.config)
    llm = build_provider(cfg, args.provider)

    result = await run_probe(llm, args.question, f"主测：{args.question}")
    await run_control(llm, args.question)

    print(f"\n{'=' * 68}\n结论\n{'=' * 68}")
    r1 = result["round1"]
    print(f"  第一轮发出 tool_calls : {'是' if r1['called'] else '否'}"
          f"（共 {len(r1['tool_calls'])} 个）")
    r2 = r1.get("round2") or {}
    if r2.get("skipped"):
        print("  第二轮收尾            : 未测（第一轮没调工具）")
    elif r2.get("error"):
        print(f"  第二轮收尾            : 端点报错 -> {r2['error'][:120]}")
    else:
        print(f"  第二轮收尾            : {'通过' if r2['ok'] else '不通过'} -> {r2['verdict']}")
        for rd in r2.get("rounds", []):
            print(f"      · 轮{rd['round']}: finish={rd['finish_reason']} "
                  f"tool_calls={rd['tool_calls']} <say>={rd['say_count']} 泄漏={rd['leaked']}")
    print()
    if r1["called"] and r2.get("ok"):
        print("  → 可以通过：该模型能自行调搜索工具并基于结果收尾。")
        print("     建议给生成器注册 web_search 工具，做 1-2 轮有界循环；")
        print("     send_meme 那套防泄漏/重答兜底直接复用。")
    elif r1["called"]:
        print("  → 第一轮没问题，但回灌收尾不可靠：保留「预搜+注入」为主路径，")
        print("     工具循环只作为可选优化，并复用现有防泄漏/重答兜底。")
    else:
        print("  → 该模型/端点在这套提示词下不主动调工具，工具循环方案不可行；")
        print("     应继续用「预判断 + 结果注入」，把判断器和开关修好。")
    await llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

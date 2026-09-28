#!/usr/bin/env python3
"""群日报生成探针：复现「金句 / 逆天语录整块消失」并定位原因。

不复刻生产 prompt —— 直接调真的 `GroupDailyAnalysis.analyze()`，只把
provider 包一层记录「端点原始返回 / finish_reason / token 用量」。
消息从生产库里按 session + 日期取真数据。

用法（容器内）：
    python3 scripts/probe_group_report.py --session group_724786753
    python3 scripts/probe_group_report.py --session group_724786753 --max-tokens 6000
    python3 scripts/probe_group_report.py --session group_724786753 --dump-prompt /app/logs/report_prompt.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("probe_report")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.config_store import load_config as load_config_file  # noqa: E402
from main import _normalize_llm_provider_config  # noqa: E402
from modules.group_analysis import GroupDailyAnalysis  # noqa: E402
from modules.llm.openai_provider import create_provider  # noqa: E402
from modules.memory.storage import MemoryStorage  # noqa: E402
from modules.memory.storage import AsyncMemoryStorage  # noqa: E402


def _build_provider(cfg: dict, provider_name: str):
    """和 main._init_llm 完全同一条构造路径，避免探针自己造一个不一样的 provider。"""
    providers = dict(cfg.get("llm") or {})
    providers.update(cfg.get("providers") or {})
    raw = providers.get(provider_name)
    if not isinstance(raw, dict):
        raise SystemExit(f"配置里没有 provider「{provider_name}」")
    raw = _normalize_llm_provider_config(dict(raw))
    return create_provider(
        raw.get("provider_type", "openai_compatible"),
        {
            "provider_type": raw.get("provider_type", "openai_compatible"),
            "api_key": raw.get("api_key", ""),
            "base_url": raw.get("base_url", ""),
            "model": raw.get("model", ""),
            "timeout": raw.get("timeout", 120),
            "temperature": raw.get("temperature", 0.8),
            "max_tokens": raw.get("max_tokens", 2000),
            "top_p": raw.get("top_p", 0.9),
            "web_search": raw.get("web_search", False),
            "max_output_tokens": raw.get("max_output_tokens"),
            "reasoning_effort": raw.get("reasoning_effort", "low"),
            "search_temperature": raw.get("search_temperature"),
        },
    ), raw


class _Recorder:
    """包装（非替换）provider.chat，记录每次端点原始返回。"""

    def __init__(self, provider):
        self._provider = provider
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._provider, name)

    async def chat(self, request, *args, **kwargs):
        response = await self._provider.chat(request, *args, **kwargs)
        content = getattr(response, "content", "") or ""
        self.calls.append(
            {
                "finish_reason": getattr(response, "finish_reason", "?"),
                "usage": dict(getattr(response, "usage", {}) or {}),
                "model": getattr(response, "model", "?"),
                "content_len": len(content),
                "content": content,
                "request_max_tokens": getattr(request, "max_tokens", None),
            }
        )
        return response


def _report_config(cfg: dict) -> dict:
    """照抄 main 里 `_group_analysis_config` 的取法，保证预算/上限一致。"""
    analysis_cfg = (cfg.get("memory") or {}).get("group_analysis", {}) or {}
    return {
        "max_messages": max(20, min(5000, int(analysis_cfg.get("max_messages", 500)))),
        "max_prompt_chars": max(4000, min(60000, int(analysis_cfg.get("max_prompt_chars", 24000)))),
        "max_topics": max(1, min(10, int(analysis_cfg.get("max_topics", 5)))),
        "max_unhinged_quotes": max(
            1, min(8, int(analysis_cfg.get("max_unhinged_quotes", 4)))
        ),
        "unhinged_min_score": max(
            0, min(100, int(analysis_cfg.get("unhinged_min_score", 75)))
        ),
        "max_titles": max(1, min(8, int(analysis_cfg.get("max_titles", 5)))),
        "max_tokens": max(400, min(6000, int(analysis_cfg.get("max_tokens", 2400)))),
        "retries": max(1, min(8, int(analysis_cfg.get("retries", 2)))),
    }


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="", help="会话 id，如 group_724786753")
    ap.add_argument("--date", default="", help="YYYY-MM-DD，默认今天")
    ap.add_argument("--provider", default="primary", help="provider 名（默认 primary）")
    ap.add_argument("--max-tokens", type=int, default=0, help="覆盖日报预算")
    ap.add_argument("--retries", type=int, default=0, help="覆盖重试次数")
    ap.add_argument("--dump-prompt", default="", help="把最后一次真实 prompt 写到这个文件")
    ap.add_argument("--json", dest="json_out", default="", help="把原始返回写到这个文件")
    args = ap.parse_args()

    cfg = load_config_file(REPO / "config" / "config.yaml")
    rc = _report_config(cfg)
    if args.max_tokens > 0:
        rc["max_tokens"] = args.max_tokens
    if args.retries > 0:
        rc["retries"] = args.retries

    provider, raw_cfg = _build_provider(cfg, args.provider)
    print("=" * 72)
    print(f"provider      : {args.provider} | type={raw_cfg.get('provider_type')} "
          f"| model={raw_cfg.get('model')} | url={raw_cfg.get('base_url')}")
    print(f"web_search    : {raw_cfg.get('web_search')} | reasoning_effort={raw_cfg.get('reasoning_effort')}")
    print(f"日报预算      : max_tokens={rc['max_tokens']} retries={rc['retries']} "
          f"max_prompt_chars={rc['max_prompt_chars']}")
    print(f"逆天语录      : 上限={rc['max_unhinged_quotes']} 条，门槛分={rc['unhinged_min_score']}")

    memory_cfg = cfg.get("memory") or {}
    storage = AsyncMemoryStorage(
        MemoryStorage(
            memory_cfg.get("db_path", "data/memory.db"),
            share_across_sessions=bool(memory_cfg.get("share_across_sessions", False)),
        )
    )

    target = args.date or datetime.now().strftime("%Y-%m-%d")
    day = datetime.strptime(target, "%Y-%m-%d")
    since = day.replace(hour=0, minute=0, second=0, microsecond=0)
    until = since + timedelta(days=1)

    if not args.session:
        sessions = await storage.get_group_analysis_sessions()
        print("可用会话      :", [s.get("session") or s.get("source_session") for s in sessions][:10])
        return

    messages = await storage.get_group_analysis_messages(
        args.session, since=since, until=until, limit=rc["max_messages"]
    )
    humans = GroupDailyAnalysis.human_messages(messages)
    print(f"素材          : {len(messages)} 条原始 / {len(humans)} 条群友发言 （{target}）")
    if not humans:
        print("没有素材，换个 --date 或 --session")
        return
    print("=" * 72)

    recorder = _Recorder(provider)
    report = await GroupDailyAnalysis.analyze(
        messages,
        provider=recorder,
        max_chars=rc["max_prompt_chars"],
        max_topics=rc["max_topics"],
        max_unhinged_quotes=rc["max_unhinged_quotes"],
        unhinged_min_score=rc["unhinged_min_score"],
        max_titles=rc["max_titles"],
        max_tokens=rc["max_tokens"],
        retries=rc["retries"],
        bot_name="爱丽丝",
        bot_persona="",
    )

    for i, call in enumerate(recorder.calls, 1):
        print(f"\n----- 端点返回 #{i} -----")
        print(f"finish_reason={call['finish_reason']} | model={call['model']} | "
              f"字符数={call['content_len']} | usage={call['usage']}")
        tail = call["content"][-260:]
        print(f"尾部 260 字符：{tail!r}")

    # 真实 prompt 长度（只取一次，用于和预算对比）
    if args.dump_prompt:
        prompt = GroupDailyAnalysis.build_prompt(
            GroupDailyAnalysis._source_messages(messages),
            GroupDailyAnalysis.build_statistics(humans),
            max_chars=rc["max_prompt_chars"],
            max_topics=rc["max_topics"],
            max_unhinged_quotes=rc["max_unhinged_quotes"],
            unhinged_min_score=rc["unhinged_min_score"],
            max_titles=rc["max_titles"],
            bot_name="爱丽丝",
            bot_persona="",
        )
        Path(args.dump_prompt).write_text(prompt, encoding="utf-8")
        print(f"\nprompt 已写入 {args.dump_prompt}（{len(prompt)} 字符）")

    print("\n" + "=" * 72)
    print("analyze() 结果：")
    print(f"  analysis_error = {report.get('analysis_error')!r}")
    print(f"  title          = {report.get('title')!r}")
    for key in ("topics", "profiles", "unhinged_quotes"):
        items = report.get(key) or []
        print(f"  {key:15s}= {len(items)} 条")
        if key == "unhinged_quotes":
            for item in items:
                print(f"      · {item.get('score')} 分｜{item.get('content')!r}")
    qr = report.get("quality_review") or {}
    print(f"  quality_review = {len(qr.get('dimensions') or [])} 个维度")
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"provider": raw_cfg.get("model"), "tuning": rc,
                 "calls": recorder.calls, "report": report},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"原始返回已写入 {args.json_out}")


if __name__ == "__main__":
    asyncio.run(main_async())

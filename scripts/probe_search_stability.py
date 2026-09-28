"""诊断：同一问题、同一配置，服务端 web_search 两次给出不同答案，差在哪一环。

不做「能力有无」的判断（那个已由 verify_prod_server_search.py 覆盖），
这里只测量**同一输入下的输出方差**，并把方差拆到三个可归因的环节：

  1. 检索词漂移 —— 模型自己发出的 query（web_search_call.action.query）
  2. 来源漂移   —— 引用到的 url/title（message.annotations）
  3. 答案漂移   —— 最终 output_text 里的 <say>

三组对照，用来把「端点缺陷」和「问题本身的时效歧义」分开：
  A 时效问题（崩铁 up 角色）× N  —— 已知答案不稳定
  B 稳定事实（赤道周长）  × N     —— 若这组稳定，说明端点是确定的，方差来自问题类型
  C 时效问题 × N，去掉时间锚点    —— 验证「系统提示里的当前日期」是否就是稳定器

用法（容器内，读挂载进来的生产配置）：
    docker exec alice-chat-bot python3 /app/scripts/probe_search_stability.py
    docker exec alice-chat-bot python3 /app/scripts/probe_search_stability.py \\
        --samples 5 --question "崩铁现在up角色是谁"
    # 对照推理档位（low / minimal / medium / high / adaptive）
    docker exec alice-chat-bot python3 /app/scripts/probe_search_stability.py --effort adaptive
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/app")

import aiohttp  # noqa: E402

from core.config_store import load_config  # noqa: E402

# 与生产同形的系统提示骨架：人设占位 + 时间锚点 + <say> 输出契约。
# 时间锚点对应 modules/memory/context.py:542 的 [当前时间] 行。
BASE_INSTRUCTIONS = (
    "你是群里的成员之一，用自己的口吻说话，别像客服。\n"
    "{time_anchor}\n"
    "要发给群友的话必须放进 <say>...</say>；标签以外任何内容都不会发送。"
)

SAY_RE = re.compile(r"<say>(.*?)</say>", re.S)


def time_anchor() -> str:
    now = dt.datetime.now()
    weekday = "周" + "一二三四五六日"[now.weekday()]
    return f"[当前时间] {now.year}年{now.month}月{now.day}日 {weekday} {now.strftime('%H:%M')}"


def collect_keys(node, wanted: set[str], out: list, depth: int = 0) -> None:
    """递归收集指定名字的键的字符串值（不猜结构，结构变了也不至于漏抓）。

    注意去重但仍保持出现顺序：同一 url 在多个片段里重复引用很常见。
    """
    if depth > 12:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if k in wanted and isinstance(v, str) and v.strip() and v not in out:
                out.append(v)
            collect_keys(v, wanted, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            collect_keys(v, wanted, out, depth + 1)


async def one_shot(session, url, key, model, question, *, effort, use_date, max_out=2000):
    anchor = time_anchor() if use_date else "(未提供当前时间)"
    body = {
        "model": model,
        "instructions": BASE_INSTRUCTIONS.format(time_anchor=anchor),
        "input": [{"role": "user", "content": question}],
        "max_output_tokens": max_out,
        "tools": [{"type": "web_search"}],
    }
    if effort and effort != "adaptive":
        body["reasoning"] = {"effort": effort}

    async with session.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json=body,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        raw = await resp.text()
        if resp.status != 200:
            return {"ok": False, "status": resp.status, "error": raw[:400]}
        payload = json.loads(raw)

    items = [i for i in (payload.get("output") or []) if isinstance(i, dict)]
    types = [i.get("type") for i in items]
    usage = payload.get("usage") or {}

    queries: list[str] = []
    urls: list[str] = []
    titles: list[str] = []
    collect_keys(items, {"query"}, queries)
    collect_keys(items, {"url"}, urls)
    collect_keys(items, {"title", "name"}, titles)

    text = payload.get("output_text") or ""
    says = [s.strip() for s in SAY_RE.findall(text) if s.strip()]

    return {
        "ok": True,
        "status": payload.get("status"),
        "types": types,
        "searches": sum(1 for t in types if t == "web_search_call"),
        "queries": queries,
        "urls": urls,
        "titles": titles,
        "says": says,
        "text": text.strip(),
        "in": usage.get("input_tokens", 0),
        "out": usage.get("output_tokens", 0),
    }


def domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else url[:40]


def build_provider():
    """按 main._init_llm 对 primary 的做法构造 provider，保证与生产同路径。"""
    from main import _normalize_llm_provider_config
    from modules.llm.openai_provider import create_provider

    cfg = load_config("/app/config/config.yaml")
    primary = dict((cfg.get("llm") or {}).get("primary") or {})
    pc = _normalize_llm_provider_config(primary)
    return pc, create_provider(
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
            "reasoning_effort": pc.get("reasoning_effort", "low"),
            "search_temperature": pc.get("search_temperature"),
        },
    )


def is_judge(request) -> bool:
    return any("检索判断器" in str(getattr(m, "content", "")) for m in request.messages)


def prod_context_prompt(question: str) -> str:
    """生产形态的上下文提示：带 [当前时间] 锚点（modules/memory/context.py:542）。"""
    now = dt.datetime.now()
    weekday = "周" + "一二三四五六日"[now.weekday()]
    return (
        f"[当前时间] {now.year}年{now.month}月{now.day}日 {weekday} {now.strftime('%H:%M')}\n"
        f"[刚刚] 小明(对你说)：{question}"
    )


def _shape(payload: dict) -> dict:
    """从原始响应里抽出可比较的要素。"""
    items = [i for i in (payload.get("output") or []) if isinstance(i, dict)]
    types = [i.get("type") for i in items]
    usage = payload.get("usage") or {}
    queries, urls, titles = [], [], []
    collect_keys(items, {"query"}, queries)
    collect_keys(items, {"url"}, urls)
    collect_keys(items, {"title", "name"}, titles)
    text = payload.get("output_text") or ""
    return {
        "ok": True,
        "status": payload.get("status"),
        "types": types,
        "searches": sum(1 for t in types if t == "web_search_call"),
        "queries": queries,
        "urls": urls,
        "titles": titles,
        "says": [s.strip() for s in SAY_RE.findall(text) if s.strip()],
        "text": text.strip(),
        "in": usage.get("input_tokens", 0),
        "out": usage.get("output_tokens", 0),
    }


async def one_via_generator(provider, question: str, *, baseline: bool) -> dict:
    """走完整 ReplyGenerator.generate 链路（含按需声明、温度压制、时效约束）。

    baseline=True 时把新增的 guardrails 换掉，等价于修复前的行为——同一个脚本、
    同一份配置下做 A/B，避免拿两次不同环境的运行互相比。
    """
    from modules.reply.generator import ReplyGenerator

    generator = ReplyGenerator(llm_provider=provider)
    seen = []
    original_chat = provider.chat

    async def spy(request):
        response = await original_chat(request)
        seen.append((request, response))
        return response

    provider.chat = spy
    if baseline:
        generator._apply_server_search_guardrails = lambda request: request.temperature
    try:
        reply = await generator.generate(
            context_prompt=prod_context_prompt(question),
            current_message=question,
            direction="to_bot",
        )
    finally:
        provider.chat = original_chat

    if isinstance(reply, (list, tuple)):
        reply = "".join(str(part) for part in reply)
    main = [(r, resp) for r, resp in seen if not is_judge(r)]
    if not main:
        return {"ok": False, "status": "no-main-request", "error": "未捕获主回复请求"}

    request, response = main[-1]
    payload = getattr(response, "raw_response", None) or {}
    shaped = _shape(payload) if payload else {"ok": True, "types": [], "searches": 0,
                                             "queries": [], "urls": [], "titles": [],
                                             "says": [], "text": "", "in": 0, "out": 0,
                                             "status": getattr(response, "finish_reason", "?")}
    declared = any(
        isinstance(t, dict) and t.get("type") == "web_search"
        for t in (getattr(request, "tools", None) or [])
    )
    shaped["declared"] = declared
    shaped["temperature"] = getattr(request, "temperature", None)
    shaped["guardrail"] = any(
        getattr(m, "role", "") == "system" and "本轮已开启联网检索" in str(getattr(m, "content", ""))
        for m in (getattr(request, "messages", None) or [])
    )
    shaped["reply"] = reply
    if reply and not shaped["says"]:
        shaped["says"] = [str(reply)]
    return shaped


def show(tag: str, idx: int, r: dict) -> None:
    print(f"\n  [{tag} #{idx}]")
    if not r.get("ok"):
        print(f"    ✗ 失败（{r.get('status')}）：{r.get('error')}")
        return
    print(f"    status={r['status']}  搜索={r['searches']}轮  "
          f"input={r['in']:,}  output={r['out']:,}")
    if "declared" in r:
        print(f"    声明web_search={r['declared']}  温度={r['temperature']}  "
              f"时效约束={'有' if r['guardrail'] else '无'}")
    print(f"    输出项     = {r['types']}")
    if r["queries"]:
        print(f"    检索词     = {r['queries']}")
    else:
        print("    检索词     = (未抓到 query 字段)")
    doms = [domain(u) for u in r["urls"]]
    print(f"    引用来源   = {len(r['urls'])} 条  {doms[:10]}")
    for t in r["titles"][:5]:
        print(f"      · {t[:70]}")
    print(f"    答案       = {r['says'] or [r['text'][:200]]}")


def verdict(tag: str, runs: list[dict]) -> dict:
    ok = [r for r in runs if r.get("ok")]
    if not ok:
        print(f"\n  【{tag}】全部失败")
        return {"n": 0}
    answers = [tuple(r["says"]) if r["says"] else (r["text"][:120],) for r in ok]
    qsets = [tuple(sorted(set(r["queries"]))) for r in ok]
    dset = [frozenset(domain(u) for u in r["urls"]) for r in ok]
    n_search = [r["searches"] for r in ok]

    uniq_ans = len(set(answers))
    uniq_q = len(set(qsets))
    print(f"\n  【{tag}】样本={len(ok)}")
    print(f"    搜索轮数      : {n_search}  (min={min(n_search)} max={max(n_search)})")
    print(f"    不同答案数    : {uniq_ans}/{len(ok)}")
    print(f"    不同检索词组数: {uniq_q}/{len(ok)}")
    declared = [r.get("declared") for r in ok if "declared" in r]
    if declared:
        temps = sorted({r.get("temperature") for r in ok if "temperature" in r})
        guards = [r.get("guardrail") for r in ok if "guardrail" in r]
        print(f"    声明 web_search: {declared}")
        print(f"    本轮温度       : {temps}")
        print(f"    时效约束       : {guards}")
    if len(ok) > 1:
        # 注意 dset 里是 frozenset，set.intersection/union 只接受 set 实例，
        # 直接展开会抛 "descriptor 'intersection' ... doesn't apply to a 'frozenset'"。
        sets = [set(d) for d in dset]
        common = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        print(f"    来源重合      : 交集 {len(common)} / 并集 {len(union)} 个域名")
        if common:
            print(f"      共同来源    : {sorted(common)[:8]}")
    return {
        "n": len(ok),
        "uniq_answers": uniq_ans,
        "uniq_queries": uniq_q,
        "search_rounds": n_search,
    }


def score_json(path: str, expects: list[str]) -> int:
    """对已存下的 run JSON 按「要点是否命中」打分。

    为什么需要它：`uniq_answers` 是整串比较，而群里说话的人称/译名本来就会
    浮动（真珠/珍珠、绯英/艾凡妮莎、千冶·刃/莫特纳克斯·布莱德），整串比较会
    把「事实已经收敛」误判成「答案还是各不相同」。要判断修复有没有用，得看
    关键事实有没有命中，而不是字符串是否逐字相同。
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    runs_by_tag = data.get("runs") or {}
    print("=" * 74)
    print(f"要点命中打分：{path}")
    print(f"  baseline={data.get('baseline')}  via_generator={data.get('via_generator')}")
    print(f"  要点={expects}")
    print("=" * 74)
    total_ok = 0
    total_n = 0
    for tag, runs in runs_by_tag.items():
        ok = [r for r in runs if r.get("ok")]
        if not ok:
            continue
        print(f"\n  【{tag}】样本={len(ok)}")
        for needle in expects:
            hits = 0
            for r in ok:
                blob = " ".join(list(r.get("says") or []) + [r.get("text") or "",
                                                             str(r.get("reply") or "")])
                if needle in blob:
                    hits += 1
            total_ok += hits
            total_n += len(ok)
            print(f"    「{needle}」命中 {hits}/{len(ok)}")
    if total_n:
        print(f"\n  → 合计命中率 {total_ok}/{total_n} = {total_ok / total_n:.0%}")
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/app/config/config.yaml")
    ap.add_argument("--base", default="https://api.minimax.cn")
    ap.add_argument("--effort", default=None,
                    help="覆盖推理档位：low/minimal/medium/high/adaptive（adaptive=不传）")
    ap.add_argument("--question", default=None, help="只跑这一个问题")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--no-date", action="store_true", help="去掉时间锚点")
    ap.add_argument("--via-generator", action="store_true",
                    help="走完整 ReplyGenerator.generate 链路（含温度压制与时效约束）")
    ap.add_argument("--baseline", action="store_true",
                    help="配合 --via-generator：跳过新增的 guardrails，等价修复前行为")
    ap.add_argument("--score-json", default=None,
                    help="只对已存下的 run JSON 打分，不发任何请求")
    ap.add_argument("--expect", default="",
                    help="逗号分隔的要点，配合 --score-json 统计命中率")
    args = ap.parse_args()

    if args.score_json:
        expects = [e.strip() for e in (args.expect or "").split(",") if e.strip()]
        return score_json(args.score_json, expects)

    cfg = load_config(args.config)
    primary = (cfg.get("llm") or {}).get("primary") or {}
    key = str(primary.get("api_key") or "")
    model = str(primary.get("model") or "MiniMax-M3.1-Flash-Preview")
    url = args.base.rstrip("/") + "/v1/responses"
    effort = args.effort if args.effort is not None else primary.get("reasoning_effort", "low")
    if not key:
        raise SystemExit("配置里没有 llm.primary.api_key")

    # (标签, 问题, 次数, 是否给时间锚点)
    if args.question:
        plan = [("自定义", args.question, args.samples, not args.no_date)]
    elif args.via_generator:
        plan = [("时效问题", "崩铁现在up角色是谁", args.samples, True)]
    else:
        plan = [
            ("A 时效问题", "崩铁现在up角色是谁", args.samples or 4, True),
            ("B 稳定事实", "赤道周长是多少公里", 2, True),
            ("C 时效问题-无时间锚点", "崩铁现在up角色是谁", 2, False),
        ]

    print("=" * 74)
    print("服务端 web_search 答案稳定性诊断")
    if args.via_generator:
        print(f"  模式=完整 generate() 链路"
              f"{'（baseline：跳过 guardrails）' if args.baseline else '（已启用修复）'}")
    print(f"  端点={url}  模型={model}  推理档位={effort}")
    print(f"  时间锚点={time_anchor()}")
    print("=" * 74)

    generator_provider = None
    all_runs: dict[str, list[dict]] = {}
    async with aiohttp.ClientSession() as session:
        if args.via_generator:
            generator_provider = build_provider()[1]
        for tag, question, n, use_date in plan:
            print(f"\n{'=' * 74}\n{tag}：「{question}」 × {n}"
                  f"  时间锚点={'有' if use_date else '无'}\n{'=' * 74}")
            runs = []
            for i in range(1, n + 1):
                try:
                    if args.via_generator:
                        r = await one_via_generator(generator_provider, question,
                                                    baseline=args.baseline)
                    else:
                        r = await one_shot(session, url, key, model, question,
                                           effort=effort, use_date=use_date)
                except Exception as exc:  # noqa: BLE001 单个样本失败不该毁掉整轮对照
                    r = {"ok": False, "status": "exc", "error": repr(exc)[:300]}
                try:
                    show(tag, i, r)
                except Exception as exc:  # noqa: BLE001 打印失败也要留下记录
                    print(f"\n  [{tag} #{i}] 打印失败：{exc!r}")
                runs.append(r)
            all_runs[tag] = runs
        if generator_provider is not None:
            await generator_provider.close()

    print(f"\n{'=' * 74}\n汇总\n{'=' * 74}")
    summary = {tag: verdict(tag, runs) for tag, runs in all_runs.items()}

    a = summary.get("A 时效问题") or {}
    b = summary.get("B 稳定事实") or {}
    c = summary.get("C 时效问题-无时间锚点") or {}
    print(f"\n{'=' * 74}\n归因\n{'=' * 74}")
    if a.get("n") and b.get("n"):
        if a["uniq_answers"] > 1 and b["uniq_answers"] == 1:
            print("  → 稳定事实组答案唯一，说明端点本身是确定的；")
            print("    方差只出现在时效问题上 —— 属「问题类型」而非「端点缺陷」。")
        elif a["uniq_answers"] > 1 and b["uniq_answers"] > 1:
            print("  → 连稳定事实都漂，指向端点/采样层面的普遍不确定性。")
        else:
            print("  → 本轮未复现时效问题的漂移，需加大样本量。")
    if c.get("n") and a.get("n"):
        if c["uniq_answers"] > a["uniq_answers"]:
            print("  → 去掉时间锚点后答案更散，说明「当前日期」是有效的稳定器。")
        else:
            print("  → 有无时间锚点对答案离散度影响不明显。")
    g = summary.get("时效问题") or {}
    if g.get("n"):
        label = "baseline（修复前行为）" if args.baseline else "已启用修复"
        print(f"  → 本轮为 {label}：{g['uniq_answers']}/{g['n']} 个不同答案。")
        print("     修复前基线是 4/4 不同答案（见 logs/search_stability_run1.log）。")

    out = (f"/app/logs/search_stability_ab_{'baseline' if args.baseline else 'fixed'}.json"
           if args.via_generator else "/app/logs/search_stability.json")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"effort": effort, "via_generator": args.via_generator,
                       "baseline": args.baseline, "summary": summary, "runs": all_runs},
                      fh, ensure_ascii=False, indent=2)
        print(f"\n  原始记录已写入 {out}")
    except OSError as exc:
        print(f"\n  原始记录写入失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

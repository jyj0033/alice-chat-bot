#!/usr/bin/env python3
"""把某个群里的成员，对齐到他们在别的群（含 NapCat 时代）的身份。

官方通道的群标识是 ``group_openid``，一个人在不同群里 openid 相同、但
画像和记忆是按群分桶的。所以把人拉进新群之后，会出现「人还是那个人，
画像却是空白的」。这个脚本做三件事：

1. **认人**：用群昵称去比对别处的画像和 ``person_aliases`` 别名表，
   找出 QQ 号候选；
2. **绑定**：确认后把 QQ 号的历史记忆改写到 openid（复用 identity.bind_one）；
3. **移植画像**：把这个人在别处的画像复制一份到目标群作用域，原来的
   画像保留不动（只复制，不搬走）。

默认 dry-run，只打印将要发生什么。``--apply`` 才写库，并且先整库备份。

用法::

    # 看看能认出谁
    python3 scripts/match_group_people.py --group 0EED38B0F55E000F60E601F536A71116

    # 认人 + 绑定 + 复制画像
    python3 scripts/match_group_people.py --group 0EED... --apply

    # 顺便把每个人最近 30 条历史也带过去
    python3 scripts/match_group_people.py --group 0EED... --apply --copy-history 30
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.identity import (  # noqa: E402
    OPENID_RE,
    QQ_RE,
    backup_db,
    bind_one,
    load_mapping,
    norm_name,
)

DEFAULT_DB = "/app/data/memory.db"
DEFAULT_MAP = "/app/data/person_id_map.json"


def _meta(row) -> dict:
    try:
        data = json.loads(row["metadata"] or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def group_people(conn: sqlite3.Connection, session: str) -> list[dict]:
    """目标群里发过言的人：sender_id + 见过的昵称 + 消息数。"""
    people: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT metadata FROM memories WHERE source_session=? AND json_valid(metadata)",
        (session,),
    ):
        m = _meta(r)
        sid = str(m.get("sender_id") or "")
        if not sid or m.get("is_bot"):
            continue
        item = people.setdefault(sid, {"sender_id": sid, "names": set(), "count": 0})
        name = norm_name(m.get("sender_name"))
        if name:
            item["names"].add(name)
        item["count"] += 1
    return sorted(people.values(), key=lambda x: -x["count"])


def name_index(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """昵称 -> 已知 sender_id 集合（来自所有画像和别名表）。"""
    index: dict[str, set[str]] = {}
    for r in conn.execute(
        "SELECT metadata FROM memories WHERE json_valid(metadata) "
        "AND json_extract(metadata,'$.profile')=1"
    ):
        m = _meta(r)
        sid = str(m.get("sender_id") or "")
        name = norm_name(m.get("sender_name"))
        if sid and name:
            index.setdefault(name, set()).add(sid)
    for r in conn.execute("SELECT alias, qq_id FROM person_aliases WHERE enabled=1"):
        name = norm_name(r["alias"])
        sid = str(r["qq_id"] or "")
        if name and sid:
            index.setdefault(name, set()).add(sid)
    return index


def profiles_of(conn: sqlite3.Connection, sender_id: str, exclude: str) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT * FROM memories WHERE json_valid(metadata) "
        "AND json_extract(metadata,'$.profile')=1 "
        "AND json_extract(metadata,'$.sender_id')=? "
        "AND source_session<>? ORDER BY last_accessed DESC",
        (sender_id, exclude),
    ).fetchall()
    return rows


def history_of(
    conn: sqlite3.Connection, sender_id: str, exclude: str, limit: int
) -> list[sqlite3.Row]:
    if limit <= 0:
        return []
    return conn.execute(
        "SELECT * FROM memories WHERE json_valid(metadata) "
        "AND json_extract(metadata,'$.sender_id')=? "
        "AND source_session<>? AND memory_type IN ('episodic','semantic') "
        "AND json_extract(metadata,'$.profile') IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (sender_id, exclude, limit),
    ).fetchall()


def copy_row(conn: sqlite3.Connection, row, session: str, kind: str, origin: str) -> None:
    meta = _meta(row)
    meta["imported_from"] = origin
    meta["imported_kind"] = kind
    meta["imported_at"] = datetime.now().isoformat(timespec="seconds")
    if kind == "profile":
        meta["profile_session"] = session
    conn.execute(
        "INSERT INTO memories (content, memory_type, importance, created_at, "
        "last_accessed, tags, source_session, metadata, embedding) "
        "VALUES (?,?,?,COALESCE(?,CURRENT_TIMESTAMP),?,?,?,?,?)",
        (
            row["content"],
            row["memory_type"],
            row["importance"],
            row["created_at"],
            datetime.now().isoformat(timespec="seconds"),
            row["tags"],
            session,
            json.dumps(meta, ensure_ascii=False),
            row["embedding"],
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="目标群的 group_openid（不带 group_ 前缀）")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--apply", action="store_true", help="真正写库（默认 dry-run）")
    ap.add_argument("--copy-history", type=int, default=0, metavar="N",
                    help="额外复制每人最近 N 条历史（默认 0，不搬原始对话）")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"[!] 数据库不存在：{args.db}")
        return 1

    session = f"group_{args.group}"
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    people = group_people(conn, session)
    if not people:
        print(f"[!] {session} 里还没有任何人的发言记录。")
        return 1

    mapping = load_mapping(args.map)
    reverse = {v: k for k, v in mapping.items()}
    index = name_index(conn)

    print(f"库：{args.db}")
    print(f"群：{session}    发言人数：{len(people)}")
    print(f"模式：{'APPLY' if args.apply else 'DRY-RUN'}"
          f"{'   复制历史 ' + str(args.copy_history) + ' 条/人' if args.copy_history else ''}")
    print()

    plan: list[dict] = []
    for person in people:
        sid = person["sender_id"]
        names = sorted(person["names"])
        kind = "openid" if OPENID_RE.match(sid) else ("QQ号" if QQ_RE.match(sid) else "其它")

        qq_cands: set[str] = set()
        for name in names:
            for other in index.get(name, set()):
                if other == sid:
                    continue
                # 只要 QQ 号，openid 说明已经是对齐过的人
                if QQ_RE.match(other):
                    qq_cands.add(other)

        bound_qq = reverse.get(sid, "") if kind == "openid" else ""
        plan.append(
            {
                "sender_id": sid,
                "names": names,
                "count": person["count"],
                "kind": kind,
                "qq": bound_qq,
                "candidates": sorted(qq_cands),
            }
        )

    # ---------- 打印计划 ----------
    for p in plan:
        head = f"  {p['names'][0][:16] if p['names'] else '(无名)':<18} {p['kind']:<6} {p['sender_id'][:12]}… 消息{p['count']}"
        if p["qq"]:
            n_prof = len(profiles_of(conn, p["sender_id"], session))
            print(f"{head}  已绑定 QQ {p['qq']}，别处有 {n_prof} 份画像")
        elif p["kind"] == "QQ号":
            nb = [c for c in p["candidates"]]
            print(f"{head}  还是 QQ 号，缺 openid（等他自己说话）")
        elif len(p["candidates"]) == 1:
            print(f"{head}  → 唯一候选 QQ {p['candidates'][0]}")
        elif len(p["candidates"]) > 1:
            print(f"{head}  → 多候选 {p['candidates']}（跳过，需人工指定）")
        else:
            print(f"{head}  → 未找到对应 QQ 号")

    if not args.apply:
        print()
        print("DRY-RUN 结束，未改动任何数据。确认无误后加 --apply。")
        return 0

    if not args.no_backup:
        print(f"\n备份到 {backup_db(args.db)} …")

    bound = 0
    copied_profile = 0
    copied_history = 0
    for p in plan:
        target_sid = p["sender_id"]

        # 1) 认人 + 绑定
        if not p["qq"] and len(p["candidates"]) == 1:
            try:
                result = bind_one(conn, args.map, p["candidates"][0], p["sender_id"])
                bound += 1
                print(f"  ✓ 绑定 QQ {result['qq']} → {p['sender_id'][:12]}…"
                      f"（改写 {result['updated_memories']} 条记忆）")
            except ValueError as exc:
                print(f"  ! 绑定失败 {p['candidates'][0]}：{exc}")
                continue
        elif p["qq"]:
            target_sid = p["sender_id"]

        # 2) 移植画像（只复制，不动原件）
        existing = conn.execute(
            "SELECT 1 FROM memories WHERE source_session=? AND json_valid(metadata) "
            "AND json_extract(metadata,'$.profile')=1 "
            "AND json_extract(metadata,'$.sender_id')=? LIMIT 1",
            (session, target_sid),
        ).fetchone()
        if not existing:
            rows = profiles_of(conn, target_sid, session)
            if rows:
                copy_row(conn, rows[0], session, "profile", rows[0]["source_session"])
                copied_profile += 1
                print(f"  ✓ 画像移植 {p['names'][0][:12] if p['names'] else target_sid[:8]}…"
                      f" ← {rows[0]['source_session'][:24]}")

        # 3) 可选：带一部分历史
        if args.copy_history > 0:
            hist = history_of(conn, target_sid, session, args.copy_history)
            for row in hist:
                copy_row(conn, row, session, "history", row["source_session"])
            copied_history += len(hist)

    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(
        f"\n完成：绑定 {bound} 人，移植画像 {copied_profile} 份"
        f"{'，复制历史 ' + str(copied_history) + ' 条' if copied_history else ''}。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

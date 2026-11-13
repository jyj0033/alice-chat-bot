"""NapCat(QQ号) 与官方通道(openid) 的身份对齐。

官方通道里 openid 是单向的，平台不提供反查 QQ 号的手段，所以「同一个人」
只能靠昵称 + 人工确认来建立。本模块提供：

* ``pending_candidates``  找出还没对上号的老画像，并给出昵称候选；
* ``bind_identity``       把一个人的历史记忆从 QQ 号改写成 openid；
* ``merge_duplicate_profiles`` 同一人同群出现多条画像时保留最全的一条。

迁移只改 ID，不改记忆内容，可重复执行（幂等）。
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

QQ_RE = re.compile(r"^\d{5,12}$")
OPENID_RE = re.compile(r"^[0-9A-Fa-f]{32}$")


def norm_name(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "").strip()


def load_mapping(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}
    raw = data.get("mapping", data) if isinstance(data, dict) else {}
    out: dict[str, str] = {}
    for qq, oid in raw.items():
        if QQ_RE.match(str(qq)) and OPENID_RE.match(str(oid)):
            out[str(qq)] = str(oid).upper()
    return out


def save_mapping(path: str | Path, mapping: dict[str, str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "NapCat 时代的 QQ 号 -> 官方 openid 映射。新来的人在这里补一行，下次迁移脚本会带上。",
        "mapping": dict(sorted(mapping.items())),
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def profile_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, metadata, content, last_accessed, source_session FROM memories "
        "WHERE json_valid(metadata) AND json_extract(metadata,'$.profile')=1 "
        "ORDER BY last_accessed DESC"
    ).fetchall()


def _load_meta(raw: str | None) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def openid_activity(conn: sqlite3.Connection) -> dict[str, dict]:
    """官方通道里每个 openid 的昵称集合与消息数。"""
    seen: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT metadata FROM memories WHERE json_valid(metadata) "
        "AND source_session LIKE 'group_%' AND json_extract(metadata,'$.sender_id') IS NOT NULL"
    ):
        m = _load_meta(r["metadata"])
        sid = str(m.get("sender_id") or "")
        if not OPENID_RE.match(sid):
            continue
        if m.get("is_bot"):
            continue
        entry = seen.setdefault(sid, {"names": set(), "count": 0, "last": ""})
        name = norm_name(m.get("sender_name"))
        if name:
            entry["names"].add(name)
        entry["count"] += 1
        ts = str(m.get("timestamp") or "")
        if ts and ts > entry["last"]:
            entry["last"] = ts
    return seen


def pending_candidates(conn: sqlite3.Connection, mapping: dict[str, str]) -> list[dict]:
    """还没迁移的老画像 + 昵称能对上的 openid 候选。"""
    activity = openid_activity(conn)
    name_index: dict[str, list[str]] = defaultdict(list)
    for oid, info in activity.items():
        for name in info["names"]:
            name_index[name].append(oid)

    bound_qq = set(mapping)
    bound_openid = set(mapping.values())

    # 同一个 QQ 号可能有多条画像（多个群），按 QQ 聚合
    by_qq: dict[str, dict] = {}
    for r in profile_rows(conn):
        m = _load_meta(r["metadata"])
        sid = str(m.get("sender_id") or "")
        if not QQ_RE.match(sid) or sid in bound_qq:
            continue
        item = by_qq.setdefault(
            sid,
            {
                "qq": sid,
                "names": set(),
                "sessions": set(),
                "profile_ids": [],
                "last_active": "",
                "fact_count": 0,
            },
        )
        name = norm_name(m.get("sender_name"))
        if name:
            item["names"].add(name)
        item["sessions"].add(str(m.get("profile_session") or ""))
        item["profile_ids"].append(r["id"])
        item["fact_count"] += int(m.get("fact_count") or 0)
        ts = str(r["last_accessed"] or "")
        if ts > item["last_active"]:
            item["last_active"] = ts

    out = []
    for qq, item in by_qq.items():
        cands = []
        for name in item["names"]:
            for oid in name_index.get(name, []):
                if oid in bound_openid:
                    continue
                info = activity.get(oid, {})
                cands.append(
                    {
                        "openid": oid,
                        "name": next(iter(sorted(info.get("names") or {name}))),
                        "messages": info.get("count", 0),
                        "matched_name": name,
                    }
                )
        # 去重
        uniq = {c["openid"]: c for c in cands}
        cands = sorted(uniq.values(), key=lambda c: -c["messages"])
        out.append(
            {
                "qq": qq,
                "names": sorted(item["names"]),
                "sessions": sorted(s for s in item["sessions"] if s),
                "profile_ids": item["profile_ids"],
                "fact_count": item["fact_count"],
                "last_active": item["last_active"],
                "candidates": cands,
                "confidence": "high" if len(cands) == 1 else ("low" if cands else "none"),
            }
        )
    out.sort(key=lambda x: (x["confidence"] != "high", x["last_active"]), reverse=False)
    return out


def merge_duplicate_profiles(
    conn: sqlite3.Connection, *, apply: bool = True
) -> list[dict]:
    rows = conn.execute(
        "SELECT id, metadata, length(content) AS n FROM memories "
        "WHERE json_valid(metadata) AND json_extract(metadata,'$.profile')=1"
    ).fetchall()
    groups: dict[tuple[str, str], list[tuple[int, int, dict, int]]] = defaultdict(list)
    for mid, metadata, n in rows:
        m = _load_meta(metadata)
        key = (str(m.get("sender_id") or ""), str(m.get("profile_session") or ""))
        groups[key].append((int(m.get("fact_count") or 0), int(n or 0), m, mid))

    merges = []
    for (sid, sess), items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        facts: list[str] = []
        for _, _, m, _ in items:
            for fact in m.get("source_facts") or []:
                if fact not in facts:
                    facts.append(fact)
        keep = items[0]
        drop = [it[3] for it in items[1:]]
        merges.append({"sender_id": sid, "session": sess, "keep": keep[3], "drop": drop})
        if not apply:
            continue
        keep_meta = dict(keep[2])
        keep_meta["source_facts"] = facts[:80]
        keep_meta["fact_count"] = len(facts)
        conn.execute(
            "UPDATE memories SET metadata=? WHERE id=?",
            (json.dumps(keep_meta, ensure_ascii=False), keep[3]),
        )
        for did in drop:
            row = conn.execute("SELECT metadata FROM memories WHERE id=?", (did,)).fetchone()
            dm = _load_meta(row[0] if row else None)
            dm["profile"] = False
            dm["merged_into"] = keep[3]
            conn.execute(
                "UPDATE memories SET metadata=? WHERE id=?",
                (json.dumps(dm, ensure_ascii=False), did),
            )
    return merges


def apply_mapping(
    conn: sqlite3.Connection, mapping: dict[str, str]
) -> tuple[int, int]:
    """把映射表里所有人一次性改写。返回 (更新记忆行数, 合并画像组数)。"""
    updates = []
    for r in conn.execute(
        "SELECT id, metadata FROM memories WHERE json_valid(metadata)"
    ):
        m = _load_meta(r["metadata"])
        sid = str(m.get("sender_id") or "")
        if sid in mapping:
            m["sender_id"] = mapping[sid]
            updates.append((json.dumps(m, ensure_ascii=False), r["id"]))
    for metadata, mid in updates:
        conn.execute("UPDATE memories SET metadata=? WHERE id=?", (metadata, mid))
    for qq, oid in mapping.items():
        conn.execute("UPDATE person_aliases SET qq_id=? WHERE qq_id=?", (oid, qq))
        conn.execute("UPDATE profile_suppressions SET sender_id=? WHERE sender_id=?", (oid, qq))
        conn.execute("UPDATE profile_attempts SET sender_id=? WHERE sender_id=?", (oid, qq))
    merges = merge_duplicate_profiles(conn, apply=True)
    conn.commit()
    return len(updates), len(merges)


def bind_one(
    conn: sqlite3.Connection,
    map_path: str | Path,
    qq: str,
    openid: str,
) -> dict:
    """把一个人的历史记忆绑到 openid，并写回映射文件。"""
    qq = str(qq).strip()
    openid = str(openid).strip().upper()
    if not QQ_RE.match(qq):
        raise ValueError("QQ 号格式不对")
    if not OPENID_RE.match(openid):
        raise ValueError("openid 应为 32 位十六进制")

    mapping = load_mapping(map_path)
    existing = mapping.get(qq)
    if existing and existing != openid:
        raise ValueError(f"该 QQ 已绑定到 {existing}，请先解绑")
    others = [k for k, v in mapping.items() if v == openid and k != qq]
    if others:
        raise ValueError(f"该 openid 已绑定 QQ {others[0]}")

    with conn:
        updated, merges = apply_mapping(conn, {qq: openid})
    mapping[qq] = openid
    save_mapping(map_path, mapping)
    return {"qq": qq, "openid": openid, "updated_memories": updated, "merged_profiles": merges}


def backup_db(db_path: str | Path) -> str:
    db = Path(db_path)
    target = db.with_name(db.name + f".bak.identity-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(db, target)
    return str(target)

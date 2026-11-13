#!/usr/bin/env python3
"""纠正群会话映射：把历史记忆挪回它真正属于的那个官方群。

背景：NapCat 时期的数字群号迁到官方 `group_openid` 时，映射搞反了 ——
`724786753` 的历史被塞进了「盟约」(2E65…)，实际它属于「守护全世界最好
的玲」(0EED…)；而「盟约」对应的是另一个老群 `333863057`。

判断依据是「独有成员」：守护玲里发言的玄坛持念使 / 潮汐漫岸 / 宫园薰 只在
724786753 出现过，盟约里的 喵～晓森 / 贝利亚 / (ಡωಡ) 只在 333863057 出现过。

本脚本默认 dry-run，只打印将要移动的条数。``--apply`` 才写库，并先整库备份。
重复执行是安全的（第二次没有可移动的行）。

用法::

    python3 scripts/fix_group_mapping.py            # 看计划
    python3 scripts/fix_group_mapping.py --apply    # 执行
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.identity import backup_db  # noqa: E402

DEFAULT_DB = "/app/data/memory.db"

MENGYUE = "group_2E654444EB4A706936CB1C8F0B750047"   # 盟约（官方）
SHOUHU = "group_0EED38B0F55E000F60E601F536A71116"    # 守护全世界最好的玲（官方）
OLD_724 = "group_724786753"                           # → 守护玲
OLD_333 = "group_333863057"                           # → 盟约

# 盟约是 11-12 12:31 才把机器人拉进去的，这个时间点之前的记录都来自老群迁移。
MENGYUE_START = "2026-11-12 12:00:00"


def count(conn, sql, params=()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--apply", action="store_true", help="真正写库（默认 dry-run）")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"[!] 数据库不存在：{args.db}")
        return 1

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    print(f"库：{args.db}    模式：{'APPLY' if args.apply else 'DRY-RUN'}\n")

    # ---- 1. 清掉之前「按人复制」留下的副本，避免和即将迁入的历史重复 ----
    imported_mem = count(
        conn,
        "SELECT COUNT(*) FROM memories WHERE source_session=? AND json_valid(metadata) "
        "AND json_extract(metadata,'$.imported_kind')='history'",
        (SHOUHU,),
    )
    imported_prof = count(
        conn,
        "SELECT COUNT(*) FROM memories WHERE source_session=? AND json_valid(metadata) "
        "AND json_extract(metadata,'$.imported_kind')='profile'",
        (SHOUHU,),
    )
    print(f"1) 守护玲里按人复制的临时副本：历史 {imported_mem} 条 / 画像 {imported_prof} 份 → 删除")

    # ---- 2. 盟约名下、今天 12 点前的记录，其实是 724786753 的历史 → 守护玲 ----
    moved_724 = count(
        conn,
        "SELECT COUNT(*) FROM memories WHERE source_session=? AND created_at<?",
        (MENGYUE, MENGYUE_START),
    )
    prof_724 = count(
        conn,
        "SELECT COUNT(*) FROM memories WHERE source_session=? AND json_valid(metadata) "
        "AND json_extract(metadata,'$.profile_session')=?",
        (MENGYUE, OLD_724),
    )
    print(f"2) {OLD_724} → 守护玲：记忆 {moved_724} 条 / 画像 {prof_724} 份")

    # ---- 3. 333863057 还挂在老数字群号下 → 盟约 ----
    moved_333 = count(
        conn, "SELECT COUNT(*) FROM memories WHERE source_session=?", (OLD_333,)
    )
    prof_333 = count(
        conn,
        "SELECT COUNT(*) FROM memories WHERE source_session=? AND json_valid(metadata) "
        "AND json_extract(metadata,'$.profile_session')=?",
        (OLD_333, OLD_333),
    )
    print(f"3) {OLD_333} → 盟约：记忆 {moved_333} 条 / 画像 {prof_333} 份")

    # ---- 4. 每个群里同一个人只留最新一份画像 ----
    dup = count(
        conn,
        "SELECT COUNT(*) FROM (SELECT source_session, "
        "json_extract(metadata,'$.sender_id') sid, COUNT(*) c FROM memories "
        "WHERE json_valid(metadata) AND json_extract(metadata,'$.profile')=1 "
        "GROUP BY source_session, sid HAVING c > 1)",
    )
    print(f"4) 需要去重的画像组：{dup} 组")

    if not args.apply:
        print("\nDRY-RUN 结束，未改动任何数据。确认后加 --apply。")
        return 0

    if not args.no_backup:
        print(f"\n备份到 {backup_db(args.db)} …")

    now = datetime.now().isoformat(timespec="seconds")

    with conn:
        # 1) 删除临时副本
        conn.execute(
            "DELETE FROM memories WHERE source_session=? AND json_valid(metadata) "
            "AND json_extract(metadata,'$.imported_kind') IN ('history','profile')",
            (SHOUHU,),
        )

        # 2) 724786753 的历史：盟约 → 守护玲
        conn.execute(
            "UPDATE memories SET source_session=? WHERE source_session=? AND created_at<?",
            (SHOUHU, MENGYUE, MENGYUE_START),
        )
        conn.execute(
            "UPDATE memories SET metadata=json_set(metadata,'$.profile_session',?) "
            "WHERE source_session=? AND json_valid(metadata) "
            "AND json_extract(metadata,'$.profile_session')=?",
            (SHOUHU, SHOUHU, OLD_724),
        )

        # 3) 333863057 → 盟约
        conn.execute(
            "UPDATE memories SET source_session=? WHERE source_session=?",
            (MENGYUE, OLD_333),
        )
        conn.execute(
            "UPDATE memories SET metadata=json_set(metadata,'$.profile_session',?) "
            "WHERE source_session=? AND json_valid(metadata) "
            "AND json_extract(metadata,'$.profile_session')=?",
            (MENGYUE, MENGYUE, OLD_333),
        )

        # 4) 画像去重：同群同人保留 last_accessed 最新的一条
        rows = conn.execute(
            "SELECT source_session, json_extract(metadata,'$.sender_id') sid "
            "FROM memories WHERE json_valid(metadata) "
            "AND json_extract(metadata,'$.profile')=1 "
            "GROUP BY source_session, sid HAVING COUNT(*)>1"
        ).fetchall()
        dropped = 0
        for r in rows:
            ids = conn.execute(
                "SELECT id FROM memories WHERE source_session=? AND json_valid(metadata) "
                "AND json_extract(metadata,'$.profile')=1 "
                "AND json_extract(metadata,'$.sender_id')=? "
                "ORDER BY last_accessed DESC, id DESC",
                (r["source_session"], r["sid"]),
            ).fetchall()
            for extra in ids[1:]:
                conn.execute("DELETE FROM memories WHERE id=?", (extra["id"],))
                dropped += 1
        print(f"去重删除画像 {dropped} 份")

        conn.execute(
            "UPDATE memories SET last_accessed=? WHERE source_session IN (?,?) "
            "AND json_valid(metadata) AND json_extract(metadata,'$.profile')=1",
            (now, SHOUHU, MENGYUE),
        )

    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print("\n=== 完成后的分布 ===")
    for r in conn.execute(
        "SELECT source_session, COUNT(*) n FROM memories GROUP BY source_session "
        "ORDER BY n DESC"
    ):
        print(f"  {str(r['source_session'])[:46]:<46} {r['n']}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

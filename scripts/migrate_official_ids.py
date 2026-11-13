#!/usr/bin/env python3
"""把 NapCat 时代的 QQ 号记忆迁移到官方通道的 openid。

核心逻辑在 ``core/identity.py``，本脚本只是命令行入口：默认 dry-run，
``--apply`` 才写库并先做整库备份。

用法::

    python3 scripts/migrate_official_ids.py --dry-run
    python3 scripts/migrate_official_ids.py --dry-run --auto-match
    python3 scripts/migrate_official_ids.py --apply
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.identity import (  # noqa: E402
    apply_mapping,
    backup_db,
    load_mapping,
    merge_duplicate_profiles,
    pending_candidates,
)

DEFAULT_DB = "/app/data/memory.db"
DEFAULT_MAP = "/app/data/person_id_map.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--apply", action="store_true", help="真正写库（默认 dry-run）")
    ap.add_argument("--dry-run", action="store_true", help="只预演，与默认行为一致")
    ap.add_argument("--auto-match", action="store_true", help="用昵称找候选，只打印")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"[!] 数据库不存在：{args.db}")
        return 1

    mapping = load_mapping(args.map)
    print(f"库：{args.db}")
    print(f"映射：{len(mapping)} 条   模式：{'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.auto_match:
        pending = pending_candidates(conn, mapping)
        high = [p for p in pending if p["confidence"] == "high"]
        low = [p for p in pending if p["confidence"] == "low"]
        none_ = [p for p in pending if p["confidence"] == "none"]
        print("=== 昵称自动匹配候选（只读，需人工确认） ===")
        for p in high:
            c = p["candidates"][0]
            print(f"  高置信  {p['qq']:<12} {p['names'][0][:18]:<20} -> {c['openid']}  (消息{c['messages']})")
        for p in low:
            print(f"  多候选  {p['qq']:<12} {p['names'][0][:18]:<20} -> {[c['openid'][:8] + '…' for c in p['candidates']]}")
        print(f"  一对一 {len(high)} / 多候选 {len(low)} / 无候选 {len(none_)}")
        print()

    dup = merge_duplicate_profiles(conn, apply=False)
    print(f"=== 重复画像（同人同群多条）：{len(dup)} 组 ===")
    for d in dup[:15]:
        print(f"  {d['sender_id'][:16]}… @ {d['session'][:24]:<26} 保留 id={d['keep']} 合并 {d['drop']}")
    if len(dup) > 15:
        print(f"  … 其余 {len(dup) - 15} 组")
    print()

    if not args.apply:
        print("DRY-RUN 结束，未改动任何数据。确认无误后加 --apply。")
        return 0

    if not mapping:
        print("[!] 映射为空，无可迁移内容。")
        return 0

    if not args.no_backup:
        print(f"备份到 {backup_db(args.db)} …")
    updated, merges = apply_mapping(conn, mapping)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(f"完成：更新 {updated} 条记忆，合并 {merges} 组重复画像，别名/抑制记录已同步。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

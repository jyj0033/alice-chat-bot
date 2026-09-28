"""看一眼线上历史图片描述长什么样，以及新代码上线后有没有变化。

用法（容器内）：python3 /app/scripts/inspect_image_descriptions.py
"""
from __future__ import annotations

import os
import sqlite3
import sys

DB = os.environ.get("ALICE_DB", "/app/data/memory.db")
PLACEHOLDER_HINTS = ("看不清画面", "画面是[动画表情]", "[图片]", "看不清")


def main() -> int:
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    tables = [r[0] for r in db.execute("select name from sqlite_master where type='table'")]
    print("表:", tables)
    for table in tables:
        cols = [c[1] for c in db.execute(f"PRAGMA table_info({table})")]
        text_cols = [c for c in cols if any(k in c for k in ("content", "summary", "text"))]
        if not text_cols:
            continue
        where = " or ".join(f"{c} like '%一张图%'" for c in text_cols)
        try:
            rows = db.execute(f"select * from {table} where {where}").fetchall()
        except sqlite3.Error as exc:
            print(f"--- {table}: 查询失败 {exc}")
            continue
        print(f"\n--- {table} 命中 {len(rows)} 行（列：{cols}）")
        for row in rows[-12:]:
            payload = {c: row[c] for c in row.keys() if isinstance(row[c], str) and "一张图" in row[c]}
            for col, value in payload.items():
                text = value.replace("\n", " ")[:150]
                bad = any(h in value for h in PLACEHOLDER_HINTS)
                print(f"    [{'占位' if bad else '有内容'}] {col}: {text}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

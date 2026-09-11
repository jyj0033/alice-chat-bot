"""
记忆存储层 - SQLite 实现
支持三层记忆架构：工作记忆 / 情景记忆 / 语义记忆
"""
import asyncio
import json
import logging
import math
import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _db_locked(method):
    """Serialize access to the shared SQLite connection.

    The async facade runs synchronous storage methods in a thread pool, while
    the dashboard may use the same storage from another event loop.  SQLite
    connections are not safe to use concurrently just because
    ``check_same_thread`` is disabled, so every database-facing method uses a
    process-local re-entrant lock.
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._db_lock:
            return method(self, *args, **kwargs)
    return wrapped


def _strip_speaker(content: str) -> str:
    """去掉记忆内容开头的「昵称：」前缀（昵称可能随改群名片变化，比较内容即可）。"""
    if "：" in content:
        head, rest = content.split("：", 1)
        # 只有短前缀（像昵称）才剥离；消息正文本身含冒号时保留完整正文
        if len(head) <= 40 and not head.startswith(("http", "[", "【")):
            return rest.strip()
    return content.strip()


def _content_similarity(a: str, b: str) -> float:
    """基于字符 bigram 的相似度（0~1），比 difflib 对短句更稳。"""
    a, b = _strip_speaker(a), _strip_speaker(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    grams_a = {a[i:i + 2] for i in range(len(a) - 1)}
    grams_b = {b[i:i + 2] for i in range(len(b) - 1)}
    if not grams_a or not grams_b:
        return 1.0 if a == b else 0.0
    inter = len(grams_a & grams_b)
    return (2.0 * inter) / (len(grams_a) + len(grams_b))


@dataclass
class Memory:
    """记忆单元"""
    id: Optional[int] = None
    content: str = ""
    memory_type: str = "working"  # working / episodic / semantic
    importance: float = 0.5  # 0.0 - 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    source_session: str = ""  # 来源会话
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "tags": self.tags,
            "source_session": self.source_session,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        data = data.copy()
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_accessed" in data and isinstance(data["last_accessed"], str):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class MemoryStorage:
    """记忆存储"""

    def __init__(self, db_path: str = "data/memory.db", share_across_sessions: bool = False):
        self.db_path = db_path
        self.share_across_sessions = bool(share_across_sessions)
        self._db_lock = threading.RLock()
        self._ensure_dir()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        # 嵌入服务（可选，未配置时语义检索回退 TF-IDF）
        self._embedding_service = None
        # 内存向量缓存：memory_id -> list[float]（LRU，有上限，防长期运行内存膨胀）
        self._embed_cache: OrderedDict[int, list] = OrderedDict()
        self._embed_cache_max = 2048
        logger.info(f"记忆存储已初始化：{db_path}")

    def _cache_embed(self, memory_id: int, vector: list) -> None:
        """写向量缓存（LRU 淘汰最旧，避免无限增长）"""
        with self._db_lock:
            self._embed_cache[memory_id] = vector
            self._embed_cache.move_to_end(memory_id)
            while len(self._embed_cache) > self._embed_cache_max:
                self._embed_cache.popitem(last=False)

    def _get_embed(self, memory_id: int) -> Optional[list]:
        """读向量缓存（命中即刷新为最近使用）"""
        with self._db_lock:
            v = self._embed_cache.get(memory_id)
            if v is not None:
                self._embed_cache.move_to_end(memory_id)
            return v

    def set_embedding_service(self, service) -> None:
        """注入嵌入+重排服务"""
        self._embedding_service = service

    def _ensure_dir(self) -> None:
        """确保目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @_db_locked
    def _init_tables(self) -> None:
        """初始化数据库表"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'working',
                importance REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT DEFAULT '[]',
                source_session TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)

        # 迁移：旧库补 embedding 列（存 JSON 数组或 null）
        cols = [r["name"] for r in self.conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "embedding" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")

        # 创建索引
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_session ON memories(source_session)
        """)
        # 会话历史分页/按会话查类型的常见查询：会话+类型组合索引
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_type
            ON memories(source_session, memory_type, created_at DESC)
        """)

        # 群聊黑话表：群内特有的梗、内部称呼、缩写。字面意思和实际意思不同，
        # LLM 单看字面会理解错，所以单独存一份可人工校订的词表。
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                meaning TEXT NOT NULL,
                session TEXT DEFAULT '',
                example TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1,
                source TEXT DEFAULT 'auto',
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(term, session)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_glossary_session
            ON glossary(session, enabled)
        """)

        # 删除画像时保留一个"从何时起不再自动重建"的标记；新素材出现后才允许
        # 重新生成，避免用户刚删掉的旧画像在下一轮提炼中立刻复活。
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profile_suppressions (
                sender_id TEXT NOT NULL,
                source_session TEXT NOT NULL DEFAULT '',
                suppressed_at TIMESTAMP NOT NULL,
                PRIMARY KEY(sender_id, source_session)
            )
        """)
        # 记录同一批素材已经尝试过画像提炼。LLM 返回空/拒绝或调用失败时，
        # 重启后也不要因为定时任务再次对完全相同的素材重复收费；有新素材或
        # 手动 force 时由画像提炼逻辑显式解除这个抑制。
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profile_attempts (
                sender_id TEXT NOT NULL,
                source_session TEXT NOT NULL DEFAULT '',
                self_count INTEGER NOT NULL DEFAULT 0,
                daily_count INTEGER NOT NULL DEFAULT 0,
                attempted_at TIMESTAMP NOT NULL,
                PRIMARY KEY(sender_id, source_session)
            )
        """)

        self.conn.commit()

    # === 群聊黑话（glossary） ===

    @_db_locked
    def upsert_slang(
        self,
        term: str,
        meaning: str,
        session: str = "",
        example: str = "",
        source: str = "auto",
    ) -> int:
        """新增或更新一条黑话。

        人工校订过的词条（source=manual）不会被自动提取覆盖释义，
        避免每天的定时任务把人改好的解释又冲掉。
        """
        term = (term or "").strip()
        meaning = (meaning or "").strip()
        if not term or not meaning:
            return 0
        now = datetime.now().isoformat(sep=' ')
        row = self.conn.execute(
            "SELECT id, source FROM glossary WHERE term = ? AND session = ?",
            (term, session),
        ).fetchone()
        if row:
            if row["source"] == "manual" and source == "auto":
                return row["id"]
            self.conn.execute(
                "UPDATE glossary SET meaning = ?, example = COALESCE(NULLIF(?, ''), example),"
                " source = ?, updated_at = ? WHERE id = ?",
                (meaning, example, source, now, row["id"]),
            )
            self.conn.commit()
            return row["id"]
        cursor = self.conn.execute(
            "INSERT INTO glossary (term, meaning, session, example, source, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (term, meaning, session, example, source, now, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    @_db_locked
    def list_slang(self, session: str = "", enabled_only: bool = False) -> list[dict]:
        """列出黑话（session 为空时返回全部，便于 Web 管理）。"""
        clauses, params = [], []
        if session:
            clauses.append("(session = ? OR session = '')")
            params.append(session)
        if enabled_only:
            clauses.append("enabled = 1")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM glossary {where} ORDER BY hit_count DESC, updated_at DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    @_db_locked
    def update_slang(self, slang_id: int, **fields) -> bool:
        """更新单条黑话；人工编辑内容后将词条标记为 manual。"""
        allowed = {"term", "meaning", "example", "enabled"}
        sets, params = [], []
        content_edited = False
        for key, value in fields.items():
            if key in allowed and value is not None:
                sets.append(f"{key} = ?")
                params.append(int(value) if key == "enabled" else str(value).strip())
                if key != "enabled":
                    content_edited = True
        if not sets:
            return False
        # 人工编辑过释义/例句的自动词条，后续审核也必须保护起来；单纯启停不改变来源。
        if content_edited:
            sets.append("source = 'manual'")
        sets.append("updated_at = ?")
        params.append(datetime.now().isoformat(sep=' '))
        params.append(slang_id)
        cursor = self.conn.execute(
            f"UPDATE glossary SET {', '.join(sets)} WHERE id = ?", params
        )
        self.conn.commit()
        return cursor.rowcount > 0

    @_db_locked
    def delete_slang(self, slang_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM glossary WHERE id = ?", (slang_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    @_db_locked
    def delete_auto_slang(self, slang_ids: list[int]) -> int:
        """批量删除自动提取的词条，绝不触碰人工词条。"""
        ids = set()
        for value in slang_ids or []:
            try:
                ids.add(int(value))
            except (TypeError, ValueError):
                continue
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        cursor = self.conn.execute(
            f"DELETE FROM glossary WHERE source = 'auto' AND id IN ({placeholders})",
            tuple(sorted(ids)),
        )
        self.conn.commit()
        return cursor.rowcount

    @_db_locked
    def match_slang(self, text: str, session: str = "", limit: int = 12) -> list[dict]:
        """挑出文本里出现过的黑话。

        只注入命中的词条，不是把整张词表塞进提示词——词表会越来越大，
        全量注入既浪费 token 又会干扰模型。
        """
        if not text:
            return []
        matched = []
        for row in self.list_slang(session=session, enabled_only=True):
            if row["term"] and row["term"] in text:
                matched.append(row)
        # 长词优先：命中「舟舟老师」时不必再解释「舟舟」
        matched.sort(key=lambda r: len(r["term"]), reverse=True)
        return matched[:limit]

    @_db_locked
    def bump_slang_hits(self, slang_ids: list) -> int:
        """记录命中次数，Web 端据此排序，也能看出哪些词是活的。"""
        ids = [i for i in (slang_ids or []) if i]
        if not ids:
            return 0
        self.conn.executemany(
            "UPDATE glossary SET hit_count = hit_count + 1 WHERE id = ?",
            [(i,) for i in ids],
        )
        self.conn.commit()
        return len(ids)

    @_db_locked
    def store(self, memory: Memory) -> int:
        """存储记忆"""
        cursor = self.conn.execute("""
            INSERT INTO memories (content, memory_type, importance, tags, source_session, metadata, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.content,
            memory.memory_type,
            memory.importance,
            json.dumps(memory.tags),
            memory.source_session,
            json.dumps(memory.metadata),
            memory.created_at.isoformat(sep=' '),
            memory.last_accessed.isoformat(sep=' '),
        ))
        self.conn.commit()
        memory.id = cursor.lastrowid
        logger.debug(f"记忆已保存：编号={memory.id}，类型={memory.memory_type}")
        return memory.id

    @_db_locked
    def retrieve(
        self,
        query: str = "",
        memory_type: Optional[str] = None,
        session: str = "",
        limit: int = 10
    ) -> list[Memory]:
        """检索记忆"""
        conditions = []
        params = []

        if query:
            conditions.append("content LIKE ?")
            params.append(f"%{query}%")

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if session:
            conditions.append("source_session = ?")
            params.append(session)

        sql = "SELECT * FROM memories"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        memories = []
        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            if self._is_retrievable_memory(memory):
                memories.append(memory)

        return memories

    @_db_locked
    def get_recent(self, memory_type: str, limit: int = 50) -> list[Memory]:
        """获取最近的记忆"""
        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE memory_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (memory_type, limit))

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    @_db_locked
    def store_group_analysis_message(self, memory: Memory) -> int:
        """保存群分析专用消息，不参与普通长期记忆检索。"""
        memory.memory_type = "group_analysis"
        metadata = dict(memory.metadata or {})
        message_id = str(metadata.get("message_id") or "").strip()
        if message_id:
            try:
                row = self.conn.execute(
                    """
                    SELECT id FROM memories
                    WHERE memory_type = 'group_analysis'
                      AND source_session = ?
                      AND json_valid(metadata)
                      AND CAST(json_extract(metadata, '$.message_id') AS TEXT) = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (memory.source_session, message_id),
                ).fetchone()
            except sqlite3.OperationalError:
                row = None
            if row:
                memory.id = row["id"]
                return int(row["id"])

        cursor = self.conn.execute(
            """
            INSERT INTO memories (
                content, memory_type, importance, tags, source_session,
                metadata, created_at, last_accessed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.content,
                memory.memory_type,
                min(1.0, max(0.0, float(memory.importance))),
                json.dumps(memory.tags, ensure_ascii=False),
                memory.source_session,
                json.dumps(metadata, ensure_ascii=False),
                memory.created_at.isoformat(sep=" "),
                memory.last_accessed.isoformat(sep=" "),
            ),
        )
        self.conn.commit()
        memory.id = cursor.lastrowid
        return int(cursor.lastrowid)

    @_db_locked
    def get_group_analysis_messages(
        self,
        session: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[Memory]:
        """按时间正序读取群分析消息，供日报生成使用。"""
        session = str(session or "").strip()
        if not session:
            return []
        limit = max(1, min(int(limit), 5000))
        clauses = ["memory_type = 'group_analysis'", "source_session = ?"]
        params: list = [session]
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since.isoformat(sep=" "))
        if until is not None:
            clauses.append("created_at < ?")
            params.append(until.isoformat(sep=" "))
        rows = self.conn.execute(
            f"""
            SELECT * FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    @_db_locked
    def get_group_analysis_sessions(
        self, since: Optional[datetime] = None
    ) -> list[dict]:
        """列出指定时间后有群分析消息的群，会话按最近活跃排序。"""
        clauses = [
            "memory_type = 'group_analysis'",
            "source_session LIKE 'group_%'",
        ]
        params: list = []
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since.isoformat(sep=" "))
        rows = self.conn.execute(
            f"""
            SELECT source_session AS session,
                   COUNT(*) AS message_count,
                   MAX(created_at) AS last_active
            FROM memories
            WHERE {' AND '.join(clauses)}
            GROUP BY source_session
            ORDER BY last_active DESC
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    @_db_locked
    def store_group_analysis_report(self, memory: Memory) -> int:
        """按会话和报告日期覆盖保存日报，避免定时任务重复刷屏。"""
        memory.memory_type = "group_report"
        metadata = dict(memory.metadata or {})
        report_date = str(metadata.get("report_date") or "").strip()
        existing = None
        if report_date:
            try:
                existing = self.conn.execute(
                    """
                    SELECT id FROM memories
                    WHERE memory_type = 'group_report'
                      AND source_session = ?
                      AND json_valid(metadata)
                      AND CAST(json_extract(metadata, '$.report_date') AS TEXT) = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (memory.source_session, report_date),
                ).fetchone()
            except sqlite3.OperationalError:
                existing = None

        values = (
            memory.content,
            memory.importance,
            json.dumps(memory.tags, ensure_ascii=False),
            json.dumps(metadata, ensure_ascii=False),
            memory.created_at.isoformat(sep=" "),
            memory.last_accessed.isoformat(sep=" "),
        )
        if existing:
            self.conn.execute(
                """
                UPDATE memories
                SET content = ?, importance = ?, tags = ?, metadata = ?,
                    created_at = ?, last_accessed = ?
                WHERE id = ?
                """,
                (*values, existing["id"]),
            )
            self.conn.commit()
            memory.id = existing["id"]
            return int(existing["id"])

        cursor = self.conn.execute(
            """
            INSERT INTO memories (
                content, memory_type, importance, tags, source_session,
                metadata, created_at, last_accessed
            ) VALUES (?, 'group_report', ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.content,
                memory.importance,
                json.dumps(memory.tags, ensure_ascii=False),
                memory.source_session,
                json.dumps(metadata, ensure_ascii=False),
                memory.created_at.isoformat(sep=" "),
                memory.last_accessed.isoformat(sep=" "),
            ),
        )
        self.conn.commit()
        memory.id = cursor.lastrowid
        return int(cursor.lastrowid)

    @_db_locked
    def get_group_analysis_report(
        self, session: str, report_date: str
    ) -> Optional[Memory]:
        """读取某群某天的日报。"""
        row = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE memory_type = 'group_report'
              AND source_session = ?
              AND json_valid(metadata)
              AND CAST(json_extract(metadata, '$.report_date') AS TEXT) = ?
            ORDER BY created_at DESC, id DESC LIMIT 1
            """,
            (str(session or ""), str(report_date or "")),
        ).fetchone()
        return self._row_to_memory(row) if row else None

    @_db_locked
    def get_group_analysis_reports(
        self, session: str = "", limit: int = 30
    ) -> list[Memory]:
        """读取日报历史，最新的排在前面。"""
        limit = max(1, min(int(limit), 100))
        clauses = ["memory_type = 'group_report'"]
        params: list = []
        if session:
            clauses.append("source_session = ?")
            params.append(str(session))
        rows = self.conn.execute(
            f"""
            SELECT * FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    @_db_locked
    def delete_group_analysis_before(self, before: datetime) -> int:
        """清理过期的分析原始消息和日报。"""
        cursor = self.conn.execute(
            """
            DELETE FROM memories
            WHERE memory_type IN ('group_analysis', 'group_report')
              AND created_at < ?
            """,
            (before.isoformat(sep=" "),),
        )
        self.conn.commit()
        return int(cursor.rowcount or 0)

    @_db_locked
    def get_profile_scopes(
        self, limit: Optional[int] = None, share_across_sessions: bool = False
    ) -> list[tuple[str, str]]:
        """列出有情景素材的用户/会话范围，不用全局消息上限挤掉冷门用户。"""
        if limit is not None:
            limit = max(1, min(int(limit), 100000))
        limit_sql = " LIMIT ?" if limit is not None else ""
        if share_across_sessions:
            group_sql = "CAST(json_extract(metadata, '$.sender_id') AS TEXT)"
            select_sql = f"""
                SELECT {group_sql} AS sender_id, '' AS source_session
                FROM memories
                WHERE memory_type = 'episodic'
                  AND json_valid(metadata)
                  AND json_extract(metadata, '$.sender_id') IS NOT NULL
                  AND COALESCE(json_extract(metadata, '$.is_bot'), 0) = 0
                  AND COALESCE(json_extract(metadata, '$.profile_context_only'), 0) = 0
                GROUP BY {group_sql}
                ORDER BY MAX(created_at) DESC
                {limit_sql}
            """
        else:
            sender_sql = "CAST(json_extract(metadata, '$.sender_id') AS TEXT)"
            select_sql = f"""
                SELECT {sender_sql} AS sender_id, source_session
                FROM memories
                WHERE memory_type = 'episodic'
                  AND json_valid(metadata)
                  AND json_extract(metadata, '$.sender_id') IS NOT NULL
                  AND COALESCE(json_extract(metadata, '$.is_bot'), 0) = 0
                  AND COALESCE(json_extract(metadata, '$.profile_context_only'), 0) = 0
                GROUP BY {sender_sql}, source_session
                ORDER BY MAX(created_at) DESC
                {limit_sql}
            """
        try:
            params = (limit,) if limit is not None else ()
            rows = self.conn.execute(select_sql, params).fetchall()
            return [
                (str(row["sender_id"] or ""), row["source_session"] or "")
                for row in rows
                if row["sender_id"]
            ]
        except sqlite3.OperationalError:
            # 极旧 SQLite 没有 JSON1 时保留兼容路径；正常环境走上面的精确查询。
            rows = self.conn.execute("""
                SELECT source_session, metadata, MAX(created_at) AS latest
                FROM memories
                WHERE memory_type = 'episodic'
                GROUP BY source_session, metadata
                ORDER BY latest DESC
            """).fetchall()
            scopes = []
            seen = set()
            for row in rows:
                try:
                    metadata = json.loads(row["metadata"] or "{}") or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    if metadata.get("is_bot") or metadata.get("profile_context_only"):
                        continue
                    sender_id = str(metadata.get("sender_id") or "")
                except (TypeError, ValueError):
                    sender_id = ""
                scope = "" if share_across_sessions else (row["source_session"] or "")
                key = (sender_id, scope)
                if sender_id and key not in seen:
                    seen.add(key)
                    scopes.append(key)
                    if limit is not None and len(scopes) >= limit:
                        break
            return scopes

    @_db_locked
    def get_profile_materials(
        self,
        sender_id: str,
        source_session: str = "",
        share_across_sessions: bool = False,
        limit: int = 2000,
        after: Optional[datetime] = None,
    ) -> list[Memory]:
        """按用户（及默认会话范围）取画像素材，避免全局最近消息截断用户。"""
        sender_id = str(sender_id or "").strip()
        if not sender_id:
            return []
        limit = max(1, min(int(limit), 5000))
        clauses = [
            "memory_type = 'episodic'",
            "json_valid(metadata)",
            "CAST(json_extract(metadata, '$.sender_id') AS TEXT) = ?",
            "COALESCE(json_extract(metadata, '$.is_bot'), 0) = 0",
            "COALESCE(json_extract(metadata, '$.profile_context_only'), 0) = 0",
        ]
        params: list = [sender_id]
        if not share_across_sessions:
            clauses.append("source_session = ?")
            params.append(source_session or "")
        if after is not None:
            clauses.append("created_at > ?")
            params.append(after.isoformat(sep=" "))
        where = " AND ".join(clauses)
        sql = f"""
            SELECT * FROM memories
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        try:
            rows = self.conn.execute(sql, (*params, limit)).fetchall()
        except sqlite3.OperationalError:
            # JSON1 不可用时退化为按会话取候选，再用 Python 检查 sender_id。
            fallback_clauses = ["memory_type = 'episodic'"]
            fallback_params: list = []
            if not share_across_sessions:
                fallback_clauses.append("source_session = ?")
                fallback_params.append(source_session or "")
            if after is not None:
                fallback_clauses.append("created_at > ?")
                fallback_params.append(after.isoformat(sep=" "))
            fallback_where = " AND ".join(fallback_clauses)
            rows = self.conn.execute(f"""
                SELECT * FROM memories
                WHERE {fallback_where}
                ORDER BY created_at DESC, id DESC
            """, fallback_params).fetchall()
            return [
                memory for memory in (self._row_to_memory(row) for row in rows)
                if str((memory.metadata or {}).get("sender_id") or "") == sender_id
                and not (memory.metadata or {}).get("is_bot")
                and not (memory.metadata or {}).get("profile_context_only")
            ][:limit]
        return [self._row_to_memory(row) for row in rows]

    @_db_locked
    def get_profile_context(
        self,
        memory_id: int,
        before: int = 2,
        after: int = 2,
        since: Optional[datetime] = None,
    ) -> list[Memory]:
        """读取一条画像候选在同一会话的前后文，用于消歧，不改变素材归属。"""
        try:
            memory_id = int(memory_id)
        except (TypeError, ValueError):
            return []
        target_row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ? AND memory_type = 'episodic'",
            (memory_id,),
        ).fetchone()
        if not target_row:
            return []

        try:
            before = max(0, min(int(before), 5))
        except (TypeError, ValueError):
            before = 2
        try:
            after = max(0, min(int(after), 5))
        except (TypeError, ValueError):
            after = 2
        session = target_row["source_session"] or ""
        created_at = target_row["created_at"] or ""
        rows = []
        if before:
            before_sql = """
                SELECT * FROM memories
                WHERE memory_type = 'episodic'
                  AND source_session = ?
                  AND (created_at < ? OR (created_at = ? AND id < ?))
            """
            before_params = [session, created_at, created_at, memory_id]
            if since is not None:
                before_sql += " AND created_at > ?"
                before_params.append(since.isoformat(sep=" "))
            before_sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
            before_params.append(before)
            rows.extend(self.conn.execute(before_sql, before_params).fetchall())
        rows.reverse()
        rows.append(target_row)
        if after:
            after_sql = """
                SELECT * FROM memories
                WHERE memory_type = 'episodic'
                  AND source_session = ?
                  AND (created_at > ? OR (created_at = ? AND id > ?))
            """
            after_params = [session, created_at, created_at, memory_id]
            if since is not None:
                after_sql += " AND created_at > ?"
                after_params.append(since.isoformat(sep=" "))
            after_sql += " ORDER BY created_at ASC, id ASC LIMIT ?"
            after_params.append(after)
            rows.extend(self.conn.execute(after_sql, after_params).fetchall())

        # 如果消息保存了引用消息 ID，优先把被引用消息补进来；历史数据没有该字段
        # 时仍依靠前后邻居，不影响旧库兼容。
        try:
            target_meta = json.loads(target_row["metadata"] or "{}") or {}
        except (TypeError, ValueError):
            target_meta = {}
        if not isinstance(target_meta, dict):
            target_meta = {}
        reply_to_id = str(target_meta.get("reply_to_id") or "").strip()
        if reply_to_id:
            try:
                reply_sql = """
                    SELECT * FROM memories
                    WHERE memory_type = 'episodic'
                      AND source_session = ?
                      AND json_valid(metadata)
                      AND CAST(json_extract(metadata, '$.message_id') AS TEXT) = ?
                """
                reply_params = [session, reply_to_id]
                if since is not None:
                    reply_sql += " AND created_at > ?"
                    reply_params.append(since.isoformat(sep=" "))
                reply_sql += " ORDER BY created_at DESC, id DESC LIMIT 1"
                reply_row = self.conn.execute(reply_sql, reply_params).fetchone()
            except sqlite3.OperationalError:
                reply_row = None
                candidates = self.conn.execute("""
                    SELECT * FROM memories
                    WHERE memory_type = 'episodic' AND source_session = ?
                """, (session,)).fetchall()
                for candidate in candidates:
                    if since is not None and (candidate["created_at"] or "") <= since.isoformat(sep=" "):
                        continue
                    try:
                        candidate_meta = json.loads(candidate["metadata"] or "{}") or {}
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(candidate_meta, dict):
                        continue
                    if str(candidate_meta.get("message_id") or "") == reply_to_id:
                        reply_row = candidate
                        break
            if reply_row and not any(row["id"] == reply_row["id"] for row in rows):
                rows.append(reply_row)

        rows.sort(key=lambda row: (row["created_at"] or "", row["id"]))
        return [self._row_to_memory(row) for row in rows]

    @_db_locked
    def get_profiles(self) -> list[Memory]:
        """所有用户画像（semantic 记忆且 metadata.profile=True），按最近更新排序。

        用 Python 过滤（semantic 数量少），避免依赖 json_extract 的兼容性。
        """
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE memory_type = 'semantic'"
        ).fetchall()
        profiles = [
            m for m in (self._row_to_memory(r) for r in rows)
            if (m.metadata or {}).get("profile")
        ]
        profiles.sort(key=lambda m: m.last_accessed, reverse=True)
        return profiles

    @_db_locked
    def get_profiles_page(
        self, limit: int = 30, offset: int = 0, query: str = ""
    ) -> tuple[list[Memory], int]:
        """分页读取用户画像，供 Dashboard 使用，可按 QQ/昵称/画像内容检索。"""
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        query = str(query or "").strip()[:80]
        where = (
            "memory_type = 'semantic' AND json_valid(metadata) "
            "AND json_extract(metadata, '$.profile') = 1"
        )
        params: list = []
        if query:
            like = f"%{query}%"
            where += " AND (content LIKE ? OR metadata LIKE ?)"
            params.extend([like, like])
        try:
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE {where}", params
            ).fetchone()[0]
            rows = self.conn.execute(f"""
                SELECT * FROM memories
                WHERE {where}
                ORDER BY last_accessed DESC, id DESC
                LIMIT ? OFFSET ?
            """, (*params, limit, offset)).fetchall()
        except sqlite3.OperationalError:
            profiles = self.get_profiles()
            if query:
                needle = query.casefold()
                profiles = [
                    p for p in profiles
                    if needle in (p.content or "").casefold()
                    or needle in str((p.metadata or {}).get("sender_id") or "").casefold()
                    or needle in str((p.metadata or {}).get("sender_name") or "").casefold()
                ]
            return profiles[offset:offset + limit], len(profiles)
        return [self._row_to_memory(row) for row in rows], total

    @_db_locked
    def get_profile_suppressions(self) -> dict[tuple[str, str], datetime]:
        """读取已删除画像的重建抑制时间。"""
        rows = self.conn.execute(
            "SELECT sender_id, source_session, suppressed_at FROM profile_suppressions"
        ).fetchall()
        result = {}
        for row in rows:
            try:
                result[(str(row["sender_id"]), row["source_session"] or "")] = datetime.fromisoformat(
                    row["suppressed_at"]
                )
            except (TypeError, ValueError):
                continue
        return result

    @_db_locked
    def clear_profile_suppression(self, sender_id: str, source_session: str = "") -> None:
        self.conn.execute(
            "DELETE FROM profile_suppressions WHERE sender_id = ? AND source_session = ?",
            (str(sender_id), source_session or ""),
        )
        self.conn.commit()

    @_db_locked
    def get_profile_attempts(self) -> dict[tuple[str, str], tuple[int, int]]:
        """读取每个画像范围最近一次失败/无结果的素材计数。"""
        rows = self.conn.execute(
            "SELECT sender_id, source_session, self_count, daily_count FROM profile_attempts"
        ).fetchall()
        result = {}
        for row in rows:
            try:
                result[(str(row["sender_id"]), row["source_session"] or "")] = (
                    int(row["self_count"] or 0),
                    int(row["daily_count"] or 0),
                )
            except (TypeError, ValueError):
                continue
        return result

    @_db_locked
    def set_profile_attempt(
        self,
        sender_id: str,
        source_session: str = "",
        self_count: int = 0,
        daily_count: int = 0,
    ) -> None:
        """保存最近一次画像提炼尝试，避免无变化时重复调用 LLM。"""
        self.conn.execute(
            "INSERT OR REPLACE INTO profile_attempts "
            "(sender_id, source_session, self_count, daily_count, attempted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(sender_id),
                source_session or "",
                max(0, int(self_count)),
                max(0, int(daily_count)),
                datetime.now().isoformat(sep=" "),
            ),
        )
        self.conn.commit()

    @_db_locked
    def clear_profile_attempt(self, sender_id: str, source_session: str = "") -> None:
        self.conn.execute(
            "DELETE FROM profile_attempts WHERE sender_id = ? AND source_session = ?",
            (str(sender_id), source_session or ""),
        )
        self.conn.commit()

    @_db_locked
    def update_memory(self, memory: Memory) -> bool:
        """整条更新已有记忆的字段（用户画像重新提炼时复用原 id，不产生孤儿记录）。"""
        if not memory.id:
            return False
        self.conn.execute("""
            UPDATE memories
            SET content = ?, importance = ?, tags = ?, source_session = ?,
                metadata = ?, created_at = ?, last_accessed = ?
            WHERE id = ?
        """, (
            memory.content,
            memory.importance,
            json.dumps(memory.tags),
            memory.source_session,
            json.dumps(memory.metadata),
            memory.created_at.isoformat(sep=' '),
            memory.last_accessed.isoformat(sep=' '),
            memory.id,
        ))
        self.conn.commit()
        return True

    @_db_locked
    def get_all(self, limit: int = 1000, memory_type: Optional[str] = None) -> list[Memory]:
        """获取长期记忆（默认情景+语义；指定类型时只返回该类型），按重要性排序"""
        if memory_type:
            where, params = "memory_type = ?", (memory_type,)
        else:
            where, params = "memory_type IN ('episodic', 'semantic')", ()
        cursor = self.conn.execute(f"""
            SELECT * FROM memories
            WHERE {where}
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """, (*params, limit))
        return [
            memory for memory in (self._row_to_memory(row) for row in cursor.fetchall())
            if self._is_retrievable_memory(memory)
        ]

    @_db_locked
    def get_memories_page(
        self,
        session: str = "",
        query: str = "",
        memory_types: Optional[list[str]] = None,
        limit: int = 30,
        offset: int = 0,
        half_life_days: float = 30.0,
        decay_presets: Optional[dict] = None,
    ) -> tuple[list[Memory], int]:
        """按筛选条件分页读取长期记忆，并返回未截断的总数。

        排序在 SQLite 层完成，避免 Dashboard 为了显示一页数据把整个记忆库
        读入 Python。优先按动态有效重要性排序；极旧 SQLite 不支持数学函数时
        回退到基础重要性排序，不影响分页和筛选。
        """
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))

        selected_types = []
        for value in memory_types or []:
            value = str(value).strip()
            if value and value not in selected_types:
                selected_types.append(value)

        clauses = []
        filter_params = []
        if selected_types:
            placeholders = ", ".join("?" for _ in selected_types)
            clauses.append(f"memory_type IN ({placeholders})")
            filter_params.extend(selected_types)
        else:
            clauses.append("memory_type IN ('episodic', 'semantic', 'session_summary')")

        if session:
            clauses.append("source_session = ?")
            filter_params.append(session)

        query = (query or "").strip()
        if query:
            clauses.append("content LIKE ?")
            filter_params.append(f"%{query}%")

        where = " AND ".join(clauses)
        total = self.conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {where}",
            tuple(filter_params),
        ).fetchone()[0]

        order_types = selected_types or ["episodic", "semantic", "session_summary"]
        order_params = []
        case_parts = []
        fallback_half_life = max(0.1, float(half_life_days))
        presets = decay_presets or {}
        for memory_type in order_types:
            preset = presets.get(memory_type, {}) or {}
            if not isinstance(preset, dict):
                preset = {}
            try:
                type_half_life = max(
                    0.1, float(preset.get("half_life_days", fallback_half_life))
                )
            except (TypeError, ValueError):
                type_half_life = fallback_half_life
            case_parts.append("WHEN ? THEN ?")
            order_params.extend([memory_type, type_half_life])
        order_params.append(fallback_half_life)
        half_life_case = " ".join(case_parts)
        effective_order = (
            "importance * pow(0.5, max(0.0, julianday('now') - "
            "julianday(COALESCE(last_accessed, CURRENT_TIMESTAMP))) / "
            f"CASE memory_type {half_life_case} ELSE ? END) DESC"
        )

        select_sql = f"""
            SELECT * FROM memories
            WHERE {where}
            ORDER BY {effective_order}, created_at DESC, id DESC
            LIMIT ? OFFSET ?
        """
        try:
            cursor = self.conn.execute(
                select_sql,
                (*filter_params, *order_params, limit, offset),
            )
        except sqlite3.OperationalError as exc:
            if "no such function" not in str(exc).lower():
                raise
            cursor = self.conn.execute(f"""
                SELECT * FROM memories
                WHERE {where}
                ORDER BY importance DESC, created_at DESC, id DESC
                LIMIT ? OFFSET ?
            """, (*filter_params, limit, offset))

        return [self._row_to_memory(row) for row in cursor.fetchall()], total

    @_db_locked
    def list_sessions(self, limit: int = 50) -> list[dict]:
        """列出所有出现过消息的会话（含私聊），按最近活跃排序。

        重启后内存上下文窗口为空，Web 会话管理靠这里恢复历史会话。
        """
        cursor = self.conn.execute("""
            SELECT source_session AS session,
                   COUNT(*) AS message_count,
                   MAX(created_at) AS last_active
            FROM memories
            WHERE source_session != ''
              AND memory_type NOT IN ('group_analysis', 'group_report')
            GROUP BY source_session
            ORDER BY last_active DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    @_db_locked
    def retrieve_session_recent(
        self, session: str, limit: int = 5, memory_type: Optional[str] = None
    ) -> list[Memory]:
        """获取某个会话最近的记忆（可按类型过滤；指定类型时按时间倒序，未指定按重要性）"""
        if memory_type:
            cursor = self.conn.execute("""
                SELECT * FROM memories
                WHERE source_session = ? AND memory_type = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            """, (session, memory_type, limit))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM memories
                WHERE source_session = ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (session, limit))

        return [
            memory for memory in (self._row_to_memory(row) for row in cursor.fetchall())
            if self._is_retrievable_memory(memory)
        ]

    @_db_locked
    def count_session_messages(
        self, session: str, before: Optional[datetime] = None
    ) -> int:
        """统计某个会话的 episodic 消息数（可选：仅统计早于某个时间点的历史）。"""
        if before is not None:
            cursor = self.conn.execute("""
                SELECT COUNT(*) FROM memories
                WHERE source_session = ? AND memory_type = 'episodic'
                  AND created_at < ?
            """, (session, before.isoformat(sep=' ')))
        else:
            cursor = self.conn.execute("""
                SELECT COUNT(*) FROM memories
                WHERE source_session = ? AND memory_type = 'episodic'
            """, (session,))
        return cursor.fetchone()[0]

    @_db_locked
    def get_session_messages(
        self,
        session: str,
        limit: int = 30,
        offset: int = 0,
        before: Optional[datetime] = None,
    ) -> list[Memory]:
        """分页获取某个会话的历史消息记录（episodic 即一条消息）。

        与内存窗口以"最早一条窗口消息时间"为界拼接：窗口消息是最新日志，
        before 之后的历史是更早的旧消息，二者不重叠。
        """
        if before is not None:
            cursor = self.conn.execute("""
                SELECT * FROM memories
                WHERE source_session = ? AND memory_type = 'episodic'
                  AND created_at < ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
            """, (session, before.isoformat(sep=' '), limit, offset))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM memories
                WHERE source_session = ? AND memory_type = 'episodic'
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
            """, (session, limit, offset))

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    @_db_locked
    def find_similar(self, session: str, sender_id: str, content: str,
                     memory_type: str = "episodic", lookback: int = 300) -> Optional[Memory]:
        """写入去重：在同会话同发送者的已有记忆中找与 content 高度近似的一条。

        - 记忆 content 形如「昵称：内容」，昵称可能因改群名片而变化，
          所以只比较"内容"部分，且发送者按稳定的 sender_id 匹配。
        - 相似度 ≥ 0.85 视为重复，返回该记忆；否则返回 None。
        - 只扫最近 lookback 条（新消息大概率与近期内容重复，远了没意义）。
        """
        content = (content or "").strip()
        if not content or not sender_id:
            return None
        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE source_session = ? AND memory_type = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session, memory_type, lookback))
        best: Optional[Memory] = None
        best_ratio = 0.0
        for row in cursor.fetchall():
            mem = self._row_to_memory(row)
            meta = mem.metadata or {}
            if str(meta.get("sender_id") or "") != str(sender_id):
                continue
            ratio = _content_similarity(mem.content, content)
            if ratio > best_ratio:
                best_ratio = ratio
                best = mem
            if best_ratio >= 1.0:
                break
        return best if best_ratio >= 0.85 else None

    @_db_locked
    def update_access(self, memory_id: int) -> None:
        """更新访问时间"""
        now = datetime.now().isoformat(sep=" ")
        self.conn.execute("""
            UPDATE memories SET last_accessed = ? WHERE id = ?
        """, (now, memory_id))
        self.conn.commit()

    @_db_locked
    def update_access_many(self, memory_ids: list[int]) -> int:
        """批量刷新实际被召回记忆的访问时间，不改变重要性。"""
        ids = sorted({int(mid) for mid in (memory_ids or []) if mid})
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        now = datetime.now().isoformat(sep=" ")
        cursor = self.conn.execute(
            f"UPDATE memories SET last_accessed = ? WHERE id IN ({placeholders})",
            (now, *ids),
        )
        self.conn.commit()
        return cursor.rowcount

    @_db_locked
    def bump_memories(self, memory_ids: list[int], importance_boost: float = 0.01) -> int:
        """检索反馈强化：批量刷新被召回记忆的 last_accessed 并轻微提升 importance。

        - 时间衰减只看 last_accessed：刷新后常用记忆保持新鲜，冷门记忆自然淡出。
        - importance 封顶 0.95，避免长期强化导致记忆永不衰减。
        - 单条 UPDATE，命中数通常 ≤10，代价可忽略。
        """
        if not memory_ids:
            return 0
        ids = [i for i in memory_ids if i]
        if not ids:
            return 0
        now = datetime.now().isoformat(sep=' ')
        for mid in ids:
            self.conn.execute(
                "UPDATE memories SET last_accessed = ?, importance = MIN(0.95, importance + ?) WHERE id = ?",
                (now, importance_boost, mid),
            )
        self.conn.commit()
        return len(ids)

    @_db_locked
    def update_importance(self, memory_id: int, importance: float) -> None:
        """更新重要性"""
        self.conn.execute("""
            UPDATE memories SET importance = ? WHERE id = ?
        """, (importance, memory_id))
        self.conn.commit()

    @_db_locked
    def delete(self, memory_id: int) -> bool:
        """删除记忆；删除用户画像时记录重建抑制点。"""
        row = self.conn.execute(
            "SELECT memory_type, source_session, metadata FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row and row["memory_type"] == "semantic":
            try:
                metadata = json.loads(row["metadata"] or "{}") or {}
            except (TypeError, ValueError):
                metadata = {}
            if metadata.get("profile") and metadata.get("sender_id"):
                scope = "" if self.share_across_sessions else (row["source_session"] or "")
                self.conn.execute(
                    "INSERT OR REPLACE INTO profile_suppressions "
                    "(sender_id, source_session, suppressed_at) VALUES (?, ?, ?)",
                    (str(metadata["sender_id"]), scope, datetime.now().isoformat(sep=" ")),
                )
                self.conn.execute(
                    "DELETE FROM profile_attempts WHERE sender_id = ? AND source_session = ?",
                    (str(metadata["sender_id"]), scope),
                )
        cursor = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    @_db_locked
    def cleanup_old(self, days: int = 30) -> int:
        """清理旧记忆"""
        cursor = self.conn.execute("""
            DELETE FROM memories
            WHERE memory_type IN ('working', 'episodic')
            AND last_accessed < datetime('now', '-' || ? || ' days')
            AND importance < 0.3
        """, (days,))
        self.conn.commit()
        deleted = cursor.rowcount
        logger.info(f"已清理 {deleted} 条过期记忆")
        return deleted

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """将数据库行转换为 Memory 对象"""
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else datetime.now(),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            source_session=row["source_session"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    # ==================== 向量检索 + 时间衰减 ====================

    _JIEBA = None
    _NUMPY = None
    _token_pattern = re.compile(r'[a-zA-Z0-9_]+')

    @classmethod
    def _load_deps(cls):
        """惰性加载 jieba/numpy，未安装时返回 False，退回 LIKE 检索"""
        if cls._NUMPY is None:
            try:
                import numpy
                cls._NUMPY = numpy
            except ImportError:
                cls._NUMPY = False
        if cls._JIEBA is None:
            try:
                import jieba
                # 关闭 jieba 的调试日志
                jieba.setLogLevel(logging.WARNING)
                cls._JIEBA = jieba
            except ImportError:
                cls._JIEBA = False
        return bool(cls._JIEBA and cls._NUMPY)

    def _tokenize(self, text: str) -> list[str]:
        """中文文本分词：
        - jieba 可用时用 jieba，并补充中文 bigram 提高召回
          （jieba 对同一词在不同上下文可能切成不同结果，bigram 兜底保证匹配）
        - 无 jieba 时退化：英文/数字整词 + 中文相邻字符 bigram
        """
        text = (text or "").lower()
        if not text:
            return []
        words = []
        if self._JIEBA:
            for part in text.split():
                if not part:
                    continue
                words.extend(
                    w for w in self._JIEBA.cut(part) if w.strip() and not w.isspace()
                )
        else:
            words = self._token_pattern.findall(text)

        # 中文 bigram 补充（jieba 与退化路径都加，保证一致性）
        chinese_chars = [c for c in text if '一' <= c <= '鿿']
        words.extend(
            chinese_chars[i] + chinese_chars[i + 1]
            for i in range(len(chinese_chars) - 1)
        )

        # 去重保序
        seen = set()
        out = []
        for w in words:
            if w and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    @staticmethod
    def _decay_factor(last_accessed: datetime, now: datetime = None, half_life_days: float = 30.0) -> float:
        """时间衰减因子：半衰期为 half_life_days 天。
        30天前访问的记忆 → 0.5，60天前 → 0.25，越久越弱。
        """
        now = now or datetime.now()
        age_days = max(0.0, (now - last_accessed).total_seconds()) / 86400.0
        return 0.5 ** (age_days / max(0.1, float(half_life_days)))

    def _effective_importance(
        self,
        memory: Memory,
        now: datetime = None,
        half_life_days: float = 30.0,
        presets: Optional[dict] = None,
    ) -> float:
        """有效重要性 = 基础 importance × 时间衰减"""
        preset = (presets or {}).get(memory.memory_type, {})
        half_life_days = preset.get("half_life_days", half_life_days)
        return memory.importance * self._decay_factor(memory.last_accessed, now, half_life_days)

    # ---------- 嵌入向量持久化 ----------

    @_db_locked
    def get_embedding(self, memory_id: int) -> Optional[list]:
        """从数据库读取某条记忆的嵌入向量"""
        try:
            row = self.conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row and row["embedding"]:
                return json.loads(row["embedding"])
        except Exception:
            pass
        return None

    @_db_locked
    def update_embedding(self, memory_id: int, vector: list) -> None:
        """持久化某条记忆的嵌入向量"""
        try:
            self.conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (json.dumps(vector), memory_id),
            )
            self.conn.commit()
            self._cache_embed(memory_id, vector)
        except Exception as e:
            logger.error(f"保存记忆 {memory_id} 的向量失败：{e}")

    def clear_embed_cache(self, ids: set = None) -> None:
        """清理向量缓存（删除记忆后调用）"""
        with self._db_lock:
            if ids is None:
                self._embed_cache.clear()
            else:
                for mid in ids:
                    self._embed_cache.pop(mid, None)

    @_db_locked
    def semantic_search(
        self,
        query: str,
        session: str = "",
        limit: int = 5,
        top_k_candidates: int = 200,
        half_life_days: float = 30.0,
        similarity_weight: float = 0.85,
        decay_presets: Optional[dict] = None,
    ) -> list[Memory]:
        """语义检索：TF-IDF + 余弦相似度，结合时间衰减后的重要性排序。

        排序策略：
        - 只返回有相似度命中的记忆：score = weight×sim + (1-weight)×有效重要性
        - 完全无关的记忆不伪装成召回结果，由调用方决定是否使用会话最近记忆兜底

        无 jieba/numpy 时退回 LIKE 检索。
        """
        if not self._load_deps():
            logger.warning("jieba/numpy 未安装，退回 LIKE 检索")
            return self.retrieve(query=query, session=session, limit=limit)

        now = datetime.now()
        candidates = self._get_candidates(session, top_k_candidates)
        if not candidates:
            return []

        docs = [m.content for m in candidates]
        doc_tokens = [self._tokenize(d) for d in docs]
        query_tokens = self._tokenize(query)

        # 构建词典 + DF（文档频率）
        vocab: dict[str, int] = {}
        df: dict[str, int] = {}
        for toks in doc_tokens:
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        n_docs = len(docs)

        # 构建 TF-IDF 矩阵
        numpy = self._NUMPY
        X = numpy.zeros((n_docs, len(vocab)), dtype=numpy.float32)
        for i, toks in enumerate(doc_tokens):
            if not toks:
                continue
            tf = {t: toks.count(t) / len(toks) for t in set(toks)}
            for t, f in tf.items():
                idx = vocab.get(t)
                if idx is None:
                    continue
                idf = math.log(n_docs / (1 + df.get(t, 0))) + 1.0
                X[i, idx] = f * idf

        # 查询向量
        q = numpy.zeros(len(vocab), dtype=numpy.float32)
        for t in set(query_tokens):
            idx = vocab.get(t)
            if idx is None:
                continue
            idf = math.log(n_docs / (1 + df.get(t, 0))) + 1.0
            q[idx] = (query_tokens.count(t) / max(1, len(query_tokens))) * idf

        # 余弦相似度
        q_norm = numpy.linalg.norm(q)
        if q_norm == 0:
            # 查询词完全不在记忆词典里时，不能把高重要性但无关的内容
            # 冒充相关记忆；调用方会走“会话最近记忆”兜底。
            return []

        row_norms = numpy.linalg.norm(X, axis=1)
        denom = row_norms * q_norm
        similarities = numpy.zeros(n_docs)
        mask = denom > 1e-9
        similarities[mask] = numpy.sum(X[mask] * q, axis=1) / denom[mask]

        # 只保留命中记忆；无关内容交给上层的会话近期消息兜底。
        HIT_THRESHOLD = 0.05
        hits = []
        for i, m in enumerate(candidates):
            sim = float(similarities[i])
            eff_imp = self._effective_importance(
                m, now, half_life_days, presets=decay_presets
            )
            if sim > HIT_THRESHOLD:
                score = similarity_weight * sim + (1 - similarity_weight) * eff_imp
                hits.append((score, m))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in hits[:limit]]

    @_db_locked
    def _get_candidates(self, session: str = "", top_k: int = 200) -> list[Memory]:
        """取候选记忆：默认只取当前会话，显式共享时才补其他会话。

        默认严格按会话隔离，避免个人信息和群内话题跨边界泄漏。若显式开启
        ``share_across_sessions``，才恢复旧的跨会话共享策略。
        """
        if session and not self.share_across_sessions:
            cursor = self.conn.execute("""
                SELECT * FROM memories
                WHERE memory_type IN ('episodic', 'semantic', 'session_summary')
                  AND source_session = ?
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """, (session, top_k))
            return [
                memory for memory in (self._row_to_memory(r) for r in cursor.fetchall())
                if self._is_retrievable_memory(memory)
            ]

        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE memory_type IN ('episodic', 'semantic', 'session_summary')
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
        """, (top_k,))
        rows = cursor.fetchall()
        memories = [
            memory for memory in (self._row_to_memory(r) for r in rows)
            if self._is_retrievable_memory(memory)
        ]

        if not session:
            return memories

        # 群会话：过滤掉私聊来源记忆，避免把私聊内容带进群聊上下文。
        if session.startswith("group_"):
            memories = [
                m for m in memories
                if not (m.source_session or "").startswith("private_")
            ]

        session_mem = [m for m in memories if m.source_session == session]
        others = [m for m in memories if m.source_session != session]
        # 会话内优先，剩余名额补齐
        return session_mem + others[:max(0, top_k - len(session_mem))]

    @staticmethod
    def _is_retrievable_memory(memory: Memory) -> bool:
        """管理用上下文素材保留在数据库，但不直接喂给回复模型。"""
        meta = memory.metadata or {}
        if memory.memory_type in {"group_analysis", "group_report"}:
            return False
        if meta.get("profile_context_only"):
            return False
        return not (
            memory.memory_type == "semantic"
            and meta.get("profile")
            and meta.get("warnings")
        )

    @_db_locked
    def apply_time_decay(
        self,
        half_life_days: float = 30.0,
        min_importance: float = 0.1,
        max_age_days: float = 180.0,
        presets: Optional[dict] = None,
    ) -> dict:
        """批量检查时间衰减并删除已经失效的记忆。

        有效重要性在检索时按 ``importance × 时间因子`` 动态计算，不能把
        衰减后的值再次写回 importance；否则每次定时任务都会重复乘一次，
        造成远快于配置的指数衰减。

        - 有效重要性 < min_importance 且超过 max_age_days 的删除
        - 返回 {'decayed': n, 'deleted': n}

        支持按 memory_type 分级（presets），让不同性质的记忆寿命不同：
          presets = {type: {"half_life_days": .., "max_age_days": ..}}
        未在 presets 里的类型使用默认参数。
        """
        now = datetime.now()
        rows = self.conn.execute(
            "SELECT id, memory_type, importance, last_accessed FROM memories"
        ).fetchall()

        decayed = 0
        deleted = 0
        deletes: list[int] = []
        for row in rows:
            preset = (presets or {}).get(row["memory_type"], {})
            hl = float(preset.get("half_life_days", half_life_days))
            ma = float(preset.get("max_age_days", max_age_days))
            mid = row["id"]
            importance = row["importance"]
            last_acc = None
            if row["last_accessed"]:
                try:
                    last_acc = datetime.fromisoformat(row["last_accessed"])
                except ValueError:
                    pass
            last_acc = last_acc or now

            age_days = max(0.0, (now - last_acc).total_seconds()) / 86400.0
            new_importance = importance * (0.5 ** (age_days / max(0.1, hl)))

            if new_importance < min_importance and age_days > ma:
                deletes.append(mid)
                deleted += 1
                self._embed_cache.pop(mid, None)
            elif new_importance < importance - 1e-6:
                decayed += 1

        if deletes:
            self.conn.executemany(
                "DELETE FROM memories WHERE id = ?", [(mid,) for mid in deletes]
            )

        self.conn.commit()
        if deleted:
            logger.info(f"记忆衰减检查完成：{decayed} 条过期，删除 {deleted} 条")
        return {"decayed": decayed, "deleted": deleted}

    @_db_locked
    def close(self) -> None:
        """关闭连接"""
        self.conn.close()


class AsyncMemoryStorage:
    """异步记忆存储封装"""

    # 向量召回候选数（重排的输入规模）
    VECTOR_RECALL_K = 30

    def __init__(self, storage: MemoryStorage):
        self._storage = storage
        self._embed_lock = asyncio.Lock()

    async def store(self, memory: Memory) -> int:
        """异步存储，并在服务可用时生成嵌入向量"""
        mid = await asyncio.to_thread(self._storage.store, memory)
        service = self._storage._embedding_service
        if service and service.enabled and memory.content:
            try:
                vecs = await service.embed([memory.content])
                if vecs and vecs[0]:
                    # 数据库写入也放到线程池，不能在事件循环里直接操作
                    # 与 QQ/Dashboard 共享的 SQLite 连接。
                    await asyncio.to_thread(self._storage.update_embedding, mid, vecs[0])
            except Exception as e:
                logger.debug(f"保存时生成向量失败，已跳过：{e}")
        return mid

    async def retrieve(
        self,
        query: str = "",
        memory_type: Optional[str] = None,
        session: str = "",
        limit: int = 10
    ) -> list[Memory]:
        """异步检索"""
        return await asyncio.to_thread(
            self._storage.retrieve, query, memory_type, session, limit
        )

    async def get_recent(self, memory_type: str, limit: int = 50) -> list[Memory]:
        """异步获取最近记忆"""
        return await asyncio.to_thread(self._storage.get_recent, memory_type, limit)

    async def store_group_analysis_message(self, memory: Memory) -> int:
        """异步保存群分析专用消息，不生成嵌入向量。"""
        return await asyncio.to_thread(
            self._storage.store_group_analysis_message, memory
        )

    async def get_group_analysis_messages(
        self,
        session: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[Memory]:
        """异步读取群分析消息。"""
        return await asyncio.to_thread(
            self._storage.get_group_analysis_messages,
            session,
            since,
            until,
            limit,
        )

    async def get_group_analysis_sessions(
        self, since: Optional[datetime] = None
    ) -> list[dict]:
        """异步列出有群分析消息的群。"""
        return await asyncio.to_thread(
            self._storage.get_group_analysis_sessions, since
        )

    async def store_group_analysis_report(self, memory: Memory) -> int:
        """异步保存日报。"""
        return await asyncio.to_thread(
            self._storage.store_group_analysis_report, memory
        )

    async def get_group_analysis_report(
        self, session: str, report_date: str
    ) -> Optional[Memory]:
        """异步读取某群某日的日报。"""
        return await asyncio.to_thread(
            self._storage.get_group_analysis_report, session, report_date
        )

    async def get_group_analysis_reports(
        self, session: str = "", limit: int = 30
    ) -> list[Memory]:
        """异步读取日报历史。"""
        return await asyncio.to_thread(
            self._storage.get_group_analysis_reports, session, limit
        )

    async def delete_group_analysis_before(self, before: datetime) -> int:
        """异步清理过期群分析数据。"""
        return await asyncio.to_thread(
            self._storage.delete_group_analysis_before, before
        )

    async def get_profile_scopes(
        self, limit: Optional[int] = None, share_across_sessions: bool = False
    ) -> list[tuple[str, str]]:
        """异步列出有画像素材的用户/会话范围。"""
        return await asyncio.to_thread(
            self._storage.get_profile_scopes, limit, share_across_sessions
        )

    async def get_profile_materials(
        self,
        sender_id: str,
        source_session: str = "",
        share_across_sessions: bool = False,
        limit: int = 2000,
        after: Optional[datetime] = None,
    ) -> list[Memory]:
        """异步读取单个用户/会话的画像素材。"""
        return await asyncio.to_thread(
            self._storage.get_profile_materials,
            sender_id,
            source_session,
            share_across_sessions,
            limit,
            after,
        )

    async def get_profile_context(
        self,
        memory_id: int,
        before: int = 2,
        after: int = 2,
        since: Optional[datetime] = None,
    ) -> list[Memory]:
        """异步读取画像候选的同会话前后文。"""
        return await asyncio.to_thread(
            self._storage.get_profile_context, memory_id, before, after, since
        )

    async def get_profiles(self) -> list[Memory]:
        """异步获取所有用户画像（semantic 记忆）"""
        return await asyncio.to_thread(self._storage.get_profiles)

    async def get_profiles_page(
        self, limit: int = 30, offset: int = 0, query: str = ""
    ) -> tuple[list[Memory], int]:
        """异步分页获取用户画像。"""
        return await asyncio.to_thread(
            self._storage.get_profiles_page, limit, offset, query
        )

    async def get_profile_suppressions(self) -> dict[tuple[str, str], datetime]:
        """异步读取画像删除抑制记录。"""
        return await asyncio.to_thread(self._storage.get_profile_suppressions)

    async def clear_profile_suppression(
        self, sender_id: str, source_session: str = ""
    ) -> None:
        """异步清除画像删除抑制记录。"""
        await asyncio.to_thread(
            self._storage.clear_profile_suppression, sender_id, source_session
        )

    async def get_profile_attempts(self) -> dict[tuple[str, str], tuple[int, int]]:
        """异步读取画像提炼尝试记录。"""
        return await asyncio.to_thread(self._storage.get_profile_attempts)

    async def set_profile_attempt(
        self,
        sender_id: str,
        source_session: str = "",
        self_count: int = 0,
        daily_count: int = 0,
    ) -> None:
        """异步保存画像提炼尝试记录。"""
        await asyncio.to_thread(
            self._storage.set_profile_attempt,
            sender_id,
            source_session,
            self_count,
            daily_count,
        )

    async def clear_profile_attempt(
        self, sender_id: str, source_session: str = ""
    ) -> None:
        """异步清除画像提炼尝试记录。"""
        await asyncio.to_thread(
            self._storage.clear_profile_attempt, sender_id, source_session
        )

    async def update_memory(self, memory: Memory) -> bool:
        """异步整条更新已有记忆（用户画像复用原 id）"""
        return await asyncio.to_thread(self._storage.update_memory, memory)

    async def get_all(self, limit: int = 1000, memory_type: Optional[str] = None) -> list[Memory]:
        """异步获取长期记忆（默认情景+语义；指定类型时只返回该类型）"""
        return await asyncio.to_thread(self._storage.get_all, limit, memory_type)

    async def get_memories_page(
        self,
        session: str = "",
        query: str = "",
        memory_types: Optional[list[str]] = None,
        limit: int = 30,
        offset: int = 0,
        half_life_days: float = 30.0,
        decay_presets: Optional[dict] = None,
    ) -> tuple[list[Memory], int]:
        """异步分页获取长期记忆。"""
        return await asyncio.to_thread(
            self._storage.get_memories_page,
            session,
            query,
            memory_types,
            limit,
            offset,
            half_life_days,
            decay_presets,
        )

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        """异步列出所有出现过消息的会话，按最近活跃排序（Web 会话管理用）"""
        return await asyncio.to_thread(self._storage.list_sessions, limit)

    async def retrieve_session_recent(
        self, session: str, limit: int = 5, memory_type: Optional[str] = None
    ) -> list[Memory]:
        """异步获取某个会话最近的记忆（可按类型过滤）"""
        return await asyncio.to_thread(self._storage.retrieve_session_recent, session, limit, memory_type)

    async def count_session_messages(
        self, session: str, before: Optional[datetime] = None
    ) -> int:
        """异步统计会话的 episodic 消息数（Web 会话管理分页用）"""
        return await asyncio.to_thread(self._storage.count_session_messages, session, before)

    async def get_session_messages(
        self, session: str, limit: int = 30, offset: int = 0,
        before: Optional[datetime] = None,
    ) -> list[Memory]:
        """异步分页获取会话的历史消息（episodic 记忆）"""
        return await asyncio.to_thread(
            self._storage.get_session_messages, session, limit, offset, before
        )

    async def find_similar(self, session: str, sender_id: str, content: str,
                           memory_type: str = "episodic", lookback: int = 300) -> Optional[Memory]:
        """异步写入去重：返回同会话同发送者的近重复记忆，无则 None"""
        return await asyncio.to_thread(
            self._storage.find_similar, session, sender_id, content, memory_type, lookback
        )

    async def semantic_search(
        self,
        query: str,
        session: str = "",
        limit: int = 5,
        top_k_candidates: int = 200,
        half_life_days: float = 30.0,
        similarity_weight: float = 0.85,
        decay_presets: Optional[dict] = None,
    ) -> list[Memory]:
        """两阶段语义检索：
        1. 有嵌入服务：向量召回 top-30 → 重排 → top-N
        2. 否则回退 TF-IDF
        """
        service = self._storage._embedding_service
        if service and service.enabled:
            try:
                result = await self._vector_search(
                    query, session, limit, top_k_candidates,
                    half_life_days, similarity_weight, decay_presets,
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"向量检索失败，回退 TF-IDF：{e}")

        return await asyncio.to_thread(
            self._storage.semantic_search, query, session, limit,
            top_k_candidates, half_life_days, similarity_weight, decay_presets,
        )

    async def _vector_search(
        self,
        query: str,
        session: str,
        limit: int,
        top_k_candidates: int,
        half_life_days: float,
        similarity_weight: float,
        decay_presets: Optional[dict],
    ) -> Optional[list[Memory]]:
        """向量召回 + 重排。任一环节失败返回 None（调用方回退 TF-IDF）"""
        service = self._storage._embedding_service
        if not service or not service.enabled:
            return None
        try:
            similarity_weight = min(1.0, max(0.0, float(similarity_weight)))
        except (TypeError, ValueError):
            similarity_weight = 0.85

        candidates = await asyncio.to_thread(
            self._storage._get_candidates, session, top_k_candidates
        )
        if not candidates:
            return []

        # 1. 查询向量
        query_vec = await service.embed([query])
        if not query_vec:
            return None

        # 2. 文档向量（缓存优先，缺的批量补齐并持久化）
        vecs = await self._ensure_vectors(candidates)
        if vecs is None:
            return None

        # 3. 余弦相似度召回 top_k
        import numpy as np
        q = np.array(query_vec[0], dtype=np.float32)
        M = np.array(vecs, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return None
        norms = np.linalg.norm(M, axis=1)
        denom = norms * qn
        sims = np.zeros(len(candidates))
        mask = denom > 1e-9
        sims[mask] = np.sum(M[mask] * q, axis=1) / denom[mask]

        # 先按相似度取候选；跨会话共享开启时也不能让无关的本会话内容
        # 抢走真正相关的其他会话记忆。
        order = list(range(len(candidates)))
        order.sort(key=lambda i: -sims[i])
        vec_recalled = [candidates[i] for i in order[:self.VECTOR_RECALL_K]]

        # 3.5 词法召回（TF-IDF）与向量召回合并：人名/游戏黑话等专有名词靠字面匹配兜底
        recalled_by_id = {m.id: m for m in vec_recalled}
        try:
            lexical = await asyncio.to_thread(
                self._storage.semantic_search, query, session,
                limit=self.VECTOR_RECALL_K, top_k_candidates=top_k_candidates,
                half_life_days=half_life_days, similarity_weight=similarity_weight,
                decay_presets=decay_presets,
            )
            for m in lexical:
                if m.id not in recalled_by_id:
                    recalled_by_id[m.id] = m
        except Exception as e:
            logger.debug(f"词法召回失败，已跳过：{e}")
        recalled = list(recalled_by_id.values())

        # 4. 重排
        docs = [m.content for m in recalled]
        reranked_idx = await service.rerank(query, docs, top_n=limit)
        if reranked_idx is not None:
            ranked = [recalled[i] for i in reranked_idx if 0 <= i < len(recalled)]
            # 重排服务只返回顺序，不一定返回分数；用排名近似相关度，
            # 再混入有效重要性，让 half_life/similarity_weight 在嵌入模式下
            # 仍然生效，避免冷门旧记忆永远压过新事实。
            now = datetime.now()
            denom = max(1, len(ranked) - 1)
            ranked = [
                memory for _, memory in sorted(
                    enumerate(ranked),
                    key=lambda item: (
                        similarity_weight * (1.0 - item[0] / denom)
                        + (1.0 - similarity_weight)
                        * self._storage._effective_importance(
                            item[1], now, half_life_days, presets=decay_presets
                        )
                    ),
                    reverse=True,
                )
            ]
            return ranked[:limit]

        # 重排失败：退回向量相似度排序
        now = datetime.now()
        weight = similarity_weight
        ordered = sorted(
            recalled,
            key=lambda m: (
                weight * sims[candidates.index(m)]
                + (1.0 - weight)
                * self._storage._effective_importance(
                    m, now, half_life_days, presets=decay_presets
                )
            ),
            reverse=True,
        )
        return ordered[:limit]

    async def _ensure_vectors(self, memories: list) -> Optional[list]:
        """确保候选记忆都有向量：缓存/DB 优先，缺失的批量嵌入并持久化"""
        service = self._storage._embedding_service
        storage = self._storage

        vecs = []
        missing = []
        missing_idx = []
        for i, m in enumerate(memories):
            v = storage._get_embed(m.id)
            if v is None:
                v = await asyncio.to_thread(storage.get_embedding, m.id)
                if v is not None:
                    storage._cache_embed(m.id, v)
            if v is not None:
                vecs.append(v)
            else:
                vecs.append(None)
                missing.append(m)
                missing_idx.append(i)

        if missing:
            # 批量嵌入缺失部分（限流：分批）
            async with self._embed_lock:
                texts = [m.content for m in missing]
                new_vecs = await service.embed_many(texts)
                if new_vecs is None:
                    return None
                for m, v in zip(missing, new_vecs):
                    if v:
                        await asyncio.to_thread(storage.update_embedding, m.id, v)
            # 填回
            for k, i in enumerate(missing_idx):
                v = storage._get_embed(missing[k].id)
                vecs[i] = v

        if any(v is None for v in vecs):
            return None
        return vecs

    async def apply_time_decay(
        self, half_life_days: float = 30.0, min_importance: float = 0.1, max_age_days: float = 180.0,
        presets: Optional[dict] = None,
    ) -> dict:
        """异步时间衰减（支持按类型分级）"""
        return await asyncio.to_thread(
            self._storage.apply_time_decay, half_life_days, min_importance, max_age_days, presets
        )

    async def update_access(self, memory_id: int) -> None:
        """异步更新访问"""
        await asyncio.to_thread(self._storage.update_access, memory_id)

    async def update_access_many(self, memory_ids: list[int]) -> int:
        """异步批量更新实际召回记忆的访问时间。"""
        return await asyncio.to_thread(self._storage.update_access_many, memory_ids)

    async def bump_memories(self, memory_ids: list[int], importance_boost: float = 0.01) -> int:
        """异步检索反馈强化"""
        if not memory_ids:
            return 0
        return await asyncio.to_thread(self._storage.bump_memories, memory_ids, importance_boost)

    async def delete(self, memory_id: int) -> bool:
        """异步删除（同时清理向量缓存）"""
        ok = await asyncio.to_thread(self._storage.delete, memory_id)
        if ok:
            self._storage.clear_embed_cache({memory_id})
        return ok

    async def close(self) -> None:
        """异步关闭底层 SQLite 连接。"""
        await asyncio.to_thread(self._storage.close)

    # === 群聊黑话（异步包装） ===

    async def upsert_slang(self, term: str, meaning: str, session: str = "",
                           example: str = "", source: str = "auto") -> int:
        return await asyncio.to_thread(
            self._storage.upsert_slang, term, meaning, session, example, source
        )

    async def list_slang(self, session: str = "", enabled_only: bool = False) -> list:
        return await asyncio.to_thread(self._storage.list_slang, session, enabled_only)

    async def update_slang(self, slang_id: int, **fields) -> bool:
        return await asyncio.to_thread(
            lambda: self._storage.update_slang(slang_id, **fields)
        )

    async def delete_slang(self, slang_id: int) -> bool:
        return await asyncio.to_thread(self._storage.delete_slang, slang_id)

    async def delete_auto_slang(self, slang_ids: list[int]) -> int:
        return await asyncio.to_thread(self._storage.delete_auto_slang, slang_ids)

    async def match_slang(self, text: str, session: str = "", limit: int = 12) -> list:
        return await asyncio.to_thread(self._storage.match_slang, text, session, limit)

    async def bump_slang_hits(self, slang_ids: list) -> int:
        return await asyncio.to_thread(self._storage.bump_slang_hits, slang_ids)

"""
记忆存储层 - SQLite 实现
支持三层记忆架构：工作记忆 / 情景记忆 / 语义记忆
"""
import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._ensure_dir()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        logger.info(f"Memory storage initialized at {db_path}")

    def _ensure_dir(self) -> None:
        """确保目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

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

        # 创建索引
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_session ON memories(source_session)
        """)

        self.conn.commit()

    def store(self, memory: Memory) -> int:
        """存储记忆"""
        cursor = self.conn.execute("""
            INSERT INTO memories (content, memory_type, importance, tags, source_session, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            memory.content,
            memory.memory_type,
            memory.importance,
            json.dumps(memory.tags),
            memory.source_session,
            json.dumps(memory.metadata),
        ))
        self.conn.commit()
        memory.id = cursor.lastrowid
        logger.debug(f"Stored memory: {memory.id}, type={memory.memory_type}")
        return memory.id

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
            memories.append(self._row_to_memory(row))

        return memories

    def get_recent(self, memory_type: str, limit: int = 50) -> list[Memory]:
        """获取最近的记忆"""
        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE memory_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (memory_type, limit))

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def update_access(self, memory_id: int) -> None:
        """更新访问时间"""
        self.conn.execute("""
            UPDATE memories SET last_accessed = CURRENT_TIMESTAMP WHERE id = ?
        """, (memory_id,))
        self.conn.commit()

    def update_importance(self, memory_id: int, importance: float) -> None:
        """更新重要性"""
        self.conn.execute("""
            UPDATE memories SET importance = ? WHERE id = ?
        """, (importance, memory_id))
        self.conn.commit()

    def delete(self, memory_id: int) -> bool:
        """删除记忆"""
        cursor = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

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
        logger.info(f"Cleaned up {deleted} old memories")
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

    def close(self) -> None:
        """关闭连接"""
        self.conn.close()


class AsyncMemoryStorage:
    """异步记忆存储封装"""

    def __init__(self, storage: MemoryStorage):
        self._storage = storage

    async def store(self, memory: Memory) -> int:
        """异步存储"""
        return await asyncio.to_thread(self._storage.store, memory)

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

    async def update_access(self, memory_id: int) -> None:
        """异步更新访问"""
        await asyncio.to_thread(self._storage.update_access, memory_id)

    async def delete(self, memory_id: int) -> bool:
        """异步删除"""
        return await asyncio.to_thread(self._storage.delete, memory_id)

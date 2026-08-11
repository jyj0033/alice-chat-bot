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
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._ensure_dir()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        # 嵌入服务（可选，未配置时语义检索回退 TF-IDF）
        self._embedding_service = None
        # 内存向量缓存：memory_id -> list[float]（LRU，有上限，防长期运行内存膨胀）
        self._embed_cache: OrderedDict[int, list] = OrderedDict()
        self._embed_cache_max = 2048
        logger.info(f"Memory storage initialized at {db_path}")

    def _cache_embed(self, memory_id: int, vector: list) -> None:
        """写向量缓存（LRU 淘汰最旧，避免无限增长）"""
        self._embed_cache[memory_id] = vector
        self._embed_cache.move_to_end(memory_id)
        while len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.popitem(last=False)

    def _get_embed(self, memory_id: int) -> Optional[list]:
        """读向量缓存（命中即刷新为最近使用）"""
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

        self.conn.commit()

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
        return [self._row_to_memory(row) for row in cursor.fetchall()]

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
            GROUP BY source_session
            ORDER BY last_active DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

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

        return [self._row_to_memory(row) for row in cursor.fetchall()]

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

    def update_access(self, memory_id: int) -> None:
        """更新访问时间"""
        self.conn.execute("""
            UPDATE memories SET last_accessed = CURRENT_TIMESTAMP WHERE id = ?
        """, (memory_id,))
        self.conn.commit()

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
        return 0.5 ** (age_days / half_life_days)

    def _effective_importance(self, memory: Memory, now: datetime = None, half_life_days: float = 30.0) -> float:
        """有效重要性 = 基础 importance × 时间衰减"""
        return memory.importance * self._decay_factor(memory.last_accessed, now, half_life_days)

    # ---------- 嵌入向量持久化 ----------

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
            logger.error(f"Failed to save embedding for memory {memory_id}: {e}")

    def clear_embed_cache(self, ids: set = None) -> None:
        """清理向量缓存（删除记忆后调用）"""
        if ids is None:
            self._embed_cache.clear()
        else:
            for mid in ids:
                self._embed_cache.pop(mid, None)

    def semantic_search(
        self,
        query: str,
        session: str = "",
        limit: int = 5,
        top_k_candidates: int = 200,
        half_life_days: float = 30.0,
        similarity_weight: float = 0.85,
    ) -> list[Memory]:
        """语义检索：TF-IDF + 余弦相似度，结合时间衰减后的重要性排序。

        排序策略（两段式）：
        - 有相似度命中（sim > 阈值）的记忆：score = weight×sim + (1-weight)×有效重要性
        - 完全无关的记忆（sim≈0）排在命中记忆之后，按有效重要性排序
        这样精确相关的记忆不会被"高重要性但无关"的记忆挤掉。

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
            # 查询词完全不在记忆词典里（全为新词）：按有效重要性兜底返回
            scored = sorted(
                ((self._effective_importance(m, now, half_life_days), m) for m in candidates),
                key=lambda x: x[0], reverse=True,
            )
            return [m for _, m in scored[:limit]]

        row_norms = numpy.linalg.norm(X, axis=1)
        denom = row_norms * q_norm
        similarities = numpy.zeros(n_docs)
        mask = denom > 1e-9
        similarities[mask] = numpy.sum(X[mask] * q, axis=1) / denom[mask]

        # 两段式排序：命中的优先，无关记忆按有效重要性垫底
        HIT_THRESHOLD = 0.05
        hits = []
        misses = []
        for i, m in enumerate(candidates):
            sim = float(similarities[i])
            eff_imp = self._effective_importance(m, now, half_life_days)
            if sim > HIT_THRESHOLD:
                score = similarity_weight * sim + (1 - similarity_weight) * eff_imp
                hits.append((score, m))
            else:
                misses.append((eff_imp, m))

        hits.sort(key=lambda x: x[0], reverse=True)
        misses.sort(key=lambda x: x[0], reverse=True)

        result = [m for _, m in hits[:limit]]
        if len(result) < limit:
            result.extend(m for _, m in misses[: limit - len(result)])
        return result

    def _get_candidates(self, session: str = "", top_k: int = 200) -> list[Memory]:
        """取候选记忆：会话内优先，不足补其他会话。

        会话隔离：群会话不检索私聊来源的记忆（私事不外泄）；私聊会话仍可检索
        群聊记忆（更贴心）；跨群记忆共享保持不变（爱丽丝是统一人格）。
        """
        cursor = self.conn.execute("""
            SELECT * FROM memories
            WHERE memory_type IN ('episodic', 'semantic', 'session_summary')
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
        """, (top_k,))
        rows = cursor.fetchall()
        memories = [self._row_to_memory(r) for r in rows]

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

    def apply_time_decay(
        self,
        half_life_days: float = 30.0,
        min_importance: float = 0.1,
        max_age_days: float = 180.0,
        presets: Optional[dict] = None,
    ) -> dict:
        """批量应用时间衰减：
        - 每条记忆 importance 按 last_accessed 半衰期衰减
        - 衰减后 importance < min_importance 且超过 max_age_days 的删除
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
        updates: list[tuple] = []
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
            new_importance = importance * (0.5 ** (age_days / hl))

            if new_importance < min_importance and age_days > ma:
                deletes.append(mid)
                deleted += 1
                self._embed_cache.pop(mid, None)
            elif abs(new_importance - importance) > 1e-6:
                updates.append((round(new_importance, 4), mid))
                decayed += 1

        # 批量执行，避免逐条 UPDATE 的开销
        if updates:
            self.conn.executemany(
                "UPDATE memories SET importance = ? WHERE id = ?", updates
            )
        if deletes:
            self.conn.executemany(
                "DELETE FROM memories WHERE id = ?", [(mid,) for mid in deletes]
            )

        self.conn.commit()
        if decayed or deleted:
            logger.info(f"Time decay applied: {decayed} decayed, {deleted} deleted")
        return {"decayed": decayed, "deleted": deleted}

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
                    self._storage.update_embedding(mid, vecs[0])
            except Exception as e:
                logger.debug(f"Embedding on store skipped: {e}")
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

    async def get_profiles(self) -> list[Memory]:
        """异步获取所有用户画像（semantic 记忆）"""
        return await asyncio.to_thread(self._storage.get_profiles)

    async def update_memory(self, memory: Memory) -> bool:
        """异步整条更新已有记忆（用户画像复用原 id）"""
        return await asyncio.to_thread(self._storage.update_memory, memory)

    async def get_all(self, limit: int = 1000, memory_type: Optional[str] = None) -> list[Memory]:
        """异步获取长期记忆（默认情景+语义；指定类型时只返回该类型）"""
        return await asyncio.to_thread(self._storage.get_all, limit, memory_type)

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
        similarity_weight: float = 0.6,
    ) -> list[Memory]:
        """两阶段语义检索：
        1. 有嵌入服务：向量召回 top-30 → 重排 → top-N
        2. 否则回退 TF-IDF
        """
        service = self._storage._embedding_service
        if service and service.enabled:
            try:
                result = await self._vector_search(
                    query, session, limit, top_k_candidates
                )
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"Vector search failed, fallback to TF-IDF: {e}")

        return await asyncio.to_thread(
            self._storage.semantic_search, query, session, limit,
            top_k_candidates, half_life_days, similarity_weight,
        )

    async def _vector_search(
        self, query: str, session: str, limit: int, top_k_candidates: int
    ) -> Optional[list[Memory]]:
        """向量召回 + 重排。任一环节失败返回 None（调用方回退 TF-IDF）"""
        service = self._storage._embedding_service
        if not service or not service.enabled:
            return None

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

        # 会话内优先，再按相似度排序取候选
        order = list(range(len(candidates)))
        order.sort(key=lambda i: (candidates[i].source_session != session, -sims[i]))
        vec_recalled = [candidates[i] for i in order[:self.VECTOR_RECALL_K]]

        # 3.5 词法召回（TF-IDF）与向量召回合并：人名/游戏黑话等专有名词靠字面匹配兜底
        recalled_by_id = {m.id: m for m in vec_recalled}
        try:
            lexical = await asyncio.to_thread(
                self._storage.semantic_search, query, session,
                limit=self.VECTOR_RECALL_K, top_k_candidates=top_k_candidates,
                half_life_days=30.0, similarity_weight=0.85,
            )
            for m in lexical:
                if m.id not in recalled_by_id:
                    recalled_by_id[m.id] = m
        except Exception as e:
            logger.debug(f"Lexical recall skipped: {e}")
        recalled = list(recalled_by_id.values())

        # 4. 重排
        docs = [m.content for m in recalled]
        reranked_idx = await service.rerank(query, docs, top_n=limit)
        if reranked_idx is not None:
            ranked = [recalled[i] for i in reranked_idx if i < len(recalled)]
            return ranked[:limit]

        # 重排失败：退回向量相似度排序
        ordered = sorted(recalled, key=lambda m: sims[candidates.index(m)], reverse=True)
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
                        storage.update_embedding(m.id, v)  # 内部走 _cache_embed
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

    async def bump_memories(self, memory_ids: list[int], importance_boost: float = 0.01) -> int:
        """异步检索反馈强化"""
        if not memory_ids:
            return 0
        return await asyncio.to_thread(self._storage.bump_memories, memory_ids, importance_boost)

    async def delete(self, memory_id: int) -> bool:
        """异步删除（同时清理向量缓存）"""
        ok = await asyncio.to_thread(self._storage.delete, memory_id)
        if ok:
            self._storage._embed_cache.pop(memory_id, None)
        return ok

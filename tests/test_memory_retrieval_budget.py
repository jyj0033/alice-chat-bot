"""记忆检索预算、相关度门槛及媒体归档回填的离线回归。"""

import asyncio
from datetime import datetime
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from modules.memory.embedding import EmbeddingRerankService
from modules.memory.storage import AsyncMemoryStorage, Memory, MemoryStorage


@pytest.fixture
def storage(tmp_path):
    value = MemoryStorage(str(tmp_path / "memory.db"))
    try:
        yield value
    finally:
        value.close()


def remembered(storage, content="旧话题", session="group_a", vector=None, **kwargs):
    memory = Memory(
        content=content, source_session=session, memory_type="episodic", **kwargs,
    )
    storage.store(memory)
    if vector is not None:
        storage.update_embedding(memory.id, vector)
    return memory


def legacy_service():
    return SimpleNamespace(
        enabled=True,
        embed=AsyncMock(return_value=[[1.0, 0.0]]),
        embed_many=AsyncMock(side_effect=AssertionError("查询不能补文档向量")),
        rerank=AsyncMock(return_value=[0]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_stage", ["embed", "rerank"])
async def test_remote_timeout_returns_prepared_lexical_result(storage, slow_stage):
    memory = remembered(storage, vector=[1.0, 0.0])
    service = legacy_service()
    cancelled = asyncio.Event()

    async def wait_forever(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    setattr(service, slow_stage, AsyncMock(side_effect=wait_forever))
    storage.set_embedding_service(service)
    facade = AsyncMemoryStorage(storage)
    with patch.object(storage, "semantic_search", return_value=[memory]) as lexical:
        results = await asyncio.wait_for(
            facade.semantic_search("当前问题", "group_a", retrieval_timeout=0.1),
            timeout=1.0,
        )
    assert [item.id for item in results] == [memory.id]
    lexical.assert_called_once()
    assert cancelled.is_set()
    service.embed_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_total_budget_also_limits_unfinished_lexical_fallback(storage):
    release = threading.Event()
    finished = threading.Event()

    def slow_lexical(*args, **kwargs):
        try:
            release.wait(1.0)
            return []
        finally:
            finished.set()

    try:
        with patch.object(storage, "semantic_search", side_effect=slow_lexical):
            results = await asyncio.wait_for(
                AsyncMemoryStorage(storage).semantic_search("问题", retrieval_timeout=0.05),
                timeout=0.5,
            )
        assert results == []
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 1.0)


@pytest.mark.asyncio
async def test_missing_vectors_use_lexical_without_any_embedding_request(storage):
    memory = remembered(storage, content="鸣潮角色菲比")
    service = legacy_service()
    storage.set_embedding_service(service)
    with patch.object(storage, "semantic_search", return_value=[memory]):
        results = await AsyncMemoryStorage(storage).semantic_search("菲比", "group_a")
    assert [item.id for item in results] == [memory.id]
    service.embed.assert_not_awaited()
    service.embed_many.assert_not_awaited()
    service.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_candidate_is_recalled_lexically_without_blocking_other_vectors(storage):
    remembered(storage, content="另一段旧事", vector=[1.0, 0.0])
    missing = remembered(storage, content="鸣潮角色菲比")
    service = legacy_service()
    service.rerank.return_value = [1]
    storage.set_embedding_service(service)
    with patch.object(storage, "semantic_search", return_value=[missing]):
        results = await AsyncMemoryStorage(storage).semantic_search("菲比", "group_a")
    assert [item.id for item in results] == [missing.id]
    service.embed.assert_awaited_once_with(["菲比"])
    service.embed_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_unrelated_vectors_are_not_forced_into_memory_context(storage):
    remembered(storage, vector=[0.0, 1.0], importance=0.99)
    service = legacy_service()
    storage.set_embedding_service(service)
    with patch.object(storage, "semantic_search", return_value=[]):
        results = await AsyncMemoryStorage(storage).semantic_search("全新话题", "group_a")
    assert results == []
    service.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_low_rerank_score_filters_even_high_cosine_and_lexical_hit(storage):
    memory = remembered(storage, vector=[1.0, 0.0], importance=0.99)
    service = legacy_service()
    service.rerank_with_scores = AsyncMock(return_value=[(0, 0.01)])
    storage.set_embedding_service(service)
    with patch.object(storage, "semantic_search", return_value=[memory]):
        results = await AsyncMemoryStorage(storage).semantic_search("当前问题", "group_a")
    assert results == []
    service.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_relevance_thresholds_are_configurable(storage):
    memory = remembered(storage, vector=[0.8, 0.6])
    service = legacy_service()
    service.rerank_with_scores = AsyncMock(return_value=[(0, 0.08)])
    storage.set_embedding_service(service)
    facade = AsyncMemoryStorage(storage)
    with patch.object(storage, "semantic_search", return_value=[]):
        assert await facade.semantic_search("当前问题", "group_a", min_similarity=0.9) == []
        assert await facade.semantic_search("当前问题", "group_a") == []
        results = await facade.semantic_search("当前问题", "group_a", min_rerank_score=0.05)
    assert [item.id for item in results] == [memory.id]


@pytest.mark.asyncio
async def test_legacy_index_only_reranker_still_works_without_fabricated_scores(storage):
    memory = remembered(storage, vector=[1.0, 0.0])
    service = legacy_service()
    storage.set_embedding_service(service)
    with patch.object(storage, "semantic_search", return_value=[]):
        results = await AsyncMemoryStorage(storage).semantic_search("当前问题", "group_a")
    assert [item.id for item in results] == [memory.id]
    service.rerank.assert_awaited_once()


@pytest.mark.asyncio
async def test_rerank_failure_uses_only_relevant_candidates(storage):
    match = remembered(storage, content="相关内容", vector=[1.0, 0.0])
    remembered(storage, content="完全无关", vector=[0.0, 1.0], importance=0.99)
    service = legacy_service()
    service.rerank.return_value = None
    storage.set_embedding_service(service)
    with patch.object(storage, "semantic_search", return_value=[]):
        results = await AsyncMemoryStorage(storage).semantic_search("当前问题", "group_a")
    assert [item.id for item in results] == [match.id]


@pytest.mark.asyncio
async def test_parent_cancellation_is_not_swallowed(storage):
    memory = remembered(storage, vector=[1.0, 0.0])
    entered = asyncio.Event()

    async def wait_forever(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    service = legacy_service()
    service.embed.side_effect = wait_forever
    storage.set_embedding_service(service)
    with patch.object(storage, "semantic_search", return_value=[memory]):
        task = asyncio.create_task(AsyncMemoryStorage(storage).semantic_search("当前问题", "group_a"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_archive_backfill_changes_only_exact_group_message_content(storage):
    timestamp = datetime(2026, 10, 1, 12, 0)
    target = Memory(
        content="一张图，看不清画面。", source_session="group_a",
        created_at=timestamp, last_accessed=timestamp,
        metadata={"message_id": "photo", "sender_id": "user", "is_bot": False},
    )
    storage.store_group_analysis_message(target)
    other_group = Memory(content="别群原文", source_session="group_b", metadata={"message_id": "photo"})
    storage.store_group_analysis_message(other_group)
    other_message = Memory(content="同群别条", source_session="group_a", metadata={"message_id": "other"})
    storage.store_group_analysis_message(other_message)
    episodic = remembered(storage, content="普通记忆", metadata={"message_id": "photo"})
    before = dict(storage.conn.execute("SELECT * FROM memories WHERE id = ?", (target.id,)).fetchone())

    changed = await AsyncMemoryStorage(storage).update_group_analysis_message_content(
        "group_a", "photo", "一张图，画面是鸣潮技能界面。",
    )
    assert changed
    after = dict(storage.conn.execute("SELECT * FROM memories WHERE id = ?", (target.id,)).fetchone())
    assert after.pop("content") == "一张图，画面是鸣潮技能界面。"
    before.pop("content")
    assert after == before
    for memory in (other_group, other_message, episodic):
        assert storage.conn.execute("SELECT content FROM memories WHERE id = ?", (memory.id,)).fetchone()[0] == memory.content
    assert not storage.update_group_analysis_message_content("group_a", "missing", "新正文")
    assert not storage.update_group_analysis_message_content("group_a", "", "新正文")
    assert not storage.update_group_analysis_message_content("", "photo", "新正文")


class FakeResponse:
    status = 200

    def __init__(self, data):
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return self.data


@pytest.mark.asyncio
async def test_embedding_service_preserves_scores_and_old_index_api():
    service = EmbeddingRerankService({"api_key": "offline-test-key"})
    response = FakeResponse({"results": [
        {"index": 0, "relevance_score": 0.03},
        {"index": 1, "relevance_score": 0.9},
        {"index": 10, "relevance_score": 0.99},
        {"index": True, "relevance_score": 0.99},
        {"index": 2, "relevance_score": "nan"},
    ]})
    service._get_session = AsyncMock(return_value=SimpleNamespace(post=lambda *args, **kwargs: response))
    assert await service.rerank_with_scores("问题", ["甲", "乙", "丙"]) == [(1, 0.9), (0, 0.03)]
    assert await service.rerank("问题", ["甲", "乙", "丙"]) == [1, 0]


@pytest.mark.asyncio
async def test_embedding_service_does_not_invent_scores_for_legacy_response():
    service = EmbeddingRerankService({"api_key": "offline-test-key"})
    response = FakeResponse({"results": [{"index": 1}, {"index": 0}]})
    service._get_session = AsyncMock(return_value=SimpleNamespace(post=lambda *args, **kwargs: response))
    assert await service.rerank_with_scores("问题", ["甲", "乙"]) == [(1, None), (0, None)]
    assert await service.rerank("问题", ["甲", "乙"]) == [1, 0]


# ---------- 模型思考痕迹不能泄漏进纪要或上下文 ----------

def test_strips_closed_think_block():
    from modules.llm.base import strip_reasoning_traces
    text = "<think>Let me analyze this chat log:\n1. x</think>群里在聊鸣潮抽卡。"
    assert strip_reasoning_traces(text) == "群里在聊鸣潮抽卡。"


def test_strips_truncated_think_block():
    # max_tokens 截断时 </think> 可能根本没输出出来
    from modules.llm.base import strip_reasoning_traces
    assert strip_reasoning_traces("<think>Let me analyze this chat log:\n\n1. x") == ""


def test_strip_keeps_normal_digest_untouched():
    from modules.llm.base import strip_reasoning_traces
    text = "【群聊纪要 9月26日 18:05】群里先聊了开黑，然后转到配队。"
    assert strip_reasoning_traces(text) == text


def test_polluted_digest_is_filtered_when_injected_into_context():
    from datetime import timedelta
    from modules.memory.context import ContextManager
    from modules.llm.base import strip_reasoning_traces
    now = datetime.now()
    polluted = Memory(
        content=("【群聊纪要 9月26日 18:05】<think>The chat is about a game"
                 " that's abo</think>群里讨论了配队。"),
        memory_type="session_summary", created_at=now - timedelta(days=2),
    )
    assert strip_reasoning_traces(polluted.content) == "【群聊纪要 9月26日 18:05】群里讨论了配队。"
    prompt = ContextManager().build_context_prompt(
        "group_1", bot_name="爱丽丝", memories=[polluted], max_messages=5)
    assert "<think>" not in prompt
    assert "The chat is about" not in prompt
    assert "群里讨论了配队" in prompt

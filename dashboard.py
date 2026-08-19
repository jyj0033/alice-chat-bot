"""
爱丽丝 (Alice) - Web 管理面板
"""
import logging
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
import uvicorn
import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

app = FastAPI(title="爱丽丝 - Alice")
bot_instance = None
LOG_BUFFER = deque(maxlen=1000)


class DashboardLogHandler(logging.Handler):
    """保留最近日志，未配置文件日志时仍可在 Web 页面查看。"""

    def emit(self, record):
        try:
            LOG_BUFFER.append(self.format(record))
        except Exception:
            self.handleError(record)


_dashboard_log_handler = DashboardLogHandler()
_dashboard_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logging.getLogger().addHandler(_dashboard_log_handler)


def set_bot(bot):
    global bot_instance, CONFIG_FILE
    bot_instance = bot
    # 与 Bot 实际加载路径保持一致，兼容 `main.py -c custom.yaml`。
    CONFIG_FILE = Path(bot.config_path)


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


def deep_merge(base, changes):
    """递归合并配置，页面只提交某个子面板时不丢失同级高级参数。"""
    result = dict(base) if isinstance(base, dict) else {}
    for key, value in (changes or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def normalize_llm_config(config):
    """把旧版扁平 llm 配置映射为 Provider 结构。"""
    raw = config.get('llm', {}) if isinstance(config, dict) else {}
    if not isinstance(raw, dict):
        return {}
    legacy_keys = {'api_key', 'base_url', 'model', 'provider_type'}
    if legacy_keys.intersection(raw) and not any(
        isinstance(value, dict) for value in raw.values()
    ):
        normalized = {'primary': dict(raw)}
        config['llm'] = normalized
        return normalized
    return raw


HTML_CONTENT = open(BASE_DIR / "templates" / "dashboard.html", 'r', encoding='utf-8').read()


@app.get("/")
async def root():
    return HTMLResponse(content=HTML_CONTENT)


# API 端点
@app.get("/api/config")
async def get_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    # 隐藏 API Keys
    if 'llm' in config:
        for name, provider in normalize_llm_config(config).items():
            if isinstance(provider, dict) and 'api_key' in provider:
                provider['api_key'] = '********' if provider['api_key'] else ''
    if 'memory' in config and isinstance(config.get('memory'), dict):
        emb = config['memory'].get('embedding')
        if isinstance(emb, dict) and emb.get('api_key'):
            emb['api_key'] = '********'
    if 'qq' in config and isinstance(config.get('qq'), dict):
        if config['qq'].get('access_token'):
            config['qq']['access_token'] = '********'
    if 'image' in config and isinstance(config.get('image'), dict):
        vision = config['image'].get('vision')
        if isinstance(vision, dict) and vision.get('api_key'):
            vision['api_key'] = '********'
    if 'search' in config and isinstance(config.get('search'), dict):
        search = config['search']
        if isinstance(search.get('llm'), dict) and search['llm'].get('api_key'):
            search['llm']['api_key'] = '********'
        for name in ('bocha', 'doubao'):
            backend = (search.get('backends') or {}).get(name)
            if isinstance(backend, dict) and backend.get('api_key'):
                backend['api_key'] = '********'
    return config


@app.post("/api/config")
async def update_config(request: Request):
    try:
        data = await request.json()
        current = load_config()

        # LLM 配置 - 支持多 provider
        if 'llm' in data:
            llm = normalize_llm_config(current)
            current['llm'] = llm
            for name, provider_data in data['llm'].items():
                if isinstance(provider_data, dict):
                    provider = llm.setdefault(name, {})
                    if provider_data.get('api_key') and not str(provider_data['api_key']).startswith('*'):
                        provider['api_key'] = provider_data['api_key']
                    for key in ['base_url', 'model', 'temperature', 'max_tokens', 'timeout', 'provider_type', 'enabled']:
                        if key in provider_data:
                            provider[key] = provider_data[key]

        # QQ 配置
        if 'qq' in data:
            qq = current.setdefault('qq', {})
            for key in ['ws_url', 'ws_host', 'ws_port', 'access_token', 'self_id']:
                if key in data['qq']:
                    value = data['qq'][key]
                    if key == 'access_token' and str(value).startswith('*'):
                        continue
                    qq[key] = value

        # 发言设置
        if 'speaking' in data:
            current['speaking'] = deep_merge(current.get('speaking', {}), data['speaking'])

        # 记忆设置
        if 'memory' in data:
            current_mem = current.get('memory', {})
            # 深合并 embedding，保留未变动的 key
            if isinstance(data['memory'].get('embedding'), dict):
                new_emb = data['memory']['embedding'].copy()
                if new_emb.get('api_key', '').startswith('*'):
                    old_emb = current_mem.get('embedding', {})
                    new_emb['api_key'] = old_emb.get('api_key', '')
                data['memory']['embedding'] = new_emb
            current['memory'] = deep_merge(current_mem, data['memory'])

        # Bot 基本信息
        if 'bot' in data:
            current['bot'] = {**current.get('bot', {}), **data['bot']}

        # 人格配置
        if 'personality' in data:
            current['personality'] = deep_merge(
                current.get('personality', {}), data['personality']
            )

        # 群组设置
        if 'groups' in data:
            current['groups'] = data['groups']

        # 情感系统设置
        if 'emotion' in data:
            current['emotion'] = deep_merge(current.get('emotion', {}), data['emotion'])

        # 注意力系统设置
        if 'attention' in data:
            current['attention'] = deep_merge(current.get('attention', {}), data['attention'])

        # 疲劳系统设置
        if 'fatigue' in data:
            current['fatigue'] = deep_merge(current.get('fatigue', {}), data['fatigue'])

        # 冷却系统设置
        if 'cooldown' in data:
            current['cooldown'] = deep_merge(current.get('cooldown', {}), data['cooldown'])

        # 思考延迟设置
        if 'thinking' in data:
            current['thinking'] = deep_merge(current.get('thinking', {}), data['thinking'])

        # 打字风格设置
        if 'typing_style' in data:
            current['typing_style'] = deep_merge(current.get('typing_style', {}), data['typing_style'])

        if 'conversation_floor' in data:
            current['conversation_floor'] = deep_merge(
                current.get('conversation_floor', {}), data['conversation_floor']
            )

        if 'rich_media' in data:
            current['rich_media'] = deep_merge(
                current.get('rich_media', {}), data['rich_media']
            )

        # 图片/视觉模型配置
        if 'image' in data:
            current_image = current.get('image', {})
            if isinstance(data['image'].get('vision'), dict):
                new_vision = data['image']['vision'].copy()
                if str(new_vision.get('api_key', '')).startswith('*'):
                    new_vision['api_key'] = (current_image.get('vision') or {}).get('api_key', '')
                data['image']['vision'] = new_vision
            current['image'] = deep_merge(current_image, data['image'])

        # 联网搜索配置
        if 'search' in data:
            current_search = current.get('search', {})
            new_search = data['search']
            # 回填掩码过的 api_key
            if isinstance(new_search.get('llm'), dict):
                if str(new_search['llm'].get('api_key', '')).startswith('*'):
                    new_search['llm']['api_key'] = (current_search.get('llm') or {}).get('api_key', '')
            backends = new_search.get('backends') or {}
            for name in ('bocha', 'doubao'):
                if isinstance(backends.get(name), dict):
                    if str(backends[name].get('api_key', '')).startswith('*'):
                        backends[name]['api_key'] = (
                            (current_search.get('backends') or {}).get(name, {}) or {}
                        ).get('api_key', '')
            current['search'] = deep_merge(current_search, new_search)

        save_config(current)
        return {"success": True, "message": "配置已保存"}
    except Exception as e:
        return {"success": False, "error": str(e) if e else "Unknown error"}


@app.get("/api/status")
async def get_status():
    if not bot_instance:
        return {"status": "not_initialized", "running": False}
    rich_media = {}
    recognition_history = []
    if bot_instance.qq_adapter:
        enricher = getattr(bot_instance.qq_adapter, 'rich_media_enricher', None)
        if enricher:
            rich_media = enricher.statistics()
            try:
                recognition_history = enricher.recognition_history(20)
            except Exception as e:
                logger.warning("recognition_history failed: %s", e)
    emotion = {
        "energy": 0.7,
        "engagement": 0.5,
        "mood": 0.5,
        "emotion": "neutral",
    }
    manager = getattr(bot_instance, 'emotional_manager', None)
    states = getattr(manager, '_states', {}) if manager else {}
    if states:
        # 仪表盘展示最近活跃会话的状态，比固定默认值更能反映当前运行情况。
        latest = max(states.values(), key=lambda state: state._last_update or 0)
        mood_values = {
            "happy": 0.8,
            "excited": 0.9,
            "calm": 0.6,
            "neutral": 0.5,
            "anxious": 0.35,
            "bored": 0.25,
            "tired": 0.2,
        }
        emotion = {
            "energy": latest.energy,
            "engagement": latest.engagement,
            "mood": mood_values.get(latest.current_emotion.value, 0.5),
            "emotion": latest.current_emotion.value,
        }

    return {
        "status": "running" if bot_instance._running else "stopped",
        "running": bot_instance._running,
        "qq": {
            "connected": bot_instance.qq_adapter.is_connected if bot_instance.qq_adapter else False,
            "messages_sent": bot_instance.qq_adapter.messages_sent if bot_instance.qq_adapter else 0,
            "messages_received": bot_instance.qq_adapter.messages_received if bot_instance.qq_adapter else 0,
        },
        "runtime": {
            "active_sessions": len(bot_instance.context_manager._windows) if bot_instance.context_manager else 0,
            "pending_replies": len(bot_instance._reply_tasks),
            "pending_api_calls": len(bot_instance.qq_adapter._pending_api) if bot_instance.qq_adapter else 0,
        },
        "rich_media": rich_media,
        "recognition_history": recognition_history,
        "emotion": emotion,
    }


@app.get("/api/personality")
async def get_personality():
    if not bot_instance or not bot_instance.personality:
        # 返回默认人格配置
        return {
            "name": "爱丽丝",
            "nickname": "小艾",
            "age_range": "20-25",
            "avatar_description": "",
            "background": "一个活泼可爱、喜欢聊天的女孩",
            "traits": {
                "openness": 0.7,
                "conscientiousness": 0.5,
                "extraversion": 0.8,
                "agreeableness": 0.85,
                "neuroticism": 0.3
            },
            "interested_topics": ["美食", "旅行", "音乐", "电影", "闲聊"],
            "bored_topics": ["广告", "政治"],
            "humor_style": "dry",
            "taboo_topics": [],
            "catchphrases": [],
            "emoji_set": ["😅", "🤔", "😂", "👍", "🙄"],
            "description": ""
        }
    personality = bot_instance.personality.to_dict()
    personality["description"] = getattr(bot_instance.personality, 'description', '')
    return personality


@app.get("/api/sessions")
async def get_sessions():
    """会话列表 = 当前活跃(内存窗口) + 历史会话(SQLite记忆)。

    重启后内存窗口为空，靠历史记忆恢复会话管理，避免"什么都看不到"。
    """
    if not bot_instance:
        return {"sessions": []}

    seen = set()
    sessions = []

    # 1. 当前活跃窗口（内存），优先展示
    if bot_instance.context_manager:
        for sid in bot_instance.context_manager._windows.keys():
            seen.add(sid)
            sessions.append({"session": sid, "active": True})

    # 2. 历史会话（SQLite 记忆），重启后仍可恢复
    try:
        if bot_instance.memory_storage:
            for row in await bot_instance.memory_storage.list_sessions(limit=50):
                sid = row.get("session") or ""
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                sessions.append({
                    "session": sid,
                    "active": False,
                    "message_count": row.get("message_count", 0),
                    "last_active": (row.get("last_active") or ""),
                })
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")

    return {"sessions": sessions}


@app.get("/api/context/{session_id}")
async def get_context(session_id: str, page: int = 1, page_size: int = 30):
    """会话消息（分页）。

    - page=1 且内存窗口有消息：返回窗口内的最近聊天记录；
      更早的历史由「加载更早」分页拉取（SQLite episodic，即消息）。
    - 重启后窗口为空：直接分页返回 SQLite 历史，避免什么都看不到。
    - 绝不一次把全部消息载入前端。
    """
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    empty = {"messages": [], "total": 0, "page": page, "page_size": page_size, "has_more": False}
    if not bot_instance or not bot_instance.memory_storage:
        return empty

    window_msgs = []
    if bot_instance.context_manager:
        window = bot_instance.context_manager.get_window(session_id)
        window_msgs = list(window.messages)  # 时间正序（旧→新）

    storage = bot_instance.memory_storage

    async def history_count(before=None):
        try:
            return await storage.count_session_messages(session_id, before=before)
        except Exception:
            return 0

    async def history_page(offset, limit, before=None):
        try:
            return await storage.get_session_messages(session_id, limit=limit, offset=offset, before=before)
        except Exception:
            return []

    # 内存窗口非空：窗口日志在前，历史记忆在后（以最早窗口消息时间为界，天然不重叠）
    if window_msgs:
        oldest_ts = window_msgs[0].timestamp
        db_total = await history_count(before=oldest_ts)
        total = len(window_msgs) + db_total

        if page == 1:
            msgs = [_window_msg_to_dict(m) for m in window_msgs]
            return {
                "messages": msgs,
                "total": total,
                "page": page,
                "page_size": page_size,
                "has_more": db_total > 0,
            }

        offset = (page - 2) * page_size  # page2 起从历史第一页开始
        rows = await history_page(offset, page_size, before=oldest_ts)
        rows = rows[::-1]  # 库内为倒序（新→旧），翻转为升序，前端前置后严格「旧在上、新在下」
        msgs = [_memory_to_msg_dict(m) for m in rows]
        return {
            "messages": msgs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": (page - 1) * page_size < db_total,
        }

    # 窗口为空（如重启后）：直接读历史
    db_total = await history_count()
    offset = (page - 1) * page_size
    rows = await history_page(offset, page_size)
    rows = rows[::-1]  # 同上：库内倒序翻转为升序，前端按时间正序展示
    msgs = [_memory_to_msg_dict(m) for m in rows]
    return {
        "messages": msgs,
        "total": db_total,
        "page": page,
        "page_size": page_size,
        "has_more": page * page_size < db_total,
    }


def _window_msg_to_dict(m) -> dict:
    return {
        "sender": m.sender_name,
        "content": m.content,
        "is_bot": m.is_bot,
        "timestamp": m.timestamp.isoformat(),
        "from_history": False,
    }


def _memory_to_msg_dict(mem) -> dict:
    """episodic 记忆即一条消息：content 形如「昵称：内容」，metadata 保留发送者信息。

    bot 自己发出的消息 metadata 带 is_bot=True，据此渲染为 bot 发言侧。
    """
    content = mem.content or ""
    meta = mem.metadata or {}
    sender = meta.get("sender_name") or ""
    is_bot = bool(meta.get("is_bot"))
    body = content
    if sender and content.startswith(sender + "："):
        body = content[len(sender) + 1:]
    fallback = "爱丽丝" if is_bot else "历史记录"
    return {
        "sender": sender or fallback,
        "content": body or content,
        "is_bot": is_bot,
        "timestamp": mem.created_at.isoformat(),
        "from_history": True,
    }


@app.get("/api/memories")
async def get_memories(session: str = "", q: str = "", type: str = "all", limit: int = 100):
    """获取长期记忆列表（支持按会话/关键词/类型过滤，含时间衰减后的有效重要性）
    type: all=长期记忆+群聊纪要, memory=仅长期记忆, digest=仅群聊纪要
    """
    if not bot_instance or not bot_instance.memory_storage:
        return {"memories": [], "total": 0}

    from datetime import datetime

    try:
        if type == "digest":
            all_mem = await bot_instance.memory_storage.get_all(limit=1000, memory_type="session_summary")
            allowed_types = {"session_summary"}
        elif type == "memory":
            all_mem = await bot_instance.memory_storage.get_all(limit=1000)
            allowed_types = {"episodic", "semantic"}
        else:
            regular = await bot_instance.memory_storage.get_all(limit=1000)
            digests = await bot_instance.memory_storage.get_all(limit=1000, memory_type="session_summary")
            all_mem = list(regular) + list(digests)
            allowed_types = {"episodic", "semantic", "session_summary"}
    except Exception as e:
        logger.error(f"Failed to load memories: {e}")
        return {"memories": [], "total": 0}

    now = datetime.now()
    result = []
    q_lower = (q or "").strip().lower()
    for m in all_mem:
        if m.memory_type not in allowed_types:
            continue
        if session and m.source_session != session:
            continue
        if q_lower and q_lower not in m.content.lower():
            continue
        eff = bot_instance.memory_storage._storage._effective_importance(m, now, bot_instance.memory_half_life_days)
        result.append({
            "id": m.id,
            "content": m.content,
            "importance": round(m.importance, 3),
            "effective_importance": round(eff, 3),
            "memory_type": m.memory_type,
            "tags": list(m.tags or []),
            "created_at": m.created_at.isoformat(),
            "last_accessed": m.last_accessed.isoformat(),
            "source_session": m.source_session,
            "sender_name": (m.metadata or {}).get("sender_name", ""),
        })

    # 默认按有效重要性排序
    result.sort(key=lambda x: x["effective_importance"], reverse=True)
    result = result[:limit]
    return {"memories": result, "total": len(result)}


@app.get("/api/profiles")
async def get_profiles():
    """所有用户画像（semantic 记忆，按最近更新排序）"""
    if not bot_instance or not bot_instance.memory_storage:
        return {"profiles": [], "total": 0}
    try:
        profiles = await bot_instance.memory_storage.get_profiles()
        result = []
        for p in profiles:
            meta = p.metadata or {}
            result.append({
                "id": p.id,
                "sender_id": meta.get("sender_id", ""),
                "sender_name": meta.get("sender_name", "") or "未知用户",
                "content": p.content,
                "fact_count": meta.get("fact_count", 0),
                "daily_count": meta.get("daily_count", 0),
                # 提炼素材原句：用于在页面上核对"哪句话导致了这个结论"
                "source_facts": meta.get("source_facts", []) or [],
                "source_daily": meta.get("source_daily", []) or [],
                "warnings": meta.get("warnings", []) or [],
                # 画像是历次结论迭代出来的，展示累积轨迹而不是单次快照
                "first_distilled_at": meta.get("first_distilled_at", "") or p.created_at.isoformat(),
                "last_distilled_at": meta.get("last_distilled_at", "") or p.created_at.isoformat(),
                "distill_count": meta.get("distill_count", 0) or 0,
                "previous_summary": meta.get("previous_summary", "") or "",
                "mbti": meta.get("mbti") or None,
                "created_at": p.created_at.isoformat(),
                # 注意：last_accessed 是"最近被检索召回"的时间（记忆保鲜会刷新它），
                # 不代表重新提炼过，前端不要再标成"更新"
                "last_accessed": p.last_accessed.isoformat(),
                "source_session": p.source_session,
            })
        return {"profiles": result, "total": len(result)}
    except Exception as e:
        logger.error(f"Failed to load profiles: {e}")
        return {"profiles": [], "total": 0}


@app.post("/api/profiles/distill")
async def distill_profiles():
    """手动触发一次用户画像提炼（force=True）"""
    if not bot_instance:
        return {"success": False, "error": "Bot not initialized"}
    try:
        result = await bot_instance._distill_user_profiles(force=True)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Manual profile distill failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/slang")
async def get_slang():
    """群聊黑话词表（Web 管理用，返回全部含停用项）"""
    if not bot_instance or not bot_instance.memory_storage:
        return {"slang": [], "total": 0}
    try:
        rows = await bot_instance.memory_storage.list_slang()
        return {"slang": rows, "total": len(rows)}
    except Exception as e:
        logger.error(f"Failed to load slang: {e}")
        return {"slang": [], "total": 0}


@app.post("/api/slang")
async def create_slang(request: Request):
    """手动新增一条黑话（source=manual，不会被自动提取覆盖）"""
    if not bot_instance or not bot_instance.memory_storage:
        return {"success": False, "error": "Bot not initialized"}
    try:
        data = await request.json()
        term = (data.get("term") or "").strip()
        meaning = (data.get("meaning") or "").strip()
        if not term or not meaning:
            return {"success": False, "error": "词和含义都不能为空"}
        slang_id = await bot_instance.memory_storage.upsert_slang(
            term=term,
            meaning=meaning,
            session=(data.get("session") or "").strip(),
            example=(data.get("example") or "").strip(),
            source="manual",
        )
        return {"success": bool(slang_id), "id": slang_id}
    except Exception as e:
        logger.error(f"Create slang failed: {e}")
        return {"success": False, "error": str(e)}


@app.put("/api/slang/{slang_id}")
async def edit_slang(slang_id: int, request: Request):
    """修改黑话（含启用/停用）"""
    if not bot_instance or not bot_instance.memory_storage:
        return {"success": False, "error": "Bot not initialized"}
    try:
        data = await request.json()
        ok = await bot_instance.memory_storage.update_slang(
            slang_id,
            term=data.get("term"),
            meaning=data.get("meaning"),
            example=data.get("example"),
            enabled=data.get("enabled"),
        )
        return {"success": ok}
    except Exception as e:
        logger.error(f"Update slang failed: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/slang/{slang_id}")
async def remove_slang(slang_id: int):
    """删除一条黑话"""
    if not bot_instance or not bot_instance.memory_storage:
        return {"success": False, "error": "Bot not initialized"}
    try:
        ok = await bot_instance.memory_storage.delete_slang(slang_id)
        return {"success": ok, "message": "已删除" if ok else "词条不存在"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/slang/extract")
async def extract_slang_now():
    """立即从最近聊天记录提取黑话"""
    if not bot_instance:
        return {"success": False, "error": "Bot not initialized"}
    try:
        hours = int(bot_instance._slang_config.get("lookback_hours", 48))
        result = await bot_instance.extract_slang_all(hours=hours)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Manual slang extract failed: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: int):
    """删除一条长期记忆"""
    if not bot_instance or not bot_instance.memory_storage:
        return {"success": False, "error": "Bot not initialized"}
    try:
        ok = await bot_instance.memory_storage.delete(memory_id)
        return {"success": ok, "message": "已删除" if ok else "记忆不存在"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/embedding/test")
async def test_embedding(request: Request):
    """测试嵌入模型连接"""
    try:
        data = await request.json()
        api_key = (data.get("api_key") or "").strip()
        if api_key.startswith('*'):
            api_key = str(
                load_config().get("memory", {}).get("embedding", {}).get("api_key", "")
            ).strip()
        if not api_key:
            return {"success": False, "error": "API Key 不能为空"}
        model = data.get("embed_model") or "Qwen/Qwen3-Embedding-8B"

        from modules.memory.embedding import EmbeddingRerankService
        svc = EmbeddingRerankService({
            "api_key": api_key,
            "base_url": data.get("base_url") or "https://api.siliconflow.cn/v1",
            "embed_model": model,
            "rerank_model": data.get("rerank_model") or "Pro/BAAI/bge-reranker-v2-m3",
            "input_type": data.get("input_type") or None,
            "batch_size": data.get("batch_size") or 32,
        })
        try:
            vecs = await svc.embed(["测试连接"])
            if not vecs:
                return {"success": False, "error": "嵌入服务无返回"}
            dim = len(vecs[0])
            return {"success": True, "model": model, "dim": dim,
                    "message": f"连接成功，向量维度 {dim}"}
        finally:
            await svc.close()
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


@app.get("/api/emotion/{session_id}")
async def get_emotion(session_id: str):
    if not bot_instance:
        return {"energy": 0.8, "engagement": 0.6, "mood": 0.5, "emotion": "平静"}
    if not bot_instance.emotional_manager.enabled:
        return {"energy": 0.7, "engagement": 0.5, "mood": 0.5, "emotion": "neutral"}
    return bot_instance.emotional_manager.get_state(session_id).to_dict()


@app.get("/api/providers")
async def get_providers():
    """获取所有 LLM providers"""
    config = load_config()
    providers = []
    for name, provider in normalize_llm_config(config).items():
        if isinstance(provider, dict):
            providers.append({
                "name": name,
                "base_url": provider.get('base_url', ''),
                "model": provider.get('model', ''),
                "enabled": provider.get('enabled', True),
                "provider_type": provider.get('provider_type', 'openai'),
                "has_key": bool(provider.get('api_key')),
            })
    return {"providers": providers}


@app.post("/api/providers")
async def add_provider(request: Request):
    """添加新的 LLM provider"""
    try:
        data = await request.json()
        name = data.get('name', '').strip()
        if not name:
            return {"success": False, "error": "Provider name is required"}
        if not name.replace('_', '').replace('-', '').isalnum():
            return {"success": False, "error": "Invalid provider name"}

        current = load_config()
        llm = normalize_llm_config(current)
        current['llm'] = llm

        if name in llm:
            return {"success": False, "error": f"Provider '{name}' already exists"}

        # 添加新 provider
        llm[name] = {
            'api_key': data.get('api_key', ''),
            'base_url': data.get('base_url', 'https://api.openai.com/v1'),
            'model': data.get('model', 'gpt-4o'),
            'provider_type': data.get('provider_type', 'openai'),
            'timeout': data.get('timeout', 120),
            'temperature': data.get('temperature', 0.8),
            'max_tokens': data.get('max_tokens', 2000),
            'enabled': data.get('enabled', True),
        }

        save_config(current)
        return {"success": True, "message": f"Provider '{name}' added"}
    except Exception as e:
        return {"success": False, "error": str(e) if e else "Unknown error"}


@app.delete("/api/providers/{name}")
async def delete_provider(name: str):
    """删除 LLM provider"""
    try:
        current = load_config()
        llm = normalize_llm_config(current)

        if name not in llm:
            return {"success": False, "error": f"Provider '{name}' not found"}
        if name in ('primary',):
            return {"success": False, "error": "Cannot delete primary provider"}

        del llm[name]
        save_config(current)
        return {"success": True, "message": f"Provider '{name}' deleted"}
    except Exception as e:
        return {"success": False, "error": str(e) if e else "Unknown error"}


@app.post("/api/providers/{name}/test")
async def test_provider(name: str):
    """测试 provider 连接"""
    try:
        current = load_config()
        # 先检查 llm.primary/llm.fallback，再检查 providers
        provider_config = normalize_llm_config(current).get(name)
        if not provider_config:
            provider_config = current.get('providers', {}).get(name)
        if not provider_config:
            return {"success": False, "error": f"Provider '{name}' not found"}

        from modules.llm.openai_provider import create_provider
        test_provider = create_provider(
            provider_config.get('provider_type', 'openai'),
            {
                'api_key': provider_config.get('api_key', ''),
                'base_url': provider_config.get('base_url', 'https://api.openai.com/v1'),
                'model': provider_config.get('model', 'gpt-4o'),
                'timeout': 30,
                'provider_type': provider_config.get('provider_type', 'openai'),
            }
        )

        from modules.llm.base import ChatRequest, ChatMessage
        resp = await test_provider.chat(ChatRequest(
            messages=[ChatMessage(role="user", content="Hi, respond with OK")],
            temperature=0.1,
            max_tokens=10,
        ))

        return {"success": True, "response": resp.content[:100]}
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Provider test failed: {error_msg}\n{traceback.format_exc()}")
        return {"success": False, "error": error_msg}


@app.post("/api/vision/test")
async def test_vision(request: Request):
    """测试视觉模型连接（与 LLM 相同创建方式，但用 image.vision 独立配置）"""
    try:
        data = await request.json()
        api_key = (data.get("api_key") or "").strip()
        if api_key.startswith('*'):
            api_key = str(
                load_config().get("image", {}).get("vision", {}).get("api_key", "")
            ).strip()
        if not api_key:
            return {"success": False, "error": "API Key 不能为空"}
        provider_type = data.get("provider_type") or "anthropic"
        base_url = (data.get("base_url") or "").strip()
        model = (data.get("model") or "").strip()
        if not base_url or not model:
            return {"success": False, "error": "Base URL 和模型不能为空"}

        from modules.llm.openai_provider import create_provider
        from modules.llm.base import ChatRequest, ChatMessage
        test_provider = create_provider(
            provider_type,
            {
                'api_key': api_key,
                'base_url': base_url,
                'model': model,
                'timeout': 30,
                'provider_type': provider_type,
            },
        )
        resp = await test_provider.chat(ChatRequest(
            messages=[ChatMessage(role="user", content="Hi, respond with OK")],
            temperature=0.1,
            max_tokens=10,
        ))
        return {"success": True, "response": resp.content[:100]}
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Vision test failed: {error_msg}\n{traceback.format_exc()}")
        return {"success": False, "error": error_msg}


@app.post("/api/test/generate")
async def test_generate(request: Request):
    if not bot_instance:
        return {"success": False, "error": "Bot not initialized"}
    data = await request.json()
    message = data.get("message", "")
    if not message:
        return {"success": False, "error": "Message required"}
    try:
        reply = await bot_instance.reply_generator.generate(
            context_prompt="[测试模式]",
            current_message=message,
            emotional_state=(
                bot_instance.emotional_manager.get_state("test")
                if bot_instance.emotional_manager.enabled
                else None
            )
        )
        return {"success": True, "reply": reply}
    except Exception as e:
        return {"success": False, "error": str(e) if e else "Unknown error"}


@app.get("/api/logs")
async def get_logs():
    # 必须返回 text/plain 而非裸 str，否则 FastAPI 会按 JSON 转义把换行变成字面 \n。
    for log_file in (BASE_DIR / "logs" / "bot.log", Path("/app/logs/bot.log")):
        if log_file.exists():
            lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()[-200:]
            return Response(content="\n".join(lines), media_type="text/plain; charset=utf-8")
    text = "\n".join(list(LOG_BUFFER)[-200:]) or "暂无日志"
    return Response(content=text, media_type="text/plain; charset=utf-8")


@app.post("/api/restart")
async def restart():
    import subprocess
    subprocess.Popen(["docker", "restart", "group-chat-bot"])
    return {"success": True}


def run_dashboard(bot, host: str = "0.0.0.0", port: int = 30080):
    set_bot(bot)
    logger.info(f"Starting dashboard on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

"""
爱丽丝 (Alice) - Web 管理面板
"""
import logging
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

app = FastAPI(title="爱丽丝 - Alice")
bot_instance = None


def set_bot(bot):
    global bot_instance
    bot_instance = bot


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


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
        for name, provider in config['llm'].items():
            if isinstance(provider, dict) and 'api_key' in provider:
                provider['api_key'] = '********' if provider['api_key'] else ''
    if 'memory' in config and isinstance(config.get('memory'), dict):
        emb = config['memory'].get('embedding')
        if isinstance(emb, dict) and emb.get('api_key'):
            emb['api_key'] = '********'
    return config


@app.post("/api/config")
async def update_config(request: Request):
    try:
        data = await request.json()
        current = load_config()

        # LLM 配置 - 支持多 provider
        if 'llm' in data:
            llm = current.setdefault('llm', {})
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
            for key in ['ws_url', 'access_token', 'self_id']:
                if key in data['qq']:
                    qq[key] = data['qq'][key]

        # 发言设置
        if 'speaking' in data:
            current['speaking'] = {**current.get('speaking', {}), **data['speaking']}

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
            current['memory'] = {**current_mem, **data['memory']}

        # Bot 基本信息
        if 'bot' in data:
            current['bot'] = {**current.get('bot', {}), **data['bot']}

        # 人格配置
        if 'personality' in data:
            current['personality'] = {**current.get('personality', {}), **data['personality']}

        # 群组设置
        if 'groups' in data:
            current['groups'] = data['groups']

        # 情感系统设置
        if 'emotion' in data:
            current['emotion'] = {**current.get('emotion', {}), **data['emotion']}

        # 注意力系统设置
        if 'attention' in data:
            current['attention'] = {**current.get('attention', {}), **data['attention']}

        # 疲劳系统设置
        if 'fatigue' in data:
            current['fatigue'] = {**current.get('fatigue', {}), **data['fatigue']}

        # 冷却系统设置
        if 'cooldown' in data:
            current['cooldown'] = {**current.get('cooldown', {}), **data['cooldown']}

        # 思考延迟设置
        if 'thinking' in data:
            current['thinking'] = {**current.get('thinking', {}), **data['thinking']}

        # 打字风格设置
        if 'typing_style' in data:
            current['typing_style'] = {**current.get('typing_style', {}), **data['typing_style']}

        save_config(current)
        return {"success": True, "message": "配置已保存"}
    except Exception as e:
        return {"success": False, "error": str(e) if e else "Unknown error"}


@app.get("/api/status")
async def get_status():
    if not bot_instance:
        return {"status": "not_initialized", "running": False}
    return {
        "status": "running" if bot_instance._running else "stopped",
        "running": bot_instance._running,
        "qq": {
            "connected": bot_instance.qq_adapter.is_connected if bot_instance.qq_adapter else False,
            "messages_sent": bot_instance.qq_adapter.messages_sent if bot_instance.qq_adapter else 0,
            "messages_received": bot_instance.qq_adapter.messages_received if bot_instance.qq_adapter else 0,
        }
    }


@app.get("/api/personality")
async def get_personality():
    if not bot_instance or not bot_instance.personality:
        # 返回默认人格配置
        return {
            "name": "爱丽丝",
            "nickname": "小艾",
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
            "description": ""
        }
    return {
        "name": bot_instance.personality.name,
        "nickname": bot_instance.personality.nickname,
        "background": bot_instance.personality.background,
        "traits": bot_instance.personality.traits,
        "interested_topics": bot_instance.personality.interested_topics,
        "bored_topics": bot_instance.personality.bored_topics,
        "description": bot_instance.personality.description if hasattr(bot_instance.personality, 'description') else ""
    }


@app.get("/api/sessions")
async def get_sessions():
    if not bot_instance:
        return {"sessions": []}
    return {"sessions": list(bot_instance.context_manager._windows.keys())}


@app.get("/api/context/{session_id}")
async def get_context(session_id: str):
    if not bot_instance:
        return {"messages": []}
    window = bot_instance.context_manager.get_window(session_id)
    messages = window.get_recent(100)
    return {"messages": [{"sender": m.sender_name, "content": m.content, "is_bot": m.is_bot, "timestamp": m.timestamp.isoformat()} for m in messages]}


@app.get("/api/memories")
async def get_memories(session: str = "", q: str = "", limit: int = 100):
    """获取长期记忆列表（支持按会话/关键词过滤，含时间衰减后的有效重要性）"""
    if not bot_instance or not bot_instance.memory_storage:
        return {"memories": [], "total": 0}

    from datetime import datetime

    try:
        all_mem = await bot_instance.memory_storage.get_all(limit=1000)
    except Exception as e:
        logger.error(f"Failed to load memories: {e}")
        return {"memories": [], "total": 0}

    now = datetime.now()
    result = []
    q_lower = (q or "").strip().lower()
    for m in all_mem:
        if m.memory_type not in ("episodic", "semantic"):
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
            "created_at": m.created_at.isoformat(),
            "last_accessed": m.last_accessed.isoformat(),
            "source_session": m.source_session,
            "sender_name": (m.metadata or {}).get("sender_name", ""),
        })

    # 默认按有效重要性排序
    result.sort(key=lambda x: x["effective_importance"], reverse=True)
    result = result[:limit]
    return {"memories": result, "total": len(result)}


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
        if not api_key:
            return {"success": False, "error": "API Key 不能为空"}
        model = data.get("embed_model") or "Qwen/Qwen3-Embedding-8B"

        from modules.memory.embedding import EmbeddingRerankService
        svc = EmbeddingRerankService({
            "api_key": api_key,
            "embed_model": model,
            "rerank_model": data.get("rerank_model") or "Pro/BAAI/bge-reranker-v2-m3",
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
    return bot_instance.emotional_manager.get_state(session_id).to_dict()


@app.get("/api/providers")
async def get_providers():
    """获取所有 LLM providers"""
    config = load_config()
    providers = []
    for name, provider in config.get('llm', {}).items():
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
        llm = current.setdefault('llm', {})

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
        llm = current.get('llm', {})

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
        provider_config = current.get('llm', {}).get(name)
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
            emotional_state=bot_instance.emotional_manager.get_state("test")
        )
        return {"success": True, "reply": reply}
    except Exception as e:
        return {"success": False, "error": str(e) if e else "Unknown error"}


@app.get("/api/logs")
async def get_logs():
    log_file = Path("/app/logs/bot.log")
    if log_file.exists():
        lines = log_file.read_text(encoding='utf-8').splitlines()[-200:]
        return "\n".join(lines)
    return "暂无日志"


@app.post("/api/restart")
async def restart():
    import subprocess
    subprocess.Popen(["docker", "restart", "group-chat-bot"])
    return {"success": True}


def run_dashboard(bot, host: str = "0.0.0.0", port: int = 30080):
    set_bot(bot)
    logger.info(f"Starting dashboard on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

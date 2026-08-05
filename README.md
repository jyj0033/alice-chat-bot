# 爱丽丝 (Alice) - 群聊 AI 机器人

一个模拟真实人类发言行为的 QQ 群聊 AI 机器人，基于 LLM 大语言模型驱动。

## 特性

- 🎭 **人格模拟** - 独特的人设和说话风格，不是每条消息都回复
- 🧠 **社交感知** - 智能判断何时该发言、何时该潜水
- 💭 **情感系统** - 有情绪变化，会开心、会烦、会傲娇
- 👀 **注意力机制** - 对不同用户有不同的关注度
- ⌨️ **打字风格** - 模拟人类打字习惯，包括偶尔的错字
- 🔄 **记忆系统** - 记住对话上下文和重要信息
- ⚡ **冷却机制** - 防止过度发言，保持自然感

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- NapCat (OneBot v11 QQ 机器人)
- LLM API (SiliconFlow / OpenAI / Claude 等)

### 配置

1. 复制配置文件：
```bash
cp config/config.example.yaml config/config.yaml
```

2. 编辑 `config/config.yaml`，填入你的配置：
   - LLM API Key
   - QQ Bot 信息
   - NapCat WebSocket 地址

3. 配置 NapCat（OneBot v11）并确保 WebSocket 可访问

### 启动

```bash
docker-compose up -d
```

访问管理面板：http://localhost:30080

## 项目结构

```
alice-chat-bot/
├── main.py                 # 主入口
├── config/                 # 配置文件
│   └── config.example.yaml # 配置示例
├── modules/
│   ├── llm/               # LLM 接口
│   │   ├── base.py        # 基类
│   │   ├── openai_provider.py
│   │   └── claude_provider.py
│   ├── social/            # 社交系统
│   │   ├── awareness.py   # 社交感知
│   │   ├── enhanced_decider.py  # 发言决策
│   │   ├── fatigue.py     # 疲劳系统
│   │   └── emotion.py     # 情感管理
│   ├── memory/            # 记忆系统
│   │   ├── context_manager.py
│   │   └── vector_memory.py
│   └── reply/             # 回复生成
├── core/
│   └── adapter/           # 平台适配器
│       └── qq_adapter.py  # QQ 适配器
├── dashboard.py           # Web 管理面板
└── docker-compose.yml     # Docker 部署
```

## 配置说明

### LLM 配置

```yaml
llm:
  primary:
    api_key: "your-api-key"
    base_url: "https://api.siliconflow.cn/v1"
    model: "deepseek-ai/DeepSeek-V3"
    provider_type: "openai_compatible"
```

支持以下 provider_type：
- `openai_compatible` - OpenAI 兼容格式
- `anthropic` - Claude 兼容格式

### QQ 配置

```yaml
qq:
  ws_host: "0.0.0.0"
  ws_port: 3001
  self_id: "123456789"  # Bot QQ 号
```

## 管理面板

- 状态监控
- 人格配置
- 对话上下文查看
- 情感状态查看
- LLM Provider 管理与测试
- 实时日志

## 许可

MIT License

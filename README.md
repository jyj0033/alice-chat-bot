# 爱丽丝 (Alice) - 群聊 AI 机器人

一个模拟真实人类发言行为的 QQ 群聊 AI 机器人，基于 LLM 大语言模型驱动。

## 特性

- 🎭 **人格模拟** - 独特的人设和说话风格，不是每条消息都回复
- 🧠 **社交感知** - 智能判断何时该发言、何时该潜水
- 🗣️ **群聊发言权** - 识别两人对聊、消息爆发和话题切换，避免抢话或回复过期消息
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
│   │   ├── conversation_floor.py # 群聊发言权与行为计划
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

### 富媒体消息

图片、视频、小程序、网页链接和合并转发会先转换为安全的短语义，再进入群聊上下文：

- 图片/视频的临时 URL 不会写进提示词或长期记忆；
- 小程序和 JSON 卡片只提取标题、来源等白名单字段；
- 合并转发通过 `get_forward_msg` 展开，默认最多保留 12 个节点；
- 转发、卡片和网页中的昵称、`@`、问题不会触发机器人；
- 无人提问的链接、卡片和转发默认不插话，纯图片/视频只可能偶尔短反应；
- 明确对机器人发送网页链接时才抓取标题，且会拒绝内网地址和非标准端口。

可选配置：

```yaml
rich_media:
  enabled: true
  forward:
    enabled: true
    expand_when_undirected: true
    max_nodes: 12
    max_chars: 600
    timeout: 5.0
  links:
    enabled: true
    directed_only: true
    timeout: 3.0
    max_bytes: 262144
    max_redirects: 3
    cache_ttl: 1800
  image:
    # NapCat OCR 只识别图片文字，不等同于视觉理解；默认关闭。
    ocr_enabled: false
    ocr_action: "ocr_image"
    ocr_timeout: 5.0
```

当前没有配置视觉模型时，普通图片和视频会诚实保留为 `[图片]`、`[视频]`，机器人不会猜测画面。视频不会自动下载或抽帧，避免在群聊中产生高延迟和大流量。

## 管理面板

- 状态监控
- 人格配置
- 对话上下文查看
- 情感状态查看
- LLM Provider 管理与测试
- 实时日志

## 许可

MIT License

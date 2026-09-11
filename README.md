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
- 🖼️ **表情包素材库** - 本地去重收集、分类管理和按语境发送表情包
- 📊 **群聊日报** - 统计活跃度，提炼话题、金句和活跃成员
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
│   ├── meme_manager.py    # 表情包本地存储、收集与选择
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

### 群聊目标判断

群聊消息会先由一个短 JSON 判断调用结合发送者、@、引用和最近上下文判断“主要在对谁说、Bot 是否应该接话”。这不是思维链，也不会让模型生成隐藏推理；判断失败时会保守回退到旧规则。

```yaml
conversation_judge:
  enabled: true
  provider_id: primary
  timeout: 8.0
  max_tokens: 220
  context_messages: 16
```

该配置也可以在管理面板的“高级行为”中修改；保存后目标判断器会在当前进程中立即替换。

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

- 状态监控（连接、会话、后台任务、富媒体处理计数和当前情绪）
- 人格配置（Big Five、背景、兴趣/禁忌、口头禅和 Emoji）
- 对话上下文查看
- 分群启停和独立发言概率
- 拟人行为配置（思考延迟、说话风格、注意力、情绪、疲劳与冷却）
- 群聊目标判断、发言权、富媒体、滚动纪要和记忆检索配置
- 群聊日报开关、自动发送时间和原始素材保留期配置
- 表情包素材上传、分类、搜索、删除、手动发送和自动收集配置
- 长期记忆搜索与删除、嵌入/重排服务配置
- LLM Provider 管理与测试
- QQ WebSocket 监听地址与端口配置
- 实时日志

群聊黑话默认每天自动审核一次自动提取的词条，参考最近 7 天群聊，清理明显没有依据、释义不符或属于通用词的记录。手动添加或人工编辑过的词条不会被自动删除。可在配置中调整：

```yaml
memory:
  enable_long_term_memory: true
  # 默认不把个人信息和群聊内容带到其他会话；确有统一记忆需求时再开启
  share_across_sessions: false
  slang:
    cleanup:
      enabled: true
      interval_hours: 24
      lookback_hours: 168
      max_entries: 40
```

### 表情包素材库

表情包采用轻量本地集成：图片保存在 `data/memes/`，清单保存在同目录的 `catalog.json`，按 SHA-256 去重，不依赖云图床或额外向量服务。每张素材除了分类，还可以填写“大致含义 / 适用场景”，用于后台理解、搜索和自动选图。管理面板的“表情包”页面支持上传、分类、新建分类、搜索、分页预览、修改含义、删除和选择群聊发送。

自动收集默认关闭。开启后，Bot 会在富媒体安全下载完成后，后台收集市场表情，以及带有明显表情/梗图信号的普通图片；能根据识别摘要或发送文字判断为开心、无语、吐槽、鼓励、卖萌、震惊时会直接归入对应分类，只有语义不足或冲突时才放入“待整理”，并把图片识别摘要作为待确认的含义。普通图片默认不会全部收集；截图、码图、过长手机界面图和无语义图片会被护栏跳过。可以在管理面板中谨慎开启普通图片收集，并限制群号、每日数量和同群冷却时间。自动发送也默认关闭，开启后只有当回复生成器判断确实适合并根据分类、含义和短编号选择了素材时才会发送一张，不会把图片能力变成每轮固定动作。

```yaml
meme_manager:
  enabled: true
  storage_path: data/memes
  auto_collect_enabled: false
  auto_send_enabled: false
  collect_private: false
  collect_scope: []
  collect_plain_images: false
  skip_screenshots: true
  min_collect_dimension: 64
  max_collect_dimension: 2400
  max_collect_pixels: 6000000
  default_category: 待整理
  max_image_bytes: 8388608
  max_images_per_message: 2
  daily_collect_limit: 80
  collect_cooldown_seconds: 15
  auto_send_cooldown_seconds: 60
```

表情库配置在管理面板保存后，开关、收集范围、每日上限和冷却会立即作用于当前进程；存储路径仍建议重启后生效。自动收集只复用现有富媒体下载和 OneBot 图片发送链路，不会把图片 URL、原始二进制或内部选择标记写进普通对话提示词。

### 群聊日报

日报采用轻量化实现：完整群消息单独保存，不进入 Bot 的普通记忆召回；生成时先做本地统计，再让当前 LLM 以 Bot 的第一人称提炼标题、副标题、带参与者和过程的今日话题、群友画像、原话金句和聊天质量锐评。图片由本地 Pillow 绘制，不引入浏览器、漫画或额外平台依赖。

日报只通过管理面板中的“立即生成”按钮或后台定时任务触发，群内聊天不会触发日报。每次只分析当天 00:00 至今的消息，默认至少需要 10 条群友消息。报告会渲染成明亮、梦幻、圆角优先的群聊小记 PNG，使用 Lucide 风格线性图标、柔和气泡和手帐式卡片，包含日记摘要、话题过程、群友画像、金句、最多 5 条“逆天语录”和聊天质量锐评；逆天语录必须能回溯到真实群消息，并按逆天程度排序，素材不足时宁缺毋滥。展示层不使用表情符号，长文本会自动换行或截断，图片生成失败时回退为文本。LLM 不可用时仍会发送 Bot 口吻的本地统计结果。报告以 Bot 自己看到和感受到的口吻生成，不采用上帝视角的客观新闻稿口吻。日报会按 QQ 号获取并缓存群友头像到 `data/group_analysis/avatars`，暂时取不到时使用姓名首字占位头像，不会阻塞日报生成。

自动日报默认关闭，打开后可配置：

```yaml
memory:
  group_analysis:
    enabled: true
    auto_enabled: false
    auto_times: ["23:50"]
    min_messages: 10
    max_messages: 500
    retention_days: 30
    max_tokens: 1800
    max_report_chars: 6000
    avatars_enabled: true
    avatar_cache_days: 7
    avatar_max_count: 12
    avatar_timeout: 6
```

头像默认通过 QQ 公共头像缩略图接口获取；如果网络环境无法访问，可在 `qq.avatar_url_template` 配置自定义模板，模板支持 `{user_id}`、`{uid}` 和 `{size}` 占位符。

配置保存到 `config/config.yaml`（敏感字段会单独保存到同目录的 `secrets.yaml`）。目标判断、表情库和富媒体相关设置会在支持的组件上热更新，其余设置按页面提示在重启后完全生效。

## 许可

MIT License

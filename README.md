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
- 📊 **群聊日报** - 统计活跃度，提炼话题、逆天语录和活跃成员
- ⚡ **冷却机制** - 防止过度发言，保持自然感

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- NapCat (OneBot v11 QQ 机器人)，或 QQ 开放平台官方机器人
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

4. 为管理面板设置 Basic Auth 凭据，在项目根目录创建未提交的 `.env`：
```dotenv
ALICE_DASHBOARD_USERNAME=admin
ALICE_DASHBOARD_PASSWORD=替换为足够长的随机密码
```

### 启动

```bash
docker-compose up -d
```

访问管理面板：`http://localhost:30080`；部署到服务器后使用 `http://服务器公网IP:30080`，浏览器会提示输入上面的凭据。

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
│       ├── qq_adapter.py  # QQ 适配器（NapCat / OneBot v11）
│       └── official_adapter.py  # QQ 开放平台官方机器人适配器（不登录 QQ 号）
├── dashboard.py           # Web 管理面板
└── docker-compose.yml     # Docker 部署
```

## 配置说明

### LLM 配置

```yaml
llm:
  primary:
    api_key: "your-api-key"
    base_url: "https://api.minimax.cn/v1"
    model: "MiniMax-M3"
    provider_type: "openai_compatible"
```

支持以下 provider_type：
- `openai_compatible` - OpenAI 兼容格式
- `anthropic` - Claude 兼容格式（仅为旧配置保留；MiniMax 使用 OpenAI 兼容格式）

MiniMax 统一使用 `https://api.minimax.cn/v1`，包括普通对话、搜索和图片识别。

各个 LLM 调用点可以复用不同的 Provider。`llm_routing` 只填写 Provider 名称，留空就沿用当前激活的主模型；也可以在 Web 管理面板的“LLM 模型管理 → 功能模型分工”中选择：

```yaml
llm_routing:
  reply: primary                 # 普通回复
  conversation_judge: primary    # 消息意图分析
  reply_review:                  # 回复复核
  meme_review:                   # 表情复核
  search:                        # 搜索判断和带资料回复
  vision:                        # 图片识别
  digest:                        # 群聊纪要
  group_analysis:                # 群聊日报
  profile:                       # 用户画像
  mbti:                          # MBTI 分析
  slang:                         # 黑话提取和审核
  expression_learning:          # 表达习惯学习
```

功能路由只引用已有 Provider，不会复制 API Key 或模型参数；搜索和视觉模型原有的独立配置仍可继续使用，显式选择功能路由后才会覆盖对应入口。

### 分群人格

同一个 Bot 可以在不同群扮演不同角色。三种写法，任选：

```yaml
groups:
  "<群号或 group_openid>":
    enabled: true
    speaking_probability: 0.05
    # 写法一：引用模板文件 config/personas/<名字>.yaml（最省事）
    persona_template: suomi
    # 写法二：整段提示词，优先级最高
    persona_prompt: |
      你叫阿蓝，是这群的技术顾问。说话克制，不主动卖萌，回答偏短。
    # 写法三：字段覆盖，与全局 personality 浅合并（traits 深合并）
    personality:
      name: 阿蓝
      background: 你是这群的技术顾问，说话克制。
      traits:
        agreeableness: 0.3
    # 说话风格也能按群覆盖，字段同全局 personality.speaking_style
    speaking_style:
      max_reply_length: 40          # 插话时的单条上限
      direct_max_reply_length: 120  # 明确问 bot 时的上限
```

`persona_template` 会在 `config/personas/` 下找同名 yaml，读 `persona_prompt` 和
`speaking_style`；群内写了 `persona_prompt` 或 `speaking_style` 时以群内为准。
模板名只允许字母、数字、下划线、短横线，防止路径穿越。

没配 `persona_prompt`、`persona_template` 也没配 `personality` 的群，继续使用全局人格。
管理面板「发言设置 → 群组设置」里的「人格」按钮可以直接编辑 `persona_prompt`
和两个长度上限，也能从下拉里载入 `config/personas/` 下的现成模板；空着表示跟随全局。

**新群怎么配**：官方通道下群标识是 `group_openid`，没法从 QQ 群号反推，只能等群里
有消息推过来。所以流程是「把 Bot 拉进群 → 群里 @ 它一句 → 打开面板刷新」，这个群会以
「未配置 · 见过 N 条消息」的形式自动出现在群组列表里，点「人格」载入模板保存即可。
没动过的自动发现行不会写进配置，不会偷偷改动它们的发言概率。

**隔离范围**：

| 维度 | 隔离方式 |
|---|---|
| 对话上下文 / 记忆 | 按 session（`group_xxx`） |
| 用户画像 | 按 `(sender_id, 群)` |
| 情绪 | 按 session，`EmotionalManager._states` |
| 注意力 | 按群，`AttentionManager._group_states` |
| 疲劳 | 按 session，`FatigueManager._session_states` |
| 发言权 | 无状态，按本群最近消息实时算 |
| 发言概率 | 按群配置 `speaking_probability` |
| 人格提示词 | 按群 `persona_prompt` / `persona_template` / `personality` |
| 说话风格 | 按群 `speaking_style` |

官方通道下群号是 32 位 `group_openid`，不是 QQ 群号。

### 新群接老数据

把人拉进新群之后，人和历史都在，但「画像」是按群存的，所以新群里对同一个人
一无所知。用 `scripts/match_group_people.py` 补上：

```bash
# 1) 先看能认出谁（默认 dry-run，不改数据）
docker exec alice-chat-bot python /app/scripts/match_group_people.py \
    --group 0EED38B0F55E000F60E601F536A71116

# 2) 认人 + 绑定 openid + 把画像复制到新群
docker exec alice-chat-bot python /app/scripts/match_group_people.py \
    --group 0EED38B0F55E000F60E601F536A71116 --apply

# 3) 顺便带一部分原始历史过去（每人最近 N 条）
docker exec alice-chat-bot python /app/scripts/match_group_people.py \
    --group 0EED38B0F55E000F60E601F536A71116 --apply --copy-history 30
```

它做三件事：

| 步骤 | 行为 |
|---|---|
| 认人 | 用群昵称比对别处的画像和 `person_aliases` 别名表，只认「唯一候选」 |
| 绑定 | 把 QQ 号的历史记忆改写到 openid，写回 `data/person_id_map.json` |
| 移植画像 | 把别处的画像**复制**一份到目标群，原件保留（只复制不搬走） |

没匹配上的（昵称改过、或者从没在别的群发过言）会跳过，等他再说话重跑一次即可。
多人同名会有多候选，脚本不猜，需要人工在面板里指定。

### 群会话映射（NapCat 数字群号 → group_openid）

一个人的 openid 在同一个 AppID 下是固定的，但**群**的标识从数字群号换成了
`group_openid`，两者无法互相推导，只能按成员构成人工对一次。对错了会把历史
挂到别的群名下，表现是「群里的人它一个都不认识」。

对的办法是看**独有成员**：只在某个老群里出现过的人，如果开始在新群发言，就能
确定这个新群对应哪个老群。`scripts/fix_group_mapping.py` 是纠错用的，里面写死
了本项目当前的对应关系：

```python
MENGYUE   = "group_2E654444..."   # 盟约（官方）
SHOUHU    = "group_0EED38B0..."   # 守护全世界最好的玲（官方）
OLD_724   = "group_724786753"     # → 守护玲
OLD_333   = "group_333863057"     # → 盟约
```

默认 dry-run，`--apply` 会先整库备份。当前核对结果：

| 官方群 | 群名 | 成员数 | 对应 NapCat 群号 | 记忆条数 |
|---|---|---|---|---|
| `2E654444…` | 盟约 | 44 | 333863057 | 5638 |
| `0EED38B0…` | 守护全世界最好的玲 | 36 | 724786753 | 18205 |

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
- 人格配置（Big Five、背景、兴趣/禁忌和幽默风格；不依赖固定口头禅或 Emoji）
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

日报采用轻量化实现：完整群消息单独保存，不进入 Bot 的普通记忆召回；生成时先做本地统计，再让当前 LLM 以 Bot 的第一人称提炼标题、副标题、带参与者和过程的今日话题、群友画像、原话逆天语录和聊天质量锐评。图片由本地 Pillow 绘制，不引入浏览器、漫画或额外平台依赖。

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

## QQ 通道：NapCat 与官方机器人

`qq.platform` 决定走哪条通道，两条通道对外接口一致，上层业务代码不用区分：

- `napcat`（默认）：登录一个 QQ 号，走 OneBot v11 协议。账号可能被风控掐断，
  需要人工扫码恢复。
- `official`：QQ 开放平台官方机器人，只用 `AppID` / `AppSecret`，
  **不登录任何 QQ 号**，因此不存在账号掉线；代价是群成员/QQ 号拿不到、
  没有戳一戳和合并转发、主动消息受用户开关限制。

```yaml
qq:
  platform: official
  self_id: "<AppID>"
  official:
    app_id: "<AppID>"
    app_secret: "<AppSecret>"
```

完整的 ID 体系映射、功能降级对照、部署与验收步骤见
[docs/official-qq-platform.md](docs/official-qq-platform.md)。

## 许可

MIT License

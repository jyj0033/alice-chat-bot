# Docker 部署说明

## 快速启动

### 1. 准备配置文件

创建 `config/config.yaml`：

```yaml
llm:
  primary:
    provider_type: "openai_compatible"
    api_key: "your-api-key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    enabled: true

qq:
  ws_host: "0.0.0.0"  # Bot 作为 WebSocket 服务端
  ws_port: 3001
  self_id: "你的QQ号"

bot:
  nickname: "小明"

speaking:
  trigger_keywords:
    - "小明"
    - "bot"
```

### 2. 构建并启动

在项目根目录创建 `.env`，设置管理面板账号和密码：

```dotenv
ALICE_DASHBOARD_USERNAME=admin
ALICE_DASHBOARD_PASSWORD=替换为足够长的随机密码
```

密码未配置时，管理面板会拒绝请求。

```bash
# 启动（仅Bot + Dashboard）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 3. 访问 Dashboard

```
http://<服务器公网IP>:30080
```

浏览器会弹出 Basic Auth 凭据框。账号密码从项目根目录 `.env` 读取；首次部署前请设置强密码，并在云服务器安全组/防火墙开放 TCP/30080。

### 4. LLM 配置：统一走 Responses 通道

所有模型（主模型、回复模型、视觉模型）统一使用 `provider_type: responses`：

```yaml
llm:
  primary:
    provider_type: "responses"
    base_url: "https://api.minimax.cn/v1"
    model: "MiniMax-M3.1-Flash"
    reasoning_effort: "low"     # low 是端点接受的最低档；none 会被拒绝
    enabled: true
```

`reasoning_effort` 不是可选项而是保命参数：判断器、视觉识别、日报这类「输出必须是
JSON、预算又小」的任务，如果思考链把 token 预算吃光，正文会被截断，解析器只能按设计
拒绝写残缺 JSON，最终退化成「看不清画面」这类占位符。`effort: low` 把它压到最低。

### 5. 联网检索走模型自带的服务端搜索

**外部搜索后端（博查 / 豆包）已下线并从代码中移除**，`config.yaml` 里不再有 `search:`
段，Dashboard 也不再暴露对应配置项。现在联网只有一条路：主模型的服务端 `web_search` 工具。

```yaml
llm:
  qianyemeng:
    provider_type: "responses"
    web_search: true        # 该 Provider 支持服务端搜索
```

按需声明：只有检索判断器认为这条消息需要查资料时，才会把 `{"type":"web_search"}` 加进
本次请求的 tools；闲聊不会触发联网。检索判断器同时负责本地资料库（全局热梗 / 游戏
资料库）的命中，两者是同一个闸门，不要关掉。

需要给判断器单独指定模型时，在 `llm_routing.search` 里填 Provider 名称（留空即复用
主模型）。

### 6. 引用回复策略

爱丽丝不是每条回复都挂引用。判定规则：

| 场景 | 是否引用 | 原因 |
|---|---|---|
| 对方 @ / 回复爱丽丝（含私聊） | 是 | 对方要知道她在回哪一句，不引用像自言自语 |
| 群聊插话 + 目标消息距今 ≥ 60 秒 | 是 | 对方可能已经忘了，引用帮定位 |
| 群聊插话 + 近 30 秒内 ≥5 条消息且 ≥2 个人在说 | 是 | 群里刷太快，指代容易糊，引用消歧 |
| 群聊插话 + 紧邻的即时对话 | 否 | 真人不会每条都挂引用 |
| 纯表情包回复 | 同上 | 和文字回复用同一套策略 |

阈值在 `config.yaml` 的 `reply.quote`：

```yaml
reply:
  quote:
    enabled: true           # 总开关；false = 完全不使用引用
    always: false           # true = 每次都引用（引入策略前的行为，用于对照）
    stale_seconds: 60
    busy_window_seconds: 30
    busy_message_count: 5
    busy_speaker_count: 2   # 一对一连发不算刷屏
```

一对一连发不算「刷屏」：一个人连发 5 条和 5 个人各发 1 条，对「她在回谁」的
指代难度完全不同，所以 busy 条件同时看条数和说话人数。

每次发送都会打一行 `[引用] 引用=是/否 原因=xxx`，长期记忆的 `metadata.quoted`
也记了真实是否引用过 —— 注意 `metadata.reply_to_id` 有兜底（没引用时记的是
触发消息），**拿它统计引用率会得到恒等于 100% 的假数字**，要用 `quoted`。

### 7. 日志

`docker compose up -d --force-recreate` 会重建容器，**`docker logs` 的历史会
随旧容器一起消失**。所以应用除了 stdout 还会写一份滚动文件日志到挂载卷：

```
logs/bot.log            # 当前日志（单文件上限 20MB）
logs/bot.log.1 ... .5   # 最多保留 5 个历史文件
```

跨重启查历史用文件日志，不要用 `docker logs --since`：

```bash
tail -f logs/bot.log
grep '\[引用\]' logs/bot.log | tail -20
```

需要保留某个时间窗口的 stdout 日志时，重建前先手动导出一份：

```bash
docker logs alice-chat-bot > logs/docker-$(date +%Y%m%d-%H%M%S).log 2>&1
```

---

## 带 NapCat 的完整部署

编辑 `docker-compose.yml` 取消 NapCat 注释：

```bash
docker-compose up -d
```

---

## 常用命令

```bash
# 重新构建镜像
docker-compose build --no-cache

# 进入容器调试
docker exec -it group_chat_bot /bin/bash

# 查看实时日志
docker-compose logs -f bot

# 重启Bot
docker-compose restart bot
```

---

## 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 30080 | HTTP Dashboard | 公网管理面板（Basic Auth） |
| 3001 | OneBot WebSocket | NapCat 反向 WebSocket 连接地址 |

NapCat 在另一台机器时，把反向 WebSocket 地址设置为
`ws://<Bot机器IP>:3001`，并确保安全组/防火墙允许该来源访问 3001。

## 防火墙

```bash
# Ubuntu/Debian
sudo ufw allow 30080
sudo ufw allow 3001

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=30080/tcp
sudo firewall-cmd --permanent --add-port=3001/tcp
sudo firewall-cmd --reload
```

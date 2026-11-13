# QQ 开放平台官方机器人通道（替换 NapCat）

本文对应分支 `feat/official-qq-platform`。目标是把接收/发送通道从
「登录一个 QQ 号 + NapCat/OneBot v11」换成「QQ 开放平台官方机器人 API v2」，
**不再登录任何 QQ 号**，从根上消除账号被风控掐断（`KickedOffLine`）这件事。

实现见 `core/adapter/official_adapter.py`，通道选择在 `main.py::_init_qq_adapter()`。
两条通道对外接口完全一致，上层业务（决策、记忆、词条、日报、表情库）一行未改。

---

## 1. 通道切换

```yaml
qq:
  platform: official        # napcat（默认） | official
  self_id: "<AppID>"        # 官方通道下这就是机器人的 AppID
  official:
    app_id: "<AppID>"
    app_secret: "<AppSecret>"   # 会自动落到 config/secrets.yaml（600）
    api_base: "https://api.bot.qq.com"
```

`app_secret` 已加入 `core/config_store.py::SECRET_KEYS`，保存时自动分离到
`secrets.yaml`，Web 面板读取时显示为 `********`。

改 `platform` 后**重启 bot 生效**（`apply_runtime_config` 只热更新凭证类字段，
不换适配器实例）。

---

## 2. ID 体系：QQ 号 → openid

官方通道没有 QQ 号，所有标识都是 openid，而且**每个 AppID 的 openid 都不同**
（官方文档明确说明，不跨 AppID 通用）。对应关系：

| 语义 | NapCat 通道 | 官方通道 |
|---|---|---|
| 群 | `group_id` = QQ 群号（数字） | `group_id` = `group_openid`（32 位十六进制） |
| 群成员 | `user_id` = QQ 号 | `sender_id` = `member_openid` |
| 单聊对象 | `user_id` = QQ 号 | `sender_id` = `user_openid` |
| 机器人自己 | `self_id` = QQ 号 | `self_id` = AppID |

会话标识仍是 `group_<openid>` / `private_<openid>`，上层逻辑不用改。

**`qq.self_id` 必须填 AppID**：上层多处拿 `qq.self_id` 和消息里的 id 比对
（判断「这条是不是在回复机器人」「这个 @ 是不是在 @ 我」）。官方通道下
`member_openid` 和 AppID 不是一个命名空间，混用会导致这些判断永远为假。
适配器内部统一把「机器人自己」映射成 AppID，调用方看到的就是自洽的。

### 2.1 必须迁移的配置：`groups`

`config.yaml` 的 `groups` 段原本按 QQ 群号做键：

```yaml
groups:
  "724786753": { enabled: true, ... }   # NapCat
```

官方通道下 `message.group_id` 是 `group_openid`，**这个键对不上**。
群没配到时的行为是「用默认配置（启用）」，不会崩，但按群做的差异化配置会失效。

拿到 openid 的办法：机器人被拉进群后发一条消息，看 bot 日志里
`[群聊]` 那行的会话前缀（`group_XXXX`），或直接看
`GET /v2/groups/{group_openid}/info`。

### 2.2 拿不到 QQ 号

`union_user_account` 默认返回空串，且要联系平台运营才能申请。
所以官方通道下**无法把 openid 映射回 QQ 号**，历史记忆里的 QQ 号身份对新
openid 不通用。新旧记忆不会互相污染，但也不会自动接上。

---

## 3. 协议要点（实现依据）

官方文档：<https://bot.q.qq.com/wiki/develop/api-v2/>

| 项 | 值 |
|---|---|
| OpenAPI 基址 | `https://api.bot.qq.com` |
| 鉴权头 | `Authorization: QQBot <access_token>` |
| 取凭证 | `POST /app/getAppAccessToken`，body `{"appId","clientSecret"}` → `{"access_token","expires_in"}`（7200s）。**失败时 HTTP 仍是 200**，用 body 里的 `code` 判断 |
| 普通接口错误体 | `{"err_code":N,"message":"...","trace_id":"..."}` |
| 网关地址 | `GET /gateway` → `{"url":"wss://..."}` |
| 网关协议 | op 10 Hello（心跳周期 ms）→ op 2 Identify（`token` 为 `QQBot <access_token>`，`intents`，`shard:[0,1]`）→ op 1 心跳（`d` 携带最新 `s`）→ op 11 ACK；op 6 Resume / op 7 Reconnect / op 9 Invalid Session |
| 事件订阅位 | `GROUP_AND_C2C_EVENT = 1 << 25 = 33554432`（覆盖 `GROUP_AT_MESSAGE_CREATE`、`GROUP_MESSAGE_CREATE`、`C2C_MESSAGE_CREATE`、好友/入群事件） |
| 发群消息 | `POST /v2/groups/{group_openid}/messages` |
| 发单聊 | `POST /v2/users/{user_openid}/messages` |
| 富媒体 | 先 `POST /v2/groups/{group_openid}/files` 拿 `file_info`，再 `msg_type=7` + `media.file_info`；单聊用 `/v2/users/{user_openid}/files`（**两边不通用**） |
| 被动回复 | 群聊 5 分钟内 5 次、单聊 60 分钟内 4 次；`msg_id` + `msg_seq`（同 msg_id 同 seq 重复发会失败） |
| 主动消息 | 用户可在客户端关闭「允许主动发送」，关了必失败；群 60/qpm（未认证 30/qpm）、单聊 10/qps |
| 引用回复 | `message_reference.message_id` 要的是 `REFIDX_xxx`，不是 message id：他人消息从事件 `message_scene.ext` 的 `msg_idx` 取，自己发的从发消息响应 `ext_info.ref_idx` 取 |

### 3.1 引用是怎么实现的

适配器同时维护 `message_id ↔ REFIDX` 双向索引：

* 收到消息时，从 `message_scene.ext` 的 `msg_idx=REFIDX_xxx` 登记自己的 REFIDX；
* 用户引用别人时，从 `msg_elements[0].msg_idx` 或 `ext` 的 `ref_msg_idx` 反查
  被引用消息的 message_id，填进 `Message.reply_to_id`；被引用消息的作者填进
  `Message.reply_to_qq`（若作者是机器人，统一换成 AppID）。
* 发送时：优先用**被动回复**（`msg_id` + `msg_seq`，QQ 里原生引用成一条消息，
  窗口期内）；超出窗口或对象没收到过，退化成 `message_reference` 主动引用。

**被引用的正文不进正文。** 官方事件把被引用的那条放在 `msg_elements[0].content`，
早期实现把它当成普通文本段拼在说话人自己的话后面，而渲染层是 `"".join(...)`
无分隔符拼接，结果两者黏成一句：

```
自己说的：爱丽丝，来点幕后黑手的小说
引用机器人的：灵魂原来还有分辨率的吗……？
存进记忆库：爱丽丝，来点幕后黑手的小说灵魂原来还有分辨率的吗……？
```

引用的是机器人自己时更糟——机器人的话被记成了群友说的话，检索出来再喂回
模型。NapCat 时代没有这个问题（引用是 `[CQ:reply]` 段，渲染时直接跳过，只存
说话人自己的话）。

现在的约定：

* `Message.content` / `outer_text` **只有说话人自己的话**；
* 被引用原文放 `Message.quoted_content`，写记忆时进 `metadata.quoted_content`，
  不进正文；
* 上下文渲染和目标判断仍能看到引用的原话：优先按 `reply_to_id` 回查窗口内
  那条消息，回查不到时用 `quoted_content` 兜底。

### 3.2 发送端频控（`core/adapter/rate_limit.py`）

官方规则：

| 维度 | 限制 |
|---|---|
| Bot（发送方） | 企业/个人实名认证 60/qpm，未认证 30/qpm |
| 单关系（接收方） | 20/qpm；每个群 1 天最多接收 1000 条 |
| 被动消息 | 5 分钟有效，每条消息最多回 5 次 |

本实现的本地阈值比官方更严（按未认证口径留余量）：
**Bot 25/分钟、单关系 15/分钟、单会话每天 900 条**。

两种发送区别对待：

* **被动回复**（body 里带 `msg_id`，有人在等它回话）：额度不够就排队等一下，
  最多等 5 秒；等不到就放弃这一条。
* **主动消息**（插话、群日报、表情包，不带 `msg_id`）：额度不够直接跳过，
  不排队 —— 主动发言晚几秒没人等，排队反而会拖慢主流程。

被官方拒收的限频错误码（`1100100` / `504xxx` / `40034100` / `40034105` /
`40054013` / `1100304`）会单独记一条 WARNING 并计数，面板
`/api/status` 的 `qq.rate_limit` 段能直接看到「本地拦下多少次、官方拒收多少次」。

阈值可配：`qq.official.rate_limit_bot_per_minute` /
`rate_limit_session_per_minute` / `rate_limit_daily_per_session` /
`rate_limit_wait_seconds`。

### 3.3 主动消息的开关状态（`bot_state` + 四个事件）

官方对「机器人能不能主动推消息」有两套机制，我们两边都接了：

| 来源 | 接口 / 事件 | 语义 |
|---|---|---|
| 查询 | `GET /v2/groups/{gid}/bot_state` | `allow_proactive_msg`（还能不能主动推）、`recv_msg_setting`（`all` / `only_mention` / `mention_and_context`）、`member_role`（`member` / `admin` / `owner`） |
| 事件 | `GROUP_MSG_REJECT` / `GROUP_MSG_RECEIVE` | 群管理员在机器人资料页**关/开消息接收**——关掉后这个群的消息官方不再投递，机器人整个静默 |
| 事件 | `C2C_MSG_REJECT` / `C2C_MSG_RECEIVE` | 用户在资料卡关/开**主动消息推送**——关掉后给这个用户发主动消息必被拒（40034105） |

这四个事件**都在 `GROUP_AND_C2C_EVENT (1<<25)` 里**，也就是我们本来就在订阅，
只是早期代码只打了「收到未处理的群事件」就丢掉了。现在接上了：

* 事件是实时推送，优先于 `bot_state` 的 5 分钟缓存；
* 关掉之后，主动消息（插话、日报、表情包）在**发出去之前**就被拦下来，
  抛 `ProactiveBlockedError`，不再去撞官方的拒收；
* **被动回复不受影响**——群里 @ 机器人是有人等着回话的，被动回复照发。

`bot_state` 的频控是 30 QPM，结果缓存 5 分钟。查不到（网络挂了、字段没返回）
时按「允许」处理：宁可让官方拒一次，也不要误杀正常消息。

### 3.4 Markdown 消息（`msg_type=2`）

官方支持用 Markdown 发消息，body 是**嵌套对象**：

```json
{"msg_type": 2, "markdown": {"content": "## 标题\n\n- 一条"}}
```

填了 `markdown` 之后 `content` 必须留空，否则撞 22006「消息类型与内容不匹配」。

适配器接口：`send_markdown_with_id(session_id, markdown, reply_to_id=None)` /
`send_markdown(session_id, markdown)`，被动回复的 `msg_id` / `msg_seq` 照常带。

**目前接在群日报上**。日报图片依赖字体和 Pillow，环境一变就可能挂，
Markdown 是文本那条命。日报的发送格式可配：

```yaml
memory:
  group_analysis:
    report_format: image   # image（默认，发图片）／markdown（发 Markdown 文本）／both（两条都发）
```

- `image`：图片发成功就不发文本；图片失败自动回退 Markdown，再失败回退纯文本；
- `markdown`：只发 Markdown 文本（不渲染图片）；
- `both`：图片 + Markdown 文本各一条。

---

## 4. 功能降级对照

| 功能 | NapCat | 官方通道 | 说明 |
|---|---|---|---|
| 收群消息（@机器人） | ✅ | ✅ | `GROUP_AT_MESSAGE_CREATE` |
| 收群消息（全量） | ✅ | ✅ | `GROUP_MESSAGE_CREATE`，需在开放平台开启「接收所有消息」 |
| 收单聊 | ✅ | ✅ | `C2C_MESSAGE_CREATE` |
| 发文本 | ✅ | ✅ | |
| 发图片 | ✅ | ✅ | 走 `file_info`，5MB 内二进制直传，更大走分片上传 |
| 一条消息多张图 | ✅ | ❌ | 官方无此能力，拆成多条 |
| 合并转发 | ✅ | ❌ | 官方无此接口，`mode="forward"` 自动降级为逐张发送 |
| 戳一戳 | ✅ | ❌ | 官方无戳一戳接口，`poke()` 返回 False |
| 撤回自己发的消息 | ✅ | ✅ | 官方限制 2 分钟内 |
| 群成员昵称 | ✅ | ⚠️ | 消息事件自带 `author.username` 可用；成员列表接口是白名单（11253） |
| 群黑名单 | ✅ | ❌ | 11253 白名单（内邀中） |
| 群禁言 | ✅ | ⚠️ | 查询/设置均可用，要求机器人是群管理员，最长 30 天、单次 10 人；代码尚未接 |
| 群信息 / 群人数 | ✅ | ✅ | `GET /v2/groups/{openid}/info`，实测无需白名单 |
| 群禁言 | ✅ | ⚠️ | 查询/设置均可用，要求机器人是群管理员，最长 30 天、单次 10 人；代码尚未接 |
| 入群审批 | ✅ | ✅ | 申请列表、审批、自动审批策略实测可用 |
| 主动消息开关状态 | ❌ | ✅ | `bot_state` + `GROUP_MSG_REJECT`/`C2C_MSG_REJECT` 事件；关掉后主动消息提前拦截，不去撞官方拒收 |
| 成员列表 / 批量踢人 | ✅ | ❌ | 11253 白名单（内邀中） |
| QQ 号 / 头像 | ✅ | ❌ | 官方拿不到 QQ 号；头像接口需要 QQ 号，恒为空 |
| 群历史消息 API | ✅ | ❌ | 官方没有；日报只能靠 bot 自建历史表，**切换后旧的历史查不到** |
| 主动插话 / 群日报 | ✅ | ⚠️ | 用户可关「允许主动发送」，关了主动消息必失败（现在会提前拦下并记日志） |
| Markdown 消息 | ❌ | ✅ | 官方 `msg_type=2`，已接（群日报用）；NapCat 侧没接 |

---

### 4.1 实测权限边界（2026-11-13 复测，两个群都跑了一遍）

**盟约**（44 人，Bot 是 **admin**）：

| 接口 | 结果 |
|---|---|
| `GET /v2/groups/{gid}/info` | ✅ 群名/简介/分类/标签/44 人 |
| `GET /v2/groups/{gid}/bot_state` | ✅ `member_role=admin`、`recv_msg_setting=all`、`allow_proactive_msg=true` |
| `GET /v2/groups/{gid}/restrict_chat_setting` | ✅ 返回全局禁言规则 |
| `GET /v2/groups/{gid}/join_request_list` | ✅ |
| `GET /v2/groups/join_approval_strategy` | ✅ |
| `GET /v2/menu` | ✅（当前为空） |
| `GET /v2/panels?scope=group` | ✅（当前为空） |
| `POST /v2/generate_url_link` | ✅ 能出分享链接 |
| `GET /v2/groups/{gid}/members` | ❌ 11253 |
| `GET /v2/groups/{gid}/members/{openid}` | ❌ 11253 |
| `GET /v2/groups/{gid}/member_blacklist` | ❌ 11253 |

**守护全世界最好的玲**（36 人，Bot 是 **member，不是管理员**）：

| 接口 | 结果 |
|---|---|
| `GET /v2/groups/{gid}/info` | ✅ 分类「游戏」 |
| `GET /v2/groups/{gid}/bot_state` | ✅ `member_role=member`、`recv_msg_setting=all`、`allow_proactive_msg=true` |
| `GET /v2/groups/{gid}/restrict_chat_setting` | ❌ 11703 机器人不是群管理员 |
| `GET /v2/groups/{gid}/join_request_list` | ❌ 11703 机器人不是群管理员 |
| `GET /v2/groups/{gid}/members` | ❌ 11253 |

结论：
1. 群管理（禁言、撤回他人消息、入群审批）**要求机器人在该群是管理员**，
   不是管理员的群一律 11703 —— 上面「盟约」和「守护玲」的差别就在这。
2. **成员名册类接口（成员列表 / 黑名单 / 批量踢人）需要白名单**，
   与机器人是否为管理员无关。
3. `GET /users/me` 文档里写了但实际返回 404「不支持的调用」，别依赖它。

## 5. 部署步骤

### 5.1 开放平台侧

1. <https://q.qq.com/#/> 注册 → 创建机器人 → 拿到 `AppID` / `AppSecret`；
2. 机器人**要过审核**才能在正式环境工作；开发期可用沙箱域名
   （把 `qq.official.api_base` 指向沙箱地址）；
3. 手动把机器人**添加到目标群**（官方机器人是独立身份，群里会看到两个机器人）；
4. 如需全量群消息，在「接收所有消息」开关里打开。

### 5.2 服务器侧

```bash
# 1. 代码（走 bundle 中转，服务器拉 GitHub 不稳）
# 2. 配置
sudo -H sh -c 'cd /root/alice-chat-bot && python3 - <<EOF
import yaml, pathlib
p = pathlib.Path("config/config.yaml")
d = yaml.safe_load(p.read_text(encoding="utf-8"))
d.setdefault("qq", {})
d["qq"]["platform"] = "official"
d["qq"]["self_id"] = "<AppID>"
d["qq"].setdefault("official", {})
d["qq"]["official"]["app_id"] = "<AppID>"
d["qq"]["official"]["app_secret"] = "<AppSecret>"
p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
EOF'
# app_secret 落 secrets.yaml 后记得 chmod 600

# 3. 重启
docker restart alice-chat-bot

# 4. 看日志
docker logs alice-chat-bot 2>&1 | grep -E '官方|access_token|网关|就绪'
```

### 5.3 身份迁移：让老画像接上新身份

官方通道的 `sender_id` 是 openid，NapCat 时代是 QQ 号。两者的记忆都在同一个
`memory.db` 里，只是主键对不上——不改写的话，同一个人会攒出两份互不相干的画像。

映射文件：`data/person_id_map.json`

```json
{
  "note": "NapCat 时代的 QQ 号 -> 官方 openid 映射。",
  "mapping": {
    "2399634605": "4B4CF9FDD307941C394E121A5786B274"
  }
}
```

```bash
# 预演：只看会改多少条，不写库
docker exec alice-chat-bot python3 /app/scripts/migrate_official_ids.py --dry-run --auto-match

# 执行（自动整库备份到 data/memory.db.bak.identity-<时间>）
docker exec alice-chat-bot python3 /app/scripts/migrate_official_ids.py --apply
```

改写的范围：`memories.metadata.sender_id`、`person_aliases.qq_id`、
`profile_suppressions.sender_id`、`profile_attempts.sender_id`。只改 ID，
记忆内容和画像文本一字不动，可重复执行。

**日常用面板，不用手改 JSON**：管理面板侧栏「用户画像 → 身份对齐」
（`/identity`）会列出所有还没对上号的老画像，昵称能唯一命中的标「可自动匹配」，
点一下即完成绑定；同名多人或没在官方通道发过言的会标成对应状态，等他们开口后刷新即可。

`--auto-match` 只读，它拿老画像的昵称去官方通道已出现的 openid 里找同名候选，
不会自动写库——重名和改昵称都必须人来确认。

> 注意：官方通道拿不到 QQ 号（`union_user_account` 需单独申请且默认空），
> 所以不存在"自动全量对齐"，只能逐人确认一次。确认过的不会再问。

### 5.4 NapCat 下线（可选）
官方通道跑通后，NapCat 容器可以停掉：

```bash
cd /home/ubuntu/astrbot && docker compose stop napcat   # 先停，别删
```

**先别删容器和数据**：官方通道还有白名单接口没批下来时，回滚需要它。
确认官方通道稳定运行若干天后再做清理。

---

## 6. 验收清单

| 检查项 | 怎么验 |
|---|---|
| 凭证拿到了 | 日志出现「已获取官方 access_token（有效期 7200 秒）」 |
| 网关连上了 | 日志出现「官方网关已连接」+「官方通道就绪：机器人=xxx」 |
| 群里 @ 能收到 | bot 日志出现 `[群聊]` 行 |
| 回复能发出去 | 面板 `messages_sent` 递增；群里能看到机器人回复 |
| 引用渲染正确 | 回复出现在被 @ 的那条消息下面 |
| 图片能发 | 表情库/日报图片在群里正常显示 |
| 主动消息是否可用 | 用户未关闭「允许主动发送」时群日报能发出去 |
| 掉线问题消失 | 连续运行 24h 以上，无掉线、无重新登录 |

面板状态在 `qq` 子对象里：`connected` / `messages_received` / `messages_sent`。

---

## 7. 已知限制

* **没有 QQ 号**：所有历史记忆、画像按 openid 重新积累。
* **群日报依赖主动消息**：用户关掉「允许主动发送」就发不出去，只能降级为
  「收到 @ 时才回」。
* **群历史要自建**：官方没有历史消息接口，日报得靠 bot 自己落库的消息记录。
* **群成员信息拿不到**（除非拿到白名单）：@ 别人时无法显示对方群名片。
* **白名单接口**：群成员列表 / 黑名单 / 批量踢人 / `union_user_account` 都要
  另外申请，不保证给。
* **官方机器人是独立身份**，和原来的 QQ 号机器人不是同一个账号，
  群里的历史消息、好友关系都不共享。

---

## 8. 官方还有、我们没接的能力（2026-11-13 盘点）

读完全套 API v2 文档后列的。括号里是前置条件。

**读多花不了钱、可以优先做**：

| 能力 | 接口 / 事件 | 备注 |
|---|---|---|
| 群画像 | `GET /v2/groups/{gid}/info` | 除了群名，还有简介/分类/标签/人数，能直接喂给回复生成器 |
| 群指令面板 | `POST /v2/panels`（`scope=group`） | 群里出现面板，按钮点了填进输入框。**前置：要先实现命令处理**（现在 `/` `!` `#` 开头的消息会被直接跳过） |
| C2C 快捷菜单 | `PUT /v2/menu` | 仅单聊，最多 10 个按钮。`switch` 类型能在消息里带 `switch_id`，可做「允许插话」开关 |
| 消息反馈闭环 | `INTERACTION_CREATE` type=13 | 用户对机器人消息点赞/点踩，是免费的质量反馈信号 |
| 内嵌键盘 | 发消息 body 的 `keyboard` | 消息下面直接挂按钮。`action.type=2` + `enter=true` 点击即自动发送，**不需要 INTERACTION intent** |
| 单聊流式输出 | `POST /v2/users/{openid}/stream_messages` | 打字机效果，50 QPS |
| 互动召回消息 | 发消息 body 的 `is_wakeup=true` | 用户主动说过话后 30 天内可主动推 4 条，**不受「允许主动发送」开关限制** |
| 图文卡片 | 发消息 body `msg_type=8` | 标题+描述+图+链接 |
| 素材管理 | 上传后拿 `qqbot-media://` | 图片存平台，Markdown 里直接引用，不用每次发都传文件 |
| 分享链接 | `POST /v2/generate_url_link` | 带 `callback_data`，可做「扫码加好友自动带来源标记」 |
| @全体成员 | 正文里 `<qqbot-at-everyone />` | 机器人得是群管理员，每群每天 1 次 |

**要权限 / 要产品决策**：

| 能力 | 前置 |
|---|---|
| 群禁言 | `POST /v2/groups/{gid}/restrict_chat_setting`，机器人必须是管理员（盟约群可以，守护玲群不行） |
| 入群自动审批 | `POST /v2/groups/join_approval_strategy` + `strategy_id.execute` |
| 群成员增减事件 | 要订阅 `GROUP_MEMBER_EVENT (1<<24)`，**不在** `1<<25` 里 |
| 主动消息审核结果 | 要订阅 `MESSAGE_AUDIT (1<<27)` |
| 消息反馈 / 交互事件 | 要订阅 `INTERACTION (1<<26)` |

> ⚠️ **加 intents 有风险**：官方文档写明「鉴权时传递了无权限的 intents，
> websocket 会报错并直接关闭连接」。我们代码里 op 9 会清 session 重连，
> 权限不足可能变成**重连风暴**。要么先去申请权限，要么在 Identify 失败时
> 降级回基础 intents（`GROUP_AND_C2C_EVENT`）。


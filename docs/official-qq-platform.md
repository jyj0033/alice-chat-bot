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
| 群成员昵称 | ✅ | ⚠️ | 群成员接口是白名单能力（错误码 11253），拿不到就返回空串 |
| 群黑名单 / 禁言 | ✅ | ❌ | 官方侧未开放（内邀中） |
| QQ 号 / 头像 | ✅ | ❌ | 官方拿不到 QQ 号；头像接口需要 QQ 号，恒为空 |
| 群历史消息 API | ✅ | ❌ | 官方没有；日报只能靠 bot 自建历史表，**切换后旧的历史查不到** |
| 主动插话 / 群日报 | ✅ | ⚠️ | 用户可关「允许主动发送」，关了主动消息必失败 |
| Markdown 消息 | ❌ | ✅ | 官方 `msg_type=2` 支持，NapCat 侧没接 |

---

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

### 5.3 NapCat 下线（可选）

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
* **白名单接口**：群成员列表 / 黑名单 / 禁言 / `union_user_account` 都要
  另外申请，不保证给。
* **官方机器人是独立身份**，和原来的 QQ 号机器人不是同一个账号，
  群里的历史消息、好友关系都不共享。

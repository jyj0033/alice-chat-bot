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

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
http://localhost:30080
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
| 30080 | Web Dashboard | 浏览器访问 |
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

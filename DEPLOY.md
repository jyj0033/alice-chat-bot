# Docker 部署说明

## 快速启动

### 1. 准备配置文件

创建 `config/config.yaml`：

```yaml
llm:
  api_key: "your-api-key"
  model: "gpt-4o"

qq:
  ws_url: "ws://host.docker.internal:3001"  # 连接宿主机NapCat
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

## 防火墙

```bash
# Ubuntu/Debian
sudo ufw allow 30080

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=30080/tcp
sudo firewall-cmd --reload
```

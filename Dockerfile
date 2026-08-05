# 群聊Bot镜像
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p data logs

# 暴露端口
EXPOSE 30080 3001

# 启动命令
CMD ["python3", "main.py", "--dashboard", "--dashboard-host", "0.0.0.0", "--dashboard-port", "30080"]

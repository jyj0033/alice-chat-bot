# 群聊Bot镜像
FROM python:3.11-slim

WORKDIR /app

# 安装依赖（国内镜像加速）
COPY requirements.txt .
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p data logs

# 暴露端口
EXPOSE 30080 3001

# 启动命令
CMD ["python3", "main.py", "--dashboard", "--dashboard-host", "0.0.0.0", "--dashboard-port", "30080"]

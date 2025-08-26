FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 系统依赖（faiss 需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends     gcc g++ git curl ca-certificates libgomp1  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认入口：直接运行评测脚本
ENTRYPOINT ["python", "rag_vs_lc.py"]

# RAG vs 长上下文（Long-Context）入门基线

> **目的**：在 HotpotQA(distractor) 的一个子集上，对比 **RAG（检索增强）** 与 **长上下文拼接** 的 **EM/F1** 与 **平均延迟**，产出可复现的 `summary.csv`。

## 运行（本地）
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY=你的key
# 如用兼容端点（可选）：
# export OPENAI_BASE_URL=https://你的/v1
# export OPENAI_MODEL=gpt-4o-mini

# RAG（前 100 条）
python rag_vs_lc.py --method rag --limit 100 --k 3

# 长上下文（前 100 条）
python rag_vs_lc.py --method lc --limit 100 --max_chars 12000
```

输出位于：`runs/<时间戳>/rag.jsonl | lc.jsonl | summary.csv`

## 运行（Docker/Compose）
```bash
docker build -t rag-lc:0.1 .
docker run --rm -it -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd)/runs:/app/runs rag-lc:0.1 --method rag --limit 100 --k 3
docker run --rm -it -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd)/runs:/app/runs rag-lc:0.1 --method lc  --limit 100 --max_chars 12000
# 或者
docker compose up --build
```

## 结果示例
```
method,limit,avg_em,avg_f1,avg_latency_ms,k_or_maxchars
rag,100,0.28,0.45,850.3,3
lc,100,0.31,0.47,1420.6,12000
```

> 注：示例数值仅作参考，与你的模型/端点有关。

## 常见问题
- **未设置 OPENAI_API_KEY**：需设置环境变量。兼容端点还需 `OPENAI_BASE_URL`、`OPENAI_MODEL`。
- **FAISS 相似度怪**：确保对向量进行了 L2 归一化；本项目用 `IndexFlatIP`。
- **长上下文报错/截断**：控制 `--max_chars`（模型有 token 上限）。

## 下一步升级
- 改为**全局检索**（例如 Wikipedia dump / 企业文档），而不是每题在自带段落中检索。
- 加入**成本估算**（tokens 统计），绘制 Accuracy–Latency–Cost 三维权衡。
- 扩展到 **TriviaQA / NQ-open / LongBench** 子集。

——
建议阅读：LangChain/LlamaIndex 的中文教程、FAISS 入门文章，以及 Hugging Face Datasets 文档。

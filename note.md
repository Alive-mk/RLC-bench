**RLC-Bench: When to Retrieve and When to Read Long? A Quantitative Study of Accuracy–Latency–Cost**

**RAG vs Long-Context: Practical Trade-offs on Multi-Hop QA**

**Context Budgeting for LLMs: Top-k Retrieval or Long-Context Packing?**

 这个项目就是用**同一套数据**，把两种“把知识喂给大模型”的方法——**RAG（检索增强）和长上下文直塞（Long-Context）**——放到同一跑道上，**量化比较它们的准确率与耗时**，帮你决定在你的场景里该选哪条路线。

# 在干什么（What）

- 任务：多跳问答（HotpotQA 子集）。
- 两种方法：
  - **RAG**：先用向量检索挑出 **Top-K** 段落，再把这几段拼进提示里问模型。
  - **长上下文（LC）**：把能塞的相关段落尽量**一次性长文拼接**给模型。
- 度量：对每道题记录 **EM/F1（准确性）** 和 **推理延迟（ms）**，最后输出汇总表 `summary.csv`。
- 复现：固定依赖与脚本；结果逐样本落盘到 `runs/<时间戳>/*.jsonl`。

# 目的（Why）

1. **做决策**：给出“**准确率—延迟（—成本）**”的量化权衡，指导你在产品/论文里**选 RAG 还是 LC**，或两者混用的阈值。
2. **打样本**：一周内拿到第一张基准表，后续很容易扩展到**更真实的全局文档库、更多数据集**，形成论文/报告的实验骨架。
3. **工程可落地**：代码与 Docker 都给了，换模型/改参数即可在你自己的知识库上复跑。

# 你会拿到的产出（Outputs）

- `summary.csv`：两行（rag / lc）的 **EM、F1、平均延迟**；
- `rag.jsonl` / `lc.jsonl`：逐样本记录（预测、金标、延迟、上下文长度等）；
- （可选）把 `k=1/3/5`、`max_chars=6k/12k/24k` 跑成**曲线**，直观看出“多拿上下文”带来的收益与代价。

# 设计取舍（Assumptions）

- 入门版先在每题自带候选段落里检索（**闭集**、最稳），方便你先跑通；
- 下一步可扩展到**全局索引（如 Wikipedia/企业文档）**，更贴近真实检索场景；
- 成本统计（tokens）留作加分项，便于把结果转化为**费用评估**。

# 成功标准（Definition of Done）

- 跑完 100 条样本，得到一张表；通常会看到：
  - **LC** → 可能略高的 EM/F1，但**延迟更高**；
  - **RAG** → 延迟更低，对 **k** 较敏感。
- 能据此给出一句**选择建议**（例如：“我们领域里，k=3 的 RAG 已经接近 LC，但速度快 40%+”）。

如果你 ok，我们下一步就按这目标跑“第一轮 100 条”，我再教你把结果画成一张对比图，顺便做两点小消融（k 与 max_chars）。

------

```
# 1) 新建环境（Python 3.11）
conda create -n rlc-bench python=3.11 -y
conda activate rlc-bench
python -m pip install -U pip

# 2) 安装依赖（在项目根目录）
pip install -r requirements.txt

# 3) 运行
# 设置 Key（替换成你自己的）
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx" 

# 如果用的是官方 OpenAI，不用设置 base，脚本会默认走 https://api.openai.com/v1
# 如果你用的是兼容端点（比如你们的服务器/代理），就要设：
export OPENAI_BASE_URL="https://你的域名/v1"

# 可选：指定模型名
export OPENAI_MODEL="gpt-4o-mini"
```

```
echo '' >> ~/.bashrc
echo '# >>> OpenAI API config >>>' >> ~/.bashrc
echo 'export OPENAI_API_KEY="sk-feQaMdPuRBDsZg3I667f359cAb7042B98246724b794e6c0c"' >> ~/.bashrc
echo 'export OPENAI_BASE_URL="https://api.ai-gaochao.cn/v1"' >> ~/.bashrc
echo 'export OPENAI_MODEL="gpt-4o-mini"' >> ~/.bashrc
echo '# <<< OpenAI API config <<<' >> ~/.bashrc

source ~/.bashrc

```

```
脚本
#!/bin/bash
# 用法: ./setup_openai.sh sk-你的key https://你的base/v1 gpt-4o-mini

API_KEY=$1
BASE_URL=$2
MODEL=${3:-gpt-4o-mini}   # 如果没传第三个参数，默认 gpt-4o-mini

if [ -z "$API_KEY" ]; then
  echo "❌ 请传入 API Key，例如:"
  echo "   ./setup_openai.sh sk-xxxx https://api.openai.com/v1"
  exit 1
fi

# 追加到 ~/.bashrc
{
  echo ""
  echo "# >>> OpenAI API config >>>"
  echo "export OPENAI_API_KEY=\"$API_KEY\""
  if [ -n "$BASE_URL" ]; then
    echo "export OPENAI_BASE_URL=\"$BASE_URL\""
  fi
  echo "export OPENAI_MODEL=\"$MODEL\""
  echo "# <<< OpenAI API config <<<"
} >> ~/.bashrc

# 让配置立即生效
source ~/.bashrc

echo "✅ 已写入 ~/.bashrc 并生效"
echo "   OPENAI_API_KEY=${API_KEY:0:8}..."
echo "   OPENAI_BASE_URL=${BASE_URL:-默认 https://api.openai.com/v1}"
echo "   OPENAI_MODEL=$MODEL"


nano setup_openai.sh
# 粘贴上面的内容，保存退出
chmod +x setup_openai.sh

./setup_openai.sh sk-feQaMdPuRBDsZg3I667f359cAb7042B98246724b794e6c0c https://api.ai-gaochao.cn/v1 gpt-4o-mini

验证
echo $OPENAI_API_KEY
echo $OPENAI_BASE_URL
echo $OPENAI_MODEL
```

<img src="../../AppData/Roaming/Typora/typora-user-images/image-20250827115543519.png" alt="image-20250827115543519" style="zoom:67%;" />

```
你这个界面是 Vim 的 E325 报错，意思是：

👉 Vim 发现你要打开的文件 ~/.bashrc 已经有一个**交换文件（swap file）**存在：

~/.bashrc.swp

这通常表示：

你之前在 Vim 里打开过这个文件，但没正常退出（比如直接关了终端）。

或者这个文件此刻正在被另一个 Vim 实例编辑。

(O)pen Read-Only → 只读方式打开，不允许保存。

(E)dit anyway → 强行编辑，忽略那个 .swp 文件。

(R)ecover → 尝试用 .swp 文件里的内容恢复未保存的修改。

(Q)uit → 什么也不做，直接退出。

(A)bort → 中止启动。

常见处理方式

如果你确定只是上次没退出干净（没有别的 Vim 在编辑 ~/.bashrc）：
👉 按 E 就行（Edit anyway）。
然后保存时会覆盖 .swp 里的旧东西。
保存退出后再执行：

rm ~/.bashrc.swp


把多余的交换文件删掉，以后就不会再弹了。

如果你想看看 .swp 里是不是有没保存的修改：
👉 按 R（Recover）。
Vim 会用 .swp 里的内容恢复文件。
然后再手动保存一次（:wq），再删掉 .swp 文件。
```

```
运行 RAG 模式：

python rag_vs_lc.py --method rag --limit 100 --k 3


运行 长上下文 模式：

python rag_vs_lc.py --method lc --limit 100 --max_chars 12000
```

![image-20250829110255626](../../AppData/Roaming/Typora/typora-user-images/image-20250829110255626.png)

`AttributeError: _ARRAY_API not found` / `ImportError: numpy.core.multiarray failed to import` 出在 **faiss**，这是 **FAISS 和 NumPy 的二进制不匹配**（常见于在 conda 里用 pip 装了 faiss-cpu，NumPy 版本不同步）。

```
# 卸载 pip 版 faiss，避免冲突
pip uninstall -y faiss-cpu faiss

# 用 conda-forge 安装（会自动配好依赖）
conda install -y -c conda-forge faiss-cpu=1.8.* numpy=1.26.*

python - <<'PY'
import numpy, faiss, numpy as np
print("numpy:", numpy.__version__)
print("faiss:", faiss.__version__)
x = np.random.rand(5,4).astype('float32'); x/=np.linalg.norm(x,axis=1,keepdims=True)
index = faiss.IndexFlatIP(4); index.add(x); D,I = index.search(x[:1], 3)
print("ok:", D.shape, I.shape)
PY
```

```
环境重装

# 退出到(base)，多敲几次直到不再显示 (rlc-bench)
conda deactivate

# 删除旧环境（名字按你现在用的来，这里是 rlc-bench）
conda remove -n rlc-bench --all -y

conda env list    # 确认 rlc-bench 不在列表里了

# 用 conda-forge 创建新环境并一次装齐核心科学计算栈
conda create -n rlc-bench -c conda-forge -y \
  python=3.11 \
  numpy=1.26.4 scipy=1.11.4 scikit-learn=1.3.2 faiss-cpu=1.8.0 \
  pandas=2.1.4 pyarrow=14.0.2

conda activate rlc-bench
conda config --env --set channel_priority strict   # 本环境启用严格优先级
python -m pip install -U pip


# 确保 requirements.txt 只有以下内容（不要有 numpy/scipy/sklearn/faiss/pandas/pyarrow）
# datasets
# sentence-transformers
# openai>=1.0.0
# tqdm
# PyYAML
# regex

python -m pip install -r requirements.txt

自检
python - <<'PY'
import numpy as np, scipy, sklearn, faiss, pandas, pyarrow
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances

print("numpy:", np.__version__, "scipy:", scipy.__version__,
      "sklearn:", sklearn.__version__, "faiss:", faiss.__version__)
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = m.encode(["hello","world"], normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype("float32"))
print("pairwise:", pairwise_distances([[0,1],[1,0]]).shape, "faiss search:", index.search(emb[:1].astype("float32"), 2)[1])
PY

能打印出版本、pairwise: (2, 2) 和 faiss search: [[0 1]] 之类就 OK。

如仍提示 “No module named numpy”：要么环境没激活，要么 VS Code 终端没接管 conda。

先 conda activate rlc-bench && which python 看路径是否在 …/envs/rlc-bench/…。

如果 VS Code 内置终端不生效：conda init bash 后重开终端；或用
conda run -n rlc-bench python rag_vs_lc.py --method rag --limit 100 --k 3
```

![image-20250830104256973](../../AppData/Roaming/Typora/typora-user-images/image-20250830104256973.png)

```
你这条报错是 Conda 在校验下载包时发现缓存里的包损坏了：

CondaVerificationError: The package for _openmp_mutex ... appears to be corrupted.
The path 'lib/libgomp.so.1' specified in the package manifest cannot be found.


意思：~/miniconda3/pkgs/_openmp_mutex-4.5-2_gnu 这个包在本地缓存不完整（少了 lib/libgomp.so.1），导致创建环境失败。处理办法就是清掉坏缓存→重新下载。按下面一步步做：

# 退出到 base 环境
conda deactivate

# 删掉损坏的包缓存（精确删除这类目录/压缩包）
rm -rf ~/miniconda3/pkgs/_openmp_mutex-4.5-2_gnu*

# 清理索引、tarball、已下载但未用的包
conda clean --index-cache --tarballs --packages -y
# 或者一把梭：conda clean -a -y

conda update -n base -c conda-forge conda -y
conda install -n base -c conda-forge conda-libmamba-solver -y
conda config --set solver libmamba

```

------

![image-20250830113259457](../../AppData/Roaming/Typora/typora-user-images/image-20250830113259457.png)

```
method,limit,avg_em,avg_f1,avg_latency_ms,k_or_maxchars
lc,100,0.0000,0.0001,628.6,12000
method,limit,avg_em,avg_f1,avg_latency_ms,k_or_maxchars
rag,100,0.0000,0.0001,648.0,3


你这两行是两次跑出来的汇总：

lc,100,0.0000,0.0001,628.6,12000   # 长上下文：跑了100条，EM≈0，F1≈0.0001，平均延迟≈629ms，上下文上限12000字符
rag,100,0.0000,0.0001,648.0,3      # RAG：跑了100条，EM≈0，F1≈0.0001，平均延迟≈648ms，top-k=3


意思：模型给出的答案几乎都没匹配到标准答案（要么答错、答整句、要么大量输出 unknown）。延迟 0.6s 也异常快，像是在迅速返回固定样式（例如一直 unknown）。
```

看看模型到底回答了啥（unknown 比例与样例）

```
# RAG 结果抽样+unknown占比
python - <<'PY'
import json,glob,random
p=sorted(glob.glob('runs/*/rag.jsonl'))[-1]
L=[json.loads(x) for x in open(p,encoding='utf-8')]
u=sum(1 for d in L if d["pred"].strip().lower().startswith("unknown"))
print(f"RAG unknown比例: {u}/{len(L)} = {u/len(L):.1%}")
for d in random.sample(L,3):
    print("\nQ:",d["question"]);print("PRED:",d["pred"]);print("GOLD:",d["gold"])
PY

# LC 同样看一下
python - <<'PY'
import json,glob,random
p=sorted(glob.glob('runs/*/lc.jsonl'))[-1]
L=[json.loads(x) for x in open(p,encoding='utf-8')]
u=sum(1 for d in L if d["pred"].strip().lower().startswith("unknown"))
print(f"LC unknown比例: {u}/{len(L)} = {u/len(L):.1%}")
for d in random.sample(L,3):
    print("\nQ:",d["question"]);print("PRED:",d["pred"]);print("GOLD:",d["gold"])
PY

确认我们确实把“上下文”塞进去了（不是空的）

python - <<'PY'
import json,glob
p=sorted(glob.glob('runs/*/rag.jsonl'))[-1]
L=[json.loads(x) for x in open(p,encoding='utf-8')]
zero_ctx=sum(1 for d in L if d["ctx_chars"]==0 or d["num_ctx"]==0)
print(f"RAG 空上下文样本: {zero_ctx}/{len(L)}")
print("RAG 平均ctx长度:", sum(d["ctx_chars"] for d in L)//len(L))

p=sorted(glob.glob('runs/*/lc.jsonl'))[-1]
L=[json.loads(x) for x in open(p,encoding='utf-8')]
zero_ctx=sum(1 for d in L if d["ctx_chars"]==0 or d["num_ctx"]==0)
print(f"LC  空上下文样本: {zero_ctx}/{len(L)}")
print("LC  平均ctx长度:", sum(d["ctx_chars"] for d in L)//len(L))
PY

快速验证接口真的在走 OpenAI 兼容 JSON（不是 404/HTML）

curl -sS -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"'"${OPENAI_MODEL:-gpt-4o-mini}"'","messages":[{"role":"user","content":"Say OK only."}]}' \
     "$OPENAI_BASE_URL/chat/completions" | head -n 3
期待是 JSON（以 { 开头）；

如果看到 <html> 或 404，OPENAI_BASE_URL 还不对，必须用带 /v1 的真正接口（例如 https://.../v1）。
```

![image-20250831105256403](../../AppData/Roaming/Typora/typora-user-images/image-20250831105256403.png)

这是在告诉你：**模型返回的不是答案，而是一整页网站的 404 HTML**。
 页面里有 “`<div class="title">404</div>` / `Powered by aapanel`”，说明你的请求打到了一个普通网站（或面板）而不是 OpenAI 兼容的 API。

最常见原因：**`OPENAI_BASE_URL` 写错了，少了 `/v1`**。

![image-20250831110053482](../../AppData/Roaming/Typora/typora-user-images/image-20250831110053482.png)

```
conda activate rlc-bench
conda install -y -c conda-forge libstdcxx-ng=13.2.0 libgcc-ng=13.2.0 _openmp_mutex=4.5=2_gnu
#（已经有也无妨，强制对齐到 13.x）

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 持久化（以后激活环境自动生效）
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
printf 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"\n' \
  > "$CONDA_PREFIX/etc/conda/activate.d/zzz_ld_path.sh"

strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.30 || echo "missing"

```

:hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs:

![image-20250904094320169](../../AppData/Roaming/Typora/typora-user-images/image-20250904094320169.png)

**RAG (k=3)**：EM=**0.26**，F1=**0.366**，平均延迟≈**1498 ms**

**长上下文 LC (12k chars)**：EM=**0.35**，F1=**0.489**，平均延迟≈**1561 ms**

**准确率**：LC 比 RAG **高 ~0.09 EM / ~0.12 F1**，说明 *k=3 的 RAG 召回仍会漏关键信息*；LC 一次性喂更多原文，漏掉的少。

**时延**：LC 只比 RAG **慢 ~62 ms（≈4%）**，差距很小（在你这个模型和长度上，长上下文的额外开销不大）。

粗结论：**若更看重准确，LC 更有优势；若追求吞吐/成本，RAG 需要把 k 拉高来“追上”准确度**

:hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs:

把 RAG 的召回拉满一点、LC 再试更长上限——看看能否逼近/拉开差距：

```
# RAG：k 网格
python rag_vs_lc.py --method rag --limit 100 --k 5 --temperature 0.0
python rag_vs_lc.py --method rag --limit 100 --k 7 --temperature 0.0

# LC：更长的上下文（模型允许的话）
python rag_vs_lc.py --method lc  --limit 100 --max_chars 20000 --temperature 0.0
```

![image-20250904102544661](../../AppData/Roaming/Typora/typora-user-images/image-20250904102544661.png)

- **RAG (k=5)**：EM=**0.27**，F1=**0.377**，Latency≈**1544 ms**
- **RAG (k=7)**：EM=**0.27**，F1=**0.404**，Latency≈**1527 ms**
- **LC (max_chars=20k)**：EM=**0.33**，F1=**0.479**，Latency≈**1471 ms**

1. **准确率**：LC 仍然领先（F1 高 ~0.07–0.10）。RAG 把 k 从 5 → 7 后**有提升**（F1 +0.027），但 EM 没再涨，说明**召回提高了但答案抽取仍有miss**。
2. **时延**：三者都在 ~1.5s 左右，**差距很小**；你的端点下，LC 的额外开销不明显。
3. **20k 长上下文未胜过 12k**：出现“边际收益递减/可能被截断”。说明继续加长上下文不一定更准；应控制在模型 token 上限内，并优先提高信息密度（更好的检索/排序）。

:hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs:

```
# RAG 网格（看是否继续涨）
python rag_vs_lc.py --method rag --limit 100 --k 9  --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0

# LC 回退到 12k 再对比一次
python rag_vs_lc.py --method lc  --limit 100 --max_chars 12000 --temperature 0.0

```

![image-20250904104239167](../../AppData/Roaming/Typora/typora-user-images/image-20250904104239167.png)

- **RAG（k=9，bge-small）**：**EM 0.39 / F1 0.53 / 1.78s**
- **LC（12k）**：**EM 0.34 / F1 0.486 / 1.67s**

结论：在你这个端点与数据集上，**RAG 已反超 LC**（准确更高，延迟只略高≈110ms）。可以把 RAG 作为当前默认方案。

:hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs::hugs:

**把 RAG 再榨一点（+Rerank）**

两段改动即可，加一个交叉编码器做二段排序，通常 **+3～8 F1**。

```
原：
class PerSampleRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)

    def topk(self, passages: List[Tuple[str,str]], query: str, k: int=3):
        if len(passages) <= k:
            return passages
        texts = [p[1] for p in passages]
        # 归一化后用内积近似余弦
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        p_emb = self.embedder.encode(texts, normalize_embeddings=True)
        index = faiss.IndexFlatIP(p_emb.shape[1])
        index.add(p_emb.astype("float32"))
        D, I = index.search(q_emb.astype("float32"), k)
        return [passages[i] for i in I[0].tolist()]
        
后：
class PerSampleRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 rerank_model=None, rerank_topn=40):
        self.embedder = SentenceTransformer(model_name)
        self.ce = CrossEncoder(rerank_model) if rerank_model else None
        self.rerank_topn = rerank_topn

    def topk(self, passages, query, k=3):
        texts = [p[1] for p in passages]
        q = self.embedder.encode([query], normalize_embeddings=True)
        P = self.embedder.encode(texts, normalize_embeddings=True)

        import numpy as np, faiss
        index = faiss.IndexFlatIP(P.shape[1]); index.add(P.astype("float32"))
        topn = max(k, self.rerank_topn) if self.ce else k
        D, I = index.search(q.astype("float32"), topn)
        cand = [passages[i] for i in I[0].tolist()]

        if self.ce:  # 二段rerank再挑k个
            pairs = [(query, p) for _, p in cand]
            scores = self.ce.predict(pairs)  # 越大越相关
            order = np.argsort(-scores)[:k]
            return [cand[i] for i in order]
        return cand[:k]

```

```
# RAG + bge-small + rerank
python rag_vs_lc.py --method rag --limit 100 --k 7 \
  --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0
```

![image-20250904111858354](../../AppData/Roaming/Typora/typora-user-images/image-20250904111858354.png)

看到了，这次用 **bge-small + k=7** 结果掉到了 *EM 0.23 / F1 0.35*。大概率是 **bge 没用到专属的 query/passsage 提示**；bge v1.5 需要在编码时加上 `prompt_name='query'/'passage'`，否则召回会明显退步。

```
# ========== 检索器（每题在其自带段落内检索） ==========
class PerSampleRetriever:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 rerank_model: str | None = None,
                 rerank_topn: int = 40):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.rerank_topn = rerank_topn
        self.ce = None
        if rerank_model:
            try:
                from sentence_transformers import CrossEncoder
                self.ce = CrossEncoder(rerank_model)   # 二段排序模型
            except Exception as e:
                print(f"[warn] load rerank model failed: {e}")
                self.ce = None

    def topk(self, passages: List[Tuple[str, str]], query: str, k: int = 3):
        if len(passages) <= k:
            return passages

        texts = [p[1] for p in passages]

        # ---- BGE v1.5 需要 query/passsage 的 prompt_name，否则效果会掉 ----
        use_bge = "bge" in self.model_name.lower()
        q_kw = {"normalize_embeddings": True}
        p_kw = {"normalize_embeddings": True}
        if use_bge:
            q_kw["prompt_name"] = "query"
            p_kw["prompt_name"] = "passage"

        q_emb = self.embedder.encode([query], **q_kw)
        p_emb = self.embedder.encode(texts, **p_kw)

        import faiss, numpy as np
        index = faiss.IndexFlatIP(p_emb.shape[1])
        index.add(p_emb.astype("float32"))

        topn = max(k, self.rerank_topn) if self.ce else k
        D, I = index.search(q_emb.astype("float32"), topn)
        cand = [passages[i] for i in I[0].tolist()]

        if self.ce:
            pairs = [(query, p) for _, p in cand]
            scores = self.ce.predict(pairs)  # 分数越大越相关
            order = np.argsort(-scores)[:k]
            return [cand[i] for i in order]

        return cand[:k]

```

```
# bge-small + 正确 prompts
python rag_vs_lc.py --method rag --limit 100 --k 7 --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0
python rag_vs_lc.py --method rag --limit 100 --k 9 --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0

```

![image-20250904114825503](../../AppData/Roaming/Typora/typora-user-images/image-20250904114825503.png)

确实变差了，说明我们这次对 **BGE 的 prompt 用法**没打准（或版本不一致）。

![image-20250904160606320](../../AppData/Roaming/Typora/typora-user-images/image-20250904160606320.png)

 **对 BGE 做 AB 测试（关掉 prompt / 自动选择 prompt）**

然后跑两次对照：

```
# A: 关闭 BGE 的 prompt（复现你之前的好成绩）
BGE_PROMPTS=off \
python rag_vs_lc.py --method rag --limit 100 --k 9 \
  --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0

# B: 开启自动 prompt（我们刚改的自适配）
python rag_vs_lc.py --method rag --limit 100 --k 9 \
  --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0

```

![image-20250904161702502](../../AppData/Roaming/Typora/typora-user-images/image-20250904161702502.png)

你怀疑对得很有道理。回退检查后，我发现**真正的问题在于 BGE v1.5 的用法**：
 对大多数 `bge-*-en-v1.5` 模型，**只需要在“查询侧”加 `prompt_name='query'`**；**文档侧不要加任何 prompt**。我前面的版本给文档侧套了 `document/passage` 的 prompt，确实会把相似度学到的空间“扭坏”，导致召回骤降（你看到的 EM/F1 大幅下滑就是这个原因）。

下面给你一版**最小且稳妥**的检索器，保持你原始逻辑不变，只在检测到 BGE 时**对查询侧**加上 `query` prompt；文档侧不动。`rerank` 仍然是可选开关，不传就和你最初一模一样。

```
# ========== 检索器（每题在其自带段落内检索） ==========
class PerSampleRetriever:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 rerank_model: str | None = None,
                 rerank_topn: int = 40):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.ce = CrossEncoder(rerank_model) if rerank_model else None
        self.rerank_topn = rerank_topn

    def topk(self, passages, query, k: int = 3):
        if len(passages) <= k:
            return passages

        texts = [p[1] for p in passages]

        # ---- 只对 BGE v1.x 在“查询侧”加 query prompt；文档侧不加 ----
        use_bge = "bge" in self.model_name.lower()
        q_kw = {"normalize_embeddings": True}
        p_kw = {"normalize_embeddings": True}
        if use_bge:
            prompts = getattr(self.embedder, "prompts", {}) or {}
            if "query" in prompts:
                q_kw["prompt_name"] = "query"   # ✅ 只加这一处

        q_emb = self.embedder.encode([query], **q_kw)
        p_emb = self.embedder.encode(texts, **p_kw)

        import faiss, numpy as np
        index = faiss.IndexFlatIP(p_emb.shape[1])
        index.add(p_emb.astype("float32"))

        topn = max(k, self.rerank_topn) if self.ce else k
        D, I = index.search(q_emb.astype("float32"), topn)
        cand = [passages[i] for i in I[0].tolist()]

        if self.ce:
            scores = self.ce.predict([(query, p) for _, p in cand])
            order = np.argsort(-scores)[:k]
            return [cand[i] for i in order]
        return cand[:k]

```

```
# 1) bge-small + k=9（预期恢复到你之前的高分区间）
python rag_vs_lc.py --method rag --limit 100 --k 9 \
  --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0

# 2) 可选：在 1) 的基础上开 rerank（常见 +3~8 F1）
RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2" RERANK_TOPN=40 \
python rag_vs_lc.py --method rag --limit 100 --k 7 \
  --embedding "BAAI/bge-small-en-v1.5" --temperature 0.0

```


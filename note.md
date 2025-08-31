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
echo 'export OPENAI_API_KEY="sk-XRmGDCzKp0OjU9kz5cC9110a56Ba4aEbBb9e7bB559C7Ef24"' >> ~/.bashrc
echo 'export OPENAI_BASE_URL="https://apix.ai-gaochao.cn"' >> ~/.bashrc
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

./setup_openai.sh sk-XRmGDCzKp0OjU9kz5cC9110a56Ba4aEbBb9e7bB559C7Ef24 https://apix.ai-gaochao.cn gpt-4o-mini

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
import os, time, argparse, json, re, unicodedata, pathlib, sys
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# ========== 简单评测：EM / F1 ==========
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKC", s)
    # 去标点 & 冗余空格
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    # 去英文冠词
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def em(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) and gold != "" else 0.0

def f1(pred: str, gold: str) -> float:
    p_tok = normalize_text(pred).split()
    g_tok = normalize_text(gold).split()
    if not p_tok or not g_tok:
        return float(p_tok == g_tok)
    # 多集合交集计数
    from collections import Counter
    cp, cg = Counter(p_tok), Counter(g_tok)
    num_same = sum((cp & cg).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tok)
    recall = num_same / len(g_tok)
    return 2 * precision * recall / (precision + recall)

# ========== LLM 接口（OpenAI 兼容） ==========
def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        print("请先安装 openai>=1.0.0", file=sys.stderr)
        raise e
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请设置 OPENAI_API_KEY")
    return OpenAI(base_url=base_url, api_key=api_key)

def chat_complete(client, model: str, prompt: str, temperature: float=0.2, timeout: int=60) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a careful QA assistant. Only answer using the provided context. If not answerable, say 'unknown'."},
            {"role":"user","content": prompt}
        ],
        temperature=temperature,
        timeout=timeout
    )
    return resp.choices[0].message.content.strip()

# ========== 数据载入（HotpotQA distractor） ==========
@dataclass
class QASample:
    qid: str
    question: str
    answer: str
    passages: List[Tuple[str, str]]  # (title, paragraph)

def load_hotpot_subset(split: str) -> List[QASample]:
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    samples = []
    for ex in ds:
        qid = ex["id"]
        question = ex["question"]
        answer = ex.get("answer", "")
        ctx = ex["context"]  # 可能是 list，也可能是 dict{"title": [...], "sentences": [[...]]}

        paras = []
        if isinstance(ctx, dict):  # 新式：列式存储
            titles = ctx.get("title", [])
            sents_list = ctx.get("sentences", [])
            for title, sents in zip(titles, sents_list):
                para = " ".join(sents)
                if para.strip():
                    paras.append((title, para))
        else:  # 旧式：[(title, [sentences]), ...]
            for item in ctx:
                # item 可能是 (title, [sentences]) 或 {"title":..., "sentences":[...]}
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    title, sents = item[0], item[1]
                elif isinstance(item, dict):
                    title, sents = item.get("title", ""), item.get("sentences", [])
                else:
                    continue
                para = " ".join(sents)
                if para.strip():
                    paras.append((title, para))

        samples.append(QASample(qid=qid, question=question, answer=answer, passages=paras))
    return samples


# ========== 检索器（每题在其自带段落内检索） ==========
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

# ========== 提示模板 ==========
def build_prompt_rag(question: str, ctxs: List[Tuple[str,str]]) -> str:
    ctx_text = "\n\n".join([f"[{i+1}] ({t}) {p}" for i,(t,p) in enumerate(ctxs)])
    return (
        "Answer the question using ONLY the following context.\n"
        "If not answerable from the context, answer exactly: unknown.\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def pack_long_context(passages: List[Tuple[str,str]], max_chars: int=12000) -> List[Tuple[str,str]]:
    buf, total = [], 0
    for t,p in passages:
        if total + len(p) + 50 > max_chars:  # 留点余量
            break
        buf.append((t,p))
        total += len(p) + 1
    return buf if buf else passages[:1]

def build_prompt_lc(question: str, all_passages: List[Tuple[str,str]], max_chars: int=12000) -> str:
    packed = pack_long_context(all_passages, max_chars=max_chars)
    ctx_text = "\n\n".join([f"({t}) {p}" for (t,p) in packed])
    return (
        "Answer the question using ONLY the following long context.\n"
        "If not answerable from the context, answer exactly: unknown.\n\n"
        f"Long Context:\n{ctx_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

# ========== 主流程 ==========
def run(method: str, limit: int, k: int, max_chars: int, model: str, temperature: float, timeout: int, embed_name: str):
    out_dir = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{method}.jsonl"
    summ_path = out_dir / "summary.csv"

    client = get_openai_client()
    #retriever = PerSampleRetriever(embed_name)
    retriever = PerSampleRetriever(embed_name,
                               rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                               rerank_topn=int(os.getenv("RERANK_TOPN", "40")))

    # 数据：默认 HotpotQA 验证子集
    split = os.getenv("SPLIT", f"validation[:{limit}]")
    samples = load_hotpot_subset(split)

    total_em = total_f1 = 0.0
    latencies = []
    n_done = 0

    with open(log_path, "w", encoding="utf-8") as fw:
        for s in tqdm(samples[:limit], desc=f"Running {method}"):
            if method == "rag":
                ctxs = retriever.topk(s.passages, s.question, k=k)
                prompt = build_prompt_rag(s.question, ctxs)
                num_ctx = len(ctxs)
                ctx_len = sum(len(p) for _,p in ctxs)
            else:
                prompt = build_prompt_lc(s.question, s.passages, max_chars=max_chars)
                num_ctx = len(s.passages)
                ctx_len = sum(len(p) for _,p in s.passages)

            t0 = time.time()
            try:
                pred = chat_complete(client, model, prompt, temperature=temperature, timeout=timeout)
            except Exception as e:
                pred = f"[error:{e}]"
            t1 = time.time()

            gold = s.answer or ""
            _em = em(pred, gold)
            _f1 = f1(pred, gold)

            lat = (t1 - t0) * 1000
            latencies.append(lat)
            total_em += _em
            total_f1 += _f1
            n_done += 1

            rec = {
                "id": s.qid, "method": method, "question": s.question,
                "pred": pred, "gold": gold, "em": _em, "f1": _f1,
                "latency_ms": round(lat,2), "num_ctx": num_ctx, "ctx_chars": ctx_len
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg_em = total_em / max(1,n_done)
    avg_f1 = total_f1 / max(1,n_done)
    avg_lat = sum(latencies) / max(1,len(latencies))

    # 附上简表（追加写）
    header_needed = not summ_path.exists()
    with open(summ_path, "a", encoding="utf-8") as sf:
        if header_needed:
            sf.write("method,limit,avg_em,avg_f1,avg_latency_ms,k_or_maxchars\n")
        tag = k if method=='rag' else max_chars
        sf.write(f"{method},{n_done},{avg_em:.4f},{avg_f1:.4f},{avg_lat:.1f},{tag}\n")

    print(f"\n== Summary ({method}) ==")
    print(f"EM={avg_em:.4f}  F1={avg_f1:.4f}  Avg Latency={avg_lat:.1f} ms  N={n_done}")
    print(f"Logs: {log_path}\nTable: {summ_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rag","lc"], required=True, help="rag=检索增强；lc=长上下文拼接")
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "100")))
    ap.add_argument("--k", type=int, default=int(os.getenv("RAG_TOP_K","3")))
    ap.add_argument("--max_chars", type=int, default=int(os.getenv("LC_MAX_CHARS","12000")))
    ap.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE","0.2")))
    ap.add_argument("--timeout", type=int, default=int(os.getenv("TIMEOUT","60")))
    ap.add_argument("--embedding", type=str, default=os.getenv("EMBEDDING","sentence-transformers/all-MiniLM-L6-v2"))
    args = ap.parse_args()

    run(args.method, args.limit, args.k, args.max_chars, args.model, args.temperature, args.timeout, args.embedding)

if __name__ == "__main__":
    main()

import ast
import os
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from retriever import search_documents

# =========================
# 1. 설정
# =========================

EVAL_PATH = "data/retriever_eval_set_q01_q30.csv"   # 네 평가셋 파일명으로 바꾸기
KS = [3, 5, 9, 12, 15]

# 평가셋 컬럼명
QUERY_ID_COL = "query_id"
QUESTION_COL = "question"
GOLD_COL = "gold_chunks"


# =========================
# 2. 평가셋 불러오기
# =========================

evalset = pd.read_csv(EVAL_PATH)


def parse_gold_chunks(value):
    """
    gold_chunks 컬럼을 파이썬 리스트로 변환.
    지원 형태:
    - ["i_00122"]
    - ["sn_01861", "i_00131"]
    - i_00122|i_00123
    - 빈 값
    - Top-15 내 직접 정답 근거 없음
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "":
        return []

    if "Top-15" in value or "직접 정답 근거 없음" in value:
        return []

    if value.startswith("["):
        return ast.literal_eval(value)

    if "|" in value:
        return [x.strip() for x in value.split("|") if x.strip()]

    return [value]


evalset["gold_list"] = evalset[GOLD_COL].apply(parse_gold_chunks)

# gold가 없는 문항은 일반 검색 성능 평가에서 제외
eval_gold = evalset[evalset["gold_list"].apply(len) > 0].copy()

print("전체 문항 수:", len(evalset))
print("평가 대상 문항 수:", len(eval_gold))
print("gold 없는 문항:", evalset[evalset["gold_list"].apply(len) == 0][QUERY_ID_COL].tolist())


# =========================
# 3. Retriever 검색 결과에서 chunk_id 꺼내기
# =========================

def get_chunk_id(doc):
    """
    LangChain Document에서 chunk_id를 꺼내는 함수.
    네 벡터DB 구조에 따라 아래 후보 중 하나가 맞을 것.
    """
    if hasattr(doc, "id") and doc.id:
        return doc.id

    if "chunk_id" in doc.metadata:
        return doc.metadata["chunk_id"]

    if "id" in doc.metadata:
        return doc.metadata["id"]

    raise ValueError(f"chunk_id를 찾을 수 없습니다. metadata={doc.metadata}")


def search_ids(question, k):
    """
    질문을 넣으면 top-k 검색 결과의 chunk_id 리스트를 반환.
    
    아래 코드에서 vectorstore 이름만 네가 쓰는 변수명으로 바꾸면 됨.
    예: store, vectorstore, db, chroma_db 등
    """

    results = search_documents(question, k=k, threshold=-1.0)
    return [result["chunk_id"] for result in results]


# 만약 이미 retriever 변수를 만들어놨고, k만 바꿀 수 있다면 위 search_ids 대신 이걸 써도 됨.
# def search_ids(question, k):
#     retriever.search_kwargs["k"] = k
#     docs = retriever.invoke(question)
#     return [get_chunk_id(doc) for doc in docs[:k]]


# =========================
# 4. 한 문항 평가 함수
# =========================

def score_one(gold_chunks, retrieved_chunks):
    gold_set = set(gold_chunks)
    retrieved_set = set(retrieved_chunks)

    correct = gold_set & retrieved_set

    hit = int(len(correct) > 0)
    precision = len(correct) / len(retrieved_chunks) if retrieved_chunks else 0
    recall = len(correct) / len(gold_set) if gold_set else 0

    mrr = 0
    for rank, chunk_id in enumerate(retrieved_chunks, start=1):
        if chunk_id in gold_set:
            mrr = 1 / rank
            break

    return hit, precision, recall, mrr


# =========================
# 5. K별 전체 평가
# =========================

detail_rows = []

for_k_errors = []

for k in KS:
    for row in eval_gold.itertuples(index=False):
        query_id = getattr(row, QUERY_ID_COL)
        question = getattr(row, QUESTION_COL)
        gold_chunks = getattr(row, "gold_list")

        try:
            retrieved_chunks = search_ids(question, k)
            hit, precision, recall, mrr = score_one(gold_chunks, retrieved_chunks)

            detail_rows.append({
                "query_id": query_id,
                "k": k,
                "question": question,
                "gold_chunks": gold_chunks,
                "retrieved_chunks": retrieved_chunks,
                "hit": hit,
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
            })

        except Exception as e:
            for_k_errors.append({
                "query_id": query_id,
                "k": k,
                "error": str(e),
            })


detail_results = pd.DataFrame(detail_rows)

if detail_results.empty:
    error_df = pd.DataFrame(for_k_errors)
    print("평가 결과가 비었습니다. 모든 검색/평가가 실패했습니다.")
    if not error_df.empty:
        print(error_df.head(20).to_string(index=False))
        error_df.to_csv("retriever_eval_errors.csv", index=False, encoding="utf-8-sig")
    raise SystemExit(1)

summary = (
    detail_results
    .groupby("k")
    .agg(
        Hit_at_K=("hit", "mean"),
        Precision_at_K=("precision", "mean"),
        Recall_at_K=("recall", "mean"),
        MRR_at_K=("mrr", "mean"),
        evaluated_questions=("query_id", "count"),
    )
    .reset_index()
)

print("\n=== K별 요약 ===")
print(summary.to_string(index=False))

print("\n=== 문항별 상세 일부 ===")
print(detail_results[[ "query_id", "k", "gold_chunks", "retrieved_chunks", "hit", "precision", "recall", "mrr" ]].head(30).to_string(index=False))


# =========================
# 6. 결과 저장
# =========================

summary.to_csv("retriever_eval_summary_k3_k5_k9_k12_k15.csv", index=False, encoding="utf-8-sig")
detail_results.to_csv("retriever_eval_detail_k3_k5_k9_k12_k15.csv", index=False, encoding="utf-8-sig")

if for_k_errors:
    error_df = pd.DataFrame(for_k_errors)
    print("\n=== 평가 중 발생한 오류 ===")
    print(error_df.to_string(index=False))
    error_df.to_csv("retriever_eval_errors.csv", index=False, encoding="utf-8-sig")

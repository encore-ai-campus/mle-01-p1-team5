from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)


# 기존 Vector DB 불러오기
vector_db = Chroma(
    persist_directory="data/vector_db",
    embedding_function=embeddings,
    collection_name="travel_safety",
)


# 저장된 데이터 개수 확인
print("Vector DB 저장 개수:", vector_db._collection.count())


# ==========================================
# 여기만 바꿔서 검색
# ==========================================

query_id = "Q24"
country = "케냐"
query = "케냐에서 강도를 만나면 어떻게 대처해?"

top_k = 15


# ==========================================
# 검색
# ==========================================

if country:
    results = vector_db.similarity_search(
        query,
        k=top_k,
        filter={"국가명": country},
    )
else:
    results = vector_db.similarity_search(
        query,
        k=top_k,
    )


# ==========================================
# 결과 출력
# ==========================================

print("\n" + "=" * 100)
print(f"{query_id}: {query}")
print(f"국가 필터: {country}")
print("=" * 100)

print("검색 결과 개수:", len(results))


for rank, doc in enumerate(results, start=1):
    print(f"\n=== 검색 결과 {rank} ===")
    print("id:", doc.id)
    print("metadata:", doc.metadata)
    print("내용:")
    print(doc.page_content[:700])
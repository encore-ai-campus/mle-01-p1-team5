from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd

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

# query = "가나에서 택시 이용할 때 주의할 점은?"

# results = vector_db.similarity_search(
#     query,
#     k=10,
# )

# print("\n질문:", query)

# for i, doc in enumerate(results, start=1):
#     print(f"\n=== 검색 결과 {i} ===")
#     print("metadata:", doc.metadata)
#     print("내용:")
#     print(doc.page_content[:500])


questions = [
    {
        "id": "Q01",
        "country": "미합중국",
        "question": "여자 혼자 미국 여행해도 괜찮을까? 밤에는 많이 위험해?"
    },
    {
        "id": "Q02",
        "country": "미합중국",
        "question": "미국 로스앤젤레스에서 렌터카로 이동할 때 어떤점을 주의해야 해?"
    },
    {
        "id": "Q03",
        "country": "미합중국",
        "question": "미국 여행 중 경찰에게 검문을 받거나 제지를 당하면 어떻게 행동해야 해?"
    },
    {
        "id": "Q04",
        "country": "일본",
        "question": "일본 여행 중 지진이 자주 발생한다는데 많이 걱정해야 해?"
    },
    {
        "id": "Q05",
        "country": "일본",
        "question": "일본 여행 가면 한국인 관광객을 대상으로 바가지를 씌우는 경우가 많아?"
    },
    {
        "id": "Q06",
        "country": "일본",
        "question": "홋카이도에서 렌터카로 여행하려는데 한국에서 운전할 때와 비교해서 특히 조심해야 할 점이 뭐야?"
    },
    {
        "id": "Q07",
        "country": "중국",
        "question": "중국 여행 가는데 전반적인 치안은 어떤 편이야?"
    },
    {
        "id": "Q08",
        "country": "중국",
        "question": "중국에서 전동스쿠터를 면허 없이 타면 문제가 될 수 있어?"
    },
    {
        "id": "Q09",
        "country": "중국",
        "question": "티베트 라싸 같은 지역은 그냥 자유롭게 여행할 수 있는 게 아니야?"
    },
    {
        "id": "Q10",
        "country": "베트남",
        "question": "베트남에서 휴대폰 들고 걸어 다니면 날치기 위험이 커?"
    },
    {
        "id": "Q11",
        "country": "베트남",
        "question": "베트남 공항에서 Grab 직원이라고 접근하는 사람이 있으면 믿어도 돼?"
    },
    {
        "id": "Q12",
        "country": "베트남",
        "question": "베트남에서는 전자담배를 가지고만 있어도 문제가 될 수 있어?"
    },
    {
        "id": "Q13",
        "country": "태국",
        "question": "태국 여행 가는데 치안 괜찮아?"
    },
    {
        "id": "Q14",
        "country": "태국",
        "question": "태국 야시장이나 클럽처럼 사람이 많은 곳에서는 어떤 범죄를 조심해야 해?"
    },
    {
        "id": "Q15",
        "country": "필리핀",
        "question": "필리핀 요즘 치안 괜찮아? 강도나 소매치기 위험이 커?"
    },
    {
        "id": "Q16",
        "country": "필리핀",
        "question": "필리핀에서 총기나 흉기를 든 강도를 만나면 어떻게 해야 해?"
    },
    {
        "id": "Q17",
        "country": "홍콩",
        "question": "홍콩에 태풍 오면 여행 일정은 어떻게 해야 해?"
    },
    {
        "id": "Q18",
        "country": "홍콩",
        "question": "홍콩에서 분실 신고한 여권을 다시 찾았는데 그대로 써도 돼?"
    },
    {
        "id": "Q19",
        "country": "그리스",
        "question": "그리스 여행 가는데 산불이나 폭염 때문에 위험하지 않을까?"
    },
    {
        "id": "Q20",
        "country": "그리스",
        "question": "아테네 근처에서 산불 대피령이 내려지면 관광객도 바로 이동해야 해?"
    },
    {
        "id": "Q21",
        "country": "남아프리카공화국",
        "question": "남아공 자유여행은 치안 때문에 많이 위험해?"
    },
    {
        "id": "Q22",
        "country": "나이지리아",
        "question": "나이지리아는 여행을 고민할 정도로 납치나 강도 위험이 큰 편이야?"
    },
    {
        "id": "Q23",
        "country": "케냐",
        "question": "케냐 여행할 때 치안 위험은 어느 정도로 생각해야 해?"
    },
    {
        "id": "Q24",
        "country": "케냐",
        "question": "케냐에서 강도를 만나면 어떻게 대처해?"
    },
    {
        "id": "Q25",
        "country": "몰디브",
        "question": "몰디브 리조트면 치안 걱정은 거의 안 해도 돼?"
    },
    {
        "id": "Q26",
        "country": "몰디브",
        "question": "여자 혼자 몰디브 리조트에 머물 때 성범죄나 절도도 조심해야 해?"
    },
    {
        "id": "Q27",
        "country": "몽골",
        "question": "여자 혼자 몽골 여행해도 괜찮을까?"
    },
    {
        "id": "Q28",
        "country": "아랍에미리트",
        "question": "두바이 클럽에서 모르는 사람이 주는 음료를 받아 마셔도 괜찮아?"
    },
    {
        "id": "Q29",
        "country": None,
        "question": "여자 혼자 여행 중 처음 만난 사람이 주는 음료를 마셔도 괜찮을까?"
    },
    {
        "id": "Q30",
        "country": None,
        "question": "여자 혼자 여행할 때 성범죄나 강도 피해를 줄이려면 어떤 상황을 특히 피해야 해?"
    },
]


for item in questions:
    query_id = item["id"]
    country = item["country"]
    query = item["question"]

    print("\n" + "=" * 100)
    print(f"{query_id}: {query}")
    print(f"국가 필터: {country}")
    print("=" * 100)

    if country:
        results = vector_db.similarity_search(
            query,
            k=15,
            filter={"국가명": country},
        )
    else:
        results = vector_db.similarity_search(
            query,
            k=15,
        )

    print("검색 결과 개수:", len(results))

    for rank, doc in enumerate(results, start=1):
        print(f"\n=== 검색 결과 {rank} ===")
        print("id:", doc.id)
        print("metadata:", doc.metadata)
        print("내용:")
        print(doc.page_content[:700])
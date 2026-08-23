import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("financial_reports")

query = "What was total revenue growth?"

result = collection.query(
    query_embeddings=[embedder.encode(query).tolist()],
    n_results=5
)

for doc, meta in zip(
    result["documents"][0],
    result["metadatas"][0]
):
    print(
        f"[{meta['content_type']} | "
        f"{meta['doc_id']} p{meta['page_number']}] "
        f"{doc[:200]}"
    )
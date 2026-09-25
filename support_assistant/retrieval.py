from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOCS = Path("support_assistant/docs")
DB = Path("support_assistant/chroma_db")
COLLECTION = "zepto_policies"

def load_chunks():
    rows=[]
    for path in sorted(DOCS.glob("*.txt")):
        text=path.read_text(encoding="utf-8").strip()
        rows.append((path.stem, text))
    return rows

def get_collection():
    client=chromadb.PersistentClient(path=str(DB))
    return client.get_or_create_collection(COLLECTION)

def build_index():
    model=SentenceTransformer("all-MiniLM-L6-v2")
    collection=get_collection()
    rows=load_chunks()
    ids=[r[0] for r in rows]
    docs=[r[1] for r in rows]
    embeddings=model.encode(docs, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings)
    return collection

def retrieve(query, k=3):
    model=SentenceTransformer("all-MiniLM-L6-v2")
    collection=get_collection()
    if collection.count() == 0:
        collection=build_index()
    q=model.encode([query], normalize_embeddings=True).tolist()
    result=collection.query(query_embeddings=q, n_results=k)
    out=[]
    for i,doc in enumerate(result["documents"][0]):
        out.append({"id":result["ids"][0][i], "content":doc,
                    "distance":result["distances"][0][i] if result.get("distances") else None})
    return out

if __name__=="__main__":
    build_index()
    print("Indexed policy documents:", get_collection().count())

# Module 3 — Support Assistant

The corpus contains the eight exact policy documents specified by the capstone. `retrieval.py` embeds them locally with `all-MiniLM-L6-v2` and stores vectors in ChromaDB.

Architecture:

`ingestion -> embedding -> retrieval -> generation`

- Ingestion: `docs/*.txt`
- Embedding/index: `retrieval.py`
- Retrieval: ChromaDB top-3 cosine-similarity search
- Generation/routing: `graph.py`
- API: `main.py`

The LangGraph `StateGraph` contains `classify_intent`, `retrieve_and_answer`, and `direct_answer`, with a conditional edge after classification. `MOCK_LLM` is the default graded path. Policy queries use real retrieval and return a deterministic context-grounded response; general queries return the required fixed response. The final answer is validated through Pydantic.

Run locally:

```bash
python -m support_assistant.main
uvicorn support_assistant.main:app --reload
```

Docker:

```bash
docker build -t zepto-support .
docker run -p 8000:8000 zepto-support
```

Example request:

```json
{"query":"How long does delivery take?"}
```

Example general query:

```json
{"query":"What is the capital of France?"}
```

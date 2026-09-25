# Zepto Data & AI Platform

End-to-end capstone covering Data Engineering, Analytics/ML, and a grounded GenAI Support Assistant in one repository.

## Modules

- `data_pipeline/` — scrape, clean, enrich, normalize into SQLite, query with SQL and pandas.
- `analytics/` — Titanic EDA, visualization, classification, imbalance handling, tuning, regression, and model persistence.
- `support_assistant/` — local embeddings + ChromaDB retrieval + LangGraph routing + Pydantic output + FastAPI.

## Setup

Create/activate a Python 3.11 virtual environment, then:

```bash
python -m pip install -r requirements.txt
```

The project uses one consolidated `requirements.txt`.

## Run

```bash
python data_pipeline/cleaning.py
python data_pipeline/database.py
python data_pipeline/queries.py

python analytics/run_analytics.py

python support_assistant/main.py
```

For the API:

```bash
uvicorn support_assistant.main:app --reload
```

Then POST JSON to `/ask`:

```json
{"query":"How long does delivery take?"}
```

## Design decisions

### Data pipeline
The scraper uses the first five catalogue pages so the dataset is deterministic and comfortably exceeds the 60-row requirement. Detail pages provide categories. Numeric parsing is defensive; missing numeric values are median-imputed. The required fixed conversion `1 GBP = 105.50 INR` is used. SQLite is normalized into `categories` and `books` with a foreign key.

### Analytics
The raw Titanic dataset is loaded once with Seaborn and immediately saved to `analytics/titanic.csv`; all later work uses that CSV. Train/test splitting happens before model preprocessing. A `ColumnTransformer` keeps imputation, encoding, and scaling inside a train-only pipeline. The final saved artifact is the complete preprocessing + estimator pipeline.

### Support assistant
Eight supplied policy documents are embedded locally with `all-MiniLM-L6-v2` and stored in ChromaDB. LangGraph routes policy questions to retrieval and general questions directly. `MOCK_LLM` defaults to deterministic offline behavior, which is the graded baseline. The API validates the final response with Pydantic.

## Git workflow
The repository is intended to be submitted as one public GitHub repository. The required feature-branch workflow should remain visible in Git history.

## Submission Notes

The project is organized as one repository containing all three required modules.

The required fixed currency conversion baseline for Module 1 is 1 GBP = 105.50 INR.

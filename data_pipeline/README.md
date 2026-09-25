# Module 1 — Data Pipeline

Run from the repository root:

```bash
python data_pipeline/cleaning.py
python data_pipeline/database.py
python data_pipeline/queries.py
```

The first five catalogue pages produce 100 books and multiple categories. The scraper captures title, price, star rating, availability, and category. `price_gbp` is numeric, rating is integer 1–5, and `in_stock` is boolean. Numeric parse failures are median-imputed so malformed rows do not crash the pipeline. The required fixed conversion is `price_inr = price_gbp * 105.50`.

SQLite uses normalized `categories` and `books` tables with a primary-key/foreign-key relationship. Five query outputs are saved as CSV files. The final JOIN is reproduced with `pd.read_sql` and `pd.merge`.

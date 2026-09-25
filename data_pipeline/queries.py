import sqlite3
from pathlib import Path
import pandas as pd

DB = Path("data_pipeline/database/books.db")
OUT = Path("data_pipeline/query_outputs")
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = {
    "query_1_five_star_books.csv": "SELECT title, rating, price_gbp FROM books WHERE rating = 5;",
    "query_2_books_by_price.csv": "SELECT title, price_gbp, rating FROM books ORDER BY price_gbp DESC;",
    "query_3_top_5_expensive.csv": "SELECT title, price_gbp, rating FROM books ORDER BY price_gbp DESC LIMIT 5;",
    "query_4a_categories.csv": "SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
    "query_4b_price_between_20_30.csv": "SELECT title, price_gbp, rating FROM books WHERE price_gbp BETWEEN 20 AND 30 ORDER BY price_gbp;",
    "query_5_books_with_categories.csv": "SELECT b.title,b.price_gbp,b.rating,c.category_name FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.title LIMIT 10;"
}

if __name__ == "__main__":
    con = sqlite3.connect(DB)
    for filename, sql in QUERIES.items():
        df = pd.read_sql(sql, con)
        df.to_csv(OUT / filename, index=False)
        print(f"{filename}: {df.shape}")
        print(df.head())
    sql_join = pd.read_sql(
        "SELECT b.title,b.price_gbp,b.rating,c.category_name FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.title",
        con)
    books = pd.read_sql("SELECT title,price_gbp,rating,category_id FROM books", con)
    cats = pd.read_sql("SELECT category_id,category_name FROM categories", con)
    merged = pd.merge(books, cats, on="category_id", how="inner")[
        ["title","price_gbp","rating","category_name"]].sort_values("title").reset_index(drop=True)
    print("SQL JOIN shape:", sql_join.shape)
    print("Pandas merge shape:", merged.shape)
    print("SQL JOIN vs pandas.merge equivalent:", sql_join.reset_index(drop=True).equals(merged))
    con.close()

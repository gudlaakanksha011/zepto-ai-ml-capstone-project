import sqlite3
from pathlib import Path
from scraper import scrape_books
from cleaning import clean_books

DB = Path("data_pipeline/database/books.db")

def create_database():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL UNIQUE,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        availability TEXT NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY(category_id) REFERENCES categories(category_id)
    );
    """)
    con.commit()
    return con

def load_data(con, df):
    con.execute("DELETE FROM books")
    con.execute("DELETE FROM categories")
    for category in sorted(df["category"].dropna().unique()):
        con.execute("INSERT INTO categories(category_name) VALUES (?)", (category,))
    for _, r in df.iterrows():
        category_id = con.execute(
            "SELECT category_id FROM categories WHERE category_name=?", (r["category"],)
        ).fetchone()[0]
        con.execute("""
            INSERT INTO books(title,price_gbp,price_inr,rating,availability,in_stock,category_id)
            VALUES (?,?,?,?,?,?,?)
        """, (r["title"], r["price_gbp"], r["price_inr"], int(r["rating"]),
              r["availability"], int(r["in_stock"]), category_id))
    con.commit()

if __name__ == "__main__":
    df = clean_books(scrape_books())
    con = create_database()
    load_data(con, df)
    print("Books:", con.execute("SELECT COUNT(*) FROM books").fetchone()[0])
    print("Categories:", con.execute("SELECT COUNT(*) FROM categories").fetchone()[0])
    print("JOIN sample:")
    for row in con.execute("""
        SELECT b.title,b.price_gbp,b.rating,c.category_name
        FROM books b JOIN categories c ON b.category_id=c.category_id
        LIMIT 10
    """):
        print(row)
    con.close()

from scraper import scrape_books
import pandas as pd

GBP_TO_INR = 105.50

def clean_books(df):
    df = df.copy().drop_duplicates().drop_duplicates(subset=["title"])
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(df["rating"].median())
    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df = df[df["rating"].between(1, 5)]
    df = df[df["price_gbp"] > 0]
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    return df.reset_index(drop=True)

if __name__ == "__main__":
    raw = scrape_books()
    clean = clean_books(raw)
    print("Rows:", len(clean))
    print("Categories:", clean["category"].nunique())
    print(clean.isna().sum())
    print(clean.dtypes)

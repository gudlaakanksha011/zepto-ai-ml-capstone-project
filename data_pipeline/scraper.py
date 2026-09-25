import re
from urllib.parse import urljoin
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape_books(pages=5):
    rows = []
    for page_number in range(1, pages + 1):
        page_url = BASE_URL if page_number == 1 else f"{BASE_URL}catalogue/page-{page_number}.html"
        response = requests.get(page_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for book in soup.select("article.product_pod"):
            title = book.select_one("h3 a")["title"]
            price_text = book.select_one("p.price_color").get_text(strip=True)
            match = re.search(r"\d+(?:\.\d+)?", price_text)
            price_gbp = float(match.group()) if match else None
            rating_class = book.select_one("p.star-rating").get("class", [])[-1]
            rating = RATING_MAP.get(rating_class)
            availability = book.select_one("p.availability").get_text(" ", strip=True)
            in_stock = availability.lower() == "in stock"
            detail_url = urljoin(page_url, book.select_one("h3 a")["href"])
            detail = requests.get(detail_url, timeout=30)
            category = None
            if detail.ok:
                detail_soup = BeautifulSoup(detail.text, "html.parser")
                crumbs = detail_soup.select("ul.breadcrumb li")
                if len(crumbs) >= 3:
                    category = crumbs[2].get_text(strip=True)
            rows.append({"title": title, "price_gbp": price_gbp, "rating": rating,
                         "availability": availability, "in_stock": in_stock, "category": category})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = scrape_books()
    print(df.shape)
    print(df.head())
    print("Categories:", df["category"].nunique())

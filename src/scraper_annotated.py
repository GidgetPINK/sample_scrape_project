#!/usr/bin/env python3
"""
scraper_annotated.py
--------------------
A fully commented, teaching-focused version of the scraping pipeline.

Goal
----
Parse a (demo) paginated website of company records, clean/standardize
the fields, then export the results to both CSV and SQLite.

This script is intentionally written for readability and learning:
- Clear function boundaries
- Docstrings + inline comments
- Defensive parsing (graceful fallbacks)
- Simple, reproducible outputs

Project Layout (expected)
-------------------------
sample_scrape_project/
├─ site/                 # Local demo "website" with HTML pages (page1.html → page2.html → page3.html)
├─ src/
│  ├─ scraper_annotated.py   # <-- this file
│  └─ requirements.txt
├─ output/               # Results are written here
└─ data/                 # (optional) for any future inputs

How to run
----------
$ python src/scraper_annotated.py

Outputs
-------
- output/companies.csv      (clean tabular data)
- output/companies.db       (SQLite database with 'companies' table)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import sqlite3

import pandas as pd
from bs4 import BeautifulSoup

# ------------------------------
# Paths and basic configuration
# ------------------------------

# ROOT is the repository root (two levels up from this file).
ROOT = Path(__file__).resolve().parents[1]

# Folder containing our local demo "website" pages.
SITE_DIR = ROOT / "site"

# Folder where we'll write our results (CSV + SQLite DB).
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------
# Cleaning helpers
# ------------------------------
def normalize_phone(raw: str) -> str:
    """
    Normalize phone numbers into a simplified canonical format.

    Strategy
    --------
    - Keep only digits and an optional leading '+' (strip spaces, punctuation).
    - If no '+' is present and the number looks like a 10/11-digit US number,
      coerce to +1########## for consistency.
    - Return empty string if input is obviously missing (e.g., 'N/A').

    Examples
    --------
    "(555) 123-4567"   -> "+15551234567"
    "1-555-123-4567"   -> "+15551234567"
    "+44 20 7946 0958" -> "+442079460958" (keeps leading '+')
    "N/A"              -> ""
    """
    if not raw or raw.strip().lower() in {"n/a", "na", "none"}:
        return ""

    # Keep digits and '+' only. This collapses spaces, dashes, parentheses, etc.
    digits = re.sub(r"[^\d+]", "", raw)

    # If there's no '+' prefix, infer US formatting where reasonable.
    if digits and not digits.startswith("+"):
        # 11-digit starting with leading country code '1'
        if len(digits) == 11 and digits.startswith("1"):
            return f"+{digits}"
        # 10-digit US local number
        if len(digits) == 10:
            return f"+1{digits}"

    return digits


def normalize_country(raw: Optional[str]) -> str:
    """
    Standardize country names to a small controlled vocabulary.

    - Trim whitespace
    - Map common variants to canonical labels
    - Fallback to Title Case for unknowns
    """
    if not raw:
        return ""
    s = raw.strip()

    # Known variants map → canonical value
    mapping = {
        "usa": "United States",
        "united states": "United States",
        "canada": "Canada",
        "uk": "United Kingdom",
        "united kingdom": "United Kingdom",
        "mexico": "Mexico",
    }

    key = s.lower()
    if key in mapping:
        return mapping[key]

    # Fallback: Title Case (e.g., "bRaZiL" -> "Brazil")
    return s.title()


# ------------------------------
# HTML parsing helpers
# ------------------------------
def parse_company_table(html: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Parse a single HTML page for the company table.

    Parameters
    ----------
    html : str
        The full HTML content of the page.

    Returns
    -------
    rows : list of dict
        Each dict contains a single row from the table with keys:
        ["CompanyID", "CompanyName", "Category", "Email", "Phone", "Country"]
    next_href : Optional[str]
        The relative link to the next page (if a "Next" link exists), else None.
    """
    soup = BeautifulSoup(html, "lxml")

    # Locate the table by id (set in our demo pages).
    table = soup.find("table", id="company-table")
    rows: List[Dict[str, str]] = []

    if not table:
        # Defensive: if the table isn't found, return empty results.
        return rows, None

    # Iterate over all table body rows.
    tbody = table.find("tbody")
    if not tbody:
        return rows, None

    for tr in tbody.find_all("tr"):
        # Extract text from each <td>, stripping extra whitespace.
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]

        # Expecting exactly 6 columns; skip malformed rows gracefully.
        if len(cols) != 6:
            continue

        rows.append(
            {
                "CompanyID": cols[0],
                "CompanyName": cols[1],
                "Category": cols[2],
                "Email": cols[3],
                "Phone": cols[4],
                "Country": cols[5],
            }
        )

    # Find a "Next" link (if present) to handle pagination.
    next_link = soup.find("a", id="next")
    next_href = next_link["href"] if next_link and next_link.has_attr("href") else None

    return rows, next_href


# ------------------------------
# Scrape/paginate over the demo site
# ------------------------------
def scrape_demo_site(start_page: str = "page1.html") -> pd.DataFrame:
    """
    Walk the local demo site pages, parse each table, and accumulate rows.

    Parameters
    ----------
    start_page : str
        The filename of the first page to parse (within SITE_DIR).

    Returns
    -------
    pd.DataFrame
        Raw, uncleaned rows from all pages concatenated together.
    """
    current = (SITE_DIR / start_page).resolve()
    seen = set()  # Protect against accidental loops
    all_rows: List[Dict[str, str]] = []

    # Traverse pages until no "next" link is found.
    while current and current.exists():
        if current in seen:
            break  # Safety: avoid infinite loops on circular pagination
        seen.add(current)

        html = current.read_text(encoding="utf-8")
        rows, next_href = parse_company_table(html)
        all_rows.extend(rows)

        # Move to next page if available; else stop.
        current = (current.parent / next_href).resolve() if next_href else None

    # Convert the list of dicts → DataFrame
    df = pd.DataFrame(all_rows, columns=["CompanyID", "CompanyName", "Category", "Email", "Phone", "Country"])
    return df


# ------------------------------
# Cleaning / standardization pipeline
# ------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of cleaning steps to standardize the dataset.

    Steps
    -----
    - Convert CompanyID to integer (nullable Int64 for safety)
    - Trim/Title-Case names & categories
    - Lowercase & trim emails; empty strings become NaN
    - Normalize phone numbers into a consistent canonical form
    - Normalize countries to canonical labels
    - Drop duplicates on (CompanyID, Email)

    Returns
    -------
    pd.DataFrame
        Cleaned, deduplicated DataFrame.
    """
    # Convert to numeric safely; non-convertible become <NA> (nullable Int64)
    df["CompanyID"] = pd.to_numeric(df["CompanyID"], errors="coerce").astype("Int64")

    # Basic text normalization
    df["CompanyName"] = df["CompanyName"].astype(str).str.strip().str.title()
    df["Category"]    = df["Category"].astype(str).str.strip().str.title()

    # Emails: lowercase, strip spaces; coerce empty strings to NaN
    df["Email"] = (
        df["Email"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": pd.NA})
    )

    # Phones: normalize via helper; coerce empty strings to NaN
    df["Phone"] = (
        df["Phone"]
        .astype(str)
        .map(normalize_phone)
        .replace({"": pd.NA})
    )

    # Countries: controlled vocabulary via helper
    df["Country"] = df["Country"].apply(normalize_country)

    # Remove duplicate rows by a logical business key
    df = df.drop_duplicates(subset=["CompanyID", "Email"], keep="first")

    return df


# ------------------------------
# Persistence (CSV + SQLite)
# ------------------------------
def save_outputs(df: pd.DataFrame) -> None:
    """
    Save the cleaned DataFrame to CSV and a SQLite database.

    - CSV is easy for humans and Excel.
    - SQLite is great for programmatic querying and demos.
    """
    csv_path = OUTPUT_DIR / "companies.csv"
    db_path = OUTPUT_DIR / "companies.db"

    # 1) CSV export
    df.to_csv(csv_path, index=False)

    # 2) SQLite export
    with sqlite3.connect(db_path) as conn:
        df.to_sql("companies", conn, if_exists="replace", index=False)

    # Console summary
    print(f"Saved CSV    → {csv_path}")
    print(f"Saved SQLite → {db_path}")
    print("\nDataFrame info:")
    print(df.info())
    print("\nCounts by Country:")
    print(df["Country"].value_counts(dropna=False))


# ------------------------------
# Orchestration (main entry point)
# ------------------------------
def main() -> None:
    """End-to-end run: scrape → clean → persist."""
    # 1) Extract (from local demo pages)
    raw = scrape_demo_site(start_page="page1.html")

    # 2) Transform / Clean
    clean = clean_dataframe(raw)

    # 3) Load (save to CSV + SQLite)
    save_outputs(clean)


if __name__ == "__main__":
    main()

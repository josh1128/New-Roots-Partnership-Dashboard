# halifax_chamber_map.py
# Streamlit app: scrape Halifax Chamber "Not-For-Profit/Charitable" list,
# parse addresses, geocode, and map nonprofits in Halifax.
#
# Run:
#   pip install -r requirements.txt
#   streamlit run halifax_chamber_map.py
#
# Notes:
# - Uses Halifax Chamber directory page as the data source.
# - Respects geocoding rate limit (1 req/sec) and caches results.
# - Best for personal/operational use. If republishing, review Chamber terms.

from __future__ import annotations
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(page_title="Halifax Nonprofits — Chamber Directory Map", layout="wide")
st.title("Halifax Nonprofits — Chamber Directory Map")

CATEGORY_URL = "https://business.halifaxchamber.com/members/category/not-for-profit-groups-charitable-organizations-87"
SHEET_NAME = "nonprofits"  # internal only if you export
CACHE_FILE = Path("data/geocode_cache.csv")  # persisted cache (best effort)

# Halifax bounds (sanity checks)
HAL_LAT = (44.3, 45.1)
HAL_LON = (-64.2, -62.9)

# ------------------ Scrape helpers ------------------ #
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

def parse_listing_page(url: str) -> list[dict]:
    """
    Return list of {name, detail_url} from the category page.
    The page shows 'Results Found: N' and lists member cards with links.
    """
    soup = get_soup(url)
    items: list[dict] = []
    # Strategy: find all links to /members/member/...
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/members/member/" in href:
            # filter duplicates via name anchor text
            name = a.get_text(strip=True)
            if not name:
                continue
            # Avoid collecting "Visit Website" etc. Restrict: anchor inside H5 or strong-ish headings.
            parent_tag = a.find_parent()
            # Keep simple: accept unique (name, href) pairs
            items.append({"name": name, "detail_url": requests.compat.urljoin(url, href)})
    # De-dup by detail_url
    seen = set()
    uniq = []
    for it in items:
        if it["detail_url"] in seen:
            continue
        seen.add(it["detail_url"])
        uniq.append(it)
    return uniq

def text_or_none(el) -> str:
    return el.get_text(" ", strip=True) if el else ""

def parse_member_detail(url: str) -> dict:
    """
    From a member detail page, parse:
      - name (H1)
      - address (first 'maps/google' link text or nearby text)
      - website (anchor containing 'Visit Website')
      - phone(s) (naive regex over page text)
      - about/description (optional)
    """
    soup = get_soup(url)

    # Name
    h1 = soup.find(["h1", "h2"])
    name = text_or_none(h1) or ""

    # Address: often appears as an <a> with href to Google
    addr = ""
    for a in soup.find_all("a", href=True):
        if "google." in a["href"]:
            candidate = a.get_text(" ", strip=True)
            # Basic sanity: must include NS (Nova Scotia) or Halifax/Dartmouth
            if "NS" in candidate or "Halifax" in candidate or "Dartmouth" in candidate or "Bedford" in candidate:
                addr = candidate
                break

    # Website: link with text containing "Visit Website"
    website = ""
    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True).lower()
        if "visit website" in label:
            website = a["href"]
            break

    # Phone(s): simple regex over fu


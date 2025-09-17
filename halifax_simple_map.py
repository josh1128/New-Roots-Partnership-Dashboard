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

    # Phone(s): simple regex over full text
    page_text = soup.get_text(" ", strip=True)
    phones = re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", page_text)
    phone = ", ".join(sorted(set(phones)))[:80]

    # About (optional)
    about = ""
    about_hdr = soup.find(lambda t: t.name in ["h2", "h3"] and "about" in t.get_text(strip=True).lower())
    if about_hdr:
        # grab next sibling text block
        sib = about_hdr.find_next()
        about = text_or_none(sib)

    return {"org_name": name, "address": addr, "website": website, "phone": phone, "about": about, "detail_url": url}

# ------------------ Geocoding ------------------ #
@st.cache_data
def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        try:
            df = pd.read_csv(CACHE_FILE)
            if not {"address", "lat", "lon"}.issubset(df.columns):
                return pd.DataFrame(columns=["address", "lat", "lon"])
            df["address_norm"] = df["address"].astype(str).str.strip().str.lower()
            return df
        except Exception:
            return pd.DataFrame(columns=["address", "lat", "lon"])
    return pd.DataFrame(columns=["address", "lat", "lon"])

def save_cache(cache_df: pd.DataFrame):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(CACHE_FILE, index=False)
    except Exception:
        pass

@st.cache_data(show_spinner=False)
def geocode_addresses(addresses: tuple[str, ...]) -> dict[str, Tuple[Optional[float], Optional[float]]]:
    geocoder = Nominatim(user_agent="halifax_nonprofits_map")
    geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)
    out: dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for raw in addresses:
        q = f"{raw}, Halifax Regional Municipality, Nova Scotia, Canada"
        loc = geocode(q)
        if loc:
            out[raw] = (loc.latitude, loc.longitude)
        else:
            out[raw] = (None, None)
    return out

def fill_missing_coords(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    msgs = []
    cache = load_cache().copy()
    if "address_norm" not in cache.columns:
        cache["address_norm"] = cache["address"].astype(str).str.strip().str.lower()

    df["address_norm"] = df["address"].astype(str).str.strip().str.lower()
    # merge cached coords
    df = df.merge(cache[["address_norm", "lat", "lon"]].rename(columns={"lat": "lat_cached", "lon": "lon_cached"}),
                  on="address_norm", how="left")
    if "lat" not in df.columns: df["lat"] = None
    if "lon" not in df.columns: df["lon"] = None
    df["lat"] = df["lat"].fillna(df["lat_cached"])
    df["lon"] = df["lon"].fillna(df["lon_cached"])

    todo = df[df["lat"].isna() | df["lon"].isna()]["address"].dropna().unique().tolist()
    if todo:
        st.info(f"Geocoding {len(todo)} address(es)… (OpenStreetMap, 1 req/sec)")
        lookups = geocode_addresses(tuple(todo))
        # apply back + upsert cache
        for addr in todo:
            lat, lon = lookups.get(addr, (None, None))
            if lat is not None and lon is not None:
                df.loc[df["address"] == addr, "lat"] = lat
                df.loc[df["address"] == addr, "lon"] = lon
                row = {"address": addr, "lat": lat, "lon": lon, "address_norm": addr.strip().lower()}
                cache = pd.concat([cache[cache["address_norm"] != row["address_norm"]], pd.DataFrame([row])],
                                  ignore_index=True)
        save_cache(cache)
        good = sum(1 for a in todo if lookups.get(a, (None, None))[0] is not None)
        msgs.append(f"Geocoded {good}/{len(todo)} new address(es).")

    df.drop(columns=[c for c in ["lat_cached", "lon_cached", "address_norm"] if c in df.columns], inplace=True)
    return df, msgs

# ------------------ Scrape + assemble ------------------ #
@st.cache_data
def build_directory_dataframe(url: str) -> pd.DataFrame:
    listing = parse_listing_page(url)

    # De-dup by detail URL
    seen = set()
    to_visit = []
    for item in listing:
        href = item["detail_url"]
        if href in seen:
            continue
        seen.add(href)
        to_visit.append(href)

    rows = []
    # Visit each member page; be polite to the site (short sleep)
    for i, href in enumerate(to_visit, 1):
        try:
            detail = parse_member_detail(href)
            # fallback to listing name if detail missing name
            if not detail.get("org_name"):
                detail["org_name"] = next((x["name"] for x in listing if x["detail_url"] == href), "")
            rows.append(detail)
        except Exception as e:
            rows.append({"org_name": "", "address": "", "website": "", "phone": "", "about": "", "detail_url": href, "error": str(e)})
        time.sleep(0.2)  # light throttle

    df = pd.DataFrame(rows)
    # Derive area (very rough) from address text
    def infer_area(addr: str) -> str:
        a = (addr or "").lower()
        if "dartmouth" in a: return "Dartmouth"
        if "bedford" in a: return "Bedford"
        if "halifax" in a: return "Halifax"
        return "HRM"
    df["area"] = df["address"].apply(infer_area)
    return df

with st.spinner("Fetching Chamber directory…"):
    df = build_directory_dataframe(CATEGORY_URL)

# Clean + geocode
keep_cols = ["org_name", "area", "address", "website", "phone", "about", "detail_url"]
df = df[keep_cols].copy()
df, geo_msgs = fill_missing_coords(df)

# Filter out anything wildly outside Halifax (sanity)
oob = (~df["lat"].between(*HAL_LAT) | ~df["lon"].between(*HAL_LON))
oob_count = int(oob.sum())
if oob_count:
    st.warning(f"{oob_count} location(s) outside typical Halifax bounds (may be PO boxes or province-wide). They remain on map.")

# Report + preview
pins = df.dropna(subset=["lat", "lon"])
st.success(f"Loaded nonprofits: {len(df)} · Mappable pins: {len(pins)}")
for m in geo_msgs:
    st.info(m)

with st.expander("Preview data"):
    st.dataframe(df.sort_values("org_name").reset_index(drop=True), use_container_width=True)

# Optional: export to CSV for reuse
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download CSV (for your repo/app)", pins[["org_name","area","address","lat","lon","website","phone","about","detail_url"]].to_csv(index=False),
                       file_name="halifax_nonprofits_from_chamber.csv", mime="text/csv")

# ------------------ Map ------------------ #
def compute_view(d: pd.DataFrame) -> pdk.ViewState:
    if d.empty: return pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)
    return pdk.ViewState(latitude=float(d["lat"].mean()), longitude=float(d["lon"].mean()), zoom=11, pitch=0)

tooltip = {
    "html": (
        "<b>{org_name}</b><br/>"
        "{address}<br/>"
        "<i>{about}</i><br/>"
        "{phone}<br/>"
        "<a href='{website}' target='_blank'>{website}</a>"
    ),
    "style": {"backgroundColor": "white", "color": "black"},
}

layer = pdk.Layer(
    "ScatterplotLayer",
    data=pins,
    get_position='[lon, lat]',
    get_radius=100,
    pickable=True,
    auto_highlight=True,
)

deck = pdk.Deck(layers=[layer], initial_view_state=compute_view(pins), tooltip=tooltip)
st.pydeck_chart(deck, use_container_width=True)

st.caption("Data source: Halifax Chamber member directory (Not-For-Profit/Charitable). Use responsibly for outreach; verify details before contact.")

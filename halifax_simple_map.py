# halifax_chamber_map_dots.py
# Streamlit app: scrape Halifax Chamber Not-For-Profit directory, geocode, and show
# BIG, VISIBLE DOTS on a Halifax map with tooltips.
#
# New:
# - Download current nonprofits as Excel
# - Download a blank Excel template
# - Upload an Excel (template or previously downloaded) to add/edit nonprofits
#   → Newly added rows are geocoded and appear on the map
#
# Run:
#   pip install -r requirements.txt
#   streamlit run halifax_chamber_map_dots.py
#
# requirements.txt (minimum):
#   streamlit>=1.32
#   pandas>=2.0
#   pydeck>=0.9
#   beautifulsoup4>=4.12
#   lxml>=4.9
#   requests>=2.32
#   geopy>=2.4
#   openpyxl>=3.1

from __future__ import annotations

import io
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

st.set_page_config(page_title="Halifax Nonprofits — Chamber Directory (Dots)", layout="wide")
st.title("Halifax Nonprofits — Chamber Directory (Dots Map)")

CATEGORY_URL = "https://business.halifaxchamber.com/members/category/not-for-profit-groups-charitable-organizations-87"
CACHE_FILE = Path("data/geocode_cache.csv")  # persisted cache if filesystem is writable

# Halifax sanity bounds
HAL_LAT = (44.3, 45.1)
HAL_LON = (-64.2, -62.9)

# ------------------ Sidebar dot styling ------------------ #
with st.sidebar:
    st.header("Dot style")
    radius_px = st.slider("Dot radius (pixels)", 40, 220, 120, 5)
    opacity_pct = st.slider("Opacity (%)", 20, 100, 90, 5)
    color_name = st.selectbox(
        "Dot color",
        ["Electric Blue", "Orange Red", "Emerald", "Purple", "Charcoal"],
        index=0
    )

COLOR_MAP = {
    "Electric Blue": (0, 153, 255),
    "Orange Red": (255, 69, 0),
    "Emerald": (0, 178, 120),
    "Purple": (138, 43, 226),
    "Charcoal": (40, 40, 40),
}
RGB = COLOR_MAP[color_name]
ALPHA = int(round(opacity_pct / 100 * 255))


# ------------------ Scrape helpers ------------------ #
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_listing_page(url: str) -> list[dict]:
    """Return list of {name, detail_url} from the category page."""
    soup = get_soup(url)
    items: list[dict] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/members/member/" in href:
            name = a.get_text(strip=True)
            if name:
                items.append({"name": name, "detail_url": requests.compat.urljoin(url, href)})
    # de-dup by detail_url
    seen, uniq = set(), []
    for it in items:
        if it["detail_url"] in seen:
            continue
        seen.add(it["detail_url"])
        uniq.append(it)
    return uniq


def txt(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def parse_member_detail(url: str) -> dict:
    """Grab org_name, address, website, phone, about from a member page."""
    soup = get_soup(url)
    name = txt(soup.find(["h1", "h2"])) or ""

    # Address: often a link pointing to Google Maps
    addr = ""
    for a in soup.find_all("a", href=True):
        if "google." in a["href"]:
            cand = a.get_text(" ", strip=True)
            if any(k in cand for k in ["NS", "Halifax", "Dartmouth", "Bedford"]):
                addr = cand
                break

    # Website
    website = ""
    for a in soup.find_all("a", href=True):
        if "visit website" in a.get_text(" ", strip=True).lower():
            website = a["href"]
            break

    # Phones (simple regex)
    page_text = soup.get_text(" ", strip=True)
    phones = re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", page_text)
    phone = ", ".join(sorted(set(phones)))[:80]

    # About (optional)
    about = ""
    about_hdr = soup.find(lambda t: t.name in ["h2", "h3"] and "about" in t.get_text(strip=True).lower())
    if about_hdr:
        nxt = about_hdr.find_next()
        about = txt(nxt)

    return {
        "org_name": name,
        "address": addr,
        "website": website,
        "phone": phone,
        "about": about,
        "detail_url": url,
    }


# ------------------ Geocoding (cached) ------------------ #
@st.cache_data
def load_cache() -> pd.DataFrame:
    if CACHE_FILE.exists():
        try:
            c = pd.read_csv(CACHE_FILE)
            if {"address", "lat", "lon"}.issubset(c.columns):
                c["address_norm"] = c["address"].astype(str).str.strip().str.lower()
                return c
        except Exception:
            pass
    return pd.DataFrame(columns=["address", "lat", "lon", "address_norm"])


def save_cache(cache_df: pd.DataFrame):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(CACHE_FILE, index=False)
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def geocode_batch(addresses: tuple[str, ...]) -> dict[str, tuple[float | None, float | None]]:
    geocoder = Nominatim(user_agent="halifax_nonprofits_map_dots")
    geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)

    results: dict[str, tuple[float | None, float | None]] = {}
    for addr in addresses:
        if not isinstance(addr, str) or not addr.strip():
            results[addr] = (None, None)
            continue
        q = f"{addr}, Halifax Regional Municipality, Nova Scotia, Canada"
        loc = geocode(q)
        if loc:
            results[addr] = (loc.latitude, loc.longitude)
        else:
            results[addr] = (None, None)
    return results


def fill_coords(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    msgs = []
    cache = load_cache().copy()
    if "address_norm" not in cache.columns:
        cache["address_norm"] = cache["address"].astype(str).str.strip().str.lower()

    df["address"] = df["address"].fillna("").astype(str)
    df["address_norm"] = df["address"].astype(str).str.strip().str.lower()

    df = df.merge(
        cache[["address_norm", "lat", "lon"]].rename(columns={"lat": "lat_cached", "lon": "lon_cached"}),
        on="address_norm",
        how="left",
    )

    if "lat" not in df.columns:
        df["lat"] = None
    if "lon" not in df.columns:
        df["lon"] = None

    df["lat"] = df["lat"].fillna(df["lat_cached"])
    df["lon"] = df["lon"].fillna(df["lon_cached"])

    todo = df[df["lat"].isna() | df["lon"].isna()]["address"].dropna().unique().tolist()
    if todo:
        st.info(f"Geocoding {len(todo)} address(es)… (OpenStreetMap, ~1 req/sec)")
        lookups = geocode_batch(tuple(todo))
        for addr in todo:
            lat, lon = lookups.get(addr, (None, None))
            if lat is not None and lon is not None:
                df.loc[df["address"] == addr, ["lat", "lon"]] = (lat, lon)
                row = {
                    "address": addr,
                    "lat": lat,
                    "lon": lon,
                    "address_norm": addr.strip().lower(),
                }
                cache = pd.concat(
                    [cache[cache["address_norm"] != row["address_norm"]], pd.DataFrame([row])],
                    ignore_index=True,
                )
        save_cache(cache)
        good = sum(1 for a in todo if lookups.get(a, (None, None))[0] is not None)
        msgs.append(f"Geocoded {good}/{len(todo)} new address(es).")

    df.drop(columns=[c for c in ["lat_cached", "lon_cached", "address_norm"] if c in df.columns], inplace=True)
    return df, msgs


# ------------------ Build dataset (scraped) ------------------ #
@st.cache_data
def build_directory_dataframe(url: str) -> pd.DataFrame:
    listing = parse_listing_page(url)

    rows = []
    seen = set()
    for i, item in enumerate(listing, 1):
        href = item["detail_url"]
        if href in seen:
            continue
        seen.add(href)
        try:
            detail = parse_member_detail(href)
            if not detail.get("org_name"):
                detail["org_name"] = item["name"]
            rows.append(detail)
        except Exception as e:
            rows.append(
                {
                    "org_name": item["name"],
                    "address": "",
                    "website": "",
                    "phone": "",
                    "about": "",
                    "detail_url": href,
                    "error": str(e),
                }
            )
        time.sleep(0.2)  # be polite

    df = pd.DataFrame(rows)

    # naive area from address
    def infer_area(addr: str) -> str:
        a = (addr or "").lower()
        if "dartmouth" in a:
            return "Dartmouth"
        if "bedford" in a:
            return "Bedford"
        if "halifax" in a:
            return "Halifax"
        return "HRM"

    df["area"] = df["address"].apply(infer_area)
    return df


# ------------------ Excel helpers (NEW) ------------------ #
REQUIRED_COLUMNS = ["org_name", "address", "website", "phone", "about"]
OPTIONAL_COLUMNS = ["area", "detail_url", "lat", "lon"]

def make_template_df() -> pd.DataFrame:
    # Provide empty template with required + optional columns (lat/lon optional)
    cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    return pd.DataFrame(columns=cols)

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "nonprofits") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure expected columns exist; fill missing optional ones
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    # keep a consistent column order
    return df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS].copy()

def merge_uploaded(scraped: pd.DataFrame, uploaded: pd.DataFrame) -> pd.DataFrame:
    """
    Merge strategy:
    - Key is (org_name, address) case-insensitive trim
    - Uploaded rows override scraped on conflicts
    - Include rows unique to either side
    """
    a = scraped.copy()
    b = uploaded.copy()
    for d in (a, b):
        d["__key"] = (
            d["org_name"].astype(str).str.strip().str.lower()
            + " | "
            + d["address"].astype(str).str.strip().str.lower()
        )

    # Keep uploaded first to override, then add scraped that aren't in uploaded
    combined = pd.concat([b, a[~a["__key"].isin(b["__key"])]], ignore_index=True)
    combined.drop(columns=["__key"], inplace=True, errors="ignore")
    return combined


# ------------------ Main flow ------------------ #
with st.spinner("Fetching Halifax Chamber directory…"):
    df_scraped = build_directory_dataframe(CATEGORY_URL)

keep = ["org_name", "area", "address", "website", "phone", "about", "detail_url"]
df_scraped = df_scraped[keep].copy()

# Upload Excel to add/edit nonprofits (NEW)
st.subheader("Add or edit nonprofits via Excel (optional)")
st.markdown(
    "Upload an **Excel file** with columns: "
    "`org_name`, `address`, `website`, `phone`, `about` "
    "(optional: `area`, `detail_url`, `lat`, `lon`). "
    "If `lat`/`lon` are missing, the app will geocode from `address`."
)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    template_bytes = to_excel_bytes(make_template_df())
    st.download_button(
        "Download blank Excel template",
        data=template_bytes,
        file_name="nonprofits_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with c2:
    # Also offer the scraped dataset for editing
    scraped_bytes = to_excel_bytes(normalize_columns(df_scraped))
    st.download_button(
        "Download current nonprofits (Excel)",
        data=scraped_bytes,
        file_name="nonprofits_current.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

uploaded_df = None
uploaded_file = st.file_uploader("Upload Excel (.xlsx) to add/edit nonprofits", type=["xlsx"])
if uploaded_file is not None:
    try:
        df_up = pd.read_excel(uploaded_file, engine="openpyxl")
        uploaded_df = normalize_columns(df_up)
        st.success(f"Uploaded rows: {len(uploaded_df)}")
        with st.expander("Preview uploaded data"):
            st.dataframe(uploaded_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")

# Merge scraped + uploaded (uploaded overrides on key)
if uploaded_df is not None and not uploaded_df.empty:
    df_combined = merge_uploaded(df_scraped, uploaded_df)
else:
    df_combined = df_scraped.copy()

# Geocode missing coords and prep for map
df_combined, geocode_msgs = fill_coords(df_combined)

# Sanity: outside Halifax bounds
oob = (~df_combined["lat"].between(*HAL_LAT) | ~df_combined["lon"].between(*HAL_LON))
oob_count = int(oob.sum())

pins = df_combined.dropna(subset=["lat", "lon"]).copy()

st.success(f"Loaded nonprofits: {len(df_combined)} · Mappable dots: {len(pins)}")
for m in geocode_msgs:
    st.info(m)
if oob_count:
    st.warning(f"{oob_count} location(s) outside typical Halifax bounds (may be PO boxes / province-wide).")

with st.expander("Preview data"):
    st.dataframe(df_combined.sort_values("org_name").reset_index(drop=True), use_container_width=True)

# ------------------ Map (visible dots) ------------------ #
def compute_view(d: pd.DataFrame) -> pdk.ViewState:
    if d.empty:
        return pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)
    return pdk.ViewState(
        latitude=float(d["lat"].mean()),
        longitude=float(d["lon"].mean()),
        zoom=11,
        pitch=0,
    )

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

dot_color = [RGB[0], RGB[1], RGB[2], ALPHA]
dots = pdk.Layer(
    "ScatterplotLayer",
    data=pins,
    get_position='[lon, lat]',
    get_radius=radius_px,  # BIG, visible dots
    filled=True,
    stroked=False,  # no thin outlines
    get_fill_color=dot_color,  # constant color with chosen opacity
    pickable=True,
    auto_highlight=True,
)

deck = pdk.Deck(layers=[dots], initial_view_state=compute_view(pins), tooltip=tooltip)
st.pydeck_chart(deck, use_container_width=True)
st.caption("Dots are sized for visibility. Adjust size/opacity/color in the sidebar.")

# ------------------ Download the current combined dataset (NEW) ------------------ #
st.subheader("Export")
st.markdown("Download everything currently on the map (scraped + uploaded edits).")
combined_bytes = to_excel_bytes(normalize_columns(df_combined))
st.download_button(
    "Download combined nonprofits (Excel)",
    data=combined_bytes,
    file_name="nonprofits_combined.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)



# halifax_chamber_map_dots.py
# Streamlit app: scrape Halifax Chamber Not-For-Profit directory, geocode, and show
# BIG, VISIBLE DOTS on a Halifax map with tooltips.
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
        st.info(f"Geocoding {len(todo)} address(es)… (OpenStreetMap, 1 req/sec)")
        lookups = geocode_batch(tuple(todo))
        for addr in todo:
            lat, lon = lookups.get(addr, (None, None))
            if lat is not None and lon is not None:
                df.loc[df["address"] == addr, ["lat", "lon"]] = (lat, lon)
                row = {"address": addr, "lat": lat, "lon": lon, "address_norm": addr.strip().lower()}
                cache = pd.concat(
                    [cache[cache["address_norm"] != row["address_norm"]], pd.DataFrame([row])],
                    ignore_index=True,
                )
        save_cache(cache)
        good = sum(1 for a in todo if lookups.get(a, (None, None))[0] is not None)
        msgs.append(f"Geocoded {good}/{len(todo)} new address(es).")

    df.drop(columns=[c for c in ["lat_cached", "lon_cached", "address_norm"] if c in df.columns], inplace=True)
    return df, msgs


# ------------------ Build dataset ------------------ #
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
        time.sleep(0.2)

    df = pd.DataFrame(rows)

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


with st.spinner("Fetching Halifax Chamber directory…"):
    df = build_directory_dataframe(CATEGORY_URL)

keep = ["org_name", "area", "address", "website", "phone", "about", "detail_url"]
df = df[keep].copy()

df, geocode_msgs = fill_coords(df)

oob = (~df["lat"].between(*HAL_LAT) | ~df["lon"].between(*HAL_LON))
oob_count = int(oob.sum())

pins = df.dropna(subset=["lat", "lon"]).copy()

st.success(f"Loaded nonprofits: {len(df)} · Mappable dots: {len(pins)}")
for m in geocode_msgs:
    st.info(m)
if oob_count:
    st.warning(f"{oob_count} location(s) outside typical Halifax bounds (may be PO boxes / province-wide).")

with st.expander("Preview data"):
    st.dataframe(df.sort_values("org_name").reset_index(drop=True), use_container_width=True)


def compute_view(d: pd.DataFrame) -> pdk.ViewState:
    if d.empty:
        return pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)
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

dot_color = [RGB[0], RGB[1], RGB[2], ALPHA]
dots = pdk.Layer(
    "ScatterplotLayer",
    data=pins,
    get_position='[lon, lat]',
    get_radius=radius_px,
    filled=True,
    stroked=False,
    get_fill_color=dot_color,
    pickable=True,
    auto_highlight=True,
)

deck = pdk.Deck(layers=[dots], initial_view_state=compute_view(pins), tooltip=tooltip)
st.pydeck_chart(deck, use_container_width=True)
st.caption("Dots are sized for visibility. Adjust size/opacity/color in the sidebar.")


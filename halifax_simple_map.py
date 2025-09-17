# halifax_map_from_repo_geocode.py
# Streamlit app that loads an Excel/CSV from your GitHub repo and shows a Halifax map.
# You can provide ONLY "address" (plus org_name, services). If lat/lon are missing,
# the app GEOCODES the address to coordinates (cached) and plots the pins.
#
# Put your data file in the repo at ONE of these paths:
#   data/halifax_nonprofits_map.xlsx   (sheet "nonprofits")
#   halifax_nonprofits_map.xlsx        (sheet "nonprofits")
#   data/nonprofits.csv                (CSV)
#   nonprofits.csv                     (CSV)
#
# Required columns (case-insensitive; aliases handled):
#   org_name, address, services
# Optional columns:
#   lat (or latitude), lon (or long/longitude), area, website, phone, notes
#
# Run locally:
#   pip install streamlit pandas pydeck openpyxl geopy
#   streamlit run halifax_map_from_repo_geocode.py
#
# requirements.txt:
#   streamlit>=1.32
#   pandas>=2.0
#   pydeck>=0.9
#   openpyxl>=3.1
#   geopy>=2.4

from __future__ import annotations
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(page_title="Halifax Non-Profits — Map (address-only OK)", layout="wide")
st.title("Halifax Non-Profits — Map")
st.caption("You can list only **address** (plus name & services). The app geocodes missing coordinates automatically (cached).")

# -------------------------- Config -------------------------- #
DATA_PATHS_XLSX = [Path("data/halifax_nonprofits_map.xlsx"), Path("halifax_nonprofits_map.xlsx")]
DATA_PATHS_CSV  = [Path("data/nonprofits.csv"), Path("nonprofits.csv")]
SHEET_NAME = "nonprofits"

HALIFAX_LAT_RANGE = (44.3, 45.1)
HALIFAX_LON_RANGE = (-64.2, -62.9)

CACHE_PATH = Path("data/geocode_cache.csv")  # optional on-disk cache (if repo is writeable at runtime)

# ---------------------- Helpers ----------------------------- #
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    alias = {
        "organization": "org_name",
        "org": "org_name",
        "services_offered": "services",
        "latitude": "lat",
        "long": "lon",
        "longitude": "lon",
    }
    df.rename(columns={c: alias.get(c, c) for c in df.columns], inplace=True)
    for opt in ["website", "phone", "area", "notes"]:
        if opt not in df.columns:
            df[opt] = ""
    # required
    for col in ["org_name", "address", "services"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # coerce coords if present
    if "lat" in df.columns: df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns: df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df

@st.cache_data
def _load_repo_data() -> tuple[pd.DataFrame | None, str]:
    # Try Excel then CSV
    for p in DATA_PATHS_XLSX:
        if p.exists():
            df = pd.read_excel(p, sheet_name=SHEET_NAME, engine="openpyxl")
            return df, str(p)
    for p in DATA_PATHS_CSV:
        if p.exists():
            df = pd.read_csv(p)
            return df, str(p)
    return None, ""

@st.cache_data
def _load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        try:
            c = pd.read_csv(CACHE_PATH)
            if not {"address","lat","lon"}.issubset({*c.columns}):
                return pd.DataFrame(columns=["address","lat","lon"])
            c["lat"] = pd.to_numeric(c["lat"], errors="coerce")
            c["lon"] = pd.to_numeric(c["lon"], errors="coerce")
            c["address_norm"] = c["address"].str.strip().str.lower()
            return c
        except Exception:
            return pd.DataFrame(columns=["address","lat","lon"])
    return pd.DataFrame(columns=["address","lat","lon"])

def _save_cache(cache_df: pd.DataFrame):
    # Best-effort; may fail on some hosts with read-only FS
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(CACHE_PATH, index=False)
    except Exception:
        pass

def _norm_addr(s: str) -> str:
    return (s or "").strip().lower()

@st.cache_data(show_spinner=False)
def _geocode_addresses(addresses: tuple[str, ...]) -> dict[str, tuple[float | None, float | None]]:
    """
    Geocode a tuple of addresses using Nominatim with rate limit.
    Returns dict: normalized_address -> (lat, lon)
    Results cached by Streamlit across reruns.
    """
    geocoder = Nominatim(user_agent="new_roots_halifax_map")
    geocode = RateLimiter(geocoder.geocode, min_delay_seconds=1, swallow_exceptions=True)  # be polite!

    results: dict[str, tuple[float | None, float | None]] = {}
    for addr in addresses:
        a = _norm_addr(addr)
        loc = geocode(f"{addr}, Halifax, Nova Scotia, Canada")
        if loc:
            results[a] = (loc.latitude, loc.longitude)
        else:
            results[a] = (None, None)
    return results

def geocode_missing(df: pd.DataFrame, cache_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill lat/lon for rows missing coords by geocoding address; merge with cache; return df and messages."""
    msgs = []
    # Prepare cache
    cache = cache_df.copy()
    if "address_norm" not in cache.columns:
        cache["address_norm"] = cache["address"].astype(str).str.strip().str.lower()

    df["address_norm"] = df["address"].astype(str).str.strip().str.lower()

    # Use cached hits first
    df = df.merge(cache[["address_norm","lat","lon"]].rename(columns={"lat":"lat_cached","lon":"lon_cached"}),
                  on="address_norm", how="left")
    df["lat"] = df.get("lat", pd.Series([None]*len(df)))
    df["lon"] = df.get("lon", pd.Series([None]*len(df)))
    df["lat"] = df["lat"].fillna(df["lat_cached"])
    df["lon"] = df["lon"].fillna(df["lon_cached"])

    # Collect addresses still missing
    todo = df[df["lat"].isna() | df["lon"].isna()]["address"].dropna().unique().tolist()
    if todo:
        st.info(f"Geocoding {len(todo)} address(es)… (uses OpenStreetMap Nominatim; 1 req/sec)")
        # Call geocoder (cached per unique address set)
        lookups = _geocode_addresses(tuple(todo))
        # Apply back
        for addr in todo:
            a = _norm_addr(addr)
            lat, lon = lookups.get(a, (None, None))
            if lat is not None and lon is not None:
                df.loc[df["address_norm"] == a, "lat"] = lat
                df.loc[df["address_norm"] == a, "lon"] = lon
                # upsert to cache
                cache_row = {"address": addr, "lat": lat, "lon": lon, "address_norm": a}
                cache = pd.concat([cache[~(cache["address_norm"] == a)], pd.DataFrame([cache_row])], ignore_index=True)
        _save_cache(cache)
        msgs.append(f"Geocoded {sum([1 for a in todo if lookups.get(_norm_addr(a), (None, None))[0] is not None])} / {len(todo)} new address(es).")

    # Clean up
    df.drop(columns=[c for c in ["lat_cached","lon_cached","address_norm"] if c in df.columns], inplace=True)
    return df, msgs

def compute_view(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty or df["lat"].isna().all() or df["lon"].isna().all():
        return pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)
    c_lat, c_lon = float(df["lat"].mean()), float(df["lon"].mean())
    lat_span = (df["lat"].max() - df["lat"].min()) if not df["lat"].isna().all() else 0
    lon_span = (df["lon"].max() - df["lon"].min()) if not df["lon"].isna().all() else 0
    span = max(abs(lat_span), abs(lon_span))
    if span > 0.5:   zoom = 9
    elif span > 0.2: zoom = 10
    elif span > 0.08:zoom = 11
    else:            zoom = 12
    return pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=zoom, pitch=0)

def tooltip_template() -> dict:
    return {
        "html": (
            "<b>{org_name}</b><br/>"
            "{address}<br/>"
            "<i>{services}</i><br/>"
            "{phone}<br/>"
            "<a href='{website}' target='_blank'>{website}</a>"
        ),
        "style": {"backgroundColor": "white", "color": "black"},
    }

# ---------------------- Load data ---------------------------- #
raw, source = _load_repo_data()
if raw is None:
    st.error(
        "No data file found.\n\n"
        "Add **data/halifax_nonprofits_map.xlsx** (sheet: 'nonprofits') OR **data/nonprofits.csv** "
        "to this repo (or use the root equivalents)."
    )
    st.stop()

try:
    df = _normalize_columns(raw)
except Exception as e:
    st.error(str(e))
    st.write("Detected columns:", list(raw.columns))
    st.stop()

# Geocode if needed
cache_df = _load_cache()
df, geocode_msgs = geocode_missing(df, cache_df)

# Keep only rows with coordinates after geocoding
before = len(df)
df = df.dropna(subset=["lat","lon"]).copy()
after = len(df)

st.success(f"Loaded from: **{source}** · Pins rendered: **{after}** (dropped {before - after} without coords)")
for m in geocode_msgs:
    st.info(m)

# Bound warnings
oob = (~df["lat"].between(*HALIFAX_LAT_RANGE) | ~df["lon"].between(*HALIFAX_LON_RANGE)).sum()
if oob > 0:
    st.warning(f"{oob} row(s) fall outside typical Halifax bounds. Check addresses or province.")

with st.expander("Preview first 25 rows"):
    st.dataframe(df.head(25), use_container_width=True)

# ------------------------- Map ------------------------------- #
view = compute_view(df)
pins = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_radius=110,
    pickable=True,
    auto_highlight=True,
)
deck = pdk.Deck(layers=[pins], initial_view_state=view, tooltip=tooltip_template())
st.pydeck_chart(deck, use_container_width=True)

st.caption(
    "To add organizations: edit your Excel/CSV (at the paths listed above). "
    "You may leave `lat`/`lon` blank—addresses will be geocoded and cached automatically. "
    "Powered by OpenStreetMap Nominatim (polite 1 req/sec)."
)

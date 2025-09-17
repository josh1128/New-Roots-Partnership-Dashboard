# halifax_map_from_repo_fixed.py
# Streamlit app that AUTO-LOADS an Excel file from your GitHub repo and
# shows Halifax non-profits on a map with robust fixes for common data issues.
#
# Put your Excel in the repo at ONE of these paths:
#   data/halifax_nonprofits_map.xlsx   (recommended)
#   halifax_nonprofits_map.xlsx        (root fallback)
# The sheet must be named: "nonprofits"
#
# Required columns (case-insensitive; aliases handled): 
#   org_name, address, services, lat(or latitude), lon(long/longitude)
# Optional columns:
#   area, website, phone, notes
#
# Run locally:
#   pip install streamlit pandas pydeck openpyxl
#   streamlit run halifax_map_from_repo_fixed.py
#
# requirements.txt:
#   streamlit>=1.32
#   pandas>=2.0
#   pydeck>=0.9
#   openpyxl>=3.1

from __future__ import annotations
from pathlib import Path
import math
import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Halifax Non-Profits — Map (repo Excel, fixed)", layout="wide")
st.title("Halifax Non-Profits — Map (auto-load from repo, with fixes)")

# -------------------------- Config -------------------------- #
DATA_PATHS = [
    Path("data/halifax_nonprofits_map.xlsx"),  # recommended
    Path("halifax_nonprofits_map.xlsx"),       # root fallback
]
SHEET_NAME = "nonprofits"

# Halifax rough bounds (for sanity checks)
HALIFAX_LAT_RANGE = (44.3, 45.1)
HALIFAX_LON_RANGE = (-64.2, -62.9)

# ---------------------- Helper functions -------------------- #
@st.cache_data
def load_repo_excel(paths: list[Path], sheet_name: str) -> tuple[pd.DataFrame | None, str | None]:
    """Return (df, src_path_str) if found, else (None, None)."""
    for p in paths:
        if p.exists():
            df = pd.read_excel(p, sheet_name=sheet_name, engine="openpyxl")
            return df, str(p)
    return None, None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize headers and provide aliases."""
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
    df.rename(columns={c: alias.get(c, c) for c in df.columns}, inplace=True)

    # Ensure optional columns exist for tooltips
    for opt in ["website", "phone", "area", "notes"]:
        if opt not in df.columns:
            df[opt] = ""

    required = ["org_name", "address", "services"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Coerce coords if present
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df

def fix_common_coord_issues(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fix: swapped columns, positive lon sign, strings; returns (df, warnings)."""
    msgs = []
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("Columns for coordinates not found. Add columns 'lat' and 'lon' (aliases ok).")

    # Detect if lat/lon are swapped (many lats near -63 and lons near +44)
    s_lat = df["lat"].dropna()
    s_lon = df["lon"].dropna()
    if not s_lat.empty and not s_lon.empty:
        lat_mean = s_lat.mean()
        lon_mean = s_lon.mean()
        if (lat_mean < -60 and -70 < lon_mean < -40) or (44 <= lon_mean <= 46 and -70 <= lat_mean <= -60):
            # almost certainly swapped (lat looks like -63, lon looks like 44-46)
            df[["lat", "lon"]] = df[["lon", "lat"]]
            msgs.append("Detected swapped lat/lon columns — auto-corrected.")

    # Enforce Halifax sign: lon should be negative (~ -63.x)
    if (df["lon"] > 0).sum() > 0:
        # If values look like +63.x, flip to negative
        suspect = df["lon"].between(60, 70).sum()
        if suspect > 0:
            df.loc[df["lon"].between(0, 180), "lon"] = -df.loc[df["lon"].between(0, 180), "lon"].abs()
            msgs.append("Found positive longitudes ~+63 — converted to negative (Halifax).")

    # Drop rows with missing/NaN coords
    before = len(df)
    df = df.dropna(subset=["lat", "lon"]).copy()
    after = len(df)
    if after < before:
        msgs.append(f"Dropped {before - after} rows without valid lat/lon.")

    # Warn about points outside rough Halifax bounds
    out_of_bounds = (~df["lat"].between(*HALIFAX_LAT_RANGE) | ~df["lon"].between(*HALIFAX_LON_RANGE)).sum()
    if out_of_bounds > 0:
        msgs.append(f"{out_of_bounds} row(s) have coords outside Halifax bounds; they may be off-screen.")

    return df, msgs

def compute_view(df: pd.DataFrame) -> pdk.ViewState:
    """Center map on data; fallback to Halifax downtown if empty."""
    if df.empty or df["lat"].isna().all() or df["lon"].isna().all():
        return pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)

    c_lat = float(df["lat"].mean())
    c_lon = float(df["lon"].mean())

    # Choose zoom based on spatial spread (simple heuristic)
    # compute rough spread in km using naive degree distances
    lat_span = df["lat"].max() - df["lat"].min()
    lon_span = df["lon"].max() - df["lon"].min()
    span = max(abs(lat_span), abs(lon_span))

    if span > 0.5:
        zoom = 9   # very spread out
    elif span > 0.2:
        zoom = 10
    elif span > 0.08:
        zoom = 11
    else:
        zoom = 12  # tight cluster

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

# ------------------------ Load and prepare ------------------- #
df_raw, src = load_repo_excel(DATA_PATHS, SHEET_NAME)
if df_raw is None:
    st.error(
        "Could not find the Excel file in the repo.\n\n"
        "Place it at **data/halifax_nonprofits_map.xlsx** (sheet: **nonprofits**), "
        "or at repo root as **halifax_nonprofits_map.xlsx**."
    )
    st.stop()

try:
    df = normalize_columns(df_raw)
except Exception as e:
    st.error(str(e))
    st.write("Detected columns:", list(df_raw.columns))
    st.stop()

try:
    df, fix_msgs = fix_common_coord_issues(df)
except Exception as e:
    st.error(str(e))
    st.write("Detected columns after normalization:", list(df.columns))
    st.stop()

st.success(f"Loaded from: **{src}** · Pins available: **{len(df)}**")
for m in fix_msgs:
    st.info(m)

with st.expander("Preview (first 25 rows)"):
    st.dataframe(df.head(25), use_container_width=True)

# ----------------------------- Map --------------------------- #
view = compute_view(df)

pins = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_radius=100,           # bigger pins to make sure you see them
    pickable=True,
    auto_highlight=True,
)

deck = pdk.Deck(
    layers=[pins],
    initial_view_state=view,
    tooltip=tooltip_template(),
    # leave default map_style so no Mapbox token required
)

st.pydeck_chart(deck, use_container_width=True)

st.caption(
    "If pins still don’t appear: open the Preview to check coordinates. "
    "Halifax should be roughly **lat≈44.6**, **lon≈-63.6**. Positive longitudes will be auto-flipped."
)

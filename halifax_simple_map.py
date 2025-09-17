# halifax_map_from_repo.py
# Streamlit app that AUTO-LOADS an Excel file from your GitHub repo and shows a map
# of Halifax non-profits with clickable pins (tooltips).
#
# Put your Excel here in the repo (any ONE of these paths):
#   - data/halifax_nonprofits_map.xlsx
#   - halifax_nonprofits_map.xlsx
# The sheet must be named: "nonprofits"
#
# Required columns in the sheet (case-insensitive):
#   org_name, address, services, lat(or latitude), lon(long/longitude)
# Optional: area, website, phone, notes
#
# Run locally:
#   pip install streamlit pandas pydeck openpyxl
#   streamlit run halifax_map_from_repo.py
#
# On Streamlit Community Cloud:
#   - Point to this file as the main app
#   - Ensure requirements.txt includes:
#       streamlit>=1.32
#       pandas>=2.0
#       pydeck>=0.9
#       openpyxl>=3.1

from pathlib import Path
import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Halifax Non-Profits — Map (from repo Excel)", layout="wide")
st.title("Halifax Non-Profits — Map (auto-load from repo)")

# Where we look for the Excel (first path that exists wins)
DATA_PATHS = [
    Path("data/halifax_nonprofits_map.xlsx"),  # recommended
    Path("halifax_nonprofits_map.xlsx"),       # fallback at repo root
]

# ------------------------------- Helpers ---------------------------------- #
@st.cache_data
def load_repo_excel(paths: list[Path], sheet_name: str = "nonprofits") -> tuple[pd.DataFrame | None, str | None]:
    """Return (df, src_path_str) if found, else (None, None)"""
    for p in paths:
        if p.exists():
            df = pd.read_excel(p, sheet_name=sheet_name, engine="openpyxl")
            return df, str(p)
    return None, None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase headers, replace spaces with underscores, apply common aliases, coerce coords."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    alias = {
        "latitude": "lat",
        "long": "lon",
        "longitude": "lon",
        "organization": "org_name",
        "org": "org_name",
        "services_offered": "services",
    }
    df.rename(columns={c: alias.get(c, c) for c in df.columns}, inplace=True)
    # Ensure expected optional columns exist (for tooltip formatting)
    for opt in ["website", "phone", "area", "notes"]:
        if opt not in df.columns:
            df[opt] = ""
    # Required columns check
    required = ["org_name", "address", "services", "lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sheet 'nonprofits': {', '.join(missing)}")
    # Coerce coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df

def tooltip_html() -> dict:
    """Tooltip template for pydeck (shows on hover/click)."""
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

# ------------------------------ Load data --------------------------------- #
df, src = load_repo_excel(DATA_PATHS, sheet_name="nonprofits")

if df is None:
    st.error(
        "Could not find the Excel file in the repo.\n\n"
        "Place it at **data/halifax_nonprofits_map.xlsx** (sheet name: **nonprofits**), "
        "or at repo root as **halifax_nonprofits_map.xlsx**."
    )
    st.stop()

try:
    df = normalize_columns(df)
except Exception as e:
    st.error(str(e))
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Drop rows without valid coordinates
before = len(df)
df = df.dropna(subset=["lat", "lon"]).copy()
after = len(df)

st.success(f"Loaded from: **{src}** · Pins rendered: **{after}** (dropped {before - after} without lat/lon)")
with st.expander("Preview first 25 rows"):
    st.dataframe(df.head(25), use_container_width=True)

# ------------------------------- Map -------------------------------------- #
# Halifax center
view = pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)

# Simple pins
pins = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_radius=80,           # adjust pin size as needed
    pickable=True,
    auto_highlight=True,
)

deck = pdk.Deck(
    layers=[pins],
    initial_view_state=view,
    tooltip=tooltip_html(),
    # leaving default map_style so no Mapbox token is required
)

st.pydeck_chart(deck, use_container_width=True)

st.caption(
    "Tips: Keep columns `org_name, address, services, lat, lon` (plus optional `area, website, phone, notes`). "
    "Coordinates must be numeric (e.g., 44.65, -63.57)."
)

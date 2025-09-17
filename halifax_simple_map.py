# halifax_map_only.py
# Minimal Streamlit app: map of Halifax with clickable pins for non-profits.
# Edit the SAMPLE_DATA list or upload a CSV with columns:
# org_name, address, services, lat, lon, website, phone
#
# Run:
#   pip install streamlit pandas pydeck
#   streamlit run halifax_map_only.py

import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Halifax Non-Profits — Map Only", layout="wide")
st.title("Halifax Non-Profits — Map (Click pins for details)")

# ---- 1) EDITABLE SAMPLE DATA -----------------------------------------------
SAMPLE_DATA = [
    {
        "org_name": "Hope Blooms",
        "address": "Barrington St, Halifax, NS",
        "services": "Youth programs; community garden; food security",
        "lat": 44.6585, "lon": -63.5918,
        "website": "https://www.hopeblooms.ca", "phone": ""
    },
    {
        "org_name": "North End Community Health Centre (NECHC)",
        "address": "2165 Gottingen St, Halifax, NS",
        "services": "Primary health care; community outreach; social supports",
        "lat": 44.6577, "lon": -63.5887,
        "website": "https://www.nechc.com", "phone": ""
    },
    {
        "org_name": "Mi'kmaw Native Friendship Centre",
        "address": "2021 Gottingen St, Halifax, NS B3K 3B1",
        "services": "Indigenous cultural; housing & employment supports; youth",
        "lat": 44.6597, "lon": -63.5823,
        "website": "https://mymnfc.com", "phone": ""
    },
]

# ---- 2) OPTIONAL CSV UPLOAD -------------------------------------------------
st.caption("Optional: Upload your own CSV (columns: org_name, address, services, lat, lon, website, phone).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.DataFrame(SAMPLE_DATA)

# ---- 3) BASIC VALIDATION ----------------------------------------------------
required = ["org_name", "address", "services", "lat", "lon"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

df = df.dropna(subset=["lat", "lon"]).copy()

# ---- 4) MAP VIEW STATE (centered on Halifax) --------------------------------
view = pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)

# ---- 5) TOOLTIP CONTENT (shown on hover/click) ------------------------------
tooltip = {
    "html": (
        "<b>{org_name}</b><br/>"
        "{address}<br/>"
        "<i>{services}</i><br/>"
        "{phone}<br/>"
        "<a href='{website}' target='_blank'>{website}</a>"
    ),
    "style": {"backgroundColor": "white", "color": "black"}
}

# ---- 6) LAYER: clickable pins ----------------------------------------------
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_radius=80,          # adjust pin size
    pickable=True,
    auto_highlight=True,
)

# ---- 7) RENDER --------------------------------------------------------------
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    tooltip=tooltip,
    # map_style left default to avoid requiring a Mapbox token explicitly
)

st.pydeck_chart(deck, use_container_width=True)

# ---- 8) OPTIONAL: show data for quick edits (comment out if you want map-only)
with st.expander("Show data table (optional)"):
    st.dataframe(df, use_container_width=True)

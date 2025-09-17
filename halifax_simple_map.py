# halifax_heatmap.py
# Streamlit app: SIMPLE geographic heatmaps for Halifax non-profit activity.
# No Excel needed. Uses built-in sample points OR paste-your-own coordinates.
#
# Run:
#   pip install streamlit pandas pydeck
#   streamlit run halifax_heatmap.py
#
# Paste data format (no header, one per line):
#   lat,lon,weight
# Example:
#   44.6590,-63.5895,3

import streamlit as st
import pandas as pd
import pydeck as pdk
import random

st.set_page_config(page_title="Halifax Non-Profits • Geographic Heatmaps", layout="wide")
st.title("Halifax Non-Profits — Geographic Heatmaps")

st.caption(
    "Shows the concentration of **clients**, **volunteers**, or **program locations** on a map. "
    "Use the built-in sample data or paste your own `lat,lon,weight` points (no files needed)."
)

# --------------------------- Helpers --------------------------- #
def make_cluster(center_lat, center_lon, n=60, spread=0.01, w_low=1, w_high=5):
    """Generate n points around (center_lat, center_lon) with Gaussian jitter."""
    pts = []
    for _ in range(n):
        lat = random.gauss(center_lat, spread)
        lon = random.gauss(center_lon, spread)
        weight = random.randint(w_low, w_high)
        pts.append({"lat": lat, "lon": lon, "weight": weight})
    return pts

def sample_dataset(kind: str) -> pd.DataFrame:
    """
    Create small synthetic clusters around Halifax areas:
    - North End / Gottingen
    - Downtown / Waterfront
    - Dartmouth
    - Spryfield
    - Clayton Park
    """
    random.seed(42 + hash(kind) % 1000)  # stable per kind
    clusters = [
        # (lat, lon, points, spread)
        (44.6575, -63.5885, 70, 0.004),  # North End
        (44.6450, -63.5720, 40, 0.0035), # Downtown
        (44.6690, -63.5670, 30, 0.0030), # Hydrostone area
        (44.6640, -63.5675, 35, 0.0030), # Agricola/Gottingen (reinforce)
        (44.6660, -63.5695, 20, 0.0025), # Joseph Howe area
        (44.6410, -63.6160, 35, 0.0045), # Clayton Park edge
        (44.6030, -63.6200, 30, 0.0045), # Spryfield
        (44.6655, -63.5690, 25, 0.0025), # Program cluster near North End
        (44.6665, -63.5925, 25, 0.0035), # West of Gottingen
        (44.6675, -63.5810, 25, 0.0030), # North of Cogswell
    ]
    pts = []
    for (lat, lon, n, s) in clusters:
        # vary strength by dataset type
        if kind == "Clients":
            w_low, w_high = 2, 6
        elif kind == "Volunteers":
            w_low, w_high = 1, 4
        else:  # Program locations (treat as presence: heavier radius, medium weight)
            w_low, w_high = 2, 5
        pts.extend(make_cluster(lat, lon, n=n, spread=s, w_low=w_low, w_high=w_high))
    return pd.DataFrame(pts)

def parse_pasted(text: str) -> pd.DataFrame:
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            weight = float(parts[2]) if len(parts) >= 3 else 1.0
            rows.append({"lat": lat, "lon": lon, "weight": weight})
        except ValueError:
            # skip malformed lines
            continue
    return pd.DataFrame(rows, columns=["lat", "lon", "weight"])

def view_from_data(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty:
        return pdk.ViewState(latitude=44.6488, longitude=-63.5752, zoom=11, pitch=0)
    c_lat, c_lon = float(df["lat"].mean()), float(df["lon"].mean())
    lat_span = df["lat"].max() - df["lat"].min()
    lon_span = df["lon"].max() - df["lon"].min()
    span = max(abs(lat_span), abs(lon_span))
    if span > 0.5:   zoom = 9
    elif span > 0.2: zoom = 10
    elif span > 0.08:zoom = 11
    else:            zoom = 12
    return pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=zoom, pitch=0)

# --------------------------- Controls ------------------------- #
with st.sidebar:
    st.header("Controls")
    dataset_kind = st.selectbox("Dataset", ["Clients", "Volunteers", "Program locations"], index=0)
    data_source = st.radio("Data source", ["Use sample", "Paste points"], horizontal=True)
    radius_px = st.slider("Radius (pixels)", 10, 120, 60, 2)
    intensity = st.slider("Intensity", 1.0, 5.0, 2.0, 0.1)
    opacity = st.slider("Opacity", 0.2, 1.0, 0.85, 0.05)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.05, 0.01)
    show_points = st.checkbox("Overlay point markers", value=False)

# ------------------------- Data prep -------------------------- #
if data_source == "Use sample":
    df = sample_dataset(dataset_kind)
else:
    st.subheader("Paste your points (lat,lon,weight)")
    example = "44.6589,-63.5897,3\n44.6571,-63.5880,2\n44.6451,-63.5719,5"
    pasted = st.text_area("One per line (no header):", value=example, height=120)
    df = parse_pasted(pasted)
    if df.empty:
        st.warning("No valid rows parsed. Using a tiny fallback near North End.")
        df = pd.DataFrame(
            [{"lat": 44.6589, "lon": -63.5897, "weight": 3},
             {"lat": 44.6571, "lon": -63.5880, "weight": 2}]
        )

# Ensure numeric
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
df = df.dropna(subset=["lat", "lon"]).copy()

st.success(f"Points: {len(df)}")

with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

# --------------------------- Map ------------------------------ #
heat = pdk.Layer(
    "HeatmapLayer",
    data=df,
    get_position='[lon, lat]',
    get_weight="weight",
    radiusPixels=radius_px,
    intensity=intensity,
    threshold=threshold,
    opacity=opacity,
)

layers = [heat]

if show_points:
    pts = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=40,
        pickable=False,
        opacity=0.7,
    )
    layers.append(pts)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_from_data(df),
    tooltip={"html": "<b>Weight:</b> {weight}", "style": {"backgroundColor": "white", "color": "black"}},
    # default map style -> no Mapbox token required
)

st.pydeck_chart(deck, use_container_width=True)

st.caption(
    "Use **Radius** and **Intensity** to tune the heatmap. "
    "Paste your own `lat,lon,weight` points to visualize where you have the most clients, volunteers, or program locations."
)


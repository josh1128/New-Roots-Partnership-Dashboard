import streamlit as st
import pandas as pd

st.set_page_config(page_title="Halifax Non‑Profits (Simple Map)", layout="wide")
st.title("Halifax, Nova Scotia — Example Non‑Profits & Services (Simple Map)")

st.write("Quick demo with a few **example** organizations. Edit the table to add your own.")

# ---- Minimal example data (editable) ----
data = pd.DataFrame([
    {"org_name":"Hope Blooms","services":"Youth programs; community garden; food security","lat":44.6585,"lon":-63.5918},
    {"org_name":"North End Community Health Centre (NECHC)","services":"Primary health care; community outreach; social supports","lat":44.6577,"lon":-63.5887},
    {"org_name":"Mi'kmaw Native Friendship Centre","services":"Indigenous cultural; housing & employment supports; youth","lat":44.6597,"lon":-63.5823},
    {"org_name":"Parker Street Food & Furniture Bank","services":"Food bank; furniture & household items; community supports","lat":44.6528,"lon":-63.5859},
    {"org_name":"YWCA Halifax","services":"Women & family services; housing; employment; violence prevention","lat":44.6459,"lon":-63.5766},
    {"org_name":"Halifax Refugee Clinic","services":"Legal aid for refugees; settlement support","lat":44.6479,"lon":-63.5799},
    {"org_name":"YMCA Centre for Immigrant Programs","services":"Newcomer settlement; language; employment","lat":44.6425,"lon":-63.5756},
])

# Let users edit / add rows quickly
edited = st.data_editor(data, num_rows="dynamic", use_container_width=True)

st.subheader("Map")
st.map(edited.rename(columns={"lat":"latitude","lon":"longitude"}), latitude="latitude", longitude="longitude", size=70)

st.caption("Tip: Add more rows above and the map will update.")

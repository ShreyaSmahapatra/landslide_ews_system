import streamlit as st
import random
import time
import folium
import numpy as np
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("🏔️ SIH Problem Statement ID: SIH26001")
st.header("AI-Based Early Warning & Landslide Risk Monitoring System")

# 1. Setup Mock Stations in the North Eastern Region
stations = {
    "Guwahati Hills Station (Assam)": [26.1445, 91.7362],
    "Shillong Ridge Station (Meghalaya)": [25.5788, 91.8933],
    "Gangtok Slope Monitor (Sikkim)": [27.3314, 88.6138]
}

selected_station = st.selectbox("🎯 Select Active Monitoring Station:", list(stations.keys()))
coords = stations[selected_station]

# 2. Simulator & Predictive Risk Engine
if st.button("📡 Fetch Live Sensor Stream & Analyze Risk", type="primary"):
    with st.spinner("Processing telemetry through Analytical Risk Engine..."):
        time.sleep(0.5)
        
        # Telemetry Stream Simulation
        rainfall = round(random.uniform(10, 160), 2)
        moisture = round(random.uniform(40, 98), 2)
        tilt = round(random.uniform(0.1, 8.5), 2)
        
        # --- Multi-Variant Sigmoid Failure Probability Model ---
        # Replaces hard thresholds with a smooth probability mapping (0.0 to 1.0)
        z = (rainfall * 0.02) + (moisture * 0.015) + (tilt * 0.45) - 4.5
        fail_probability = 1 / (1 + np.exp(-z))
        
        # Determine Alert Levels based on Mathematical Probability
        if fail_probability > 0.75:
            status, color, alert_box = f"RED (CRITICAL RISK: {round(fail_probability*100, 1)}%)", "red", st.error
        elif fail_probability > 0.35:
            status, color, alert_box = f"YELLOW (MODERATE WATCH: {round(fail_probability*100, 1)}%)", "orange", st.warning
        else:
            status, color, alert_box = f"GREEN (STABLE / SAFE: {round(fail_probability*100, 1)}%)", "green", st.success
            
    # 3. UI Display Layout
    alert_box(f"🚨 System Status for {selected_station}: {status}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("24h Cumulative Rainfall", f"{rainfall} mm")
    col2.metric("Soil Moisture Content", f"{moisture} %")
    col3.metric("Slope Tilt Angle", f"{tilt}°")
    
    # 4. Inject Interactive Map Tracking
    st.subheader("📍 Geospatial Hazard Mapping")
    m = folium.Map(location=coords, zoom_start=11)
    folium.Marker(
        coords, 
        popup=f"Status: {status}", 
        tooltip=selected_station,
        icon=folium.Icon(color=color, icon="exclamation-sign" if color != "green" else "ok-sign")
    ).add_to(m)
    st_folium(m, width=900, height=400)

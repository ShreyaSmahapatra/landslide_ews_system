"""
=============================================================================
Smart India Hackathon (SIH) - Early Warning System (EWS) Platform
Problem Statement ID: SIH26001 | Landslide Risk & Hazard Monitoring in NER
Assigned by: Ministry of Development of North Eastern Region (MDoNER)
=============================================================================
Architecture: Full-Stack AI-GIS Landslide Early Warning & Decision Support
Engine Modules:
  1. Live Ingestion & Dynamic 3D Topographic Mesh (Plotly Surface)
  2. 3D Elevation GIS Hazard Mapping (PyDeck Column, Path, Scatter Layers)
  3. Ground Anomaly Field Portal (EXIF GPS Extraction & Video/Image Intake)
  4. Emergency Prioritization, Multilingual Hub & Offline Resilience Engine
=============================================================================
"""

import copy
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import random
import sys
logger=logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import ExifTags, Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import streamlit as st

# Import local telemetry ingestion & threshold matrix engine
try:
    from src.telemetry_monitor import (
        evaluate_threshold_matrix,
        fetch_live_station_telemetry,
    )
except ImportError:
    # Graceful standalone fallback if executed outside root
    from src.telemetry_monitor import (
        evaluate_threshold_matrix,
        fetch_live_station_telemetry,
    )

# ---------------------------------------------------------------------------
# Streamlit App Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MDoNER Landslide EWS | North-Eastern Region",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-performance dashboard UI
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.3rem;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .main-header p {
        color: #d1e3ff;
        margin-bottom: 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .status-badge-critical {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-warning {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ffe0b2;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-badge-safe {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .offline-banner {
        background-color: #fff8e1;
        border-left: 6px solid #ffa000;
        padding: 12px 20px;
        border-radius: 6px;
        color: #b78103;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Global Spatial & Geological Data Mocks (North Eastern Region)
# ---------------------------------------------------------------------------
NER_STATIONS = {
    "Guwahati Hills Station (Assam)": {
        "coords": [26.1445, 91.7362],
        "state": "Assam",
        "elevation_m": 120,
        "geology": "Pre-Cambrian Gneissic Complex",
    },
    "Shillong Ridge Station (Meghalaya)": {
        "coords": [25.5788, 91.8933],
        "state": "Meghalaya",
        "elevation_m": 1496,
        "geology": "Shillong Plateau Quartzite",
    },
    "Gangtok Slope Monitor (Sikkim)": {
        "coords": [27.3314, 88.6138],
        "state": "Sikkim",
        "elevation_m": 1650,
        "geology": "Central Crystalline Gneiss",
    },
    "Tawang Alpine Corridor (Arunachal)": {
        "coords": [27.5861, 91.8594],
        "state": "Arunachal Pradesh",
        "elevation_m": 3048,
        "geology": "Higher Himalayan Schist",
    },
    "Kohima Ridge Monitor (Nagaland)": {
        "coords": [25.6751, 94.1086],
        "state": "Nagaland",
        "elevation_m": 1444,
        "geology": "Disang Shale Formation",
    },
}

# 3D Hazard Columns (Villages and hill communities in NER)
VILLAGES_HAZARD_DATA = [
    {"name": "Rangpo Settlement", "lat": 27.1767, "lon": 88.5292, "population": 6800, "base_risk": 0.88, "state": "Sikkim"},
    {"name": "Singtam Foothills", "lat": 27.2348, "lon": 88.4977, "population": 8400, "base_risk": 0.74, "state": "Sikkim"},
    {"name": "Mangan Ridge Village", "lat": 27.5054, "lon": 88.5332, "population": 4200, "base_risk": 0.92, "state": "Sikkim"},
    {"name": "Cherrapunji Escarpment", "lat": 25.2702, "lon": 91.7323, "population": 11200, "base_risk": 0.79, "state": "Meghalaya"},
    {"name": "Mawkdok Valley Hamlets", "lat": 25.3850, "lon": 91.8020, "population": 3100, "base_risk": 0.65, "state": "Meghalaya"},
    {"name": "Dispur Hill Slopes", "lat": 26.1360, "lon": 91.7920, "population": 28000, "base_risk": 0.58, "state": "Assam"},
    {"name": "Haflong Hill Station", "lat": 25.1764, "lon": 93.0242, "population": 43000, "base_risk": 0.82, "state": "Assam"},
    {"name": "Zunheboto Ridge", "lat": 25.9723, "lon": 94.5204, "population": 15000, "base_risk": 0.71, "state": "Nagaland"},
    {"name": "Dirang River Slope", "lat": 27.3590, "lon": 92.2340, "population": 5400, "base_risk": 0.62, "state": "Arunachal"},
]

# Vulnerable Highway Corridors in NER
HIGHWAY_CORRIDORS = [
    {
        "id": "NH-10",
        "name": "NH-10 (Siliguri - Kalimpong - Gangtok)",
        "risk_base": 0.89,
        "traffic_vol": "Heavy Freight / Lifeline",
        "bypass": "Lava - Damdim Secondary Route",
        "path": [
            [88.4312, 26.7271],
            [88.4721, 26.9125],
            [88.5292, 27.1767],
            [88.6138, 27.3314],
        ],
    },
    {
        "id": "NH-2",
        "name": "NH-2 (Dibrugarh - Kohima - Imphal)",
        "risk_base": 0.68,
        "traffic_vol": "Critical Commercial",
        "bypass": "Chakhabama - Pfutsero Detour",
        "path": [
            [94.9120, 27.4728],
            [94.5204, 25.9723],
            [94.1086, 25.6751],
            [93.9368, 24.8170],
        ],
    },
    {
        "id": "NH-27",
        "name": "NH-27 (Guwahati - Shillong - Silchar)",
        "risk_base": 0.45,
        "traffic_vol": "Inter-State Trunk Corridor",
        "bypass": "Meghalaya Bypass Express Arc",
        "path": [
            [91.7362, 26.1445],
            [91.8933, 25.5788],
            [92.5120, 25.2100],
            [92.7976, 24.8333],
        ],
    },
    {
        "id": "NH-29",
        "name": "NH-29 (Dimapur - Kohima Bypass)",
        "risk_base": 0.81,
        "traffic_vol": "Essential Cargo Lifeline",
        "bypass": "Old Kuki Hill Track (Light Vehicles)",
        "path": [
            [93.7266, 25.9067],
            [93.8900, 25.7600],
            [94.1086, 25.6751],
        ],
    },
    {
        "id": "NH-102",
        "name": "NH-102 (Imphal - Moreh Border Highway)",
        "risk_base": 0.38,
        "traffic_vol": "International Trade Transit",
        "bypass": "Thoubal Rural Bypass",
        "path": [
            [93.9368, 24.8170],
            [94.0200, 24.5800],
            [94.3000, 24.2500],
        ],
    },
]

# Critical Infrastructure Assets
INFRASTRUCTURE_ASSETS = [
    {"name": "Teesta Hydro Dam Powerhouse #3", "type": "Hydroelectric Dam", "lat": 27.1820, "lon": 88.5120, "vulnerability": "High", "color": [241, 196, 15, 230]},
    {"name": "220kV Central Power Grid Tower 44", "type": "High-Voltage Power Mast", "lat": 25.6120, "lon": 91.9210, "vulnerability": "Critical", "color": [231, 76, 60, 240]},
    {"name": "Rangpo Suspension Bridge", "type": "Vital River Span", "lat": 27.1780, "lon": 88.5310, "vulnerability": "Critical", "color": [155, 89, 182, 230]},
    {"name": "SDRF Optical Fiber Repeater Node", "type": "Telecom Backhaul", "lat": 25.5910, "lon": 91.8840, "vulnerability": "Moderate", "color": [52, 152, 219, 230]},
    {"name": "Barapani Viaduct Water Conduits", "type": "Water Aqueduct", "lat": 25.6540, "lon": 91.9120, "vulnerability": "High", "color": [46, 204, 113, 230]},
]


# ---------------------------------------------------------------------------
# Session State Ingestion Initializer
# ---------------------------------------------------------------------------
if "citizen_reports" not in st.session_state:
    st.session_state.citizen_reports = [
        {
            "id": "REP-2026-0901",
            "anomaly_type": "Cracks (Tension fractures along slope)",
            "severity": "Critical - Evacuate",
            "description": "5cm longitudinal tension cracks opening along upper hillside road. Mud displacement visible.",
            "lat": 27.1812,
            "lon": 88.5345,
            "landmark": "Near Mile Marker 29, Rangpo Highway",
            "reporter": "Tashi Bhutia (SDRF Scout)",
            "timestamp": "2026-09-09 18:30 IST",
            "status": "Verified by District HQ",
            "offline_queued": False,
        },
        {
            "id": "REP-2026-0902",
            "anomaly_type": "Hydrological Changes (Sudden spring surge)",
            "severity": "Moderate",
            "description": "Natural natural hillside spring flowing heavily silted brown mud water into drainage culvert.",
            "lat": 25.5891,
            "lon": 91.8899,
            "landmark": "Lower Shillong Slope bypass",
            "reporter": "Banrap Lyngdoh (Citizen Watch)",
            "timestamp": "2026-09-09 19:15 IST",
            "status": "Under Drone Recon",
            "offline_queued": False,
        },
    ]

if "offline_mode" not in st.session_state:
    st.session_state.offline_mode = False

if "offline_report_queue" not in st.session_state:
    st.session_state.offline_report_queue = []


# ---------------------------------------------------------------------------
# Predictive AI Threat Assessment Matrix (XGBoost Architecture Mock)
# ---------------------------------------------------------------------------
def compute_xgboost_threat_matrix(
    rainfall_mm: float,
    moisture_pct: float,
    tilt_deg: float,
) -> Tuple[float, str, str, str, Dict[str, float]]:
    """
    Simulates an XGBoost gradient boosted ensemble model predicting landslide failure
    probability using meteorological saturation, pore water pressure, and inclinometer shear.

    Returns:
        prob (float): Landslide occurrence probability [0.0 - 1.0]
        badge (str): Styled text badge
        color (str): Hex color code
        action (str): Response action recommendation
        feature_weights (dict): Simulated SHAP feature importance percentages
    """
    f_rain = min(rainfall_mm / 160.0, 1.5)
    f_moist = min(moisture_pct / 100.0, 1.2)
    f_tilt = min(tilt_deg / 60.0, 1.5)
    f_interaction = f_rain * f_moist
    f_shear = np.sin(np.radians(tilt_deg)) * (f_moist**1.4)

    # Gradient boosting logit function
    raw_logit = -3.9 + (3.2 * f_rain) + (2.7 * f_moist) + (3.7 * f_tilt) + (2.3 * f_interaction) + (3.1 * f_shear)
    prob = float(1.0 / (1.0 + np.exp(-raw_logit)))
    prob = np.clip(prob, 0.02, 0.99)

    # Feature contribution breakdown (SHAP emulation)
    total_signal = max(0.1, (f_rain * 3.2) + (f_moist * 2.7) + (f_tilt * 3.7) + (f_interaction * 2.3))
    feature_weights = {
        "Precipitation Saturation (Rain 24h)": round(((f_rain * 3.2) / total_signal) * 100, 1),
        "Pore Water Pressure (Soil Moisture)": round(((f_moist * 2.7) / total_signal) * 100, 1),
        "Gravitational Shear Drift (Slope Tilt)": round(((f_tilt * 3.7) / total_signal) * 100, 1),
        "Hydrological Coupling Stress": round(((f_interaction * 2.3) / total_signal) * 100, 1),
    }

    if prob >= 0.75:
        badge = "CRITICAL"
        color = "#e74c3c"
        action = "🚨 IMMEDIATE EVACUATION ORDER: Activating High-Frequency Nodes (1-min telemetry) & Twilio SMS sirens."
    elif prob >= 0.50:
        badge = "WARNING"
        color = "#e67e22"
        action = "⚠️ HEIGHTENED WATCH: Road authorities alerted; geotechnical radar scanning shear plane."
    elif prob >= 0.28:
        badge = "WATCH"
        color = "#f1c40f"
        action = "🟡 PRECAUTIONARY ADVISORY: Restrict heavy commercial convoy traffic on mountain passes."
    else:
        badge = "SAFE"
        color = "#2ecc71"
        action = "🟢 NORMAL CONDITIONS: Telemetry within geological safety envelope. 15-minute standard heartbeat."

    return prob, badge, color, action, feature_weights


# ---------------------------------------------------------------------------
# Dynamic 3D Topographic Mesh Generator (Plotly Surface)
# ---------------------------------------------------------------------------
def generate_dynamic_3d_slope_mesh(tilt_deg: float) -> go.Figure:
    """
    Renders an interactive 3D surface plot modeling a mountain slope face.
    The physical grid geometry and tilt of the mesh deform and shift dynamically
    based on the 'Structural Slope Tilt' input slider on the dashboard to visually
    represent real-time slope sagging/slipping.
    """
    # 45x45 elevation grid representing mountain ridge domain
    x = np.linspace(-10, 10, 45)
    y = np.linspace(-10, 10, 45)
    X, Y = np.meshgrid(x, y)

    tilt_rad = np.radians(tilt_deg)

    # 1. Base Mountainous Topography with Bedrock Escarpment
    Z_base = 14.0 - (0.55 * Y) + (2.8 * np.cos(X / 3.2) * np.exp(-((Y - 2.0) ** 2) / 32.0))

    # 2. Dynamic Tilt Incline Plane
    Z_tilt = -Y * np.tan(tilt_rad * 0.72)

    # 3. Non-Linear Geological Failure & Sagging:
    # High tilt causes upper slope scarp detachment and lower toe debris accumulation
    slip_magnitude = (tilt_deg / 90.0) ** 2.2
    upper_scarp = -slip_magnitude * 8.5 / (1.0 + np.exp(-(Y - 1.2) / 1.1)) * np.exp(-(X**2) / 16.0)
    toe_accumulation = slip_magnitude * 5.8 * np.exp(-((Y + 5.2) ** 2) / 5.5) * np.exp(-(X**2) / 16.0)

    # Combined deformed elevation matrix
    Z_deformed = Z_base + Z_tilt + upper_scarp + toe_accumulation

    # Surface shear strain metric for coloring
    shear_strain = np.abs(upper_scarp) + (toe_accumulation * 0.8) + (np.sin(tilt_rad) * 4.0)

    fig = go.Figure(
        data=[
            go.Surface(
                z=Z_deformed,
                x=X,
                y=Y,
                surfacecolor=shear_strain,
                colorscale="Portland",
                colorbar=dict(
                    title=dict(text="Shear Strain / Slip Stress", side="right"),
                    thickness=15,
                    len=0.7,
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                ),
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4, roughness=0.5),
            )
        ]
    )

    fig.update_layout(
        title=f"Dynamic Geomechanical Slope Deformation (Structural Tilt: {tilt_deg:.1f}°)",
        autosize=True,
        height=520,
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(
            xaxis=dict(title="Lateral Span (X meters)"),
            yaxis=dict(title="Slope Gradient Axis (Y meters)"),
            zaxis=dict(title="Elevation (Z meters)"),
            camera=dict(
                eye=dict(x=1.6, y=-1.7, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=0.7),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# EXIF GPS Telemetry Extraction Helper
# ---------------------------------------------------------------------------
def extract_gps_from_exif(image_file: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts latitude and longitude from uploaded image EXIF metadata using Pillow.
    Handles DMS (Degrees, Minutes, Seconds) conversion with cardinal references.
    """
    try:
        image = Image.open(image_file)
        exif = image.getexif()
        if not exif:
            return None, None

        gps_info = None
        # Standard EXIF GPS IFD lookup
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                gps_info = value
                break

        # Fallback to get_ifd for modern Pillow structures
        if not gps_info and hasattr(exif, "get_ifd"):
            try:
                gps_info = exif.get_ifd(0x8825)
            except Exception:
                gps_info = None

        if not gps_info:
            return None, None

        gps_data = {}
        for sub_tag_id, sub_val in gps_info.items():
            sub_name = ExifTags.GPSTAGS.get(sub_tag_id, sub_tag_id)
            gps_data[sub_name] = sub_val

        lat_dms = gps_data.get("GPSLatitude")
        lat_ref = gps_data.get("GPSLatitudeRef", "N")
        lon_dms = gps_data.get("GPSLongitude")
        lon_ref = gps_data.get("GPSLongitudeRef", "E")

        if lat_dms and lon_dms:
            def _to_deg(dms_tuple):
                deg = float(dms_tuple[0])
                mnt = float(dms_tuple[1])
                sec = float(dms_tuple[2])
                return deg + (mnt / 60.0) + (sec / 3600.0)

            lat = _to_deg(lat_dms)
            if lat_ref in ["S", "s"]:
                lat = -lat

            lon = _to_deg(lon_dms)
            if lon_ref in ["W", "w"]:
                lon = -lon

            return round(lat, 5), round(lon, 5)
    except Exception as e:
        logger.debug(f"EXIF parsing error: {e}")
        return None, None

    return None, None


# ===========================================================================
# SIDEBAR CONTROLS & ENVIRONMENT CONFIGURATION
# ===========================================================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg",
        width=70,
    )
    st.title("🏔️ SIH EWS Engine")
    st.caption("Problem Statement: **SIH26001** | North-Eastern Region (NER)")
    st.markdown("---")

    # 1. Monitoring Station Selector
    selected_station_name = st.selectbox(
        "📍 Active Regional Telemetry Station:",
        list(NER_STATIONS.keys()),
        index=0,
    )
    station_meta = NER_STATIONS[selected_station_name]
    st.caption(
        f"**Elevation**: {station_meta['elevation_m']}m | **State**: {station_meta['state']}\n\n"
        f"**Bedrock**: *{station_meta['geology']}*"
    )
    st.markdown("---")

    # 2. Live vs. Simulation Toggle
    st.subheader("⚙️ Ingestion Mode")
    telemetry_mode = st.radio(
        "Select Telemetry Source:",
        ("🛰️ Live Sensor Stream (Open-Meteo API)", "🧪 Simulation Sandbox Mode"),
        index=0,
    )

    # Quick Scenario Presets for Sandbox Mode
    if telemetry_mode == "🧪 Simulation Sandbox Mode":
        st.markdown("**Load Rapid Risk Scenarios:**")
        preset_cols = st.columns(3)
        if preset_cols[0].button("☀️ Dry"):
            st.session_state["sb_moist"] = 28
            st.session_state["sb_rain"] = 5
            st.session_state["sb_tilt"] = 14
        if preset_cols[1].button("🌧️ Storm"):
            st.session_state["sb_moist"] = 82
            st.session_state["sb_rain"] = 75
            st.session_state["sb_tilt"] = 38
        if preset_cols[2].button("🚨 Slip"):
            st.session_state["sb_moist"] = 96
            st.session_state["sb_rain"] = 135
            st.session_state["sb_tilt"] = 54

        # Dynamic Sliders
        val_moist = st.session_state.get("sb_moist", 78)
        val_rain = st.session_state.get("sb_rain", 85)
        val_tilt = st.session_state.get("sb_tilt", 42)

        slider_moisture = st.slider(
            "💧 Soil Moisture Content (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(val_moist),
            step=0.5,
            key="slider_moist",
        )
        slider_rainfall = st.slider(
            "🌧️ Precipitation Accumulation (24h mm)",
            min_value=0.0,
            max_value=200.0,
            value=float(val_rain),
            step=1.0,
            key="slider_rain",
        )
        slider_tilt = st.slider(
            "📐 Structural Slope Tilt (degrees)",
            min_value=0.0,
            max_value=90.0,
            value=float(val_tilt),
            step=0.5,
            key="slider_tilt",
        )

        current_moisture = slider_moisture
        current_rainfall = slider_rainfall
        current_tilt = slider_tilt
        active_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    else:
        # Live Stream: Pull real-time telemetry from telemetry_monitor.py
        with st.spinner("Pinging Satellite Weather Feed..."):
            lat, lon = station_meta["coords"]
            live_payload = fetch_live_station_telemetry(lat, lon)
            current_moisture = live_payload["soil_moisture_pct"]
            current_rainfall = live_payload["rainfall_24h_mm"]
            current_tilt = live_payload["slope_tilt_deg"]
            active_timestamp = live_payload["timestamp"]

        st.success(f"Connected to Open-Meteo Gateway ({active_timestamp})")
        st.metric("Live 24h Rain Accum.", f"{current_rainfall} mm")
        st.metric("Atmospheric/Soil Saturation", f"{current_moisture} %")
        st.metric("Inclinometer Drift", f"{current_tilt}°")

    st.markdown("---")
    st.caption("🛡️ **System Status**: MDoNER Grid v3.4 Node Active | 15-sec Telemetry Polling")


# ===========================================================================
# TOP BANNER & REAL-TIME RISK METRIC MATRIX
# ===========================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🏔️ AI-Based Landslide Early Warning & Risk Monitoring System</h1>
        <p>Ministry of Development of North Eastern Region (MDoNER) | Smart India Hackathon Prototype</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compute ML Risk Severity
risk_prob, risk_badge, risk_color, risk_action, feature_weights = compute_xgboost_threat_matrix(
    current_rainfall, current_moisture, current_tilt
)

# Offline Banner Alert if enabled
if st.session_state.offline_mode:
    st.markdown(
        """
        <div class="offline-banner">
            📶 <b>LOW-NETWORK EDGE CACHE ENGAGED:</b> Running in localized offline mode. Telemetry updates and 
            field anomaly reports are cached locally on edge storage awaiting automatic cloud sync.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 4-Column Executive Metric Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric(
        label="Landslide Probability (XGBoost)",
        value=f"{risk_prob * 100:.1f}%",
        delta=f"{risk_badge} THREAT",
        delta_color="inverse" if risk_badge in ["CRITICAL", "WARNING"] else "normal",
    )
with m_col2:
    st.metric(
        label="24h Cumulative Precipitation",
        value=f"{current_rainfall:.1f} mm",
        delta="NER Critical: >80mm",
        delta_color="inverse" if current_rainfall > 80 else "normal",
    )
with m_col3:
    st.metric(
        label="Soil Moisture Content",
        value=f"{current_moisture:.1f} %",
        delta="Critical: >85%",
        delta_color="inverse" if current_moisture > 85 else "normal",
    )
with m_col4:
    st.metric(
        label="Inclinometer Slope Tilt",
        value=f"{current_tilt:.1f}°",
        delta="Critical: >45°",
        delta_color="inverse" if current_tilt > 45 else "normal",
    )


# ===========================================================================
# 4 SYSTEM TABS
# ===========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Monitor & Forecasts",
    "🗺️ GIS Hazard Map",
    "📢 Field Portal (Citizen & Official)",
    "📶 Offline Cache & Comms",
])


# ===========================================================================
# TAB 1: LIVE MONITOR & FORECASTS
# ===========================================================================
with tab1:
    st.subheader("⚡ Real-Time Sensor Telemetry & Predictive Slope Stability")

    # Alert Box with Action Protocol
    if risk_badge == "CRITICAL":
        st.error(f"**CRITICAL ALERT FOR {selected_station_name.upper()}**: {risk_action}")
    elif risk_badge == "WARNING":
        st.warning(f"**WARNING WATCH FOR {selected_station_name.upper()}**: {risk_action}")
    elif risk_badge == "WATCH":
        st.info(f"**ADVISORY NOTICE FOR {selected_station_name.upper()}**: {risk_action}")
    else:
        st.success(f"**NOMINAL CONDITIONS FOR {selected_station_name.upper()}**: {risk_action}")

    # Section 1: Dynamic 3D Mountain Slope Face Mesh & Feature Importances
    st.markdown("### 🏔️ Dynamic 3D Topographic Modeling (Plotly Surface)")
    st.write(
        "Interactive 3D representation of the mountain slope face. "
        "The physical grid geometry and tilt of the mesh deform and shift dynamically based on the "
        "**'Structural Slope Tilt'** input slider on the sidebar, simulating real-time geotechnical sagging and slipping."
    )

    t3d_col1, t3d_col2 = st.columns([3, 2])

    with t3d_col1:
        # Dynamic 3D Mesh
        fig_3d = generate_dynamic_3d_slope_mesh(current_tilt)
        st.plotly_chart(fig_3d, use_container_width=True)

    with t3d_col2:
        st.markdown("#### 🧠 XGBoost Threat Assessment Matrix")
        st.write(
            "Gradient boosted ensemble assessing multivariate geotechnical indicators with non-linear interaction terms."
        )

        # Feature Importance Progress Bars
        for feat_name, feat_pct in feature_weights.items():
            st.write(f"**{feat_name}**: `{feat_pct}%`")
            st.progress(int(np.clip(feat_pct, 0, 100)))

        st.markdown("---")
        st.markdown("#### 📐 In-Situ Structural Health Status")
        st.markdown(
            f"""
            - **Shear Displacement State**: {'⚠️ Active Slippage / Scarp Forming' if current_tilt > 45 else 'Stable Creep Equilibrium'}
            - **Pore Water Saturation Level**: {'High Liquefaction Risk' if current_moisture > 85 else 'Normal Pore Pressure'}
            - **Runoff Erosion Velocity**: {'Severe Gullying Potential' if current_rainfall > 80 else 'Standard Soil Drainage'}
            """
        )

    st.markdown("---")

    # Section 2: Weather-Linked Multi-Day Risk Projection Chart
    st.markdown("### 📈 Weather-Linked Multi-Day Risk Projection (7-Day Forecast)")
    st.write(
        "Combines incoming 7-day numerical precipitation forecasting models with baseline geological tilt degradation "
        "to calculate daily projected failure probability."
    )

    # Generate realistic multi-day forecast sequence based on current conditions
    dates = [(datetime.now() + timedelta(days=i)).strftime("%a, %b %d") for i in range(1, 8)]
    np.random.seed(42)
    # Seasonal monsoon curve with potential spike
    forecast_rain = [
        round(max(5.0, current_rainfall * factor + np.random.uniform(-10, 15)), 1)
        for factor in [0.9, 1.3, 1.7, 1.4, 0.8, 0.5, 0.3]
    ]
    # Progressive tilt degradation (cumulative shear strain if rainfall persists)
    forecast_tilt = [
        round(current_tilt + (np.cumsum([r / 50.0 for r in forecast_rain])[i]), 1)
        for i in range(7)
    ]
    # Projected probabilities
    forecast_risk = [
        round(compute_xgboost_threat_matrix(forecast_rain[i], min(99.0, current_moisture + (forecast_rain[i] * 0.15)), forecast_tilt[i])[0] * 100, 1)
        for i in range(7)
    ]

    fig_proj = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar trace: Forecasted Rainfall
    fig_proj.add_trace(
        go.Bar(
            x=dates,
            y=forecast_rain,
            name="Projected Rainfall (mm)",
            marker_color="#3498db",
            opacity=0.75,
        ),
        secondary_y=False,
    )

    # Line trace: Predicted Landslide Risk Probability
    fig_proj.add_trace(
        go.Scatter(
            x=dates,
            y=forecast_risk,
            name="Landslide Threat Probability (%)",
            mode="lines+markers+text",
            text=[f"{p}%" for p in forecast_risk],
            textposition="top center",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=8, color="#c0392b"),
        ),
        secondary_y=True,
    )

    # Critical threshold reference line
    fig_proj.add_hline(
        y=75.0,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="Critical Threshold (75%)",
        annotation_position="bottom right",
        secondary_y=True,
    )

    fig_proj.update_layout(
        title="7-Day Synchronized Hydro-Geological Forecast Model",
        xaxis_title="Forecast Timeline",
        height=380,
        margin=dict(l=20, r=20, b=20, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_proj.update_yaxes(title_text="24h Rainfall Forecast (mm)", secondary_y=False)
    fig_proj.update_yaxes(title_text="Landslide Probability (%)", range=[0, 110], secondary_y=True)

    st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown("---")

    # Section 3: Road Connectivity Status Matrix
    st.markdown("### 🛣️ North Eastern Region Road Connectivity & Corridor Status")
    st.write("Dynamic operational status of vital lifeline routes based on current risk scores and field telemetry.")

    corridor_rows = []
    for corridor in HIGHWAY_CORRIDORS:
        # Dynamic risk score factoring in current regional weather intensity
        scaled_risk = min(0.99, corridor["risk_base"] * (risk_prob / 0.5 if risk_prob > 0.3 else 0.8))
        if scaled_risk >= 0.75:
            status = "🔴 Blocked (Severe Mudflow Debris)"
            action_tag = "Emergency Diversion Active"
        elif scaled_risk >= 0.45:
            status = "🟡 Restrictive Traffic (Single Lane Convoy)"
            action_tag = "Escorted Convoy Only"
        else:
            status = "🟢 Open (Nominal Conditions)"
            action_tag = "Clear Passage"

        corridor_rows.append({
            "Corridor ID": corridor["id"],
            "Highway Stretch Name": corridor["name"],
            "Hazard Score": f"{scaled_risk * 100:.1f}%",
            "Transit Status": status,
            "Traffic Priority": corridor["traffic_vol"],
            "Emergency Bypass Route": corridor["bypass"],
            "Control Protocol": action_tag,
        })

    df_corridors = pd.DataFrame(corridor_rows)
    st.dataframe(df_corridors, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2: GIS HAZARD MAP WITH 3D ELEVATION DATA (PyDeck)
# ===========================================================================
with tab2:
    st.subheader("🗺️ 3D Geospatial Hazard Mapping (PyDeck Engine)")
    st.write(
        "Interactive 3D GIS visualization centered over the North-Eastern Region. "
        "High-risk villages and hill settlements are rendered as **elevated 3D columns** where "
        "the column elevation and red color intensity correspond directly to the computed landslide threat index."
    )

    # Layer toggles
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1:
        show_3d_columns = st.checkbox("🏛️ 3D Hazard Columns (Villages)", value=True)
    with ctrl_col2:
        show_roads = st.checkbox("🛣️ Vulnerable Highway Corridors", value=True)
    with ctrl_col3:
        show_infra = st.checkbox("⚡ Critical Infrastructure Assets", value=True)
    with ctrl_col4:
        show_reports = st.checkbox("📍 Citizen Anomaly Reports", value=True)

    # Prepare PyDeck Layers
    deck_layers = []

    # 1. 3D Hazard Columns Layer (Villages)
    if show_3d_columns:
        village_records = []
        for v in VILLAGES_HAZARD_DATA:
            # Scale village risk with current session risk
            eff_risk = min(0.99, v["base_risk"] * (0.8 + (risk_prob * 0.4)))
            
            # Color gradient: Green -> Orange -> Red
            if eff_risk >= 0.75:
                col = [231, 76, 60, 220]  # Red
            elif eff_risk >= 0.50:
                col = [243, 156, 18, 220]  # Orange
            elif eff_risk >= 0.30:
                col = [241, 196, 15, 220]  # Yellow
            else:
                col = [46, 204, 113, 220]  # Green

            village_records.append({
                "name": v["name"],
                "state": v["state"],
                "lat": v["lat"],
                "lon": v["lon"],
                "population": v["population"],
                "threat_index": round(eff_risk * 100, 1),
                "elevation": eff_risk * 8000,  # 3D Column Height
                "color": col,
            })

        df_villages = pd.DataFrame(village_records)

        layer_columns = pdk.Layer(
            "ColumnLayer",
            data=df_villages,
            get_position="[lon, lat]",
            get_elevation="elevation",
            elevation_scale=1,
            radius=2200,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        deck_layers.append(layer_columns)

    # 2. Vulnerable Roads Layer (PathLayer)
    if show_roads:
        road_records = []
        for c in HIGHWAY_CORRIDORS:
            r_risk = min(0.99, c["risk_base"] * (risk_prob / 0.5 if risk_prob > 0.3 else 0.8))
            if r_risk >= 0.75:
                r_col = [235, 59, 90, 240]  # Crimson
            elif r_risk >= 0.45:
                r_col = [250, 130, 49, 240]  # Orange
            else:
                r_col = [32, 191, 115, 240]  # Green

            road_records.append({
                "id": c["id"],
                "name": c["name"],
                "path": c["path"],
                "risk_pct": f"{r_risk * 100:.1f}%",
                "color": r_col,
            })

        df_roads = pd.DataFrame(road_records)

        layer_paths = pdk.Layer(
            "PathLayer",
            data=df_roads,
            get_path="path",
            get_color="color",
            width_scale=35,
            width_min_pixels=4,
            pickable=True,
            auto_highlight=True,
        )
        deck_layers.append(layer_paths)

    # 3. Critical Infrastructure Layer (ScatterplotLayer)
    if show_infra:
        df_infra = pd.DataFrame(INFRASTRUCTURE_ASSETS)
        layer_infra = pdk.Layer(
            "ScatterplotLayer",
            data=df_infra,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=1800,
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            get_line_color=[255, 255, 255, 255],
        )
        deck_layers.append(layer_infra)

    # 4. Citizen Anomaly Reports Layer (ScatterplotLayer)
    if show_reports and len(st.session_state.citizen_reports) > 0:
        df_citizen = pd.DataFrame(st.session_state.citizen_reports)
        layer_citizen = pdk.Layer(
            "ScatterplotLayer",
            data=df_citizen,
            get_position="[lon, lat]",
            get_color=[155, 89, 182, 240],  # Purple beacons
            get_radius=2500,
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=3,
            get_line_color=[255, 255, 255, 255],
        )
        deck_layers.append(layer_citizen)

    # Center map on active station
    center_lat, center_lon = station_meta["coords"]

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=7.8,
        pitch=48.0,  # 3D tilted perspective
        bearing=15.0,
    )

    deck = pdk.Deck(
        layers=deck_layers,
        initial_view_state=view_state,
        tooltip={
            "html": """
            <div style="background-color: #2c3e50; color: white; padding: 10px; border-radius: 6px; font-size: 12px;">
                <b>{name}</b><br/>
                <b>Threat Level / Risk:</b> {threat_index}% {risk_pct}<br/>
                <b>Asset Type / Population:</b> {type} {population}<br/>
                <b>Coordinates:</b> [{lat}, {lon}]<br/>
                <b>Status / Anomaly:</b> {anomaly_type} {severity}
            </div>
            """,
            "style": {"color": "white"},
        },
        map_style="mapbox://styles/mapbox/dark-v10",
    )

    st.pydeck_chart(deck)

    # Map Legend
    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    leg_col1.markdown("🔴 **3D Columns**: Village Risk Index (Height = Threat Level)")
    leg_col2.markdown("🛣️ **Road Paths**: Red (Blocked) / Orange (Caution) / Green (Open)")
    leg_col3.markdown("⚡ **Circles**: Critical Infrastructure (Dams, Power Masts, Bridges)")
    leg_col4.markdown("🟣 **Purple Beacons**: Citizen Ground Anomaly Reports")


# ===========================================================================
# TAB 3: FIELD PORTAL (CITIZEN & OFFICIAL REPORTING)
# ===========================================================================
with tab3:
    st.subheader("📢 Citizen & Field Official Anomaly Reporting Portal")
    st.write(
        "Empowers local citizens, village defense parties, and SDRF field personnel to upload "
        "ground anomaly sightings. Image uploads automatically extract GPS metadata via **Pillow/EXIF**, "
        "with instant fallback to manual coordinates. Verified reports dynamically populate the active 3D GIS map."
    )

    r_col1, r_col2 = st.columns([3, 2])

    with r_col1:
        st.markdown("#### 📝 Submit Ground Hazard Report")
        with st.form("ground_report_form", clear_on_submit=False):
            # Anomaly Type Dropdown
            anomaly_type = st.selectbox(
                "Select Anomaly Type:",
                [
                    "Cracks (Tension fractures along slope)",
                    "Active Slope Movement (Creep / Rockfall observed)",
                    "Blocked Road (Mudflow / Debris on corridor)",
                    "Hydrological Changes (Sudden spring surge / muddy runoff)",
                ],
            )

            severity_level = st.select_slider(
                "Observed Threat Severity:",
                options=["Advisory / Minor", "Moderate", "Critical - Evacuate Immediately"],
                value="Moderate",
            )

            description = st.text_area(
                "Ground Description & Visible Structural Impact:",
                placeholder="E.g., 8cm wide tension crack spreading across 25 meters of road verge near bridge abutment...",
            )

            # Media File Uploader
            uploaded_file = st.file_uploader(
                "Upload Geotagged Media (JPG, PNG, MP4, MOV):",
                type=["jpg", "jpeg", "png", "mp4", "mov"],
                help="GPS coordinates are extracted automatically if EXIF tags are embedded in the image.",
            )

            # Automatic EXIF Processing
            extracted_lat, extracted_lon = None, None
            if uploaded_file is not None and uploaded_file.type.startswith("image"):
                with st.spinner("Parsing image EXIF tags for GPS metadata..."):
                    extracted_lat, extracted_lon = extract_gps_from_exif(uploaded_file)

                if extracted_lat and extracted_lon:
                    st.success(f"✅ GPS Extracted from EXIF: Latitude {extracted_lat}°, Longitude {extracted_lon}°")
                else:
                    st.info("ℹ️ No EXIF GPS found in image (or camera GPS disabled). Please verify coordinates below.")

            # Coordinate Inputs with Baseline Fallback
            c_sub1, c_sub2 = st.columns(2)
            default_lat = extracted_lat if extracted_lat is not None else station_meta["coords"][0]
            default_lon = extracted_lon if extracted_lon is not None else station_meta["coords"][1]

            input_lat = c_sub1.number_input(
                "Latitude (°N):",
                value=float(default_lat),
                format="%.5f",
                help="Manual override field for exact latitude",
            )
            input_lon = c_sub2.number_input(
                "Longitude (°E):",
                value=float(default_lon),
                format="%.5f",
                help="Manual override field for exact longitude",
            )

            landmark = st.text_input("Landmark / Nearest Village:", placeholder="E.g., Near Mile 18, Shillong Bypass")
            reporter_name = st.text_input("Reporter Name / Official ID:", placeholder="E.g., Village Headman / SDRF Officer")

            submit_report = st.form_submit_button("🚀 Submit Anomaly Report", type="primary")

            if submit_report:
                if not description:
                    st.error("Please provide a brief description of the observed anomaly.")
                else:
                    new_id = f"REP-2026-{len(st.session_state.citizen_reports) + 101}"
                    new_report = {
                        "id": new_id,
                        "anomaly_type": anomaly_type,
                        "severity": severity_level,
                        "description": description,
                        "lat": input_lat,
                        "lon": input_lon,
                        "landmark": landmark if landmark else "Unspecified Landmark",
                        "reporter": reporter_name if reporter_name else "Anonymous Scout",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M IST"),
                        "status": "Pending Cloud Verification" if not st.session_state.offline_mode else "Queued in Local Edge Cache",
                        "offline_queued": st.session_state.offline_mode,
                    }

                    # Handle offline queue vs immediate live dispatch
                    if st.session_state.offline_mode:
                        st.session_state.offline_report_queue.append(new_report)
                        st.session_state.citizen_reports.append(new_report)
                        st.warning(
                            f"📶 **OFFLINE MODE ENGAGED**: Report `{new_id}` stored securely in local browser/device cache. "
                            "It will be automatically synced to MDoNER State Command Center when connection is restored."
                        )
                    else:
                        st.session_state.citizen_reports.append(new_report)
                        st.success(
                            f"✅ Report `{new_id}` successfully registered! It has been dispatched to district authorities "
                            "and dynamically anchored onto the **3D GIS Hazard Map**."
                        )

    with r_col2:
        st.markdown("#### 📋 Active Field Reports Ledger")
        st.write(f"Total Logged Reports: **{len(st.session_state.citizen_reports)}**")

        for rep in reversed(st.session_state.citizen_reports):
            is_offline = rep.get("offline_queued", False)
            with st.expander(f"{'⏳' if is_offline else '📌'} {rep['id']}: {rep['anomaly_type']}", expanded=False):
                st.write(f"**Severity**: `{rep['severity']}`")
                st.write(f"**Description**: {rep['description']}")
                st.write(f"**Location**: {rep['landmark']} (`{rep['lat']:.4f}°N, {rep['lon']:.4f}°E`)")
                st.write(f"**Reporter**: {rep['reporter']} | **Logged**: {rep['timestamp']}")
                if is_offline:
                    st.caption("📶 *Status: Queued in Local Offline Edge Cache*")
                else:
                    st.caption(f"🛡️ *Status: {rep['status']}*")


# ===========================================================================
# TAB 4: OFFLINE CACHE & MULTILINGUAL COMMS
# ===========================================================================
with tab4:
    st.subheader("📶 Offline Cache, Prioritization & Multilingual Distribution Hub")
    st.write(
        "Guarantees operational resilience in remote Himalayan terrains where cellular networks are fragile or disrupted. "
        "Provides low-bandwidth offline caching, disaster response ranking, and automated multi-lingual emergency broadcasting."
    )

    st.markdown("---")

    # Section 1: Low-Network & Offline Resilience Architecture
    st.markdown("### 1. Low-Network / Offline Resilience Architecture")
    c_off1, c_off2 = st.columns([2, 3])

    with c_off1:
        st.markdown("**Edge Network Simulation:**")
        offline_toggle = st.toggle(
            "Activate Offline Mode (Simulate Wi-Fi Blackout)",
            value=st.session_state.offline_mode,
            key="offline_toggle_btn",
        )
        if offline_toggle != st.session_state.offline_mode:
            st.session_state.offline_mode = offline_toggle
            st.rerun()

        st.write(
            f"**Current State**: {'🔴 Disconnected (Edge Cache Active)' if st.session_state.offline_mode else '🟢 Cloud Connected (Zero Latency)'}"
        )

    with c_off2:
        queued_count = len(st.session_state.offline_report_queue)
        st.metric("Pending Offline Reports in Edge Cache", f"{queued_count} Records")

        if queued_count > 0:
            if st.button("🔄 Synchronize Offline Cache to Cloud Command", type="primary"):
                with st.spinner("Pushing cached packets to MDoNER Cloud Cluster..."):
                    # Mark all as synced
                    for r in st.session_state.citizen_reports:
                        r["offline_queued"] = False
                        r["status"] = "Synced with MDoNER Cloud Server"
                    st.session_state.offline_report_queue = []
                st.success("✅ All local offline cache packets successfully synchronized with zero data loss!")
                st.rerun()
        else:
            st.caption("All field reports are currently synchronized with the cloud registry.")

    st.markdown("---")

    # Section 2: Emergency Response Prioritization Engine
    st.markdown("### 2. Emergency Response Prioritization Engine")
    st.write(
        "Auto-ranks high-risk hazard zones using a multi-criteria vulnerability index: "
        "`Vulnerability Index = 0.35 * Hazard Risk + 0.30 * Population Density + 0.25 * Road Criticality + 0.10 * Infrastructure Density`"
    )

    priority_records = []
    for v in VILLAGES_HAZARD_DATA:
        pop_norm = min(1.0, v["population"] / 30000.0)
        h_risk = min(0.99, v["base_risk"] * (0.8 + (risk_prob * 0.4)))
        road_crit = 0.85 if "Settlement" in v["name"] or "Escarpment" in v["name"] else 0.55
        infra_crit = 0.75

        # Weighted composite score
        composite_score = (0.35 * h_risk) + (0.30 * pop_norm) + (0.25 * road_crit) + (0.10 * infra_crit)

        if composite_score >= 0.70:
            protocol = "🚨 Priority 1: Deploy NDRF / SDRF Convoy & Sound Local Sirens"
            badge = "CRITICAL"
        elif composite_score >= 0.50:
            protocol = "⚠️ Priority 2: Pre-position Earth Movers (BRO) & Close Lane Traffic"
            badge = "HIGH"
        else:
            protocol = "🟡 Priority 3: Issue Village Advisory & Drone LiDAR Patrol"
            badge = "MODERATE"

        priority_records.append({
            "Rank": 0,  # Computed after sorting
            "Settlement / Hazard Zone": v["name"],
            "State": v["state"],
            "Population": v["population"],
            "Hazard Score": f"{h_risk * 100:.1f}%",
            "Priority Score": round(composite_score * 100, 1),
            "Tier": badge,
            "Mandated Emergency Protocol": protocol,
        })

    df_priority = pd.DataFrame(priority_records).sort_values(by="Priority Score", ascending=False)
    df_priority["Rank"] = [f"#{i+1}" for i in range(len(df_priority))]
    st.dataframe(df_priority, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Section 3: Multilingual Alert Distribution Hub
    st.markdown("### 3. Multilingual Alert Distribution Hub")
    st.write(
        "Automated regional broadcast engine formatting multi-lingual disaster warnings for the North-Eastern Region "
        "across **English**, **Hindi (हिन्दी)**, **Assamese (অসমীয়া)**, and **Bengali (বাংলা)**."
    )

    alert_location = selected_station_name
    lang_tab1, lang_tab2, lang_tab3, lang_tab4 = st.tabs([
        "🇬🇧 English",
        "🇮🇳 Hindi (हिन्दी)",
        "🏞️ Assamese (অসমীয়া)",
        "🌺 Bengali (বাংলা)",
    ])

    # Multilingual Templates
    msg_en = (
        f"🚨 EMERGENCY LANDSLIDE ALERT ({risk_badge}): High slope instability and heavy rainfall saturation "
        f"detected around {alert_location}. Evacuate vulnerable valley sectors immediately. Follow District Police directions."
    )
    msg_hi = (
        f"🚨 आपातकालीन भूस्खलन चेतावनी ({risk_badge}): {alert_location} के आसपास अत्यधिक वर्षा एवं ढलान विस्थापन "
        f"दर्ज किया गया है। संवेदनशील ढलानों से तुरंत सुरक्षित राहत शिविरों की ओर जाएं। एनडीआरएफ निर्देशों का पालन करें।"
    )
    msg_as = (
        f"🚨 জৰুৰীকালীন ভূমিস্খলন সতৰ্কবাৰ্তা ({risk_badge}): {alert_location} অঞ্চলত ধাৰাসাৰ বৰষুণ আৰু পাহাৰীয়া মাটিৰ স্খলনৰ "
        f"প্ৰচণ্ড আশংকা দেখা দিছে। সকলো নাগৰিকক অনতিপলমে সুৰক্ষিত আশ্ৰয় শিবিৰলৈ স্থানান্তৰিত হ'বলৈ অনুৰোধ জনোৱা হ'ল।"
    )
    msg_bn = (
        f"🚨 জরুরি ভূমিধস সতর্কতা ({risk_badge}): {alert_location} সংলগ্ন অঞ্চলে অতিবৃষ্টির কারণে বিপজ্জনক ভূমিধসের "
        f"আশঙ্কা রয়েছে। ঝুঁকিপূর্ণ ঢালু এলাকা থেকে অবিলম্বে নিকটবর্তী নিরাপদ আশ্রয়কেন্দ্রে সরে যান। প্রশাসনের নির্দেশ মেনে চলুন।"
    )

    with lang_tab1:
        st.text_area("English Emergency Broadcast Message:", value=msg_en, height=95)
    with lang_tab2:
        st.text_area("Hindi Emergency Broadcast Message (हिन्दी):", value=msg_hi, height=95)
    with lang_tab3:
        st.text_area("Assamese Emergency Broadcast Message (অসমীয়া):", value=msg_as, height=95)
    with lang_tab4:
        st.text_area("Bengali Emergency Broadcast Message (বাংলা):", value=msg_bn, height=95)

    # Multi-Channel Dispatch Trigger Simulator
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        if st.button("📲 Push Twilio SMS Gateway", type="primary"):
            st.success("✅ SMS Alert broadcasted to 14,200 registered residents via District Emergency Cell.")
    with b_col2:
        if st.button("📢 Trigger Community Sirens"):
            st.warning("⚠️ High-decibel audio sirens triggered across active valley repeater nodes.")
    with b_col3:
        if st.button("📻 Broadcast to AIR / FM"):
            st.info("ℹ️ Emergency audio warning patched into regional All India Radio & Doordarshan feeds.")
    with b_col4:
        if st.button("📱 Push WhatsApp Gov Alert"):
            st.success("✅ Official State Disaster WhatsApp bulletin broadcasted to community heads.")


# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown("---")
st.caption(
    "🏔️ **SIH Problem Statement ID: SIH26001** | AI-Based Early Warning and Landslide Risk Monitoring System in NER | "
    "Ministry of Development of North Eastern Region (MDoNER) | Scalable Full-Stack Architecture Prototype."
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_folium import st_folium
from utils.map_builder import build_mumbai_map, CONSTRUCTION_SITES
from data.aqi_fetcher import get_aqi_category

def render(station_df: pd.DataFrame):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 05 · NGT CONSTRUCTION MONITORS</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        CONSTRUCTION — NGT Mandated Site Monitors</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        National Green Tribunal 2021 order · MCGM mandated dust monitors ·
        All sites >500 sq meters required to display live AQI
        </p>
    </div>""", unsafe_allow_html=True)

    # Stats
    avg_pm10 = sum(s["pm10"] for s in CONSTRUCTION_SITES) / len(CONSTRUCTION_SITES)
    worst    = max(CONSTRUCTION_SITES, key=lambda x: x["pm10"])
    active   = sum(1 for s in CONSTRUCTION_SITES if s["status"] == "Active")

    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val,color) in zip([c1,c2,c3,c4],[
        ("SITES MONITORED", len(CONSTRUCTION_SITES), "#4caf7d"),
        ("CURRENTLY ACTIVE", active,                 "#ffc96b"),
        ("AVG PM10",        f"{avg_pm10:.0f} µg/m³", "#ff6b6b"),
        ("WHO PM10 LIMIT",  "45 µg/m³",              "#4a9eff"),
    ]):
        with col:
            st.markdown(f"""
            <div style='background:#0f1a0f;border:1px solid #1e3020;
            border-radius:4px;padding:12px 14px;margin-bottom:12px;'>
                <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
                color:#2a5040;text-transform:uppercase;letter-spacing:0.08em;
                margin-bottom:4px;'>{label}</div>
                <div style='font-family:"IBM Plex Mono",monospace;font-size:18px;
                font-weight:600;color:{color};'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # NGT info box
    st.markdown("""
    <div style='background:rgba(74,158,255,0.06);
    border:1px solid rgba(74,158,255,0.2);border-radius:4px;
    padding:12px 16px;margin-bottom:16px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#4a9eff;text-transform:uppercase;margin-bottom:6px;'>
        ⚖️ LEGAL MANDATE</div>
        <div style='font-size:12px;color:#c8e0c8;line-height:1.6;'>
        The <b>National Green Tribunal (NGT) 2021 order</b> and
        <b>MCGM 2023 extension</b> mandate that all construction sites
        above 500 sq meters in Mumbai must install real-time PM10 dust
        monitors and display readings on digital screens visible to the public.
        BREATH·MUMBAI is the only platform aggregating this data alongside
        CPCB air quality readings for a complete picture.
        </div>
    </div>""", unsafe_allow_html=True)

    # Map
    m = build_mumbai_map(station_df, show_construction=True, show_heatmap=True)
    st_folium(m, width=None, height=460, returned_objects=[])

    # Site table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:10px;'>
    ALL CONSTRUCTION SITES — PM10 READINGS</div>""",
    unsafe_allow_html=True)

    sites_sorted = sorted(CONSTRUCTION_SITES, key=lambda x: x["pm10"],
                         reverse=True)
    for site in sites_sorted:
        pm10  = site["pm10"]
        ratio = pm10 / 45  # WHO limit
        color = ("#ff4444" if pm10 > 300 else "#ff7e00" if pm10 > 250
                 else "#ffff00" if pm10 > 200 else "#4caf7d")

        st.markdown(f"""
        <div style='background:#0f1a0f;border:1px solid #1e3020;
        border-radius:4px;padding:10px 14px;margin-bottom:6px;'>
            <div style='display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px;'>
                <div>
                    <div style='font-size:13px;font-weight:500;
                    color:#c8e0c8;'>🏗️ {site["name"]}</div>
                    <div style='font-size:10px;color:#2a5040;
                    font-family:"IBM Plex Mono",monospace;'>
                    {site["status"]} · NGT Monitor ✅</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-family:"IBM Plex Mono",monospace;
                    font-size:20px;font-weight:600;color:{color};'>
                    {pm10} µg/m³</div>
                    <div style='font-size:10px;color:{color};
                    font-family:"IBM Plex Mono",monospace;'>
                    {ratio:.1f}x WHO limit</div>
                </div>
            </div>
            <div style='height:5px;background:#1e3020;border-radius:3px;'>
                <div style='height:5px;width:{min(ratio/10*100,100):.0f}%;
                background:{color};border-radius:3px;'></div>
            </div>
        </div>""", unsafe_allow_html=True)
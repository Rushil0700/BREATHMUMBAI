import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_folium import st_folium
from utils.map_builder import build_mumbai_map, build_aqi_graph
from data.aqi_fetcher import get_aqi_category, get_source_attribution

def render(station_df: pd.DataFrame, weather: dict):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 01 · LIVE INTELLIGENCE</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        HEATMAP — Live Mumbai Air Quality</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        Real-time CPCB sensor data · Construction site overlays ·
        NGT mandated monitors · Updates every hour
        </p>
    </div>""", unsafe_allow_html=True)

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        show_heat = st.toggle("AQI Heatmap", value=True)
    with col2:
        show_construction = st.toggle("Construction Sites", value=True)
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Stat cards
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        ("STATIONS LIVE", len(station_df), "#4caf7d"),
        ("CITY AVG AQI",  int(station_df["aqi"].mean()) if not station_df.empty else "N/A", "#ffc96b"),
        ("WORST ZONE",    f"{station_df.loc[station_df['aqi'].idxmax(),'station']} ({station_df['aqi'].max()})" if not station_df.empty else "N/A", "#ff6b6b"),
        ("WIND SPEED",    f"{weather.get('wind_speed','?')} m/s", "#4a9eff"),
    ]
    for col, (label, val, color) in zip([c1,c2,c3,c4], stats):
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

    # Map
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:8px;'>
    INTERACTIVE MAP — Click stations for details</div>""",
    unsafe_allow_html=True)

    m = build_mumbai_map(
        station_df,
        show_construction=show_construction,
        show_heatmap=show_heat
    )
    st_folium(m, width=None, height=520, returned_objects=[])

    # Station table + source attribution
    st.markdown("---")
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:10px;'>
    ALL STATIONS — LIVE READINGS</div>""", unsafe_allow_html=True)

    if not station_df.empty:
        sorted_df = station_df.sort_values("aqi", ascending=False)
        for _, row in sorted_df.iterrows():
            cat   = get_aqi_category(int(row["aqi"]))
            color = cat["color"]
            sources = get_source_attribution(row)
            top_source = list(sources.keys())[0] if sources else "Unknown"
            top_pct    = list(sources.values())[0] if sources else 0

            st.markdown(f"""
            <div style='background:#0f1a0f;border:1px solid #1e3020;
            border-radius:4px;padding:10px 14px;margin-bottom:6px;
            display:flex;gap:12px;align-items:center;flex-wrap:wrap;'>
                <div style='min-width:120px;'>
                    <div style='font-size:13px;font-weight:500;
                    color:#c8e0c8;'>{row["station"]}</div>
                    <div style='font-size:10px;color:#2a5040;
                    font-family:"IBM Plex Mono",monospace;'>{cat["label"]}</div>
                </div>
                <div style='font-family:"IBM Plex Mono",monospace;
                font-size:24px;font-weight:600;color:{color};
                min-width:70px;'>{int(row["aqi"])}</div>
                <div style='flex:1;'>
                    <div style='height:6px;background:#1e3020;border-radius:3px;'>
                        <div style='height:6px;width:{min(row["aqi"]/5,100):.0f}%;
                        background:{color};border-radius:3px;'></div>
                    </div>
                </div>
                <div style='font-size:11px;color:#4a7060;min-width:200px;'>
                    Top source: {top_source} ({top_pct}%)<br>
                    <span style='color:#2a5040;'>{cat["advice"]}</span>
                </div>
            </div>""", unsafe_allow_html=True)
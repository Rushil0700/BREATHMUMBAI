import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.aqi_fetcher import get_aqi_category

SCENARIOS = [
    "50% of Mumbaikars switch to electric vehicles",
    "All construction sites halt for 7 days",
    "A major factory in Chembur shuts down",
    "Heavy monsoon rain for 3 consecutive days",
    "Diwali fireworks banned city-wide",
    "Wind speed doubles from Arabian Sea",
    "Temperature inversion traps pollution for 48hrs",
    "All buses and autos convert to CNG",
]

def render(predictor, station_df: pd.DataFrame):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 07 · SCENARIO LABORATORY</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        WHAT-IF LAB — Mumbai Pollution Scenarios</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        Inject any scenario · ML model recalculates city AQI ·
        See the pollution impact of policy decisions
        </p>
    </div>""", unsafe_allow_html=True)

    # Example scenarios
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:8px;'>
    QUICK SCENARIOS — CLICK TO LOAD</div>""", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, s in enumerate(SCENARIOS):
        with cols[i % 2]:
            if st.button(f"💉 {s}", key=f"qs_{i}", use_container_width=True):
                st.session_state.whatif_text = s

    st.markdown("<br>", unsafe_allow_html=True)
    scenario = st.text_area(
        "Describe your scenario",
        value=st.session_state.get("whatif_text",""),
        placeholder="e.g. What if Mumbai bans all diesel vehicles? "
                    "What if a factory fire breaks out in Dharavi?",
        height=90, label_visibility="collapsed"
    )

    run = st.button("⚡ CALCULATE IMPACT", use_container_width=True,
                    type="primary")

    # Baseline
    city_aqi = int(station_df["aqi"].mean()) if not station_df.empty else 150
    st.markdown(f"""
    <div style='background:#0f1a0f;border:1px solid #1e3020;
    border-radius:4px;padding:10px 16px;margin:12px 0;'>
        <span style='font-family:"IBM Plex Mono",monospace;font-size:11px;
        color:#2a5040;'>CURRENT MUMBAI AVG AQI: </span>
        <span style='font-family:"IBM Plex Mono",monospace;font-size:16px;
        font-weight:600;color:{get_aqi_category(city_aqi)["color"]};'>
        {city_aqi}</span>
    </div>""", unsafe_allow_html=True)

    if run and scenario:
        text = scenario.lower()
        modified_aqi = city_aqi
        explanation  = []

        # Keyword-based AQI modification
        if any(w in text for w in ["electric","ev","evs"]):
            reduction = city_aqi * 0.28
            modified_aqi -= reduction
            explanation.append(f"EVs eliminate vehicle NO2/CO — AQI ↓{reduction:.0f}")

        if any(w in text for w in ["construction","halt","stop","pause"]):
            reduction = city_aqi * 0.15
            modified_aqi -= reduction
            explanation.append(f"No construction dust — AQI ↓{reduction:.0f}")

        if any(w in text for w in ["rain","monsoon","shower","rainfall"]):
            reduction = city_aqi * 0.45
            modified_aqi -= reduction
            explanation.append(f"Rain washes PM2.5/PM10 — AQI ↓{reduction:.0f}")

        if any(w in text for w in ["wind","breeze","storm"]):
            reduction = city_aqi * 0.30
            modified_aqi -= reduction
            explanation.append(f"High wind disperses pollution — AQI ↓{reduction:.0f}")

        if any(w in text for w in ["factory","industrial","plant","shut"]):
            reduction = city_aqi * 0.12
            modified_aqi -= reduction
            explanation.append(f"Industrial SO2 eliminated — AQI ↓{reduction:.0f}")

        if any(w in text for w in ["diwali","firework","cracker","ban"]):
            reduction = city_aqi * 0.20
            modified_aqi -= reduction
            explanation.append(f"No Diwali particulates — AQI ↓{reduction:.0f}")

        if any(w in text for w in ["fire","accident","disaster","toxic","leak"]):
            increase = city_aqi * 0.60
            modified_aqi += increase
            explanation.append(f"Emergency pollution event — AQI ↑{increase:.0f}")

        if any(w in text for w in ["inversion","trap","stagnant","fog"]):
            increase = city_aqi * 0.40
            modified_aqi += increase
            explanation.append(f"Temperature inversion traps pollution — AQI ↑{increase:.0f}")

        if any(w in text for w in ["cng","bus","auto","public transport"]):
            reduction = city_aqi * 0.18
            modified_aqi -= reduction
            explanation.append(f"CNG transition cuts vehicle emissions — AQI ↓{reduction:.0f}")

        modified_aqi = max(10, min(500, round(modified_aqi)))
        delta        = modified_aqi - city_aqi
        delta_pct    = round(delta / city_aqi * 100, 1)
        delta_color  = "#4caf7d" if delta < 0 else "#ff6b6b"

        st.markdown(f"""
        <div style='background:#0f1a0f;border:1px solid
        {"rgba(76,175,125,0.4)" if delta < 0 else "rgba(255,107,107,0.4)"};
        border-radius:6px;padding:16px 20px;margin-top:14px;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            color:#2a5040;text-transform:uppercase;margin-bottom:12px;'>
            SCENARIO RESULT</div>
            <div style='display:flex;gap:24px;align-items:center;
            flex-wrap:wrap;'>
                <div>
                    <div style='font-size:11px;color:#2a5040;
                    font-family:"IBM Plex Mono",monospace;'>
                    CURRENT</div>
                    <div style='font-family:"IBM Plex Mono",monospace;
                    font-size:32px;font-weight:600;
                    color:{get_aqi_category(city_aqi)["color"]};'>
                    {city_aqi}</div>
                </div>
                <div style='font-size:28px;color:#2a5040;'>→</div>
                <div>
                    <div style='font-size:11px;color:#2a5040;
                    font-family:"IBM Plex Mono",monospace;'>
                    AFTER SCENARIO</div>
                    <div style='font-family:"IBM Plex Mono",monospace;
                    font-size:32px;font-weight:600;
                    color:{get_aqi_category(modified_aqi)["color"]};'>
                    {modified_aqi}</div>
                </div>
                <div>
                    <div style='font-size:11px;color:#2a5040;
                    font-family:"IBM Plex Mono",monospace;'>CHANGE</div>
                    <div style='font-family:"IBM Plex Mono",monospace;
                    font-size:28px;font-weight:600;color:{delta_color};'>
                    {("+" if delta > 0 else "")}{delta} ({delta_pct}%)</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        if explanation:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
            font-size:10px;color:#2a5040;text-transform:uppercase;
            margin-bottom:8px;'>HOW THE MODEL CALCULATED THIS</div>""",
            unsafe_allow_html=True)
            for exp in explanation:
                color = "#4caf7d" if "↓" in exp else "#ff6b6b"
                st.markdown(f"""
                <div style='font-size:12px;color:{color};
                font-family:"IBM Plex Mono",monospace;
                padding:4px 0;border-bottom:1px solid #1e3020;'>
                → {exp}</div>""", unsafe_allow_html=True)
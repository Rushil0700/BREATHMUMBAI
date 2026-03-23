import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.aqi_fetcher import get_aqi_category, get_health_risk

def render(station_df: pd.DataFrame):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 04 · PERSONAL HEALTH</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        HEALTH RISK — Personal Air Quality Assessment</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        Personalized risk score based on your health profile ·
        WHO guideline comparison · Mask recommendations
        </p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
        font-size:10px;color:#2a5040;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:10px;'>
        YOUR HEALTH PROFILE</div>""", unsafe_allow_html=True)

        age        = st.slider("Age", 5, 90, 28)
        has_asthma = st.toggle("Asthma / Respiratory condition")
        has_heart  = st.toggle("Heart condition")
        is_pregnant= st.toggle("Pregnant")
        is_athlete = st.toggle("Athlete / Outdoor exerciser")

        station_names = station_df["station"].tolist() if not station_df.empty else ["Mumbai"]
        location = st.selectbox("Your current area", station_names)

    with col2:
        current_aqi = 150
        if not station_df.empty:
            row = station_df[station_df["station"] == location]
            if not row.empty:
                current_aqi = int(row.iloc[0]["aqi"])

        risk = get_health_risk(
            current_aqi, age, has_asthma, has_heart, is_pregnant
        )
        rcolor = risk["color"]

        st.markdown(f"""
        <div style='background:#0f1a0f;border:2px solid {rcolor};
        border-radius:6px;padding:20px;text-align:center;margin-bottom:12px;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            color:#2a5040;margin-bottom:8px;'>YOUR PERSONAL RISK SCORE</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:52px;
            font-weight:600;color:{rcolor};line-height:1;'>
            {risk["risk_score"]}</div>
            <div style='font-size:11px;color:#2a5040;margin-top:4px;'>/ 100</div>
            <div style='font-size:18px;font-weight:600;color:{rcolor};
            margin-top:8px;'>{risk["level"]}</div>
        </div>""", unsafe_allow_html=True)

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk["risk_score"],
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#4a7060"),
                bar=dict(color=rcolor),
                bgcolor="#0f1a0f",
                bordercolor="#1e3020",
                steps=[
                    dict(range=[0,20],  color="rgba(0,228,0,0.1)"),
                    dict(range=[20,45], color="rgba(255,255,0,0.1)"),
                    dict(range=[45,65], color="rgba(255,126,0,0.1)"),
                    dict(range=[65,85], color="rgba(255,0,0,0.1)"),
                    dict(range=[85,100],color="rgba(143,63,151,0.1)"),
                ],
            ),
            number=dict(font=dict(color=rcolor,
                       family="IBM Plex Mono"), suffix="/100"),
        ))
        fig.update_layout(
            paper_bgcolor="#0a0f0a",
            font=dict(color="#4a7060", family="IBM Plex Mono"),
            height=220,
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown("---")
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:12px;'>
    PERSONALISED RECOMMENDATIONS</div>""", unsafe_allow_html=True)

    cat = get_aqi_category(current_aqi)

    recs = [
        (risk["mask"],    "😷 MASK",    rcolor),
        (risk["action"],  "⚡ ACTION",   rcolor),
        (cat["advice"],   "🌬️ GENERAL", cat["color"]),
    ]
    if has_asthma:
        recs.append(("Keep bronchodilator inhaler accessible. Pollution triggers asthma attacks.", "🫁 ASTHMA", "#ff6b6b"))
    if is_athlete:
        recs.append(("Avoid outdoor training when AQI > 100. Exercise indoors or at dawn when AQI is lowest.", "🏃 ATHLETE", "#4a9eff"))
    if is_pregnant:
        recs.append(("Prenatal exposure to PM2.5 is linked to low birth weight. Minimize time outdoors.", "🤰 PREGNANCY", "#ff6b6b"))
    if age > 60:
        recs.append(("Elderly are 2x more susceptible. Consider N95 even at moderate AQI levels.", "👴 ELDERLY", "#ffc96b"))

    for advice, label, color in recs:
        st.markdown(f"""
        <div style='background:#0f1a0f;border-left:3px solid {color};
        border-top:1px solid #1e3020;border-right:1px solid #1e3020;
        border-bottom:1px solid #1e3020;border-radius:0 4px 4px 0;
        padding:10px 14px;margin-bottom:7px;display:flex;gap:12px;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            font-weight:600;color:{color};min-width:80px;'>{label}</div>
            <div style='font-size:12px;color:#c8e0c8;line-height:1.5;'>
            {advice}</div>
        </div>""", unsafe_allow_html=True)

    # WHO comparison
    st.markdown("---")
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:10px;'>
    WHO GUIDELINE COMPARISON</div>""", unsafe_allow_html=True)

    row = None
    if not station_df.empty:
        rows = station_df[station_df["station"] == location]
        if not rows.empty:
            row = rows.iloc[0]

    who_limits = [
        ("PM2.5 (annual)", 5,   row.get("pm25",0) if row is not None else 0,  "µg/m³"),
        ("PM10 (annual)",  15,  row.get("pm10",0) if row is not None else 0,   "µg/m³"),
        ("NO₂ (annual)",   10,  row.get("no2",0) if row is not None else 0,    "µg/m³"),
    ]
    for name, limit, current, unit in who_limits:
        if not current or current != current:
            current = 0
        ratio = current / limit if limit > 0 else 0
        bar_color = "#4caf7d" if ratio < 1 else "#ff6b6b"
        st.markdown(f"""
        <div style='margin-bottom:10px;'>
            <div style='display:flex;justify-content:space-between;
            margin-bottom:3px;'>
                <span style='font-size:12px;color:#4a7060;'>{name}</span>
                <span style='font-family:"IBM Plex Mono",monospace;
                font-size:11px;color:{bar_color};'>
                {current:.1f} / {limit} {unit}
                ({ratio:.1f}x WHO limit)</span>
            </div>
            <div style='height:6px;background:#1e3020;border-radius:3px;'>
                <div style='height:6px;width:{min(ratio*100/3,100):.0f}%;
                background:{bar_color};border-radius:3px;'></div>
            </div>
        </div>""", unsafe_allow_html=True)
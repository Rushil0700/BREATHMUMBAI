import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.aqi_fetcher import get_aqi_category

def render(station_df: pd.DataFrame):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 08 · AREA RANKINGS</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        LEADERBOARD — Cleanest vs Most Toxic Mumbai Areas</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        Real-time ranking of all Mumbai zones · Shareable ·
        Updated every hour from CPCB sensors
        </p>
    </div>""", unsafe_allow_html=True)

    if station_df.empty:
        st.error("No station data available")
        return

    sorted_df = station_df.sort_values("aqi", ascending=True).reset_index(drop=True)

    # Summary boxes
    best  = sorted_df.iloc[0]
    worst = sorted_df.iloc[-1]
    city_avg = int(sorted_df["aqi"].mean())

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style='background:rgba(76,175,125,0.08);
        border:1px solid rgba(76,175,125,0.3);border-radius:6px;
        padding:14px;text-align:center;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            color:#4caf7d;margin-bottom:4px;'>🥇 CLEANEST RIGHT NOW</div>
            <div style='font-size:16px;font-weight:600;color:#c8e0c8;'>
            {best["station"]}</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:22px;
            font-weight:600;color:#4caf7d;'>{int(best["aqi"])}</div>
            <div style='font-size:11px;color:#2a5040;'>{best["category"]}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        cat = get_aqi_category(city_avg)
        st.markdown(f"""
        <div style='background:#0f1a0f;border:1px solid #1e3020;
        border-radius:6px;padding:14px;text-align:center;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            color:#2a5040;margin-bottom:4px;'>🏙️ CITY AVERAGE</div>
            <div style='font-size:16px;font-weight:600;color:#c8e0c8;'>
            Mumbai</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:22px;
            font-weight:600;color:{cat["color"]};'>{city_avg}</div>
            <div style='font-size:11px;color:#2a5040;'>{cat["label"]}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style='background:rgba(255,68,68,0.08);
        border:1px solid rgba(255,68,68,0.3);border-radius:6px;
        padding:14px;text-align:center;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            color:#ff4444;margin-bottom:4px;'>☠️ MOST TOXIC RIGHT NOW</div>
            <div style='font-size:16px;font-weight:600;color:#c8e0c8;'>
            {worst["station"]}</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:22px;
            font-weight:600;color:#ff4444;'>{int(worst["aqi"])}</div>
            <div style='font-size:11px;color:#2a5040;'>{worst["category"]}</div>
        </div>""", unsafe_allow_html=True)

    # Ranking chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:10px;'>
    FULL RANKING — CLEANEST TO MOST POLLUTED</div>""",
    unsafe_allow_html=True)

    colors = [get_aqi_category(int(a))["color"] for a in sorted_df["aqi"]]

    fig = go.Figure(go.Bar(
        x=sorted_df["aqi"],
        y=sorted_df["station"],
        orientation="h",
        marker_color=colors,
        text=[f"AQI {int(a)} — {get_aqi_category(int(a))['label']}"
              for a in sorted_df["aqi"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color="#c8e0c8"),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0f0a", plot_bgcolor="#0f1a0f",
        height=460, margin=dict(l=10, r=180, t=10, b=10),
        font=dict(family="IBM Plex Mono", color="#4a7060", size=11),
        xaxis=dict(gridcolor="#1e3020", linecolor="#1e3020",
                   title="AQI", range=[0, sorted_df["aqi"].max()+80]),
        yaxis=dict(gridcolor="#1e3020", linecolor="#1e3020"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:10px;'>
    DETAILED STATION DATA</div>""", unsafe_allow_html=True)

    for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
        cat   = get_aqi_category(int(row["aqi"]))
        color = cat["color"]
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else \
                "🥉" if rank == 3 else \
                "☠️" if rank == len(sorted_df) else f"#{rank}"

        st.markdown(f"""
        <div style='background:#0f1a0f;border:1px solid #1e3020;
        border-radius:4px;padding:10px 14px;margin-bottom:5px;
        display:flex;gap:12px;align-items:center;'>
            <div style='font-size:16px;min-width:32px;'>{medal}</div>
            <div style='min-width:110px;font-size:13px;font-weight:500;
            color:#c8e0c8;'>{row["station"]}</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:20px;
            font-weight:600;color:{color};min-width:60px;'>
            {int(row["aqi"])}</div>
            <div style='flex:1;height:6px;background:#1e3020;border-radius:3px;'>
                <div style='height:6px;width:{min(row["aqi"]/5,100):.0f}%;
                background:{color};border-radius:3px;'></div>
            </div>
            <div style='font-size:11px;color:{color};min-width:110px;
            text-align:right;font-family:"IBM Plex Mono",monospace;'>
            {cat["label"]}</div>
        </div>""", unsafe_allow_html=True)
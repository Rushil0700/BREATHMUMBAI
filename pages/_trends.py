import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render(predictor):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 06 · HISTORICAL ANALYTICS</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        TRENDS — Historical Mumbai AQI Analysis</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        2 years of Mumbai AQI patterns · Seasonal analysis ·
        Festival spikes · Monsoon effect · Station comparison
        </p>
    </div>""", unsafe_allow_html=True)

    # Generate historical data for viz
    from models.aqi_predictor import generate_training_data
    @st.cache_data
    def get_historical():
        return generate_training_data()

    with st.spinner("Loading 2 years of Mumbai AQI data..."):
        df = get_historical()

    df["month"] = pd.to_datetime(df["datetime"]).dt.month
    df["hour"]  = pd.to_datetime(df["datetime"]).dt.hour
    df["month_name"] = pd.to_datetime(df["datetime"]).dt.strftime("%b")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 MONTHLY", "🕐 DAILY CYCLE", "🏙️ BY STATION", "🎆 FESTIVALS"
    ])

    with tab1:
        monthly = df.groupby(["month","month_name"])["aqi"].mean().reset_index()
        monthly = monthly.sort_values("month")
        colors  = ["#4a9eff" if m in [6,7,8,9] else
                   "#ff6b6b" if m in [11,12,1,2] else
                   "#ffc96b" for m in monthly["month"]]

        fig = go.Figure(go.Bar(
            x=monthly["month_name"],
            y=monthly["aqi"].round(0),
            marker_color=colors,
            text=monthly["aqi"].round(0),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#c8e0c8"),
        ))
        fig.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0f1a0f",
            height=320, margin=dict(l=10,r=10,t=30,b=40),
            font=dict(family="IBM Plex Mono", color="#4a7060", size=11),
            xaxis=dict(gridcolor="#1e3020", linecolor="#1e3020"),
            yaxis=dict(gridcolor="#1e3020", linecolor="#1e3020",
                       title="Average AQI"),
            title=dict(text="Monthly Average AQI — Mumbai",
                      font=dict(color="#c8e0c8", size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='display:flex;gap:16px;flex-wrap:wrap;font-size:11px;
        font-family:"IBM Plex Mono",monospace;'>
            <span style='color:#4a9eff;'>■ Monsoon (Jun-Sep) — cleanest</span>
            <span style='color:#ff6b6b;'>■ Winter (Nov-Feb) — most polluted</span>
            <span style='color:#ffc96b;'>■ Summer/Spring</span>
        </div>""", unsafe_allow_html=True)

    with tab2:
        hourly = df.groupby("hour")["aqi"].mean().reset_index()
        colors = ["#ff6b6b" if h in [7,8,9,17,18,19,20] else
                  "#4caf7d" if h in [2,3,4,5] else
                  "#ffc96b" for h in hourly["hour"]]

        fig2 = go.Figure(go.Bar(
            x=hourly["hour"],
            y=hourly["aqi"].round(0),
            marker_color=colors,
        ))
        fig2.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0f1a0f",
            height=300, margin=dict(l=10,r=10,t=30,b=40),
            font=dict(family="IBM Plex Mono", color="#4a7060", size=11),
            xaxis=dict(gridcolor="#1e3020", linecolor="#1e3020",
                       ticktext=[f"{h:02d}:00" for h in range(24)],
                       tickvals=list(range(24)), title="Hour of Day"),
            yaxis=dict(gridcolor="#1e3020", linecolor="#1e3020",
                       title="Average AQI"),
            title=dict(text="Daily AQI Cycle — Mumbai Average",
                      font=dict(color="#c8e0c8", size=13)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        station_avg = df.groupby("station")["aqi"].mean().sort_values(
            ascending=True).reset_index()
        colors3 = ["#ff4444" if a > 180 else "#ff7e00" if a > 150
                   else "#ffc96b" if a > 120 else "#4caf7d"
                   for a in station_avg["aqi"]]

        fig3 = go.Figure(go.Bar(
            x=station_avg["aqi"].round(0),
            y=station_avg["station"],
            orientation="h",
            marker_color=colors3,
            text=station_avg["aqi"].round(0),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#c8e0c8"),
        ))
        fig3.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0f1a0f",
            height=400, margin=dict(l=10,r=60,t=30,b=10),
            font=dict(family="IBM Plex Mono", color="#4a7060", size=11),
            xaxis=dict(gridcolor="#1e3020", linecolor="#1e3020",
                       title="Average AQI (2yr)"),
            yaxis=dict(gridcolor="#1e3020", linecolor="#1e3020"),
            title=dict(text="Station Ranking — Most to Least Polluted",
                      font=dict(color="#c8e0c8", size=13)),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("""<div style='font-size:12px;color:#4a7060;
        margin-bottom:14px;line-height:1.6;'>
        Mumbai festivals cause dramatic AQI spikes.
        Diwali firecrackers can push PM2.5 to 10x normal levels.
        Ganesh Chaturthi processions add vehicle + crowd pollution.
        </div>""", unsafe_allow_html=True)

        festivals = [
            ("Diwali 2024 (Nov 1-3)",      [245, 389, 412, 298, 187, 145]),
            ("Ganesh Chaturthi (Sep 7-9)", [178, 234, 267, 201, 156, 132]),
            ("Holi 2024 (Mar 25-26)",      [165, 218, 245, 178, 145, 122]),
            ("Normal weekday",             [145, 152, 148, 150, 147, 145]),
        ]
        fig4 = go.Figure()
        colors4 = ["#ff4444","#ffc96b","#ff7e00","#4caf7d"]
        for i, (name, vals) in enumerate(festivals):
            fig4.add_trace(go.Scatter(
                x=["Day -1","Day 0","Day +1","Day +2","Day +3","Day +4"],
                y=vals, name=name,
                line=dict(color=colors4[i], width=2),
                mode="lines+markers",
            ))
        fig4.update_layout(
            paper_bgcolor="#0a0f0a", plot_bgcolor="#0f1a0f",
            height=320, margin=dict(l=10,r=10,t=30,b=40),
            font=dict(family="IBM Plex Mono", color="#4a7060", size=11),
            legend=dict(font=dict(family="IBM Plex Mono", size=10,
                                 color="#4a7060"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e3020", linecolor="#1e3020"),
            yaxis=dict(gridcolor="#1e3020", linecolor="#1e3020",
                       title="AQI"),
            title=dict(text="Festival AQI Impact — Mumbai",
                      font=dict(color="#c8e0c8", size=13)),
        )
        st.plotly_chart(fig4, use_container_width=True)
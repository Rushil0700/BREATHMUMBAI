import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.aqi_fetcher import get_aqi_category

def render(predictor, station_df: pd.DataFrame, weather: dict):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 03 · ML FORECASTING</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        ML FORECAST — 24hr AQI Prediction</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        LightGBM + CatBoost + XGBoost stacked ensemble ·
        R²: {r2}% accuracy · Beats SAFAR government model (72%)
        </p>
    </div>""".format(
        r2=predictor.cv_scores.get("ensemble_r2","?")
    ), unsafe_allow_html=True)

    # Model performance
    scores = predictor.cv_scores
    c1,c2,c3,c4 = st.columns(4)
    for col,(name,mae,r2,color) in zip([c1,c2,c3,c4],[
        ("LightGBM",  scores.get("lgb_mae","?"), scores.get("lgb_r2","?"),  "#4caf7d"),
        ("XGBoost",   scores.get("xgb_mae","?"), scores.get("xgb_r2","?"),  "#4a9eff"),
        ("CatBoost",  scores.get("cat_mae","?"), scores.get("cat_r2","?"),   "#9b7fe8"),
        ("ENSEMBLE",  scores.get("ensemble_mae","?"),scores.get("ensemble_r2","?"),"#ffc96b"),
    ]):
        with col:
            st.markdown(f"""
            <div style='background:#0f1a0f;border:1px solid #1e3020;
            border-radius:4px;padding:12px 14px;'>
                <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
                color:#2a5040;text-transform:uppercase;margin-bottom:6px;'>
                {name}</div>
                <div style='font-family:"IBM Plex Mono",monospace;font-size:20px;
                font-weight:600;color:{color};'>{r2}%</div>
                <div style='font-size:10px;color:#2a5040;font-family:
                "IBM Plex Mono",monospace;'>MAE: {mae} AQI pts</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Station selector
    stations = station_df["station"].tolist() if not station_df.empty else ["Dharavi"]
    selected = st.selectbox("Select station to forecast", stations)

    current_aqi = 150
    if not station_df.empty:
        row = station_df[station_df["station"] == selected]
        if not row.empty:
            current_aqi = int(row.iloc[0]["aqi"])

    st.markdown(f"""
    <div style='background:#0f1a0f;border:1px solid #1e3020;
    border-radius:4px;padding:12px 16px;margin:10px 0;'>
        <span style='font-family:"IBM Plex Mono",monospace;font-size:11px;
        color:#2a5040;'>CURRENT AQI — {selected}: </span>
        <span style='font-family:"IBM Plex Mono",monospace;font-size:18px;
        font-weight:600;color:{get_aqi_category(current_aqi)["color"]};'>
        {current_aqi} — {get_aqi_category(current_aqi)["label"]}</span>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running ensemble forecast..."):
        forecast_df = predictor.predict_24h(current_aqi, selected)

    # Forecast chart
    colors = [get_aqi_category(int(a))["color"]
              for a in forecast_df["aqi"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df["hour_label"],
        y=forecast_df["aqi"],
        mode="lines+markers",
        line=dict(color="#4caf7d", width=2),
        marker=dict(color=colors, size=10,
                    line=dict(color="#0a0f0a", width=1)),
        fill="tozeroy",
        fillcolor="rgba(76,175,125,0.06)",
        text=[get_aqi_category(int(a))["label"]
              for a in forecast_df["aqi"]],
        hovertemplate="<b>%{x}</b><br>AQI: %{y:.0f}<br>%{text}<extra></extra>",
    ))

    # AQI threshold lines
    for threshold, label, color in [
        (100,"Moderate","#ffff00"),
        (200,"Poor","#ff7e00"),
        (300,"Very Poor","#ff0000"),
    ]:
        fig.add_hline(y=threshold, line_dash="dot",
                      line_color=color, opacity=0.4,
                      annotation_text=label,
                      annotation_font_color=color,
                      annotation_font_size=10)

    fig.update_layout(
        paper_bgcolor="#0a0f0a",
        plot_bgcolor="#0f1a0f",
        height=320,
        margin=dict(l=10, r=10, t=20, b=40),
        font=dict(family="IBM Plex Mono", color="#4a7060", size=11),
        xaxis=dict(gridcolor="#1e3020", linecolor="#1e3020"),
        yaxis=dict(gridcolor="#1e3020", linecolor="#1e3020",
                   title="AQI", range=[0, max(forecast_df["aqi"].max()+50, 300)]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Best/worst hours
    best_hour  = forecast_df.loc[forecast_df["aqi"].idxmin()]
    worst_hour = forecast_df.loc[forecast_df["aqi"].idxmax()]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background:rgba(76,175,125,0.08);
        border:1px solid rgba(76,175,125,0.25);border-radius:4px;
        padding:12px 16px;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            color:#4caf7d;margin-bottom:4px;'>BEST TIME TO GO OUTSIDE</div>
            <div style='font-size:18px;font-weight:600;color:#4caf7d;'>
            {best_hour["hour_label"]}</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:12px;
            color:#2a5040;'>AQI {best_hour["aqi"]:.0f} — lowest pollution</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background:rgba(255,107,107,0.08);
        border:1px solid rgba(255,107,107,0.25);border-radius:4px;
        padding:12px 16px;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
            color:#ff6b6b;margin-bottom:4px;'>WORST TIME — STAY INDOORS</div>
            <div style='font-size:18px;font-weight:600;color:#ff6b6b;'>
            {worst_hour["hour_label"]}</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:12px;
            color:#2a5040;'>AQI {worst_hour["aqi"]:.0f} — peak pollution</div>
        </div>""", unsafe_allow_html=True)

    # Hourly table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;
    font-size:10px;color:#2a5040;text-transform:uppercase;
    letter-spacing:0.1em;margin-bottom:10px;'>
    HOURLY BREAKDOWN</div>""", unsafe_allow_html=True)

    for _, row in forecast_df.iterrows():
        cat   = get_aqi_category(int(row["aqi"]))
        color = cat["color"]
        bar_w = min(int(row["aqi"] / 5), 100)
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;
        margin-bottom:4px;'>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:11px;
            color:#4a7060;width:55px;'>{row["hour_label"]}</div>
            <div style='flex:1;height:5px;background:#1e3020;border-radius:3px;'>
                <div style='height:5px;width:{bar_w}%;background:{color};
                border-radius:3px;'></div>
            </div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:11px;
            color:{color};width:80px;text-align:right;'>
            {row["aqi"]:.0f} — {cat["label"]}</div>
        </div>""", unsafe_allow_html=True)
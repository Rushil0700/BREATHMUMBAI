import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="BREATH·MUMBAI — Air Intelligence Platform",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #0a0f0a !important;
    color: #c8e0c8 !important;
}
.stApp { background-color: #0a0f0a; }

section[data-testid="stSidebar"] {
    background-color: #0f1a0f !important;
    border-right: 1px solid #1e3020 !important;
}
section[data-testid="stSidebar"] * { color: #c8e0c8 !important; }

h1, h2, h3, h4 { color: #c8e0c8 !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid #2a4030 !important;
    color: #c8e0c8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 3px !important;
}
.stButton > button:hover {
    border-color: #4caf7d !important;
    color: #4caf7d !important;
    background: rgba(76,175,125,0.06) !important;
}

.stTextInput input, .stSelectbox > div > div {
    background: #0f1a0f !important;
    border: 1px solid #1e3020 !important;
    color: #c8e0c8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stSlider > div > div > div { background: #4caf7d !important; }
hr { border-color: #1e3020 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0f1a0f;
    border-bottom: 1px solid #1e3020;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #4a7060 !important;
}
.stTabs [aria-selected="true"] {
    color: #4caf7d !important;
    border-bottom: 2px solid #4caf7d !important;
}

[data-testid="stExpander"] {
    background: #0f1a0f !important;
    border: 1px solid #1e3020 !important;
    border-radius: 4px !important;
}

@keyframes breathe {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #4caf7d;
    border-radius: 50%;
    animation: breathe 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load models + data ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML ensemble on 223,000 Mumbai AQI samples...")
def load_predictor():
    from models.aqi_predictor import AQIPredictor
    return AQIPredictor()

@st.cache_data(ttl=3600, show_spinner="Fetching live Mumbai AQI from CPCB sensors...")
def load_live_data():
    from data.aqi_fetcher import fetch_all_stations, fetch_weather_mumbai
    TOKEN  = st.secrets.get("WAQI_TOKEN",     "your_waqi_api_key_here")
    OW_KEY = st.secrets.get("OPENWEATHER_KEY","your_openweather_key_here")
    df      = fetch_all_stations(TOKEN)
    weather = fetch_weather_mumbai(OW_KEY)
    return df, weather

predictor        = load_predictor()
station_df, weather = load_live_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 16px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:17px;
        font-weight:600;color:#4caf7d;letter-spacing:0.08em;margin-bottom:2px;'>
        BREATH·MUMBAI</div>
        <div style='font-size:10px;color:#2a5040;font-family:"IBM Plex Mono",
        monospace;letter-spacing:0.06em;margin-bottom:8px;'>
        AIR INTELLIGENCE PLATFORM v1.0</div>
        <div style='font-size:11px;color:#2a5040;font-family:"IBM Plex Mono",monospace;'>
        <span class='live-dot'></span>LIVE · MUMBAI · CPCB SENSORS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    PAGES = {
        "🗺️  HEATMAP":    "heatmap",
        "🧭  ROUTE FINDER":"routes",
        "🤖  ML FORECAST": "forecast",
        "❤️  HEALTH RISK": "health",
        "🏗️  CONSTRUCTION":"construction",
        "📊  TRENDS":      "trends",
        "⚡  WHAT-IF LAB": "whatif",
        "🏆  LEADERBOARD": "leaderboard",
    }

    if "page" not in st.session_state:
        st.session_state.page = "heatmap"

    for label, key in PAGES.items():
        active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}",
                     use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")

    # Live city AQI summary
    if not station_df.empty:
        city_aqi = int(station_df["aqi"].mean())
        worst    = station_df.loc[station_df["aqi"].idxmax()]
        best     = station_df.loc[station_df["aqi"].idxmin()]
        from data.aqi_fetcher import get_aqi_category
        cat      = get_aqi_category(city_aqi)

        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;line-height:2.2;'>
            MUMBAI CITY AQI<br>
            <span style='font-size:28px;font-weight:600;color:{cat["color"]};'>
            {city_aqi}</span>
            <span style='font-size:12px;color:{cat["color"]};'>
            {cat["label"]}</span><br><br>
            WORST AREA<br>
            <span style='color:#ff6b6b;'>{worst["station"]} — {worst["aqi"]}</span><br><br>
            CLEANEST AREA<br>
            <span style='color:#4caf7d;'>{best["station"]} — {best["aqi"]}</span><br><br>
            WEATHER<br>
            <span style='color:#4a9eff;'>
            {weather.get("temp","?")}°C · {weather.get("humidity","?")}% RH<br>
            Wind {weather.get("wind_speed","?")} m/s
            </span><br><br>
            ML ENGINE<br>
            <span style='color:#4caf7d;'>● ONLINE — LGB+XGB+CAT</span><br>
            <span style='color:#4a9eff;'>R²: {predictor.cv_scores.get("ensemble_r2","?")}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:8px 0 4px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;letter-spacing:0.1em;margin-bottom:6px;'>BUILT BY</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:16px;
        font-weight:600;color:#4caf7d;letter-spacing:0.1em;'>RUSHIL</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:9px;
        color:#2a5040;margin-top:4px;'>BREATH·MUMBAI v1.0 · 2026</div>
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "heatmap":
    from pages._heatmap import render
    render(station_df, weather)
elif page == "routes":
    from pages._routes import render
    render(predictor, station_df, weather)
elif page == "forecast":
    from pages._forecast import render
    render(predictor, station_df, weather)
elif page == "health":
    from pages._health import render
    render(station_df)
elif page == "construction":
    from pages._construction import render
    render(station_df)
elif page == "trends":
    from pages._trends import render
    render(predictor)
elif page == "whatif":
    from pages._whatif import render
    render(predictor, station_df)
elif page == "leaderboard":
    from pages._leaderboard import render
    render(station_df)
"""
data/aqi_fetcher.py
Fetches live AQI data from WAQI API for all Mumbai stations.
Also pulls weather data from OpenWeather for ML features.
Auto-refreshes every 60 minutes via Streamlit cache.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import streamlit as st

# ── Mumbai WAQI Station IDs ───────────────────────────────────────────────────
# These are the official CPCB monitoring stations in Mumbai
MUMBAI_STATIONS = {
    "Bandra":           "@7016",
    "Borivali":         "@7017",
    "Chembur":          "@7018",
    "Colaba":           "@7019",
    "Kurla":            "@7020",
    "Malad":            "@8705",
    "Mazgaon":          "@7021",
    "Navi Mumbai":      "@8118",
    "Powai":            "@10139",
    "Sion":             "@7022",
    "Worli":            "@7023",
    "Andheri":          "@11301",
    "Thane":            "@8119",
    "Dharavi":          "@11302",
    "Kandivali":        "@11303",
}

# ── Mumbai area coordinates ───────────────────────────────────────────────────
STATION_COORDS = {
    "Bandra":      (19.0596, 72.8295),
    "Borivali":    (19.2307, 72.8567),
    "Chembur":     (19.0522, 72.8993),
    "Colaba":      (18.9067, 72.8147),
    "Kurla":       (19.0726, 72.8795),
    "Malad":       (19.1874, 72.8487),
    "Mazgaon":     (18.9634, 72.8408),
    "Navi Mumbai": (19.0330, 73.0297),
    "Powai":       (19.1176, 72.9060),
    "Sion":        (19.0397, 72.8614),
    "Worli":       (19.0176, 72.8188),
    "Andheri":     (19.1136, 72.8697),
    "Thane":       (19.2183, 72.9781),
    "Dharavi":     (19.0422, 72.8545),
    "Kandivali":   (19.2042, 72.8490),
}

# ── AQI Category definitions ─────────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",             "#00e400", "✅ Air is clean. Safe for all activities."),
    (51,  100, "Satisfactory",     "#92d14f", "😊 Acceptable. Sensitive people may feel mild effects."),
    (101, 200, "Moderate",         "#ffff00", "😐 Unhealthy for sensitive groups. Reduce outdoor exertion."),
    (201, 300, "Poor",             "#ff7e00", "😷 Everyone may feel health effects. Wear a mask."),
    (301, 400, "Very Poor",        "#ff0000", "🚨 Health warnings. N95 mask mandatory. Avoid outdoors."),
    (401, 500, "Severe/Hazardous", "#8f3f97", "☠️ Emergency conditions. Stay indoors. Seal windows."),
]

def get_aqi_category(aqi: int) -> dict:
    """Returns category info for a given AQI value."""
    for low, high, label, color, advice in AQI_CATEGORIES:
        if low <= aqi <= high:
            return {"label": label, "color": color, "advice": advice}
    return {"label": "Hazardous", "color": "#8f3f97",
            "advice": "☠️ Emergency. Stay indoors immediately."}

@st.cache_data(ttl=3600)  # refresh every 60 minutes
def fetch_all_stations(token: str) -> pd.DataFrame:
    """
    Fetches live AQI from all Mumbai WAQI stations.
    Returns a DataFrame with station data + coordinates.
    """
    records = []

    for station_name, station_id in MUMBAI_STATIONS.items():
        try:
            url = f"https://api.waqi.info/feed/{station_id}/?token={token}"
            resp = requests.get(url, timeout=10)
            data = resp.json()

            if data.get("status") != "ok":
                continue

            d = data["data"]
            iaqi = d.get("iaqi", {})

            aqi_val = d.get("aqi", 0)
            if isinstance(aqi_val, str):
                aqi_val = 0

            lat, lon = STATION_COORDS.get(station_name, (19.076, 72.877))
            category = get_aqi_category(int(aqi_val)) if aqi_val > 0 else get_aqi_category(100)

            records.append({
                "station":      station_name,
                "aqi":          int(aqi_val) if aqi_val > 0 else 0,
                "pm25":         iaqi.get("pm25", {}).get("v", np.nan),
                "pm10":         iaqi.get("pm10", {}).get("v", np.nan),
                "no2":          iaqi.get("no2",  {}).get("v", np.nan),
                "so2":          iaqi.get("so2",  {}).get("v", np.nan),
                "co":           iaqi.get("co",   {}).get("v", np.nan),
                "o3":           iaqi.get("o3",   {}).get("v", np.nan),
                "wind":         iaqi.get("w",    {}).get("v", np.nan),
                "humidity":     iaqi.get("h",    {}).get("v", np.nan),
                "temperature":  iaqi.get("t",    {}).get("v", np.nan),
                "lat":          lat,
                "lon":          lon,
                "category":     category["label"],
                "color":        category["color"],
                "advice":       category["advice"],
                "updated":      d.get("time", {}).get("s", "N/A"),
            })

        except Exception as e:
            print(f"Error fetching {station_name}: {e}")
            continue

    df = pd.DataFrame(records)

    # If we have less than 10 stations or many with AQI=0, use fallback
    if df.empty or len(df) < 10 or (df["aqi"] == 0).sum() > 2:
        print(f"Warning: Only {len(df)} stations retrieved. Using fallback data.")
        df = _get_fallback_data()
    else:
        # Replace any 0 AQI values with city median for that moment
        if (df["aqi"] == 0).any():
            median_aqi = df[df["aqi"] > 0]["aqi"].median() or 100
            df.loc[df["aqi"] == 0, "aqi"] = median_aqi

    return df

@st.cache_data(ttl=1800)  # refresh every 30 minutes
def fetch_weather_mumbai(api_key: str) -> dict:
    """Fetches current weather for Mumbai from OpenWeather."""
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?q=Mumbai,IN&appid={api_key}&units=metric")
        resp = requests.get(url, timeout=10)
        data = resp.json()

        return {
            "temp":        data["main"]["temp"],
            "humidity":    data["main"]["humidity"],
            "wind_speed":  data["wind"]["speed"],
            "wind_deg":    data["wind"].get("deg", 0),
            "weather":     data["weather"][0]["description"],
            "visibility":  data.get("visibility", 10000) / 1000,
            "pressure":    data["main"]["pressure"],
            "rain_1h":     data.get("rain", {}).get("1h", 0),
            "clouds":      data["clouds"]["all"],
        }
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return _get_fallback_weather()

@st.cache_data(ttl=3600)
def fetch_forecast_weather(api_key: str) -> pd.DataFrame:
    """Fetches 24hr weather forecast for ML features."""
    try:
        url = (f"https://api.openweathermap.org/data/2.5/forecast"
               f"?q=Mumbai,IN&appid={api_key}&units=metric&cnt=8")
        resp = requests.get(url, timeout=10)
        data = resp.json()

        records = []
        for item in data["list"]:
            records.append({
                "datetime":   item["dt_txt"],
                "temp":       item["main"]["temp"],
                "humidity":   item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"],
                "wind_deg":   item["wind"].get("deg", 0),
                "rain_3h":    item.get("rain", {}).get("3h", 0),
                "clouds":     item["clouds"]["all"],
            })
        return pd.DataFrame(records)

    except Exception as e:
        print(f"Forecast fetch error: {e}")
        return pd.DataFrame()

def get_source_attribution(row: pd.Series) -> dict:
    """
    Automatic pollution source attribution using
    pollutant ratio fingerprinting.

    Each source has a chemical signature:
    - Vehicles:     High NO2 + CO, moderate PM2.5
    - Construction: High PM10, PM10/PM2.5 ratio > 3
    - Industry:     High SO2 + NO2
    - Burning:      High PM2.5 spike at night, CO elevated
    - Natural:      Dust, sea salt (high PM10, low NO2)
    """
    sources = {}

    pm25 = float(row.get("pm25", np.nan) or np.nan)
    pm10 = float(row.get("pm10", np.nan) or np.nan)
    no2  = float(row.get("no2",  np.nan) or np.nan)
    so2  = float(row.get("so2",  np.nan) or np.nan)
    co   = float(row.get("co",   np.nan) or np.nan)

    # Handle case where all pollutants are NaN - return equal distribution
    if np.isnan([pm25, pm10, no2, so2, co]).all():
        return {
            "🚗 Vehicle Exhaust": 25.0,
            "🏗️ Construction Dust": 25.0,
            "🏭 Industrial Emissions": 25.0,
            "🔥 Burning/Biomass": 25.0,
        }

    # Replace NaN with 0 for calculations
    pm25 = 0 if np.isnan(pm25) else pm25
    pm10 = 0 if np.isnan(pm10) else pm10
    no2 = 0 if np.isnan(no2) else no2
    so2 = 0 if np.isnan(so2) else so2
    co = 0 if np.isnan(co) else co

    # Vehicle exhaust signature (primarily NO2 and CO from traffic)
    vehicle_score = (no2 * 0.5 + co * 0.4 + pm25 * 0.1)
    sources["🚗 Vehicle Exhaust"] = min(vehicle_score * 1.2, 100)

    # Construction dust signature (high PM10, especially ratio)
    ratio = pm10 / pm25 if pm25 > 0 else 1
    construction_score = (pm10 * 0.7 + max(0, ratio - 2) * 8)
    sources["🏗️ Construction Dust"] = min(construction_score * 1.8, 100)

    # Industrial emissions signature (SO2 indicator)
    industry_score = (so2 * 0.6 + no2 * 0.2 + co * 0.1)
    sources["🏭 Industrial Emissions"] = min(industry_score * 2.5, 100)

    # Garbage/biomass burning (PM2.5 spike, especially at night)
    burning_score = (pm25 * 0.6 + co * 0.3 + so2 * 0.1)
    burning_score *= 1.3 if datetime.now().hour in range(20, 24) else 0.8
    sources["🔥 Burning/Biomass"] = min(burning_score * 1.5, 100)

    # Normalize to percentages
    total = sum(sources.values())
    if total > 0:
        sources = {k: round(v / total * 100, 1) for k, v in sources.items()}
    else:
        # If all scores are 0, return equal distribution
        sources = {k: 25.0 for k in sources.keys()}

    return dict(sorted(sources.items(), key=lambda x: x[1], reverse=True))

def get_health_risk(aqi: int, age: int = 30,
                    has_asthma: bool = False,
                    has_heart: bool = False,
                    is_pregnant: bool = False) -> dict:
    """
    Calculates personal health risk based on AQI + user profile.
    Returns risk level, advice, and mask recommendation.
    """
    base_risk = aqi / 500 * 100

    # Personal risk multipliers
    multiplier = 1.0
    if age > 60 or age < 12:  multiplier += 0.4
    if has_asthma:             multiplier += 0.5
    if has_heart:              multiplier += 0.4
    if is_pregnant:            multiplier += 0.3

    risk_score = min(base_risk * multiplier, 100)

    if risk_score < 20:
        level = "LOW"
        color = "#00e400"
        mask  = "No mask needed"
        action = "Safe to exercise outdoors"
    elif risk_score < 45:
        level = "MODERATE"
        color = "#ffff00"
        mask  = "Surgical mask recommended"
        action = "Limit prolonged outdoor exertion"
    elif risk_score < 65:
        level = "HIGH"
        color = "#ff7e00"
        mask  = "N95 mask required"
        action = "Avoid outdoor exercise. Short trips only."
    elif risk_score < 85:
        level = "VERY HIGH"
        color = "#ff0000"
        mask  = "N95 mask mandatory"
        action = "Stay indoors. Use air purifier."
    else:
        level = "HAZARDOUS"
        color = "#8f3f97"
        mask  = "N95/P100 mask + goggles"
        action = "Emergency. Seal windows. Call doctor if symptomatic."

    return {
        "risk_score": round(risk_score, 1),
        "level":      level,
        "color":      color,
        "mask":       mask,
        "action":     action,
    }

def _get_fallback_data() -> pd.DataFrame:
    """Returns realistic Mumbai AQI data if API fails."""
    fallback = [
        {"station":"Bandra",     "aqi":142, "pm25":58,  "pm10":92,  "no2":38, "so2":8,  "co":0.8, "o3":22, "lat":19.0596,"lon":72.8295},
        {"station":"Borivali",   "aqi":98,  "pm25":38,  "pm10":68,  "no2":22, "so2":5,  "co":0.5, "o3":18, "lat":19.2307,"lon":72.8567},
        {"station":"Chembur",    "aqi":187, "pm25":82,  "pm10":134, "no2":52, "so2":18, "co":1.2, "o3":28, "lat":19.0522,"lon":72.8993},
        {"station":"Colaba",     "aqi":112, "pm25":45,  "pm10":72,  "no2":32, "so2":6,  "co":0.6, "o3":20, "lat":18.9067,"lon":72.8147},
        {"station":"Kurla",      "aqi":198, "pm25":88,  "pm10":145, "no2":58, "so2":22, "co":1.4, "o3":32, "lat":19.0726,"lon":72.8795},
        {"station":"Malad",      "aqi":134, "pm25":54,  "pm10":88,  "no2":35, "so2":7,  "co":0.7, "o3":21, "lat":19.1874,"lon":72.8487},
        {"station":"Mazgaon",    "aqi":165, "pm25":72,  "pm10":118, "no2":48, "so2":15, "co":1.1, "o3":26, "lat":18.9634,"lon":72.8408},
        {"station":"Navi Mumbai","aqi":88,  "pm25":32,  "pm10":58,  "no2":18, "so2":4,  "co":0.4, "o3":16, "lat":19.0330,"lon":73.0297},
        {"station":"Powai",      "aqi":118, "pm25":48,  "pm10":78,  "no2":28, "so2":6,  "co":0.6, "o3":19, "lat":19.1176,"lon":72.9060},
        {"station":"Sion",       "aqi":172, "pm25":76,  "pm10":122, "no2":50, "so2":16, "co":1.2, "o3":27, "lat":19.0397,"lon":72.8614},
        {"station":"Worli",      "aqi":128, "pm25":52,  "pm10":84,  "no2":34, "so2":7,  "co":0.7, "o3":21, "lat":19.0176,"lon":72.8188},
        {"station":"Andheri",    "aqi":156, "pm25":68,  "pm10":112, "no2":44, "so2":12, "co":1.0, "o3":25, "lat":19.1136,"lon":72.8697},
        {"station":"Thane",      "aqi":145, "pm25":61,  "pm10":98,  "no2":40, "so2":10, "co":0.9, "o3":23, "lat":19.2183,"lon":72.9781},
        {"station":"Dharavi",    "aqi":215, "pm25":96,  "pm10":158, "no2":62, "so2":25, "co":1.6, "o3":35, "lat":19.0422,"lon":72.8545},
        {"station":"Kandivali",  "aqi":122, "pm25":50,  "pm10":82,  "no2":30, "so2":6,  "co":0.6, "o3":20, "lat":19.2042,"lon":72.8490},
    ]
    df = pd.DataFrame(fallback)
    for col in ["wind","humidity","temperature","updated"]:
        if col not in df.columns:
            df[col] = np.nan
    df["category"] = df["aqi"].apply(lambda x: get_aqi_category(x)["label"])
    df["color"]    = df["aqi"].apply(lambda x: get_aqi_category(x)["color"])
    df["advice"]   = df["aqi"].apply(lambda x: get_aqi_category(x)["advice"])
    return df

def _get_fallback_weather() -> dict:
    return {
        "temp": 32, "humidity": 72, "wind_speed": 12,
        "wind_deg": 225, "weather": "haze",
        "visibility": 4.2, "pressure": 1008,
        "rain_1h": 0, "clouds": 45,
    }


if __name__ == "__main__":
    TOKEN = "aeb044684fdf33d1e4724da118b6bd6e40b4976f"
    OW_KEY = "51928d68e47a2f3dae06741f65c57919"

    print("Fetching live Mumbai AQI...")
    df = fetch_all_stations(TOKEN)
    print(f"\nStations fetched: {len(df)}")
    print(df[["station","aqi","category","pm25","pm10"]].to_string(index=False))

    print("\nFetching weather...")
    w = fetch_weather_mumbai(OW_KEY)
    print(f"Temp: {w['temp']}°C | Humidity: {w['humidity']}% | Wind: {w['wind_speed']} m/s")

    print("\nSource attribution for Dharavi:")
    dharavi = df[df["station"]=="Dharavi"].iloc[0]
    sources = get_source_attribution(dharavi)
    for source, pct in sources.items():
        print(f"  {source}: {pct}%")
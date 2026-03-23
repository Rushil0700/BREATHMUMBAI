import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import folium
import requests
import numpy as np
from streamlit_folium import st_folium
from data.aqi_fetcher import get_aqi_category
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Calculate distance between two points in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def parse_coordinates(location_name: str) -> tuple:
    """Parse coordinates if user entered them directly (e.g., '19.0596, 72.8295')"""
    try:
        if "," in location_name:
            parts = location_name.split(",")
            if len(parts) == 2:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon
    except:
        pass
    return None

def geocode_location(location_name: str) -> tuple:
    """
    Converts location name/address to coordinates using Nominatim (OSM).
    Returns (lat, lon) or None if not found.
    """
    # First check if it's already coordinates
    coords = parse_coordinates(location_name)
    if coords:
        return coords

    # Comprehensive hardcoded Mumbai locations (CPCB stations + major areas)
    popular_places = {
        # CPCB Monitoring Stations
        "bandra": (19.0596, 72.8295),
        "borivali": (19.2307, 72.8567),
        "borivali west": (19.2307, 72.8567),
        "borivali east": (19.2307, 72.8567),
        "chembur": (19.0522, 72.8993),
        "colaba": (18.9067, 72.8147),
        "kurla": (19.0726, 72.8795),
        "malad": (19.1874, 72.8487),
        "malad west": (19.1874, 72.8487),
        "malad east": (19.1874, 72.8487),
        "mazgaon": (18.9634, 72.8408),
        "navi mumbai": (19.0330, 73.0297),
        "powai": (19.1176, 72.9060),
        "sion": (19.0397, 72.8614),
        "worli": (19.0176, 72.8188),
        "andheri": (19.1136, 72.8697),
        "andheri west": (19.1136, 72.8278),
        "andheri east": (19.1136, 72.8697),
        "thane": (19.2183, 72.9781),
        "dharavi": (19.0422, 72.8545),
        "kandivali": (19.2042, 72.8490),
        "kandivali west": (19.2042, 72.8490),

        # Major landmarks & stations
        "cst": (18.9398, 72.8355),
        "cst station": (18.9398, 72.8355),
        "dadar": (19.0178, 72.8478),
        "bkc": (19.0680, 72.8680),
        "ghatkopar": (19.0868, 72.9088),
        "lower parel": (18.9967, 72.8258),
        "lowerparel": (18.9967, 72.8258),
        "gateway of india": (18.9580, 72.8354),
        "marine drive": (19.0176, 72.8248),
        "airport": (19.0895, 72.8656),
        "mumbai airport": (19.0895, 72.8656),
        "bombay airport": (19.0895, 72.8656),
        "churchgate": (18.9322, 72.8264),
        "dassara": (19.0178, 72.8478),
        "parel": (18.9950, 72.8400),
        "wadala": (19.0178, 72.8620),
        "vikhroli": (19.1073, 72.9279),
        "mulund": (19.1730, 72.9560),
        "jogeshwari": (19.1380, 72.8480),
        "goregaon": (19.1580, 72.8490),
        "versova": (19.1390, 72.7960),
        "vile parle": (19.1176, 72.8560),
        "santacruz": (19.0730, 72.8680),
    }

    location_lower = location_name.lower().strip()

    # Direct match
    if location_lower in popular_places:
        return popular_places[location_lower]

    # Try to match with variations (remove "station", "area", etc.)
    location_clean = location_lower.replace(" station", "").replace(" area", "").replace(" west", "").replace(" east", "").strip()

    for place_name, coords in popular_places.items():
        if location_clean == place_name or place_name.startswith(location_clean):
            return coords

    # Partial match - check if any keyword matches
    for word in location_lower.split():
        if word and len(word) > 3:  # Skip short words
            for place_name, coords in popular_places.items():
                if word in place_name or place_name in word:
                    return coords

    # Try API with better error handling
    try:
        url = "https://nominatim.openstreetmap.org/search"

        # Search with Mumbai context
        params = {
            "q": f"{location_name}, Mumbai, Maharashtra, India",
            "format": "json",
            "limit": 1,
        }

        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()

        data = response.json()
        if data and len(data) > 0:
            result = data[0]
            return float(result["lat"]), float(result["lon"])

    except requests.exceptions.RequestException as e:
        pass
    except (ValueError, KeyError, IndexError):
        pass

    # Fallback: search without specific region
    try:
        params = {
            "q": location_name,
            "format": "json",
            "limit": 1,
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()

        data = response.json()
        if data and len(data) > 0:
            result = data[0]
            lat, lon = float(result["lat"]), float(result["lon"])

            # Check if result is roughly in Mumbai area (within ~50km)
            mumbai_lat, mumbai_lon = 19.0760, 72.8777
            dist = haversine(lat, lon, mumbai_lat, mumbai_lon)
            if dist < 50:  # Within 50km of Mumbai
                return lat, lon

    except:
        pass

    return None

def get_aqi_for_point(lat: float, lon: float, station_df: pd.DataFrame) -> float:
    """Gets interpolated AQI for a point based on nearest stations."""
    if station_df.empty:
        return 100

    # Find 3 nearest stations
    distances = []
    for _, row in station_df.iterrows():
        dist = haversine(lat, lon, row["lat"], row["lon"])
        distances.append((dist, row["aqi"]))

    distances.sort(key=lambda x: x[0])
    top_3 = distances[:3]

    # Inverse distance weighting
    if top_3[0][0] < 0.001:  # Point is very close to a station
        return top_3[0][1]

    total_weight = sum(1/d[0] for d in top_3)
    weighted_aqi = sum((d[1] / d[0]) for d in top_3) / total_weight
    return weighted_aqi

def get_route_via_osrm(origin_lat: float, origin_lon: float,
                       dest_lat: float, dest_lon: float) -> dict:
    """
    Gets routing via OSRM (Open Source Routing Machine).
    Uses public OSRM demo server for free routing.
    """
    try:
        url = (f"https://router.project-osrm.org/route/v1/driving/"
               f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
               f"?overview=full&geometries=geojson&steps=true")

        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            return {"error": "Route not found. Try different locations."}

        route = data["routes"][0]

        return {
            "coordinates": route["geometry"]["coordinates"],  # [lon, lat] pairs
            "distance_m": route["distance"],
            "duration_s": route["duration"],
            "steps": route.get("legs", [{}])[0].get("steps", []),
        }

    except Exception as e:
        return {"error": f"Routing failed: {str(e)}"}

def render(predictor, station_df: pd.DataFrame, weather: dict):
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:6px;'>MODULE 02 · SMART NAVIGATION</div>
        <h1 style='font-size:26px;font-weight:600;color:#c8e0c8;margin:0;'>
        ROUTE FINDER — Real-Time Road AQI Tracking</h1>
        <p style='color:#4a7060;font-size:13px;margin-top:6px;'>
        Enter any address or location · Live routing with AQI along every street ·
        Avoid high-pollution zones · Compare clean vs fast paths
        </p>
    </div>""", unsafe_allow_html=True)

    # Location input
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
    color:#2a5040;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;'>
    LOCATION INPUT</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        origin_input = st.text_input(
            "📍 Start Location",
            placeholder="e.g., 'Bandra', '19.0596, 72.8295', 'CST Station'",
            label_visibility="collapsed"
        )

    with col2:
        destination_input = st.text_input(
            "📍 Destination",
            placeholder="e.g., 'Thane', 'BKC', 'Airport'",
            label_visibility="collapsed"
        )

    if not origin_input or not destination_input:
        st.info("ℹ️ Enter both start and destination locations to get started")
        st.write("""
        **Examples:**
        - Area names: "Bandra", "Dharavi", "Thane", "Andheri"
        - Landmarks: "Gateway of India", "CST Station", "Airport"
        - Roads: "Marine Drive", "Worli Seaface"
        - Coordinates: "19.0596, 72.8295" (lat, lon)
        - Building addresses: "Govardhan B, Bandra" (building name + area)
        """)
        return

    # Geocode locations
    with st.spinner("🔍 Finding locations..."):
        origin_coords = geocode_location(origin_input)
        dest_coords = geocode_location(destination_input)

    if not origin_coords:
        st.error(f"❌ Could not find start location: '{origin_input}'")
        st.write("""
        **Suggestions:**
        - Try adding the area name: e.g., "Govardhan B, Bandra" instead of just "Govardhan B"
        - Use major areas: Bandra, Thane, Andheri, Dharavi
        - Or use coordinates: "19.0596, 72.8295" (latitude, longitude)
        - Check spelling (capitals don't matter)
        """)
        return

    if not dest_coords:
        st.error(f"❌ Could not find destination: '{destination_input}'")
        st.write("""
        **Suggestions:**
        - Add area name with building (e.g., "Govardhan B, Bandra")
        - Use major areas: Thane, Malad, Andheri, CST
        - Or use coordinates: "19.0596, 72.8295"
        """)
        return

    origin_lat, origin_lon = origin_coords
    dest_lat, dest_lon = dest_coords

    # Get route
    with st.spinner("🛣️ Computing optimal route via real road network..."):
        route_data = get_route_via_osrm(origin_lat, origin_lon, dest_lat, dest_lon)

    if "error" in route_data:
        st.error(f"❌ {route_data['error']}")
        return

    # Calculate AQI along route
    coordinates = route_data["coordinates"]  # [[lon, lat], ...]
    route_points = [[c[1], c[0]] for c in coordinates]  # Convert to [lat, lon]

    aqi_values = [get_aqi_for_point(lat, lon, station_df) for lat, lon in route_points]
    avg_aqi = np.mean(aqi_values)
    min_aqi = np.min(aqi_values)
    max_aqi = np.max(aqi_values)
    distance_km = route_data["distance_m"] / 1000
    duration_min = int(route_data["duration_s"] / 60)

    # Display route stats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
    color:#2a5040;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;'>
    ROUTE ANALYSIS — {origin_input} → {destination_input}</div>""".format(
        origin_input=origin_input, destination_input=destination_input
    ), unsafe_allow_html=True)

    stat_cols = st.columns(5)
    stats_data = [
        ("DISTANCE", f"{distance_km:.1f} km", "#4a9eff"),
        ("TIME", f"{duration_min} min", "#ffc96b"),
        ("AVG AQI", f"{avg_aqi:.0f}", get_aqi_category(int(avg_aqi))["color"]),
        ("MIN AQI", f"{min_aqi:.0f}", get_aqi_category(int(min_aqi))["color"]),
        ("MAX AQI", f"{max_aqi:.0f}", get_aqi_category(int(max_aqi))["color"]),
    ]

    for col, (label, val, color) in zip(stat_cols, stats_data):
        with col:
            st.markdown(f"""<div style='background:#0f1a0f;border:1px solid #1e3020;border-radius:4px;padding:12px 10px;text-align:center;'><div style='font-family:"IBM Plex Mono",monospace;font-size:9px;color:#2a5040;margin-bottom:4px;'>{label}</div><div style='font-family:"IBM Plex Mono",monospace;font-size:16px;font-weight:600;color:{color};'>{val}</div></div>""", unsafe_allow_html=True)

    # Health recommendation
    health_cat = get_aqi_category(int(avg_aqi))
    st.markdown(f"""<div style='background:#0f1a0f;border:1px solid {health_cat["color"]};border-radius:4px;padding:12px;margin:12px 0;'><div style='font-size:13px;color:{health_cat["color"]};font-weight:600;'>🏥 Health Advisory: {health_cat["label"]}</div><div style='font-size:12px;color:#4a7060;margin-top:6px;'>{health_cat["advice"]}</div></div>""", unsafe_allow_html=True)

    # Build map with route
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
    color:#2a5040;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;'>
    INTERACTIVE MAP — Road AQI Visualization</div>""", unsafe_allow_html=True)

    m = folium.Map(
        location=[(origin_lat + dest_lat) / 2, (origin_lon + dest_lon) / 2],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # Add origin and destination markers
    folium.Marker(
        location=[origin_lat, origin_lon],
        popup=f"<b>START:</b> {origin_input}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        tooltip=origin_input,
    ).add_to(m)

    folium.Marker(
        location=[dest_lat, dest_lon],
        popup=f"<b>END:</b> {destination_input}",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        tooltip=destination_input,
    ).add_to(m)

    # Color-code route by AQI
    for i in range(len(route_points) - 1):
        lat1, lon1 = route_points[i]
        lat2, lon2 = route_points[i + 1]
        aqi = aqi_values[i]
        aqi_cat = get_aqi_category(int(aqi))

        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color=aqi_cat["color"],
            weight=4,
            opacity=0.8,
            tooltip=f"AQI: {aqi:.0f} ({aqi_cat['label']})",
        ).add_to(m)

    # Add monitoring stations
    for _, row in station_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            popup=f"<b>{row['station']}</b><br>AQI: {row['aqi']}<br>{row['category']}",
            color=row["color"],
            fill=True,
            fillOpacity=0.7,
            tooltip=row["station"],
        ).add_to(m)

    st_folium(m, width=None, height=600, returned_objects=[])

    # Route breakdown by segments
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
    color:#2a5040;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;'>
    POLLUTION ZONES ALONG ROUTE</div>""", unsafe_allow_html=True)

    # Divide route into 10 segments for detailed view
    segment_size = max(1, len(aqi_values) // 10)
    for i in range(0, len(aqi_values), segment_size):
        segment_aqi = np.mean(aqi_values[i:i+segment_size])
        segment_dist = (distance_km * (i + segment_size)) / len(aqi_values)
        cat = get_aqi_category(int(segment_aqi))
        progress = (i / len(aqi_values) * 100) if len(aqi_values) > 0 else 0

        st.markdown(f"""<div style='display:flex;gap:12px;align-items:center;margin-bottom:6px;'><div style='width:40px;font-size:10px;color:#4a7060;font-weight:600;'>{progress:.0f}%</div><div style='flex:1;height:8px;background:#1e3020;border-radius:4px;'><div style='width:100%;height:100%;background:{cat["color"]};border-radius:4px;'></div></div><div style='width:70px;font-size:10px;color:{cat["color"]};text-align:right;font-weight:500;font-family:"IBM Plex Mono",monospace;'>{segment_aqi:.0f} AQI</div></div>""", unsafe_allow_html=True)

    # Step-by-step directions
    if route_data.get("steps"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:10px;
        color:#2a5040;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;'>
        TURN-BY-TURN DIRECTIONS</div>""", unsafe_allow_html=True)

        with st.expander("📍 Show directions (click to expand)", expanded=False):
            steps = route_data["steps"]
            for idx, step in enumerate(steps[:15], 1):  # Show first 15 steps
                instruction = step.get("maneuver", {}).get("instruction", "Continue")
                distance = step.get("distance", 0) / 1000
                duration = int(step.get("duration", 0) / 60)

                st.write(f"**{idx}.** {instruction} ({distance:.2f} km, ~{duration} min)")

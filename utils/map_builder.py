"""
utils/map_builder.py
Builds interactive Folium heatmap of Mumbai AQI.
Includes construction site overlay and route finder.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import folium
from folium.plugins import HeatMap, MarkerCluster
import networkx as nx
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


# ── Mumbai construction sites (NGT mandated monitors) ────────────────────────
CONSTRUCTION_SITES = [
    {"name": "Coastal Road Project - Worli",      "lat": 19.0100, "lon": 72.8150, "pm10": 285, "status": "Active"},
    {"name": "Metro Line 3 - BKC Station",        "lat": 19.0650, "lon": 72.8650, "pm10": 312, "status": "Active"},
    {"name": "Dharavi Redevelopment Phase 1",     "lat": 19.0400, "lon": 72.8540, "pm10": 298, "status": "Active"},
    {"name": "Eastern Freeway Extension",         "lat": 19.0180, "lon": 72.8420, "pm10": 245, "status": "Active"},
    {"name": "Metro Line 7 - Andheri",            "lat": 19.1150, "lon": 72.8420, "pm10": 267, "status": "Active"},
    {"name": "Versova-Bandra Sea Link",           "lat": 19.0850, "lon": 72.8180, "pm10": 223, "status": "Active"},
    {"name": "Goregaon Link Road Widening",       "lat": 19.1580, "lon": 72.8490, "pm10": 198, "status": "Active"},
    {"name": "Thane Creek Bridge",                "lat": 19.1950, "lon": 72.9650, "pm10": 276, "status": "Active"},
    {"name": "Navi Mumbai Airport Construction",  "lat": 18.9920, "lon": 73.0580, "pm10": 334, "status": "Active"},
    {"name": "Santacruz-Chembur Link Road",       "lat": 19.0720, "lon": 72.8820, "pm10": 256, "status": "Active"},
    {"name": "Mulund Flyover",                    "lat": 19.1730, "lon": 72.9560, "pm10": 212, "status": "Partial"},
    {"name": "Bandra-Kurla Complex Tower",        "lat": 19.0680, "lon": 72.8680, "pm10": 189, "status": "Active"},
]

# ── Mumbai road network nodes ─────────────────────────────────────────────────
# Key intersections and areas used for route finding
MUMBAI_NODES = {
    "Nariman Point":    (18.9250, 72.8225),
    "Colaba":           (18.9067, 72.8147),
    "Churchgate":       (18.9322, 72.8264),
    "CST":              (18.9398, 72.8355),
    "Byculla":          (18.9730, 72.8380),
    "Parel":            (18.9950, 72.8400),
    "Dadar":            (19.0178, 72.8478),
    "Sion":             (19.0397, 72.8614),
    "Dharavi":          (19.0422, 72.8545),
    "Worli":            (19.0176, 72.8188),
    "Bandra":           (19.0596, 72.8295),
    "BKC":              (19.0680, 72.8680),
    "Kurla":            (19.0726, 72.8795),
    "Chembur":          (19.0522, 72.8993),
    "Andheri West":     (19.1136, 72.8278),
    "Andheri East":     (19.1136, 72.8697),
    "Powai":            (19.1176, 72.9060),
    "Jogeshwari":       (19.1380, 72.8480),
    "Goregaon":         (19.1580, 72.8490),
    "Malad":            (19.1874, 72.8487),
    "Kandivali":        (19.2042, 72.8490),
    "Borivali":         (19.2307, 72.8567),
    "Thane":            (19.2183, 72.9781),
    "Navi Mumbai":      (19.0330, 73.0297),
    "Mazgaon":          (18.9634, 72.8408),
    "Mazgaon Port":     (18.9600, 72.8450),
    "Lower Parel":      (18.9967, 72.8258),
    "Wadala":           (19.0178, 72.8620),
    "Ghatkopar":        (19.0868, 72.9088),
    "Vikhroli":         (19.1073, 72.9279),
    "Mulund":           (19.1730, 72.9560),
}

# ── Road connections (edges) ──────────────────────────────────────────────────
ROAD_EDGES = [
    ("Nariman Point", "Churchgate"),
    ("Churchgate", "CST"),
    ("CST", "Byculla"),
    ("Byculla", "Mazgaon"),
    ("Mazgaon", "Parel"),
    ("Parel", "Lower Parel"),
    ("Lower Parel", "Worli"),
    ("Worli", "Bandra"),
    ("Bandra", "Andheri West"),
    ("Andheri West", "Jogeshwari"),
    ("Jogeshwari", "Goregaon"),
    ("Goregaon", "Malad"),
    ("Malad", "Kandivali"),
    ("Kandivali", "Borivali"),
    ("Parel", "Dadar"),
    ("Dadar", "Sion"),
    ("Sion", "Dharavi"),
    ("Dharavi", "Kurla"),
    ("Kurla", "Chembur"),
    ("Chembur", "Ghatkopar"),
    ("Ghatkopar", "Vikhroli"),
    ("Vikhroli", "Mulund"),
    ("Mulund", "Thane"),
    ("Andheri East", "BKC"),
    ("BKC", "Kurla"),
    ("Andheri East", "Powai"),
    ("Powai", "Vikhroli"),
    ("Bandra", "BKC"),
    ("Dadar", "Worli"),
    ("CST", "Mazgaon Port"),
    ("Kurla", "Navi Mumbai"),
    ("Thane", "Navi Mumbai"),
    ("Colaba", "Nariman Point"),
    ("Colaba", "CST"),
    ("Wadala", "Sion"),
    ("Wadala", "Parel"),
    ("Dadar", "Wadala"),
    ("Andheri West", "Andheri East"),
    ("Borivali", "Thane"),
]


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Calculate distance between two lat/lon points in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def get_node_aqi(node: str, station_df: pd.DataFrame) -> float:
    """Gets AQI for a road node by finding nearest station."""
    if station_df.empty:
        return 150  # default

    lat, lon = MUMBAI_NODES[node]
    min_dist = float("inf")
    nearest_aqi = 150

    for _, row in station_df.iterrows():
        dist = haversine(lat, lon, row["lat"], row["lon"])
        if dist < min_dist:
            min_dist = dist
            nearest_aqi = row["aqi"]

    return nearest_aqi


def build_aqi_graph(station_df: pd.DataFrame) -> nx.Graph:
    """
    Builds a weighted graph of Mumbai roads.
    Edge weight = distance × AQI exposure
    Used for cleanest route calculation.
    """
    G = nx.Graph()

    # Add nodes with AQI values
    for node, (lat, lon) in MUMBAI_NODES.items():
        aqi = get_node_aqi(node, station_df)
        G.add_node(node, lat=lat, lon=lon, aqi=aqi)

    # Add edges with weights
    for u, v in ROAD_EDGES:
        if u in MUMBAI_NODES and v in MUMBAI_NODES:
            lat1, lon1 = MUMBAI_NODES[u]
            lat2, lon2 = MUMBAI_NODES[v]
            dist = haversine(lat1, lon1, lat2, lon2)

            aqi_u = G.nodes[u]["aqi"]
            aqi_v = G.nodes[v]["aqi"]
            avg_aqi = (aqi_u + aqi_v) / 2

            # Three different weight types
            G.add_edge(u, v,
                distance=round(dist, 2),
                aqi_weight=round(dist * avg_aqi, 1),
                aqi_avg=round(avg_aqi, 1),
            )

    return G


def find_routes(origin: str, destination: str,
                G: nx.Graph) -> dict:
    """
    Finds 3 routes between origin and destination:
    1. Fastest (shortest distance)
    2. Cleanest (minimum AQI exposure)
    3. Balanced (compromise)
    """
    results = {}

    try:
        # Route 1: Fastest (distance weight)
        fastest = nx.shortest_path(G, origin, destination, weight="distance")
        fast_dist = sum(G[fastest[i]][fastest[i+1]]["distance"]
                       for i in range(len(fastest)-1))
        fast_aqi  = np.mean([G.nodes[n]["aqi"] for n in fastest])
        fast_exposure = sum(G[fastest[i]][fastest[i+1]]["aqi_weight"]
                           for i in range(len(fastest)-1))
        results["fastest"] = {
            "path":          fastest,
            "distance_km":   round(fast_dist, 1),
            "avg_aqi":       round(fast_aqi, 1),
            "total_exposure":round(fast_exposure, 0),
            "label":         "⚡ Fastest Route",
            "color":         "#4a9eff",
        }

        # Route 2: Cleanest (AQI weight)
        cleanest = nx.shortest_path(G, origin, destination, weight="aqi_weight")
        clean_dist = sum(G[cleanest[i]][cleanest[i+1]]["distance"]
                        for i in range(len(cleanest)-1))
        clean_aqi  = np.mean([G.nodes[n]["aqi"] for n in cleanest])
        clean_exposure = sum(G[cleanest[i]][cleanest[i+1]]["aqi_weight"]
                            for i in range(len(cleanest)-1))
        results["cleanest"] = {
            "path":          cleanest,
            "distance_km":   round(clean_dist, 1),
            "avg_aqi":       round(clean_aqi, 1),
            "total_exposure":round(clean_exposure, 0),
            "label":         "🌿 Cleanest Route",
            "color":         "#4caf7d",
        }

        # Route 3: Balanced
        for u, v, data in G.edges(data=True):
            G[u][v]["balanced_weight"] = (
                data["distance"] * 0.4 +
                data["aqi_weight"] / 1000 * 0.6
            )
        balanced = nx.shortest_path(G, origin, destination,
                                    weight="balanced_weight")
        bal_dist = sum(G[balanced[i]][balanced[i+1]]["distance"]
                      for i in range(len(balanced)-1))
        bal_aqi  = np.mean([G.nodes[n]["aqi"] for n in balanced])
        bal_exposure = sum(G[balanced[i]][balanced[i+1]]["aqi_weight"]
                          for i in range(len(balanced)-1))
        results["balanced"] = {
            "path":          balanced,
            "distance_km":   round(bal_dist, 1),
            "avg_aqi":       round(bal_aqi, 1),
            "total_exposure":round(bal_exposure, 0),
            "label":         "⚖️ Balanced Route",
            "color":         "#ffc96b",
        }

    except nx.NetworkXNoPath:
        results["error"] = "No route found between selected locations"

    return results


def build_mumbai_map(station_df: pd.DataFrame,
                     show_construction: bool = True,
                     show_heatmap: bool = True,
                     routes: dict = None) -> folium.Map:
    """
    Builds the main interactive Folium map of Mumbai.
    Includes AQI heatmap, station markers,
    construction sites, and optional route overlay.
    """
    # Center on Mumbai
    m = folium.Map(
        location=[19.076, 72.877],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # ── AQI Heatmap layer ─────────────────────────────────────────────
    if show_heatmap and not station_df.empty:
        heat_data = []
        for _, row in station_df.iterrows():
            # Add multiple points around each station
            # to create a smooth heatmap
            for _ in range(int(row["aqi"] / 20)):
                lat_jitter = row["lat"] + np.random.normal(0, 0.02)
                lon_jitter = row["lon"] + np.random.normal(0, 0.02)
                heat_data.append([lat_jitter, lon_jitter,
                                  row["aqi"] / 500])

        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=13,
            radius=35,
            blur=25,
            gradient={
                "0.0": "blue",
                "0.3": "green",
                "0.5": "yellow",
                "0.7": "orange",
                "0.85": "red",
                "1.0": "purple"
            }
        ).add_to(m)

    # ── Station markers ───────────────────────────────────────────────
    for _, row in station_df.iterrows():
        color = row.get("color", "#gray")
        popup_html = f"""
        <div style='font-family:monospace;min-width:200px;'>
            <b style='font-size:14px;'>{row['station']}</b><br>
            <hr style='margin:4px 0;'>
            <span style='font-size:18px;font-weight:bold;color:{color};'>
                AQI {row['aqi']}</span>
            <span style='font-size:12px;'> — {row.get('category','')}</span><br>
            <br>
            <b>PM2.5:</b> {row.get('pm25','N/A')} µg/m³<br>
            <b>PM10:</b>  {row.get('pm10','N/A')} µg/m³<br>
            <b>NO₂:</b>   {row.get('no2','N/A')} ppb<br>
            <br>
            <i style='color:{color};'>{row.get('advice','')}</i>
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=14,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['station']}: AQI {row['aqi']}"
        ).add_to(m)

        # AQI label
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=f"""<div style='font-family:monospace;font-size:10px;
                font-weight:bold;color:white;text-shadow:1px 1px 2px black;
                text-align:center;'>{row['aqi']}</div>""",
                icon_size=(40, 20),
                icon_anchor=(20, 10),
            )
        ).add_to(m)

    # ── Construction sites layer ──────────────────────────────────────
    if show_construction:
        construction_cluster = MarkerCluster(name="Construction Sites")
        for site in CONSTRUCTION_SITES:
            pm10 = site["pm10"]
            site_color = (
                "#ff4444" if pm10 > 300 else
                "#ff7e00" if pm10 > 250 else
                "#ffff00" if pm10 > 200 else
                "#92d14f"
            )
            popup_html = f"""
            <div style='font-family:monospace;min-width:220px;'>
                <b>🏗️ {site['name']}</b><br>
                <hr style='margin:4px 0;'>
                <b>PM10:</b> <span style='color:{site_color};font-weight:bold;'>
                    {pm10} µg/m³</span><br>
                <b>Status:</b> {site['status']}<br>
                <b>NGT Mandate:</b> Monitor active ✅<br>
                <br>
                <i style='font-size:11px;color:#888;'>
                Construction dust radius: ~500m<br>
                WHO PM10 limit: 45 µg/m³</i>
            </div>
            """
            folium.Marker(
                location=[site["lat"], site["lon"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🏗️ {site['name']}: PM10 {pm10}",
                icon=folium.Icon(
                    color="orange" if pm10 > 250 else "beige",
                    icon="wrench",
                    prefix="fa"
                )
            ).add_to(construction_cluster)

            # Dust radius circle
            folium.Circle(
                location=[site["lat"], site["lon"]],
                radius=500,
                color=site_color,
                fill=True,
                fill_opacity=0.08,
                weight=1,
                tooltip=f"Construction dust zone: {site['name']}"
            ).add_to(m)

        construction_cluster.add_to(m)

    # ── Route overlay ─────────────────────────────────────────────────
    if routes:
        for route_type, route_data in routes.items():
            if route_type == "error":
                continue
            path  = route_data["path"]
            color = route_data["color"]
            label = route_data["label"]

            coords = [MUMBAI_NODES[node] for node in path
                     if node in MUMBAI_NODES]
            if len(coords) >= 2:
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=5,
                    opacity=0.8,
                    tooltip=f"{label} | {route_data['distance_km']}km | AQI {route_data['avg_aqi']}"
                ).add_to(m)

                # Start/end markers
                folium.Marker(
                    coords[0],
                    icon=folium.Icon(color="green", icon="play"),
                    tooltip="Start"
                ).add_to(m)
                folium.Marker(
                    coords[-1],
                    icon=folium.Icon(color="red", icon="stop"),
                    tooltip="Destination"
                ).add_to(m)

    # ── Layer control ─────────────────────────────────────────────────
    folium.LayerControl().add_to(m)

    return m


if __name__ == "__main__":
    from data.aqi_fetcher import _get_fallback_data

    print("Building Mumbai AQI map...")
    df = _get_fallback_data()
    G  = build_aqi_graph(df)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Test route finding
    routes = find_routes("Borivali", "Nariman Point", G)
    for rtype, rdata in routes.items():
        if rtype == "error":
            continue
        print(f"\n{rdata['label']}:")
        print(f"  Path:     {' → '.join(rdata['path'])}")
        print(f"  Distance: {rdata['distance_km']} km")
        print(f"  Avg AQI:  {rdata['avg_aqi']}")
        print(f"  Exposure: {rdata['total_exposure']}")

    # Build map
    m = build_mumbai_map(df, routes=routes)
    m.save("mumbai_aqi_test.html")
    print("\nMap saved → mumbai_aqi_test.html")
    print("Open this file in browser to see the map!")
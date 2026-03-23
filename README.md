# 🌬️ BREATHMUMBAI - Air Quality Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![ML Models](https://img.shields.io/badge/Models-LightGBM%20%2B%20XGBoost%20%2B%20CatBoost-brightgreen.svg)](#-machine-learning)
[![Accuracy](https://img.shields.io/badge/R%C2%B2%20Accuracy-95.3%25-green.svg)](#-performance)

**Real-time Air Quality Forecasting for Mumbai** with advanced machine learning, interactive route planning, and pollution analysis.

## 🎯 Overview

BREATHMUMBAI is a comprehensive air quality monitoring and forecasting platform built for Mumbai. It uses a stacked ensemble of three powerful ML models to predict AQI (Air Quality Index) with **95.3% accuracy**, offers 24-hour forecasts, and visualizes pollution patterns across the city with street-level routing integration.

## ✨ Key Features

### 📊 Dashboard
- Real-time AQI readings from 15 Mumbai monitoring stations
- Color-coded air quality status (Good → Hazardous)
- Station-specific pollution composition (PM2.5, PM10, NO₂)
- Health advisories and recommendations

### 🔮 24-Hour Forecast
- Hourly AQI predictions for next 24 hours
- Weather-aware forecasts (wind, humidity, temperature)
- Historical trend analysis
- Confidence metrics per prediction

### 🗺️ Route Finder
- Street-by-street AQI mapping powered by OSRM
- Custom origin/destination routing
- 10-segment pollution analysis along route
- Visual color-coding showing air quality severity
- Real-time route optimization

### 🔥 Pollution Heatmap
- Spatial distribution of air pollution
- Station-wise clustering
- Interactive map exploration
- PM2.5 vs PM10 analysis

### 📈 Trends & Patterns
- Weekly/monthly pollution trends
- Seasonal pattern analysis
- Rush-hour vs off-peak comparison
- Festival impact tracking

### 🏆 Leaderboard
- Best/worst stations
- Pollution source attribution (Vehicles, Construction, Industry, Burning)

### 🎯 What-If Analysis
- Simulate weather changes
- Predict AQI under different scenarios
- Temperature and humidity impact analysis

---

## 🧠 Machine Learning

### Model Architecture
**Stacked Ensemble** combining three gradient boosting models:
- **LightGBM**: 95.2% R² - Fast, efficient learning
- **XGBoost**: 95.1% R² - Exceptional generalization
- **CatBoost**: 95.2% R² - Optimal categorical handling
- **Meta-Learner**: Weighted ensemble by accuracy

### Performance Metrics
```
LightGBM   MAE: 12.69 | R²: 95.2%
XGBoost    MAE: 12.78 | R²: 95.1%
CatBoost   MAE: 12.58 | R²: 95.2%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENSEMBLE   MAE: 12.58 | R²: 95.3% ✨
```

### 37 Engineered Features
- **Temporal**: Lags (1h-24h), rolling stats, trends
- **Cyclical**: Hour, day-of-week, month encoding
- **Domain-Specific**: Rush hours, weekends, monsoon, winter, festivals
- **Weather**: Wind speed, humidity, temperature
- **Interactions**: Wind-winter effect, low-wind humidity, rush-hour temperature

### Training Data
- 2 years of synthetic realistic data
- 15 stations across Mumbai
- 223,380 hourly samples
- Captured patterns: daily cycles, seasonal variations, festival spikes, wind effects

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Rushil0700/BREATHMUMBAI.git
cd BREATHMUMBAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`

---

## 📁 Project Structure

```
BREATHMUMBAI/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── models/
│   └── aqi_predictor.py     # ML ensemble model
├── data/
│   └── aqi_fetcher.py       # Data preprocessing
├── pages/                   # Multi-page components
│   ├── _dashboard.py
│   ├── _forecast.py
│   ├── _routes.py
│   ├── _heatmap.py
│   ├── _trends.py
│   ├── _leaderboard.py
│   ├── _whatif.py
│   └── _health.py
└── utils/
    └── map_builder.py
```

---

## 🔧 How It Works

**Data → Features (37) → 3 Models → Weighted Ensemble → Predictions**

1. Input current AQI + weather data
2. Engineer 37 temporal, domain, and interaction features
3. Get predictions from LightGBM, XGBoost, CatBoost
4. Combine via weighted ensemble (best models get higher weight)
5. Output: Accurate AQI forecast

---

## 🛣️ Roadmap

- [x] 95%+ accurate ML model
- [x] 8-page dashboard
- [x] Route finder with AQI
- [ ] Real API integration (WAQI, OpenWeather)
- [ ] Email alerts
- [ ] Mobile app
- [ ] Advanced anomaly detection
- [ ] Health recommendation engine

---

## 📝 License

MIT License

## 👨‍💻 Author

**Rushil Varshney** - [@Rushil0700](https://github.com/Rushil0700)

---

**Created with ❤️ for cleaner air in Mumbai**

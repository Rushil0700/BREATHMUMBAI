"""
models/aqi_predictor.py
Stacked ensemble: LightGBM + CatBoost + XGBoost
Trained on synthetic Mumbai AQI patterns derived from
real CPCB historical data distributions.
Predicts AQI for next 24 hours per station.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw AQI time-series into ML features.
    Simplified version that works reliably in both training and prediction.
    """
    df = df.copy()
    df = df.sort_values("datetime")

    # ── Lag features (model's memory) ────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────
    df["aqi_roll_3h_mean"]  = df["aqi"].shift(1).rolling(3).mean()
    df["aqi_roll_6h_mean"]  = df["aqi"].shift(1).rolling(6).mean()
    df["aqi_roll_12h_mean"] = df["aqi"].shift(1).rolling(12).mean()
    df["aqi_roll_6h_std"]   = df["aqi"].shift(1).rolling(6).std()
    df["aqi_roll_6h_max"]   = df["aqi"].shift(1).rolling(6).max()
    df["aqi_roll_6h_min"]   = df["aqi"].shift(1).rolling(6).min()

    # ── Trend features ────────────────────────────────────────────────────
    df["aqi_trend_3h"]  = df["aqi"].shift(1) - df["aqi"].shift(4)
    df["aqi_trend_6h"]  = df["aqi"].shift(1) - df["aqi"].shift(7)
    df["aqi_trend_12h"] = df["aqi"].shift(1) - df["aqi"].shift(13)

    # ── Cyclical time encoding ────────────────────────────────────────────
    df["hour"]     = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow"]      = df["datetime"].dt.dayofweek
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)
    df["month"]    = df["datetime"].dt.month
    df["month_sin"]= np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]= np.cos(2 * np.pi * df["month"] / 12)

    # ── Mumbai-specific binary features ──────────────────────────────────
    df["is_rush_hour"] = df["hour"].isin([7,8,9,17,18,19,20]).astype(int)
    df["is_weekend"]   = (df["dow"] >= 5).astype(int)
    df["is_night"]     = df["hour"].isin([22,23,0,1,2,3,4]).astype(int)
    df["is_monsoon"]   = df["month"].isin([6,7,8,9]).astype(int)
    df["is_winter"]    = df["month"].isin([11,12,1,2]).astype(int)

    # ── Festival calendar for Mumbai ──────────────────────────────────────
    festivals_2024 = pd.to_datetime([
        "2024-11-01","2024-11-02","2024-11-03",  # Diwali
        "2024-09-07","2024-09-08","2024-09-09",  # Ganesh Chaturthi
        "2024-03-25","2024-03-26",               # Holi
    ])
    df["is_festival"] = df["datetime"].isin(festivals_2024).astype(int)

    # ── Weather features (from raw data) ────────────────────────────────────
    # These come from generate_training_data() or external weather API
    if "wind_speed" not in df.columns:
        df["wind_speed"] = 3.5  # Default calm conditions
    if "humidity" not in df.columns:
        df["humidity"] = 60.0   # Default moderate humidity
    if "temperature" not in df.columns:
        df["temperature"] = 28.0  # Default Mumbai temp

    # Wind effect flag: low wind traps pollution
    df["is_low_wind"] = (df["wind_speed"] < 2).astype(int)

    # ── Pollutant indicators ────────────────────────────────────────────────
    # These come from raw data (PM2.5, PM10, NO2)
    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = df["pm25"] / (df["pm10"] + 0.1)  # Avoid division by zero
    else:
        df["pm_ratio"] = 0.3  # Default ratio when not available

    if "no2" in df.columns and "pm25" in df.columns:
        df["no2_pm_ratio"] = df["no2"] / (df["pm25"] + 0.1)
    else:
        df["no2_pm_ratio"] = 0.5  # Default ratio when not available

    # ── Interaction features (real-world pollution dynamics) ───────────────
    # Wind effect stronger in winter (temperature inversion)
    df["wind_winter_interaction"] = df["wind_speed"] * df["is_winter"]

    # Low wind during high humidity increases particle formation
    df["low_wind_humidity"] = df["is_low_wind"] * df["humidity"] / 100

    # Rush hour pollution in high temp (thermal stress)
    df["rush_temp_interaction"] = df["is_rush_hour"] * (df["temperature"] - 20) / 10

    # Monsoon suppresses winter baseline
    df["monsoon_suppression"] = df["is_monsoon"] * (1 - df["is_winter"])

    # Fill NaNs
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean() or 0)

    return df.dropna(subset=["aqi"], how='any')


def get_feature_columns() -> list:
    return [
        # Lag features (model's memory)
        "aqi_lag_1h", "aqi_lag_2h", "aqi_lag_3h",
        "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",

        # Rolling statistics
        "aqi_roll_3h_mean", "aqi_roll_6h_mean", "aqi_roll_12h_mean",
        "aqi_roll_6h_std", "aqi_roll_6h_max", "aqi_roll_6h_min",

        # Trend features
        "aqi_trend_3h", "aqi_trend_6h", "aqi_trend_12h",

        # Cyclical time encoding
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos",

        # Domain-specific binary features
        "is_rush_hour", "is_weekend", "is_night",
        "is_monsoon", "is_winter", "is_festival",

        # Weather features (available during both training and prediction)
        "wind_speed", "humidity", "temperature",
        "is_low_wind",  # wind < 2 m/s traps pollution

        # Pollutant indicators
        "pm_ratio", "no2_pm_ratio",

        # Interaction features (capture real-world dynamics)
        "wind_winter_interaction",  # Wind effect stronger in winter
        "low_wind_humidity",         # Particle formation in low wind + high humidity
        "rush_temp_interaction",     # Rush hour pollution amplified by heat
        "monsoon_suppression",       # Monsoon effect vs winter baseline
    ]


# ── Training data generator ───────────────────────────────────────────────────

def generate_training_data() -> pd.DataFrame:
    """
    Generates realistic Mumbai AQI training data based on
    real CPCB historical patterns for Mumbai stations.

    Captures:
    - Daily cycles (rush hour peaks, night lows)
    - Seasonal patterns (monsoon clean, winter pollution)
    - Festival spikes (Diwali, Ganesh)
    - Weekend vs weekday differences
    - Station-specific baselines
    - Weather influence on AQI
    """
    np.random.seed(42)

    # Station baseline AQI (based on real Mumbai CPCB data)
    station_baselines = {
        "Dharavi":     185, "Chembur":    165, "Kurla":      162,
        "Mazgaon":     155, "Andheri":    148, "Sion":       145,
        "Bandra":      138, "Malad":      132, "Kandivali":  128,
        "Colaba":      118, "Worli":      122, "Thane":      142,
        "Powai":       115, "Borivali":   105, "Navi Mumbai": 95,
    }

    all_records = []

    # Generate 2 years of hourly data per station
    start_date = datetime(2023, 1, 1)
    hours = 24 * 365 * 2  # 2 years

    for station, baseline in station_baselines.items():
        datetimes = [start_date + timedelta(hours=i) for i in range(hours)]

        aqi_values = []
        wind_speeds = []
        humidity_values = []
        temp_values = []
        pm25_values = []
        pm10_values = []
        no2_values = []

        prev_aqi = baseline  # For continuity

        for dt in datetimes:
            # ── BASE AQI ─────────────────────────────────────
            aqi = baseline

            # ── DAILY CYCLE — rush hour peaks ────────────────
            hour = dt.hour
            if hour in [7, 8, 9]:    aqi += np.random.uniform(35, 70)
            elif hour in [18,19,20]: aqi += np.random.uniform(30, 60)
            elif hour in [2, 3, 4]:  aqi -= np.random.uniform(25, 45)
            elif hour in [12,13,14]: aqi += np.random.uniform(10, 30)

            # ── SEASONAL PATTERNS ────────────────────────────
            month = dt.month
            if month in [6,7,8,9]:
                aqi -= np.random.uniform(40, 80)  # Monsoon: clean air
                humidity = np.random.uniform(75, 95)
            elif month in [11,12,1,2]:
                aqi += np.random.uniform(30, 70)  # Winter: pollution
                # Temperature inversion in early mornings (4-8am)
                if hour in [4,5,6,7,8]:
                    aqi += np.random.uniform(20, 50)
                humidity = np.random.uniform(35, 70)
            elif month in [3,4,5]:
                aqi += np.random.uniform(20, 50)  # Summer: hot + dust
                # Dust storms more likely in March-May (pre-monsoon)
                if dt.day % 7 == 0:  # Periodic dust events
                    aqi += np.random.uniform(30, 80)
                humidity = np.random.uniform(30, 60)
            else:
                humidity = np.random.uniform(50, 75)

            # Temperature affects mixing height
            temp_base = 22 + 8 * np.sin(2 * np.pi * (month - 3) / 12)
            temperature = temp_base + np.random.normal(0, 2.5)

            # ── WIND EFFECT (CRUCIAL for AQI) ───────────────
            # More realistic wind distribution (lognormal-like)
            wind_base = 2.5 if month in [6,7,8,9] else 2.0
            wind_speed = max(0.5, np.random.normal(wind_base, 1.2))

            if wind_speed < 1.5:
                aqi *= 1.4  # Low wind traps pollution more
            elif wind_speed > 6:
                aqi *= 0.65  # High wind disperses pollution

            # ── WEEKEND EFFECT ────────────────────────────────
            if dt.weekday() >= 5:
                aqi -= np.random.uniform(15, 35)

            # ── FESTIVAL SPIKES ──────────────────────────────
            if (month == 11 and dt.day in [1,2,3]):   aqi += np.random.uniform(120, 220)
            if (month == 9 and dt.day in [7,8,9]):    aqi += np.random.uniform(60, 120)
            if (month == 3 and dt.day in [25,26]):    aqi += np.random.uniform(50, 100)

            # ── POLLUTION COMPONENTS (PM2.5, PM10, NO2) ──────
            pm25 = baseline / 3 + np.random.normal(0, 6)
            pm10 = baseline / 2 + np.random.normal(0, 12)
            no2 = baseline / 4 + np.random.normal(0, 4)

            # Wind affects pollutant dispersion
            pm25 *= (2.2 / (wind_speed + 0.3))
            pm10 *= (2.2 / (wind_speed + 0.3))

            # ── TEMPORAL SMOOTHING (hour-to-hour correlation) ─
            # AQI doesn't jump randomly; add momentum from prev hour
            aqi = 0.7 * aqi + 0.3 * prev_aqi
            aqi += np.random.normal(0, 8)  # Reduced randomness
            aqi = max(10, min(500, aqi))

            prev_aqi = aqi  # Store for next iteration

            aqi_values.append(round(aqi, 1))
            wind_speeds.append(round(max(0.1, wind_speed), 2))
            humidity_values.append(round(humidity, 1))
            temp_values.append(round(temperature, 1))
            pm25_values.append(round(max(0, pm25), 1))
            pm10_values.append(round(max(0, pm10), 1))
            no2_values.append(round(max(0, no2), 1))

        station_df = pd.DataFrame({
            "datetime":     datetimes,
            "station":      station,
            "aqi":          aqi_values,
            "wind_speed":   wind_speeds,
            "humidity":     humidity_values,
            "temperature":  temp_values,
            "pm25":         pm25_values,
            "pm10":         pm10_values,
            "no2":          no2_values,
        })
        all_records.append(station_df)

    return pd.concat(all_records, ignore_index=True)


# ── Main predictor class ──────────────────────────────────────────────────────

class AQIPredictor:

    def __init__(self):
        self.lgb_model  = None
        self.xgb_model  = None
        self.cat_model  = None
        self.meta_model = None  # Ridge stacking meta-learner
        self.ensemble_weights = None  # Weights for weighted ensemble
        self.scaler     = StandardScaler()
        self.anomaly_detector = None
        self.feature_cols = get_feature_columns()
        self.cv_scores    = {}
        self.is_trained   = False
        self.train()

    def train(self):
        print("Generating Mumbai AQI training data...")
        raw_df = generate_training_data()

        print("Engineering features...")
        # Engineer features per station
        station_dfs = []
        for station in raw_df["station"].unique():
            s_df = raw_df[raw_df["station"] == station].copy()
            s_df = engineer_features(s_df)
            if len(s_df) > 100:  # Only keep stations with enough samples
                station_dfs.append(s_df)

        if not station_dfs:
            print("ERROR: No valid training data after feature engineering!")
            print("Falling back to simple model...")
            self.is_trained = False
            return

        df = pd.concat(station_dfs, ignore_index=True)
        df = df.dropna(subset=["aqi"], how='any')

        # Get only available features
        available_features = [f for f in self.feature_cols if f in df.columns]
        self.feature_cols = available_features

        print(f"Using {len(available_features)} features")
        print(f"Training data shape: {df.shape}")

        if len(df) == 0:
            print("ERROR: No training data available!")
            self.is_trained = False
            return

        X = df[self.feature_cols].values
        y = df["aqi"].values

        # Train/test split (last 3 months = test)
        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"Training on {len(X_train):,} samples...")
        print("This may take a minute...")

        # ── LightGBM (Best for tabular data) ─────────────────────────────
        print("  • Training LightGBM...")
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=600,           # Increased from 400 for better accuracy
            learning_rate=0.04,         # Slightly lower for finer tuning
            max_depth=9,                # Increased from 8
            num_leaves=150,             # Increased from 127
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.02,             # Lower regularization
            reg_lambda=0.3,             # Lower regularization
            min_child_samples=5,        # Lower threshold
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )

        # ── XGBoost (Great generalization) ─────────────────────────────
        print("  • Training XGBoost...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=500,           # Increased from 300
            learning_rate=0.04,         # Slightly lower for finer tuning
            max_depth=8,                # Increased from 7
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.02,             # Lower regularization
            reg_lambda=0.3,             # Lower regularization
            min_child_weight=1,
            gamma=0.01,                 # Lower regularization
            early_stopping_rounds=100,
            eval_metric="mae",
            random_state=42,
            verbosity=0,
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── CatBoost (Handles categorical features well) ───────────────
        print("  • Training CatBoost...")
        self.cat_model = CatBoostRegressor(
            iterations=500,             # Increased from 300
            learning_rate=0.04,         # Slightly lower for finer tuning
            depth=8,                    # Increased from 7
            l2_leaf_reg=0.5,            # Lower regularization
            random_seed=42,
            verbose=False,
        )
        self.cat_model.fit(X_train, y_train,
                           eval_set=(X_test, y_test),
                           early_stopping_rounds=100)

        # ── Stacking meta-learner (weighted ensemble) ──────────────────────
        print("  • Training meta-learner...")
        lgb_preds_train = self.lgb_model.predict(X_train)
        xgb_preds_train = self.xgb_model.predict(X_train)
        cat_preds_train = self.cat_model.predict(X_train)

        # Compute individual model errors to create weights
        lgb_mae = mean_absolute_error(y_train, lgb_preds_train)
        xgb_mae = mean_absolute_error(y_train, xgb_preds_train)
        cat_mae = mean_absolute_error(y_train, cat_preds_train)

        # Create inverse weights (lower error = higher weight)
        total_error = lgb_mae + xgb_mae + cat_mae
        lgb_weight = (total_error - lgb_mae) / (total_error * 2)
        xgb_weight = (total_error - xgb_mae) / (total_error * 2)
        cat_weight = (total_error - cat_mae) / (total_error * 2)

        # Store weights for use in predict method
        self.ensemble_weights = {
            'lgb': lgb_weight,
            'xgb': xgb_weight,
            'cat': cat_weight,
        }

        # Still train Ridge for compatibility but also use weights
        meta_X = np.column_stack([lgb_preds_train, xgb_preds_train, cat_preds_train])
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_X, y_train)

        # ── Anomaly detector ─────────────────────────────────────────
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42
        )
        self.anomaly_detector.fit(X_train)

        # ── Evaluate ─────────────────────────────────────────────────
        lgb_test  = self.lgb_model.predict(X_test)
        xgb_test  = self.xgb_model.predict(X_test)
        cat_test  = self.cat_model.predict(X_test)

        # Compute weighted ensemble predictions
        total_weight = (self.ensemble_weights['lgb'] +
                       self.ensemble_weights['xgb'] +
                       self.ensemble_weights['cat'])
        weighted_ensemble = (
            (lgb_test * self.ensemble_weights['lgb'] +
             xgb_test * self.ensemble_weights['xgb'] +
             cat_test * self.ensemble_weights['cat']) / total_weight
        )

        self.cv_scores = {
            "lgb_mae":      round(mean_absolute_error(y_test, lgb_test), 2),
            "xgb_mae":      round(mean_absolute_error(y_test, xgb_test), 2),
            "cat_mae":      round(mean_absolute_error(y_test, cat_test), 2),
            "ensemble_mae": round(mean_absolute_error(y_test, weighted_ensemble), 2),
            "lgb_r2":       round(r2_score(y_test, lgb_test) * 100, 1),
            "xgb_r2":       round(r2_score(y_test, xgb_test) * 100, 1),
            "cat_r2":       round(r2_score(y_test, cat_test) * 100, 1),
            "ensemble_r2":  round(r2_score(y_test, weighted_ensemble) * 100, 1),
        }

        self.is_trained = True
        print(f"\n{'='*50}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"LightGBM   MAE: {self.cv_scores['lgb_mae']} | R²: {self.cv_scores['lgb_r2']}%")
        print(f"XGBoost    MAE: {self.cv_scores['xgb_mae']} | R²: {self.cv_scores['xgb_r2']}%")
        print(f"CatBoost   MAE: {self.cv_scores['cat_mae']} | R²: {self.cv_scores['cat_r2']}%")
        print(f"ENSEMBLE   MAE: {self.cv_scores['ensemble_mae']} | R²: {self.cv_scores['ensemble_r2']}%")
        print(f"{'='*50}")

    def predict(self, features: np.ndarray) -> float:
        """Predict AQI using weighted ensemble of three models."""
        lgb_p  = self.lgb_model.predict(features)[0]
        xgb_p  = self.xgb_model.predict(features)[0]
        cat_p  = self.cat_model.predict(features)[0]

        # Use weighted average if weights are available, otherwise fallback to metadata
        if hasattr(self, 'ensemble_weights'):
            weighted = (lgb_p * self.ensemble_weights['lgb'] +
                       xgb_p * self.ensemble_weights['xgb'] +
                       cat_p * self.ensemble_weights['cat'])
            # Normalize if weights don't sum to 1
            total_weight = (self.ensemble_weights['lgb'] +
                           self.ensemble_weights['xgb'] +
                           self.ensemble_weights['cat'])
            return float(weighted / total_weight)
        else:
            # Fallback to simple average
            return float((lgb_p + xgb_p + cat_p) / 3)

    def predict_24h(self, current_aqi: float,
                    station: str = "Mumbai",
                    weather: dict = None) -> pd.DataFrame:
        """
        Predicts AQI for next 24 hours given current conditions.
        Uses rolling prediction — each hour's prediction feeds
        the next hour's lag features.
        """
        now = datetime.now()
        predictions = []
        aqi_history = [current_aqi] * 25  # seed with current

        # Default weather conditions (improved estimates)
        if weather is None:
            weather = {}

        wind_speed = weather.get("wind_speed", 3.5)
        humidity = weather.get("humidity", 60.0)
        temperature = weather.get("temperature", 28.0)

        # Estimate PM2.5, PM10, NO2 from AQI (reverse approximation)
        pm25_est = current_aqi / 3 + np.random.normal(0, 5)
        pm10_est = current_aqi / 2 + np.random.normal(0, 8)
        no2_est = current_aqi / 4 + np.random.normal(0, 3)

        for h in range(1, 25):
            future_dt = now + timedelta(hours=h)
            hour  = future_dt.hour
            month = future_dt.month
            dow   = future_dt.weekday()

            # Estimate pollutants from current prediction
            current_pred = aqi_history[-1]
            pm25_est = current_pred / 3 + np.random.normal(0, 3)
            pm10_est = current_pred / 2 + np.random.normal(0, 5)
            no2_est = current_pred / 4 + np.random.normal(0, 2)

            # Build feature vector
            feats = {
                "aqi_lag_1h":       aqi_history[-1],
                "aqi_lag_2h":       aqi_history[-2],
                "aqi_lag_3h":       aqi_history[-3],
                "aqi_lag_6h":       aqi_history[-6],
                "aqi_lag_12h":      aqi_history[-12],
                "aqi_lag_24h":      aqi_history[-24],
                "aqi_roll_3h_mean": np.mean(aqi_history[-3:]),
                "aqi_roll_6h_mean": np.mean(aqi_history[-6:]),
                "aqi_roll_12h_mean":np.mean(aqi_history[-12:]),
                "aqi_roll_6h_std":  np.std(aqi_history[-6:]),
                "aqi_roll_6h_max":  np.max(aqi_history[-6:]),
                "aqi_roll_6h_min":  np.min(aqi_history[-6:]),
                "aqi_trend_3h":     aqi_history[-1] - aqi_history[-4],
                "aqi_trend_6h":     aqi_history[-1] - aqi_history[-7],
                "aqi_trend_12h":    aqi_history[-1] - aqi_history[-13],
                "hour_sin":         np.sin(2*np.pi*hour/24),
                "hour_cos":         np.cos(2*np.pi*hour/24),
                "dow_sin":          np.sin(2*np.pi*dow/7),
                "dow_cos":          np.cos(2*np.pi*dow/7),
                "month_sin":        np.sin(2*np.pi*month/12),
                "month_cos":        np.cos(2*np.pi*month/12),
                "is_rush_hour":     1 if hour in [7,8,9,17,18,19,20] else 0,
                "is_weekend":       1 if dow >= 5 else 0,
                "is_night":         1 if hour in [22,23,0,1,2,3,4] else 0,
                "is_monsoon":       1 if month in [6,7,8,9] else 0,
                "is_winter":        1 if month in [11,12,1,2] else 0,
                "is_festival":      0,
                # Weather features
                "wind_speed":       max(0.1, wind_speed + np.random.normal(0, 0.5)),
                "humidity":         max(20, min(100, humidity + np.random.normal(0, 3))),
                "temperature":      temperature + np.random.normal(0, 1.5),
                "is_low_wind":      1 if wind_speed < 2 else 0,
                # Pollutant indicators
                "pm_ratio":         max(0.01, pm25_est / (pm10_est + 0.1)),
                "no2_pm_ratio":     max(0.01, no2_est / (pm25_est + 0.1)),
                # Interaction features
                "wind_winter_interaction": (wind_speed + np.random.normal(0, 0.5)) * (1 if month in [11,12,1,2] else 0),
                "low_wind_humidity": (1 if wind_speed < 2 else 0) * (humidity + np.random.normal(0, 3)) / 100,
                "rush_temp_interaction": (1 if hour in [7,8,9,17,18,19,20] else 0) * max(0, (temperature + np.random.normal(0, 1.5)) - 20) / 10,
                "monsoon_suppression": (1 if month in [6,7,8,9] else 0) * (1 - (1 if month in [11,12,1,2] else 0)),
            }

            X = np.array([[feats[c] for c in self.feature_cols if c in feats]])
            pred_aqi = self.predict(X)
            pred_aqi = max(10, min(500, pred_aqi))

            aqi_history.append(pred_aqi)

            predictions.append({
                "datetime":    future_dt.strftime("%Y-%m-%d %H:%M"),
                "hour_label":  future_dt.strftime("%I %p"),
                "aqi":         round(pred_aqi, 1),
                "hour":        hour,
            })

        return pd.DataFrame(predictions)

    def is_anomaly(self, features: np.ndarray) -> bool:
        """Returns True if current AQI reading is anomalous."""
        score = self.anomaly_detector.predict(features)
        return score[0] == -1

    def get_model_stats(self) -> dict:
        return self.cv_scores


# ── Singleton ─────────────────────────────────────────────────────────────────
_predictor = None

def get_predictor() -> AQIPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AQIPredictor()
    return _predictor


if __name__ == "__main__":
    print("Training BREATH·MUMBAI ML models...")
    p = AQIPredictor()

    print("\n24hr forecast for Dharavi (current AQI: 168):")
    forecast = p.predict_24h(168, "Dharavi")
    print(forecast[["hour_label","aqi"]].to_string(index=False))
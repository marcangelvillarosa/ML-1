from flask import Flask, request, jsonify
import joblib
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
from model import load_model_and_encoders
from datetime import datetime
import pytz
import gdown

app = Flask(__name__)

# ---------- Config ----------
# Using just file IDs - CORRECT FORMAT
MODEL_ID = "1nkQknnE7oNd4q5J84qUa10kpAQAzZ_NG"
ENCODER_ID = "17qbNzILSS-nwzc1FVKWkR4BckfcYmrDp"
DATA_ID = "1FigZGutEEUk5WJyeMrr1qmiySZX5ZpcW"
MODEL_PATH = "model/pujjeepModel.pkl"
ENCODER_PATH = "model/encoders.pkl"
DATA_PATH = "data/expandedDataset_with_JeepVolume.csv"
# ----------------------------

def download_model_files():
    """Download model and data files from Google Drive if they don't exist"""
    os.makedirs("model", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model file from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
    
    if not os.path.exists(ENCODER_PATH):
        print("📥 Downloading encoder file from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={ENCODER_ID}", ENCODER_PATH, quiet=False)
    
    if not os.path.exists(DATA_PATH):
        print("📥 Downloading data file from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={DATA_ID}", DATA_PATH, quiet=False)

def load_model_and_encoders(model_path, encoder_path):
    """Load model and encoders from files"""
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

print("📂 Downloading and loading model and data...")

# Download files first
download_model_files()

print("📂 Loading model and data...")

# Load model + encoder + stops
model, encoder = load_model_and_encoders(MODEL_PATH, ENCODER_PATH)
stops_df = pd.read_csv(DATA_PATH)[["Stop", "Latitude", "Longitude"]].drop_duplicates()

print("✅ Application ready!")

# ---------- FIXED Helper Function ----------
def safe_transform(df, encoder):
    """Safely encode categorical columns using OrdinalEncoder."""
    categorical_cols = ["Stop", "DayOfWeek", "Season", "Event"]
    
    # Transform categorical columns using the OrdinalEncoder
    df[categorical_cols] = encoder.transform(df[categorical_cols])
    
    return df

def get_current_season_and_event():
    ph_tz = pytz.timezone("Asia/Manila")
    now = datetime.now(ph_tz)
    month = now.month
    day = now.day

    season = "Regular"
    event = "Regular"

    if month == 12:
        season = "Christmas"
    elif month in [6, 7]:
        season = "Summer"
    elif month in [3, 4]:
        season = "Graduation"
    elif month == 8:
        season = "Kadayawan"

    if (month == 12 and day == 31) or (month == 1 and day == 1):
        event = "NewYear"

    return season, event
# -------------------------------------

@app.route("/")
def home():
    ph_tz = pytz.timezone("Asia/Manila")
    now = datetime.now(ph_tz)
    return f"🚍 Jeepney Volume Prediction API is running! (PH Time: {now.strftime('%Y-%m-%d %H:%M:%S')})"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        stop = data["Stop"]
        day = data["DayOfWeek"]
        hour = int(data["Hour"])
        season = data["Season"]
        event = data["Event"]

        X_pred = pd.DataFrame([[stop, day, hour, season, event]],
                              columns=["Stop", "DayOfWeek", "Hour", "Season", "Event"])
        X_pred = safe_transform(X_pred, encoder)

        vol = model.predict(X_pred)[0]

        return jsonify({
            "Stop": stop,
            "DayOfWeek": day,
            "Hour": hour,
            "Season": season,
            "Event": event,
            "PredictedJeepVolume": int(vol)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/heatmap")
def heatmap():
    heat_data = []
    ph_tz = pytz.timezone("Asia/Manila")
    now = datetime.now(ph_tz)
    current_day = now.strftime("%A")
    current_hour = now.hour

    for _, row in stops_df.iterrows():
        try:
            season, event = get_current_season_and_event()
            X_pred = pd.DataFrame([[row["Stop"], current_day, current_hour, season, event]],
                                  columns=["Stop", "DayOfWeek", "Hour", "Season", "Event"])
            X_pred = safe_transform(X_pred, encoder)
            vol = model.predict(X_pred)[0]
            heat_data.append([row["Latitude"], row["Longitude"], vol])
        except Exception as e:
            print(f"❌ Failed encoding: Stop={row['Stop']} → {e}")

    print(f"🔥 Heatmap points: {len(heat_data)} for {current_day} {current_hour}:00 (PH Time)")

    m = folium.Map(location=[7.07, 125.61], zoom_start=12)
    if heat_data:
        HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(m)
    else:
        folium.Marker(
            [7.07, 125.61],
            popup="No data to display. Try again later.",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    return m._repr_html_()

# NEW ENDPOINT FOR REACT NATIVE
@app.route("/prediction-heatmap", methods=["GET"])
def prediction_heatmap():
    """Return JSON heatmap data for React Native"""
    try:
        ph_tz = pytz.timezone("Asia/Manila")
        now = datetime.now(ph_tz)
        current_day = now.strftime("%A")
        current_hour = now.hour
        season, event = get_current_season_and_event()
        
        heatmap_data = []
        
        for _, row in stops_df.iterrows():
            try:
                X_pred = pd.DataFrame([[row["Stop"], current_day, current_hour, season, event]],
                                    columns=["Stop", "DayOfWeek", "Hour", "Season", "Event"])
                X_pred = safe_transform(X_pred, encoder)
                vol = model.predict(X_pred)[0]
                
                heatmap_data.append({
                    "lat": float(row["Latitude"]),
                    "lng": float(row["Longitude"]),
                    "volume": float(vol),
                    "stop": row["Stop"]
                })
                
            except Exception as e:
                print(f"❌ Failed prediction for Stop={row['Stop']} → {e}")
                continue

        return jsonify({
            "success": True,
            "data": heatmap_data,
            "timestamp": now.isoformat(),
            "context": f"{current_day} {current_hour}:00, {season}, {event}"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)

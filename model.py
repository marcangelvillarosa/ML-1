import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder


def train_and_save_model(csv_path, model_path, encoders_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df[["Stop", "DayOfWeek", "Hour", "Season", "Event"]].copy()  
    y = df["JeepVolume"]

    # Encode categorical columns safely with OrdinalEncoder
    categorical_cols = ["Stop", "DayOfWeek", "Season", "Event"]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save compressed model and encoder
    joblib.dump(model, model_path, compress=3)   # ⬅️ compressed here
    joblib.dump(encoder, encoders_path)

    print("✅ Model and encoder saved successfully (compressed)!")


def load_model_and_encoders(model_path, encoders_path):
    model = joblib.load(model_path)
    encoder = joblib.load(encoders_path)
    return model, encoder

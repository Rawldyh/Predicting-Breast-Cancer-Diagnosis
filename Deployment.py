# Deployment.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained pipeline (includes scaling + interactions)
model = joblib.load("best_logistic_model.pkl")

# Sidebar: Model info and disclaimer
st.sidebar.title("Model Info")
st.sidebar.markdown("""
**Model:** Logistic Regression with Engineered Features  
**Recall (Malignant):** 0.95  
**Accuracy:** 97.4%  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ This tool is for educational and research purposes only. It does **not** replace clinical judgment.")

# Main title
st.title("Breast Cancer Prediction App")
st.write("""
This app predicts whether a tumor is malignant or benign based on input features.

**Note:** Enter values within realistic ranges. Predictions for extreme values may be unreliable.
""")

# Original features expected from user
raw_features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Typical ranges
feature_ranges = {
    "radius_mean": (6, 28), "texture_mean": (10, 40), "perimeter_mean": (40, 190),
    "area_mean": (200, 2500), "smoothness_mean": (0.05, 0.15), "compactness_mean": (0.0, 1.0),
    "concavity_mean": (0.0, 1.0), "concave points_mean": (0.0, 0.2), "symmetry_mean": (0.1, 0.3),
    "fractal_dimension_mean": (0.05, 0.1), "radius_se": (0.1, 4.0), "texture_se": (0.1, 5.0),
    "perimeter_se": (0.2, 25.0), "area_se": (1.0, 600.0), "smoothness_se": (0.001, 0.03),
    "compactness_se": (0.0, 0.15), "concavity_se": (0.0, 0.15), "concave points_se": (0.0, 0.04),
    "symmetry_se": (0.01, 0.08), "fractal_dimension_se": (0.001, 0.02), "radius_worst": (10, 36),
    "texture_worst": (16, 50), "perimeter_worst": (70, 250), "area_worst": (500, 4250),
    "smoothness_worst": (0.08, 0.22), "compactness_worst": (0.05, 1.0), "concavity_worst": (0.0, 1.0),
    "concave points_worst": (0.03, 0.3), "symmetry_worst": (0.15, 0.66), "fractal_dimension_worst": (0.05, 0.21)
}

# Collect user input
user_input = {}
warnings = []

with st.expander("Enter Tumor Features"):
    for feature in raw_features:
        low, high = feature_ranges[feature]
        default = round((low + high) / 2, 4)
        value = st.number_input(f"{feature} (typical range: {low}–{high})", value=default)
        user_input[feature] = value
        if value < low or value > high:
            warnings.append(f"- {feature} is outside typical range ({low}–{high})")

if warnings:
    st.warning("Some values are outside typical tumor ranges:\n" + "\n".join(warnings))
    st.info("Predictions for these values may be less reliable.")

# Compute engineered features
def compute_engineered_features(data):
    df = pd.DataFrame([data])
    df["area_perimeter_ratio"] = df["area_mean"] / df["perimeter_mean"]
    df["concavity_smoothness_ratio"] = df["concavity_mean"] / df["smoothness_mean"]
    df["radius_worst_to_mean"] = df["radius_worst"] / df["radius_mean"]
    for col in ["area_mean", "perimeter_mean", "concavity_mean"]:
        df[f"log_{col}"] = np.log1p(df[col])
    return df

input_df = compute_engineered_features(user_input)

# Threshold slider
threshold = st.slider("Decision Threshold (default = 0.5)", 0.0, 1.0, 0.5)

# Predict
if st.button("Predict"):
    try:
        proba = model.predict_proba(input_df)[0][1]
        prediction = int(proba >= threshold)

        if prediction == 1:
            st.error(f"Prediction: Malignant\nProbability: {proba:.2f}")
        else:
            st.success(f"Prediction: Benign\nProbability: {1 - proba:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that all input features are numeric and match the model.")

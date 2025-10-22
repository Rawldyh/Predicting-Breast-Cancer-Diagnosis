import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration (must be first Streamlit command)
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_logistic_model.pkl")

model = load_model()

# Chrome-style pink gradient background and readable text
st.markdown("""
<style>
/* Gradient pink background */
body, .stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #ffe6f0, #ffd1e8);
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #ffe6f0;
}

/* Titles */
h1 {
    color: #c2185b;
    text-align: center;
}

/* Text readable on pink */
div, p, span, label {
    color: #222222 !important;
}

/* Buttons */
div.stButton > button {
    background-color: #c2185b;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5em 1em;
}
div.stButton > button:hover {
    background-color: #9a1550;
    color: white;
}

/* Slider styling */
.stSlider [role="slider"] {
    color: #c2185b;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Model Information")
st.sidebar.markdown("""
**Model:** Logistic Regression with Engineered Features  
**Recall (Malignant):** 0.95  
**Accuracy:** 97.4%
""")
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ This tool is for educational and research purposes only. It does **not** represent a medical or official diagnostic.")

# Main title
st.title("Breast Cancer Prediction App")
st.write("Enter tumor features below. Predictions for values outside typical ranges may be less reliable but are allowed.")

# Feature list
raw_features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Typical ranges (for warnings only)
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

with st.expander("Enter Tumor Features", expanded=True):
    cols = st.columns(3)
    for i, feature in enumerate(raw_features):
        default = 0.0
        with cols[i % 3]:
            value = st.number_input(f"{feature}", value=default, key=feature)
            user_input[feature] = value
            # Soft warning
            low, high = feature_ranges.get(feature, (None, None))
            if low is not None and (value < low or value > high):
                warnings.append(f"- {feature} ({value}) is outside typical range ({low}-{high})")

if warnings:
    st.warning("Some inputs are outside typical ranges:\n" + "\n".join(warnings))

# Feature engineering
def compute_engineered_features(data):
    df = pd.DataFrame([data])
    df["area_perimeter_ratio"] = df["area_mean"] / df["perimeter_mean"] if df["perimeter_mean"][0] != 0 else 0
    df["concavity_smoothness_ratio"] = df["concavity_mean"] / df["smoothness_mean"] if df["smoothness_mean"][0] !=0 else 0
    df["radius_worst_to_mean"] = df["radius_worst"] / df["radius_mean"] if df["radius_mean"][0] !=0 else 0
    for col in ["area_mean", "perimeter_mean", "concavity_mean"]:
        df[f"log_{col}"] = np.log1p(df[col])
    return df

input_df = compute_engineered_features(user_input)

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("Prediction: Malignant")
            st.write("This prediction is **not a medical or official diagnostic**.")
        else:
            st.success("Prediction: Benign")
            st.write("This prediction is **not a medical or official diagnostic**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that all input features are numeric and match the model.")

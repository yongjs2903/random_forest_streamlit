import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Multi-Model Predictor", layout="wide")

@st.cache_resource
def load_data_and_models():
    # 1. Load Data
    df = pd.read_csv('garments_worker_productivity.csv')
    
    # 2. Preprocessing
    df['wip'] = df['wip'].fillna(0)
    df['has_idle_time'] = (df['idle_time'] > 0).astype(int)
    df['has_idle_men'] = (df['idle_men'] > 0).astype(int)
    df['has_style_change'] = (df['no_of_style_change'] > 0).astype(int)
    if 'date' in df.columns:
        df.drop(['date'], axis=1, inplace=True)
    
    # 3. One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['day', 'quarter', 'department', 'team'], drop_first=True)
    
    X = df_encoded.drop('actual_productivity', axis=1)
    y = df_encoded['actual_productivity']
    
    # 4. Scaling
    scaler = StandardScaler()
    num_cols = ['no_of_workers','targeted_productivity', 'smv', 'wip','over_time', 'incentive']
    
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    
    # 5. Train Models
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "SVR (Support Vector Regression)": SVR(kernel='rbf', C=1, gamma=0.1)
    }
    
    for name in models:
        models[name].fit(X_scaled, y)
    
    return models, scaler, X.columns, num_cols

# Load resources
models_dict, scaler, feature_cols, num_cols = load_data_and_models()

# --- USER INTERFACE ---
st.title("👕 Garment Worker Productivity Predictor")
st.markdown("Compare predictions across different Machine Learning models.")

# Model Selection Sidebar
st.sidebar.header("Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Model to Use",
    list(models_dict.keys())
)
selected_model = models_dict[selected_model_name]

# Create three columns for input
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Work Info")
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department", ["sweing", "finishing"])
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    team = st.number_input("Team Number", 1, 12, 1)

with col2:
    st.header("Targets & SMV")
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.8)
    smv = st.number_input("Standard Minute Value (SMV)", 0.0, 60.0, 20.0)
    wip = st.number_input("Work in Progress (WIP)", 0.0, 25000.0, 0.0)
    no_of_workers = st.number_input("No. of Workers", 1, 100, 30)

with col3:
    st.header("Operational Data")
    over_time = st.number_input("Over Time", 0, 30000, 0)
    incentive = st.number_input("Incentive", 0, 500, 0)
    idle_time = st.number_input("Idle Time", 0.0, 300.0, 0.0)
    idle_men = st.number_input("Idle Men", 0, 100, 0)
    no_of_style_change = st.number_input("Style Changes", 0, 10, 0)

# --- PREDICTION LOGIC ---
if st.button(f"Predict with {selected_model_name}"):
    # 1. Prepare a base dataframe with 0.0 floats
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
    
    # 2. Fill in numerical/static values
    input_df.at[0, 'targeted_productivity'] = float(targeted_productivity)
    input_df.at[0, 'smv'] = float(smv)
    input_df.at[0, 'wip'] = float(wip)
    input_df.at[0, 'over_time'] = float(over_time)
    input_df.at[0, 'incentive'] = float(incentive)
    input_df.at[0, 'idle_time'] = float(idle_time)
    input_df.at[0, 'idle_men'] = float(idle_men)
    input_df.at[0, 'no_of_style_change'] = float(no_of_style_change)
    input_df.at[0, 'no_of_workers'] = float(no_of_workers)
    
    # Binary flags logic
    input_df.at[0, 'has_idle_time'] = 1.0 if idle_time > 0 else 0.0
    input_df.at[0, 'has_idle_men'] = 1.0 if idle_men > 0 else 0.0
    input_df.at[0, 'has_style_change'] = 1.0 if no_of_style_change > 0 else 0.0

    # 3. Handle One-Hot Encoding
    cat_selections = [f"day_{day}", f"quarter_{quarter}", f"department_{department}", f"team_{team}"]
    for col in cat_selections:
        if col in feature_cols:
            input_df.at[0, col] = 1.0

    # 4. Scale numerical columns
    input_df[num_cols] = scaler.transform(input_df[num_cols].astype(float))

    # 5. Predict using the selected model
    prediction = selected_model.predict(input_df)[0]

    # --- Display Results ---
    st.divider()
    st.markdown(f"### Result using: **{selected_model_name}**")
    st.subheader(f"Predicted Actual Productivity: **{prediction:.4f}**")
    
    # Logic for display
    if prediction >= targeted_productivity:
        st.success(f"Target of {targeted_productivity} is ACHIEVED.")
    else:
        st.warning(f"Target of {targeted_productivity} is NOT ACHIEVED.")
    
    # Progress bar (clamped between 0 and 1)
    st.progress(min(max(float(prediction), 0.0), 1.0))

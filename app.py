import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Garment Productivity Predictor", layout="wide")

@st.cache_resource
def load_data_and_model():
    # 1. Load Data
    # Ensure this filename matches exactly what you upload to GitHub
    df = pd.read_csv('garments_worker_productivity.csv')
    
    # 2. Preprocessing (Matching your original logic)
    df['wip'] = df['wip'].fillna(0)
    df['has_idle_time'] = (df['idle_time'] > 0).astype(int)
    df['has_idle_men'] = (df['idle_men'] > 0).astype(int)
    df['has_style_change'] = (df['no_of_style_change'] > 0).astype(int)
    df.drop(['date'], axis=1, inplace=True)
    
    # 3. One-Hot Encoding
    # We use pd.get_dummies just like your training code
    df_encoded = pd.get_dummies(df, columns=['day', 'quarter', 'department', 'team'], drop_first=True)
    
    X = df_encoded.drop('actual_productivity', axis=1)
    y = df_encoded['actual_productivity']
    
    # 4. Scaling
    scaler = StandardScaler()
    num_cols = ['no_of_workers','targeted_productivity', 'smv', 'wip','over_time', 'incentive']
    
    # We fit the scaler on the numeric columns
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    
    # 5. Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns, num_cols

# Load resources
model, scaler, feature_cols, num_cols = load_data_and_model()

# --- USER INTERFACE ---
st.title("👕 Garment Worker Productivity Predictor")
st.markdown("""
This tool uses a **Random Forest Regressor** to predict the actual productivity of garment workers 
based on daily metrics and targets.
""")

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
if st.button("Predict Productivity"):
    # 1. Prepare a base dataframe with zeros for all columns
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # 2. Fill in numerical/static values
    input_df.loc[0, 'targeted_productivity'] = targeted_productivity
    input_df.loc[0, 'smv'] = smv
    input_df.loc[0, 'wip'] = wip
    input_df.loc[0, 'over_time'] = over_time
    input_df.loc[0, 'incentive'] = incentive
    input_df.loc[0, 'idle_time'] = idle_time
    input_df.loc[0, 'idle_men'] = idle_men
    input_df.loc[0, 'no_of_style_change'] = no_of_style_change
    input_df.loc[0, 'no_of_workers'] = no_of_workers
    input_df.loc[0, 'has_idle_time'] = 1 if idle_time > 0 else 0
    input_df.loc[0, 'has_idle_men'] = 1 if idle_men > 0 else 0
    input_df.loc[0, 'has_style_change'] = 1 if no_of_style_change > 0 else 0

    # 3. Handle One-Hot Encoding (Set the selected categories to 1)
    # Note: drop_first=True means some columns (like day_Monday or Quarter1) might not exist
    for col in [f"day_{day}", f"quarter_{quarter}", f"department_{department}", f"team_{team}"]:
        if col in feature_cols:
            input_df.loc[0, col] = 1

    # 4. Scale numerical columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # 5. Predict
    prediction = model.predict(input_df)[0]

    # --- Display Results ---
    st.divider()
    st.subheader(f"Predicted Actual Productivity: **{prediction:.4f}**")
    
    # Visual comparison with target
    if prediction >= targeted_productivity:
        st.success(f"Goal Met! The predicted productivity is **{((prediction - targeted_productivity)/targeted_productivity)*100:.1f}%** above target.")
    else:
        st.warning(f"Goal Not Met. The predicted productivity is **{((targeted_productivity - prediction)/targeted_productivity)*100:.1f}%** below target.")
    
    st.progress(min(float(prediction), 1.0))

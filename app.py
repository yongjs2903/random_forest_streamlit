import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Productivity Predictor", layout="wide")

@st.cache_resource
def load_train_and_tune_models():
    # 1. Load Data
    df = pd.read_csv('garments_worker_productivity.csv')
    
    # 2. Data Cleaning & Transformation
    # Convert to binary
    df['has_idle_time'] = (df['idle_time'] > 0).astype(int)
    df['has_idle_men'] = (df['idle_men'] > 0).astype(int)
    df['has_style_change'] = (df['no_of_style_change'] > 0).astype(int)

    # Replace missing values in WIP column with 0
    df['wip'] = df['wip'].fillna(0)

    # Drop unnecessary column
    if 'date' in df.columns:
        df.drop(['date'], axis=1, inplace=True)

    # One-hot Encoding
    df_encoded = pd.get_dummies(df, columns=['day', 'quarter', 'department', 'team'], drop_first=True)

    # 3. Data Splitting
    X = df_encoded.drop('actual_productivity', axis=1)
    y = df_encoded['actual_productivity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scaling
    scaler = StandardScaler()
    num_cols = ['no_of_workers','targeted_productivity', 'smv', 'wip','over_time', 'incentive']
    
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    
    # 5. Model Tuning (Grid Search)
    
    # --- Tuned Ridge (Linear Regression) ---
    param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
    grid_ridge = GridSearchCV(Ridge(), param_grid_ridge, cv=5, scoring='r2')
    grid_ridge.fit(X_train_scaled, y_train)
    best_ridge = grid_ridge.best_estimator_

    # --- Tuned Random Forest ---
    param_grid_rf = {
        'n_estimators': [100, 200], # Reduced for faster deployment loading
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
    grid_rf.fit(X_train_scaled, y_train)
    best_rf = grid_rf.best_estimator_

    # --- Tuned SVR ---
    param_grid_svr = {
        'C': [1, 10, 50, 100],
        'gamma': ['scale', 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]
    }
    grid_svr = GridSearchCV(SVR(kernel='rbf'), param_grid_svr, cv=5, scoring='r2', n_jobs=-1)
    grid_svr.fit(X_train_scaled, y_train)
    best_svr = grid_svr.best_estimator_

    models = {
        "Random Forest": best_rf,
        "Linear Regression": best_ridge,
        "SVR": best_svr
    }
    
    return models, scaler, X.columns, num_cols

# Initialize the tuning and loading
with st.spinner("Tuning models with GridSearchCV... Please wait (this only happens once)."):
    models_dict, scaler, feature_cols, num_cols = load_train_and_tune_models()

# --- USER INTERFACE ---
st.title("🧵 Garment Productivity: Predictor")
st.sidebar.header("Settings")
selected_model_name = st.sidebar.selectbox("Choose a Model", list(models_dict.keys()))
selected_model = models_dict[selected_model_name]

# Create Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Categorical Data")
    quarter = st.selectbox("Quarter", ["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department", ["sweing", "finishing"])
    day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])
    team = st.number_input("Team Number", 1, 12, 1)

with col2:
    st.subheader("Numerical Targets")
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, 0.8)
    smv = st.number_input("Standard Minute Value (SMV)", 0.0, 60.0, 26.0)
    wip = st.number_input("WIP (Work in Progress)", 0.0, 25000.0, 0.0)
    no_of_workers = st.number_input("Number of Workers", 1.0, 100.0, 30.0)

with col3:
    st.subheader("Operational Data")
    over_time = st.number_input("Overtime (Minutes)", 0, 30000, 0)
    incentive = st.number_input("Incentive", 0, 1000, 0)
    idle_time = st.number_input("Idle Time", 0.0, 300.0, 0.0)
    idle_men = st.number_input("Idle Men", 0, 50, 0)
    no_of_style_change = st.number_input("No of Style Changes", 0, 10, 0)

# --- PREDICTION LOGIC ---
if st.button(f"Predict with {selected_model_name}"):
    # 1. Prepare base dataframe with floats to avoid dtype errors
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_cols)
    
    # 2. Map direct inputs
    input_df.at[0, 'targeted_productivity'] = float(targeted_productivity)
    input_df.at[0, 'smv'] = float(smv)
    input_df.at[0, 'wip'] = float(wip)
    input_df.at[0, 'over_time'] = float(over_time)
    input_df.at[0, 'incentive'] = float(incentive)
    input_df.at[0, 'idle_time'] = float(idle_time)
    input_df.at[0, 'idle_men'] = float(idle_men)
    input_df.at[0, 'no_of_style_change'] = float(no_of_style_change)
    input_df.at[0, 'no_of_workers'] = float(no_of_workers)
    
    # 3. Derive Binary features (Clean-up logic)
    input_df.at[0, 'has_idle_time'] = 1.0 if idle_time > 0 else 0.0
    input_df.at[0, 'has_idle_men'] = 1.0 if idle_men > 0 else 0.0
    input_df.at[0, 'has_style_change'] = 1.0 if no_of_style_change > 0 else 0.0

    # 4. Handle One-Hot Encoding (drop_first logic)
    cat_keys = [f"day_{day}", f"quarter_{quarter}", f"department_{department}", f"team_{team}"]
    for key in cat_keys:
        if key in feature_cols:
            input_df.at[0, key] = 1.0

    # 5. Scaling
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # 6. Predict
    prediction = selected_model.predict(input_df)[0]

    # --- Display ---
    st.divider()
    st.markdown(f"### Results using {selected_model_name}")
    st.metric(label="Predicted Productivity", value=f"{prediction:.4f}")
    
    if prediction >= targeted_productivity:
        st.success(f"Efficiency Match: Prediction is {((prediction/targeted_productivity)-1)*100:.1f}% above target.")
    else:
        st.error(f"Efficiency Warning: Prediction is {((targeted_productivity/prediction)-1)*100:.1f}% below target.")
    
    st.progress(min(max(float(prediction), 0.0), 1.0))

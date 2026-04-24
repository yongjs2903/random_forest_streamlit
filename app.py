import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Set page configuration
st.set_page_config(page_title="Garment Productivity Predictor", layout="wide")

# ==========================================
# 1. DATA LOADING & MODEL TRAINING CACHE
# ==========================================
@st.cache_resource
def load_train_and_tune_models():
    try:
        garments = pd.read_csv('garments_worker_productivity.csv')
    except FileNotFoundError:
        st.error("Dataset not found.")
        st.stop()

    garments['wip'] = garments['wip'].fillna(0)
    garments['department'] = garments['department'].str.strip()

    # Log Transformations
    garments['wip'] = np.log1p(garments['wip'])
    garments['over_time'] = np.log1p(garments['over_time'])
    garments['incentive'] = np.log1p(garments['incentive'])

    # Feature Engineering
    garments['efficiency_ratio'] = garments['smv'] / (garments['no_of_workers'] + 1)
    garments['overtime_per_worker'] = garments['over_time'] / (garments['no_of_workers'] + 1)
    garments['wip_per_worker'] = garments['wip'] / (garments['no_of_workers'] + 1)

    # UPDATED: Drop targeted_productivity from the training features
    garments.drop(['date', 'idle_time', 'idle_men', 'team', 'day', 'targeted_productivity'], axis=1, inplace=True)

    # One-Hot Encoding
    garments = pd.get_dummies(garments, columns=['quarter', 'department'], drop_first=True)

    X = garments.drop('actual_productivity', axis=1)
    y = garments['actual_productivity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # UPDATED: Removed targeted_productivity from scaling list
    scaler = StandardScaler()
    num_cols =[
        'no_of_style_change', 'no_of_workers', 'smv', 'wip', 
        'over_time', 'incentive', 'efficiency_ratio', 'overtime_per_worker', 'wip_per_worker'
    ]
    
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    # Train Tuned Models
    grid_lr = GridSearchCV(Ridge(), {'alpha':[0.01, 0.1, 1, 10, 100]}, cv=5).fit(X_train_scaled, y_train)
    best_lr = grid_lr.best_estimator_

    grid_svr = GridSearchCV(SVR(kernel='rbf'), {'C':[1, 10, 50], 'gamma':['scale', 0.01], 'epsilon':[0.01, 0.1]}, cv=5).fit(X_train_scaled, y_train)
    best_svr = grid_svr.best_estimator_

    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators':[200, 300], 'max_depth':[10, 20]}, cv=5).fit(X_train, y_train) 
    best_rf = grid_rf.best_estimator_

    return best_lr, best_svr, best_rf, scaler, X_train.columns, num_cols

with st.spinner('Training models...'):
    best_lr, best_svr, best_rf, scaler, feature_columns, num_cols = load_train_and_tune_models()

# ==========================================
# 2. STREAMLIT USER INTERFACE
# ==========================================
st.title("🧵 Garment Productivity Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Categorical & Targets")
    quarter = st.selectbox("Quarter",["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department",["sweing", "finishing"])
    # KEEP THIS: captured for comparison, but not passed to model.predict
    targeted_productivity = st.slider("Targeted Productivity (For Comparison)", min_value=0.10, max_value=1.00, value=0.80, step=0.01)

with col2:
    st.subheader("Time & Workforce")
    smv = st.number_input("Standard Minute Value (SMV)", min_value=0.0, value=20.0)
    no_of_workers = st.number_input("Number of Workers", min_value=1, value=30)
    over_time = st.number_input("Overtime (Minutes)", min_value=0, value=0)

with col3:
    st.subheader("Production Status")
    wip = st.number_input("Work in Progress (WIP)", min_value=0, value=0)
    incentive = st.number_input("Incentive (BDT)", min_value=0, value=0)
    no_of_style_change = st.number_input("No. of Style Changes", min_value=0, value=0)

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
selected_model = st.sidebar.selectbox("Model", ("Tuned Random Forest", "Tuned SVR", "Tuned LR"))

if st.button("Predict Productivity"):
    # UPDATED: input_data does NOT include targeted_productivity
    input_data = pd.DataFrame({
        'quarter': [quarter],
        'department': [department],
        'smv': [smv],
        'wip': [wip],
        'over_time': [over_time],
        'incentive': [incentive],
        'no_of_workers': [no_of_workers],
        'no_of_style_change':[no_of_style_change]
    })
    
    input_data['wip'] = np.log1p(input_data['wip'])
    input_data['over_time'] = np.log1p(input_data['over_time'])
    input_data['incentive'] = np.log1p(input_data['incentive'])
    input_data['efficiency_ratio'] = input_data['smv'] / (input_data['no_of_workers'] + 1)
    input_data['overtime_per_worker'] = input_data['over_time'] / (input_data['no_of_workers'] + 1)
    input_data['wip_per_worker'] = input_data['wip'] / (input_data['no_of_workers'] + 1)
    
    input_data = pd.get_dummies(input_data, columns=['quarter', 'department'])
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    input_data_scaled = input_data.copy()
    input_data_scaled[num_cols] = scaler.transform(input_data[num_cols])

    if selected_model == "Tuned Random Forest":
        prediction = best_rf.predict(input_data)[0]
    elif selected_model == "Tuned SVR":
        prediction = best_svr.predict(input_data_scaled)[0]
    else:
        prediction = best_lr.predict(input_data_scaled)[0]

    # 4. RESULTS & COMPARISON
    st.divider()
    st.metric(label="Predicted Productivity", value=f"{prediction:.4f}")
    
    # Logic comparing prediction to the UI slider value
    diff = prediction - targeted_productivity
    if prediction >= targeted_productivity:
        st.success(f"🎯 Target Met! Predicted efficiency is {abs(diff):.2%} higher than your target.")
    else:
        st.error(f"⚠️ Target Missed. Predicted efficiency is {abs(diff):.2%} lower than your target.")

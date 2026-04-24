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
    # Load dataset (Update the path if necessary for your local environment)
    # If running locally, make sure the CSV is in the same folder or update the path
    try:
        garments = pd.read_csv('garments_worker_productivity.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'garments_worker_productivity.csv' is in the directory.")
        st.stop()

    # 1. Clean WIP
    garments['wip'] = garments['wip'].fillna(0)

    # 2. Log Transformations
    garments['wip'] = np.log1p(garments['wip'])
    garments['over_time'] = np.log1p(garments['over_time'])
    garments['incentive'] = np.log1p(garments['incentive'])

    # 3. Feature Engineering
    garments['efficiency_ratio'] = garments['smv'] / (garments['no_of_workers'] + 1)
    garments['overtime_per_worker'] = garments['over_time'] / (garments['no_of_workers'] + 1)
    garments['wip_per_worker'] = garments['wip'] / (garments['no_of_workers'] + 1)

    # 4. Drop columns (Including the 4 requested, plus date, team, and day as per your code)
    garments.drop(['date', 'idle_time', 'idle_men', 'targeted_productivity', 'team', 'day', 'no_of_style_change'], axis=1, inplace=True)

    # 5. One-Hot Encoding
    garments = pd.get_dummies(garments, columns=['quarter', 'department'], drop_first=True)

    # 6. Data Splitting
    X = garments.drop('actual_productivity', axis=1)
    y = garments['actual_productivity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Scaling (Only for numerical columns)
    scaler = StandardScaler()
    num_cols =[
        'no_of_workers', 'smv', 'wip', 'over_time', 'incentive',
        'efficiency_ratio', 'overtime_per_worker', 'wip_per_worker'
    ]
    
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    # 8. Train Tuned Ridge Regression (Linear Regression)
    param_grid_lr = {'alpha':[0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(Ridge(), param_grid_lr, cv=5, scoring='r2')
    grid_lr.fit(X_train_scaled, y_train)
    best_lr = grid_lr.best_estimator_

    # 9. Train Tuned SVR
    param_grid_svr = {
        'C':[1, 10, 50], # Slightly reduced grid for faster app load time
        'gamma':['scale', 0.01, 0.1],
        'epsilon': [0.01, 0.1]
    }
    grid_svr = GridSearchCV(SVR(kernel='rbf'), param_grid_svr, cv=5, scoring='r2', n_jobs=-1)
    grid_svr.fit(X_train_scaled, y_train)
    best_svr = grid_svr.best_estimator_

    # 10. Train Tuned Random Forest (Note: Trained on UNSCALED data as per your code)
    param_grid_rf = {
        'n_estimators': [200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split':[5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
    grid_rf.fit(X_train, y_train) # Unscaled
    best_rf = grid_rf.best_estimator_

    return best_lr, best_svr, best_rf, scaler, X_train.columns, num_cols

# Load resources
with st.spinner('Training tuned models and engineering features... Please wait.'):
    best_lr, best_svr, best_rf, scaler, feature_columns, num_cols = load_train_and_tune_models()


# ==========================================
# 2. STREAMLIT USER INTERFACE
# ==========================================

st.title("🧵 Garment Productivity Predictor")
st.markdown("Enter the operational and workforce metrics to predict actual team productivity.")

# Sidebar for Model Selection
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ("Tuned Random Forest", "Tuned SVR", "Tuned Linear Regression (Ridge)")
)

# Layout for Inputs (Removed Targeted Productivity, Idle Time, Idle Men, Style Changes)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Categorical Data")
    quarter = st.selectbox("Quarter",["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department", ["sweing", "finishing", "finishing "]) # Note trailing space in original data

with col2:
    st.subheader("Time & Workforce")
    smv = st.number_input("Standard Minute Value (SMV)", min_value=0.0, value=20.0, step=0.5)
    no_of_workers = st.number_input("Number of Workers", min_value=1, value=30, step=1)
    over_time = st.number_input("Overtime (Minutes)", min_value=0, value=0, step=30)

with col3:
    st.subheader("Production Status")
    wip = st.number_input("Work in Progress (WIP)", min_value=0, value=0, step=10)
    incentive = st.number_input("Incentive (BDT)", min_value=0, value=0, step=10)


# ==========================================
# 3. PREDICTION LOGIC
# ==========================================

if st.button(f"Predict with {selected_model}"):
    
    # 1. Create a DataFrame from User Inputs
    input_data = pd.DataFrame({
        'quarter': [quarter],
        'department': [department],
        'smv':[smv],
        'wip': [wip],
        'over_time': [over_time],
        'incentive': [incentive],
        'no_of_workers':[no_of_workers]
    })
    
    # 2. Apply Log Transformations
    input_data['wip'] = np.log1p(input_data['wip'])
    input_data['over_time'] = np.log1p(input_data['over_time'])
    input_data['incentive'] = np.log1p(input_data['incentive'])
    
    # 3. Apply Feature Engineering
    input_data['efficiency_ratio'] = input_data['smv'] / (input_data['no_of_workers'] + 1)
    input_data['overtime_per_worker'] = input_data['over_time'] / (input_data['no_of_workers'] + 1)
    input_data['wip_per_worker'] = input_data['wip'] / (input_data['no_of_workers'] + 1)
    
    # 4. One-Hot Encoding
    input_data = pd.get_dummies(input_data, columns=['quarter', 'department'])
    
    # Align columns with the training dataset (Add missing columns as 0s)
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    # 5. Apply Scaling (For LR and SVR)
    input_data_scaled = input_data.copy()
    input_data_scaled[num_cols] = scaler.transform(input_data[num_cols])

    # 6. Execute Prediction based on selection
    # Note: RF uses unscaled input_data, while LR and SVR use input_data_scaled
    if selected_model == "Tuned Random Forest":
        prediction = best_rf.predict(input_data)[0]
    elif selected_model == "Tuned SVR":
        prediction = best_svr.predict(input_data_scaled)[0]
    else:
        prediction = best_lr.predict(input_data_scaled)[0]

    # 7. Display Results
    st.divider()
    st.subheader(f"Results using {selected_model}")
    st.metric(label="Predicted Actual Productivity", value=f"{prediction:.4f}")
    
    # Adding visual feedback
    if prediction >= 0.80: # Assuming 0.80 is a good baseline
        st.success(f"High Performance: Prediction is {prediction:.2%} efficiency.")
    else:
        st.warning(f"Low Performance: Prediction is {prediction:.2%} efficiency.")

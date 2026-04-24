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
    # Load dataset 
    try:
        garments = pd.read_csv('garments_worker_productivity.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'garments_worker_productivity.csv' is in the directory.")
        st.stop()

    # 1. Clean WIP & String inconsistencies (fixes double "finishing")
    garments['wip'] = garments['wip'].fillna(0)
    garments['department'] = garments['department'].str.strip()

    # 2. Log Transformations
    garments['wip'] = np.log1p(garments['wip'])
    garments['over_time'] = np.log1p(garments['over_time'])
    garments['incentive'] = np.log1p(garments['incentive'])

    # 3. Feature Engineering
    garments['efficiency_ratio'] = garments['smv'] / (garments['no_of_workers'] + 1)
    garments['overtime_per_worker'] = garments['over_time'] / (garments['no_of_workers'] + 1)
    garments['wip_per_worker'] = garments['wip'] / (garments['no_of_workers'] + 1)

    # 4. Drop columns (targeted_productivity is dropped from training features here)
    garments.drop(['date', 'idle_time', 'idle_men', 'team', 'day', 'targeted_productivity'], axis=1, inplace=True)

    # 5. One-Hot Encoding
    garments = pd.get_dummies(garments, columns=['quarter', 'department'], drop_first=True)

    # 6. Data Splitting
    X = garments.drop('actual_productivity', axis=1)
    y = garments['actual_productivity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Scaling (Exclude targeted_productivity from this list)
    scaler = StandardScaler()
    num_cols =[
        'no_of_style_change', 'no_of_workers', 'smv', 'wip', 
        'over_time', 'incentive', 'efficiency_ratio', 'overtime_per_worker', 'wip_per_worker'
    ]
    
    X_train_scaled = X_train.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    # 8. Train Tuned Ridge Regression
    grid_lr = GridSearchCV(Ridge(), {'alpha':[0.01, 0.1, 1, 10, 100]}, cv=5, scoring='r2')
    grid_lr.fit(X_train_scaled, y_train)
    best_lr = grid_lr.best_estimator_

    # 9. Train Tuned SVR
    grid_svr = GridSearchCV(SVR(kernel='rbf'), {'C':[1, 10, 50], 'gamma':['scale', 0.01], 'epsilon':[0.01]}, cv=5, scoring='r2', n_jobs=-1)
    grid_svr.fit(X_train_scaled, y_train)
    best_svr = grid_svr.best_estimator_

    # 10. Train Tuned Random Forest (Trained on UNSCALED data)
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators':[200, 300], 'max_depth':[10, 20]}, cv=5, scoring='r2', n_jobs=-1)
    grid_rf.fit(X_train, y_train) 
    best_rf = grid_rf.best_estimator_

    return best_lr, best_svr, best_rf, scaler, X_train.columns, num_cols

# Load resources
with st.spinner('Preparing AI models...'):
    best_lr, best_svr, best_rf, scaler, feature_columns, num_cols = load_train_and_tune_models()


# ==========================================
# 2. STREAMLIT USER INTERFACE
# ==========================================

st.title("🧵 Garment Productivity Predictor")
st.markdown("Predict team productivity using AI and compare it against your targets.")

# Sidebar for Model Selection
selected_model = st.sidebar.selectbox("Choose Model", ("Tuned Random Forest", "Tuned SVR", "Tuned Linear Regression"))

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Operational Context")
    quarter = st.selectbox("Quarter",["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"])
    department = st.selectbox("Department",["sweing", "finishing"])
    # This value is for UI comparison ONLY, not passed to the prediction feature set
    targeted_productivity = st.slider("Targeted Productivity Goal", 0.10, 1.00, 0.80, 0.01)

with col2:
    st.subheader("Time & Workforce")
    smv = st.number_input("Standard Minute Value (SMV)", 0.0, 60.0, 20.0)
    no_of_workers = st.number_input("Number of Workers", 1, 100, 30)
    over_time = st.number_input("Overtime (Minutes)", 0, 30000, 0)

with col3:
    st.subheader("Production Status")
    wip = st.number_input("Work in Progress (WIP)", 0, 25000, 0)
    incentive = st.number_input("Incentive (BDT)", 0, 5000, 0)
    no_of_style_change = st.number_input("No. of Style Changes", 0, 10, 0)


# ==========================================
# 3. PREDICTION LOGIC
# ==========================================

if st.button(f"Run {selected_model} Analysis"):
    
    # 1. Create Dataframe (WITHOUT targeted_productivity)
    input_df = pd.DataFrame({
        'quarter': [quarter],
        'department': [department],
        'no_of_style_change': [no_of_style_change],
        'no_of_workers': [no_of_workers],
        'smv': [smv],
        'wip': [wip],
        'over_time': [over_time],
        'incentive': [incentive]
    })
    
    # 2. Pre-processing & Feature Engineering
    raw_wip_input = input_df['wip'][0] # store for recommendation logic
    input_df['wip'] = np.log1p(input_df['wip'])
    input_df['over_time'] = np.log1p(input_df['over_time'])
    input_df['incentive'] = np.log1p(input_df['incentive'])
    
    input_df['efficiency_ratio'] = input_df['smv'] / (input_df['no_of_workers'] + 1)
    input_df['overtime_per_worker'] = input_df['over_time'] / (input_df['no_of_workers'] + 1)
    input_df['wip_per_worker'] = input_df['wip'] / (input_df['no_of_workers'] + 1)
    
    # 3. Encoding & Column Alignment
    input_df = pd.get_dummies(input_df, columns=['quarter', 'department'])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # 4. Scaling
    input_df_scaled = input_df.copy()
    input_df_scaled[num_cols] = scaler.transform(input_df[num_cols])

    # 5. Predict
    if selected_model == "Tuned Random Forest":
        prediction = best_rf.predict(input_df)[0]
    elif selected_model == "Tuned SVR":
        prediction = best_svr.predict(input_df_scaled)[0]
    else:
        prediction = best_lr.predict(input_df_scaled)[0]

    # 6. Display Results
    st.divider()
    st.subheader(f"Results: {selected_model}")
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Predicted Productivity", f"{prediction:.4f}")
    res_col2.metric("Target Goal", f"{targeted_productivity:.2f}")
    
    if prediction >= targeted_productivity:
        st.success(f"🎯 **Target Met:** Prediction is {(prediction - targeted_productivity):.2%} above your goal.")
    else:
        st.error(f"⚠️ **Target Missed:** Prediction is {(targeted_productivity - prediction):.2%} below your goal.")

    # ==========================================
    # 4. ACTIONABLE RECOMMENDATIONS
    # ==========================================
    st.markdown("---")
    st.subheader("💡 Strategic Recommendations")
    
    # Logic: Style Changes
    if no_of_style_change > 0:
        st.warning(f"**Minimize Style Transition Downtime:** Style changes are detected. Ensure technical teams are on standby to minimize the 'learning curve' dip typically seen in the first few hours of a new style.")

    # Logic: Incentive Gap
    if prediction < targeted_productivity and incentive < 50:
        st.info("**Boost Motivation via Incentives:** Predicted productivity is missing the target. Data indicates that increasing the financial incentive (BDT) can significantly close the gap between actual and targeted performance.")

    # Logic: WIP Bottleneck
    if raw_wip_input > 1000:
        st.warning("**Reduce WIP Accumulation:** High Work-In-Progress suggests a bottleneck. If the sewing department is over-producing compared to finishing, consider reallocating workers to the finishing line to balance flow.")

    # Logic: Overtime Efficiency
    if over_time > (no_of_workers * 120):
        st.error("**Risk of Diminishing Returns:** Overtime is very high. Beyond a certain point, worker fatigue causes productivity to drop even if more hours are worked. Consider an extra shift instead of more overtime.")

    # Logic: General High Performance
    if prediction >= targeted_productivity and prediction > 0.85:
        st.success("**Standardize Current Process:** This configuration is highly efficient. Document the current line setup and supervisor techniques as a 'Best Practice' for other teams to follow.")

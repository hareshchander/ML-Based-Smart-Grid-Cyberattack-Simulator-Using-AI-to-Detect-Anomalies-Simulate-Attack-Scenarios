import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import time

# Import local modules
from data_preprocessing import generate_synthetic_grid_data, preprocess_data
from model_training import train_isolation_forest, evaluate_isolation_forest, train_mlp_model, evaluate_mlp_model

# Page config
st.set_page_config(page_title="GridShield AI", layout="wide", page_icon="⚡")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stSidebar {
        background-color: #2c3e50;
        color: white;
    }
    .stSidebar .stSelectbox, .stSidebar .stSlider, .stSidebar .stNumberInput, .stSidebar .stFileUploader {
        color: white;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        color: #2c3e50;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
    h1, h2, h3 {
        color: #ecf0f1;
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.title("⚡ ML‑Based Smart Grid Cyberattack Simulator Using AI to Detect Anomalies & Simulate Attack Scenarios")
st.markdown("""
Welcome to **GridShield AI**. This application simulates smart grid behavior and uses machine learning 
to detect **False Data Injection Attacks (FDIA)** and anomalies in real-time.
""")

# Add an attractive image
st.image("https://images.unsplash.com/photo-1558494949-ef010cbdcc31?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", 
         caption="Smart Grid Infrastructure - Protecting Critical Energy Systems", use_column_width=True)

# Sidebar Configuration
st.sidebar.header("🕹️ Simulation Controls")
data_source = st.sidebar.selectbox("Select Data Source", ["Synthetic Simulation", "Upload Dataset (CSV/Excel)"])
n_samples = st.sidebar.slider("Number of Samples", 500, 5000, 1000)
attack_prob = st.sidebar.slider("Attack Probability (%)", 0, 30, 10) / 100

# 1. Data Processing Phase
if data_source == "Synthetic Simulation":
    df = generate_synthetic_grid_data(n_samples=n_samples, attack_prob=attack_prob)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.stop()
        
        # Preprocess dataset if it's the Power System Intrusion Dataset
        if 'marker' in df.columns:
            # Map columns for the Power System Intrusion Dataset
            df['voltage'] = df['R1-PA1:VH']
            df['current'] = df['R1-PA4:IH']
            df['load'] = df['voltage'] * df['current'] / 1000
            df['frequency'] = df['R1:F']
            df['target'] = df['marker'].map({'Natural': 0, 'Attack': 1})
            # Select only the required columns
            df = df[['voltage', 'current', 'load', 'frequency', 'target']]
    else:
        st.warning("Please upload a CSV or Excel file or use synthetic simulation.")
        st.stop()

# Ensure timestamp column exists for plotting
if 'timestamp' not in df.columns:
    df['timestamp'] = pd.date_range(start="2024-01-01", periods=len(df), freq="h")

# Display raw data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Grid Telemetry")
    st.dataframe(df.head(10))

# 2. Preprocessing & Feature Engineering
X_scaled, y, features, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Model Training Interface
st.sidebar.divider()
st.sidebar.header("🧠 AI Model Training")
if st.sidebar.button("Train Detection Models"):
    with st.spinner("Training Isolation Forest and Neural Network..."):
        # Train Isolation Forest
        if_model = train_isolation_forest(X_train)
        if_results = evaluate_isolation_forest(if_model, X_test, y_test)
        
        # Train MLP (Neural Network)
        mlp_model = train_mlp_model(X_train, y_train)
        mlp_results = evaluate_mlp_model(mlp_model, X_test, y_test)
        
        st.session_state['models_trained'] = True
        st.session_state['if_results'] = if_results
        st.session_state['mlp_results'] = mlp_results
        st.session_state['if_model'] = if_model
        st.session_state['mlp_model'] = mlp_model

# 4. Visualization & Real-time Simulation
st.subheader("📈 Real-time Grid Monitoring")
col1, col2 = st.columns([3, 1])

# Simulation Toggle
with col2:
    st.markdown("### Injection Console")
    inject_attack = st.button("🔴 INJECT FDIA ATTACK")
    if inject_attack:
        st.error("Cyberattack Simulated! Modifying Bus Voltages...")
        time.sleep(1)
        st.rerun()

# Time-series Plot
with col1:
    fig = px.line(df, x='timestamp', y=['voltage', 'load'], title="Grid Stability Over Time")
    
    # Highlight known attacks from the dataset
    attack_points = df[df['target'] == 1]
    if not attack_points.empty:
        fig.add_trace(go.Scatter(
            x=attack_points['timestamp'], 
            y=attack_points['voltage'],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Ground Truth Attack'
        ))
    
    st.plotly_chart(fig, use_container_width=True)

# 5. Detection Results & Comparisons
if 'models_trained' in st.session_state:
    st.subheader("📊 Model Performance Comparison")
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("### Isolation Forest (Unsupervised)")
        st.metric("Accuracy", f"{st.session_state['if_results']['accuracy']:.2%}")
        st.metric("Precision", f"{st.session_state['if_results']['precision']:.2f}")
        st.metric("Recall", f"{st.session_state['if_results']['recall']:.2f}")

    with c2:
        st.success("### MLP Neural Network (Supervised)")
        st.metric("Accuracy", f"{st.session_state['mlp_results']['accuracy']:.2%}")
        st.metric("Precision", f"{st.session_state['mlp_results']['precision']:.2f}")
        st.metric("Recall", f"{st.session_state['mlp_results']['recall']:.2f}")

    # 6. Intelligent Response Suggestion
    st.divider()
    st.subheader("🛡️ AI Response Suggestions")
    
    # Calculate most recent anomalies
    recent_anomalies = st.session_state['mlp_results']['predictions'][-10:]
    if sum(recent_anomalies) > 2:
        st.error("🚨 HIGH ALERT: Consistent anomalies detected in the last 10 intervals.")
        st.markdown("""
        **Recommended Actions:**
        1. **Isolate Node X:** Disconnect Bus 14 from the transmission network.
        2. **Verify SCADA Signatures:** Check if cryptographic keys for sensor 01 are compromised.
        3. **Switch to Backup Control:** Transition to local manual control mode.
        """)
    else:
        st.success("✅ System Status: Stable. No active cyber threats detected.")

else:
    st.info("👈 Click 'Train Detection Models' in the sidebar to begin analysis.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_synthetic_grid_data(n_samples=1000, attack_prob=0.1):
    """
    Generates synthetic smart grid data (Voltage, Current, Load, Frequency)
    to simulate normal and FDIA (False Data Injection Attack) scenarios.
    """
    np.random.seed(42)
    timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq="h")
    
    # Normal Behavior
    voltage = 230 + np.random.normal(0, 0.5, n_samples)
    current = 10 + 2 * np.sin(np.linspace(0, 4 * np.pi, n_samples)) + np.random.normal(0, 0.2, n_samples)
    load = current * voltage / 1000  # Simplified kW
    frequency = 50 + np.random.normal(0, 0.01, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'voltage': voltage,
        'current': current,
        'load': load,
        'frequency': frequency,
        'target': 0  # 0 for Normal
    })
    
    # Inject Attacks (FDIA)
    attack_indices = np.random.choice(n_samples, size=int(n_samples * attack_prob), replace=False)
    for idx in attack_indices:
        # Simulate FDIA: Maliciously scaling measurements to bypass simple range checks
        attack_type = np.random.choice(['voltage_spike', 'load_drift', 'frequency_shift'])
        if attack_type == 'voltage_spike':
            df.loc[idx, 'voltage'] += np.random.uniform(5, 15)
        elif attack_type == 'load_drift':
            df.loc[idx, 'load'] *= np.random.uniform(1.2, 1.5)
        elif attack_type == 'frequency_shift':
            df.loc[idx, 'frequency'] -= np.random.uniform(0.5, 1.0)
        
        df.loc[idx, 'target'] = 1  # 1 for Attack
        
    return df

def preprocess_data(df):
    """
    Cleans and scales data for AI models.
    """
    # Feature Engineering: Lag features and Rolling Averages
    df['voltage_roll_mean'] = df['voltage'].rolling(window=5).mean().bfill()
    df['load_lag1'] = df['load'].shift(1).bfill()
    
    # Features for model
    features = ['voltage', 'current', 'load', 'frequency', 'voltage_roll_mean', 'load_lag1']
    X = df[features]
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, features, scaler

def prepare_lstm_data(X_scaled, y, window_size=10):
    """
    Reshapes data for LSTM (samples, time_steps, features).
    """
    X_lstm, y_lstm = [], []
    for i in range(window_size, len(X_scaled)):
        X_lstm.append(X_scaled[i-window_size:i])
        y_lstm.append(y.iloc[i])
    return np.array(X_lstm), np.array(y_lstm)

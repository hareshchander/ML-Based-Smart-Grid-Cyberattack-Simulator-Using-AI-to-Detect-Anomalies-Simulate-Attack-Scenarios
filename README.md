# GridShield AI ⚡
### Intelligent Cyberattack Detection and Response Simulator for Smart Grids

GridShield AI is a web-based platform designed for power systems researchers and AI engineers to simulate, detect, and respond to cyber threats in smart grids. It specifically targets **False Data Injection Attacks (FDIA)**, which aim to deceive grid operators by subtly manipulating sensor data.

---

## 🚀 Features
- **Real-time Simulation:** Generate synthetic smart grid telemetry (Voltage, Load, Frequency, Current) with configurable attack probabilities.
- **Dual-Model Detection:**
  - **Isolation Forest:** An unsupervised approach for detecting statistical outliers.
  - **LSTM (Long Short-Term Memory):** A supervised deep learning model for time-series anomaly detection.
- **Interactive Dashboard:** built with Streamlit and Plotly for intuitive visualization of grid stability.
- **Intelligent Response:** Rule-based AI suggestions for mitigating detected attacks.

---

## 📂 Project Structure
- `app.py`: The main Streamlit application and UI logic.
- `data_preprocessing.py`: Handles synthetic data generation, feature engineering, and scaling.
- `model_training.py`: Contains the logic for training and evaluating Isolation Forest and LSTM models.
- `requirements.txt`: List of Python dependencies for deployment.

---

## 🛠️ Installation & Local Setup

1. **Clone or Download** this project to your local machine.
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset Links (Kaggle)
You can use the following datasets for more advanced testing:
1. [Smart Grid FDIA Attack Prediction](https://www.kaggle.com/datasets/afroz00/smart-grid-false-data-injection-attack-prediction)
2. [Power System Intrusion Dataset](https://www.kaggle.com/datasets/silvioquincozes/power-system-intrusion-dataset)

---

## ☁️ Deployment Instructions

### Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Upload `app.py`, `data_preprocessing.py`, `model_training.py`, and `requirements.txt`.
4. The space will automatically build and deploy!

### Streamlit Cloud
1. Push this code to a GitHub repository.
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io/).
3. Select the repository and `app.py` as the main file.
4. Click **Deploy**.

---

## 🧠 How it Works
1. **Data Generation:** The system creates a baseline of "Normal" grid behavior using sine waves for current and steady voltage.
2. **Attack Injection:** It injects "Spikes" or "Drifts" into specific features to mimic malicious sensor tampering.
3. **Detection:**
   - **Isolation Forest** looks for data points that are "few and different".
   - **LSTM** learns the temporal patterns and identifies sequences that deviate from historical norms.
4. **Response:** If anomalies are detected consecutively, the AI suggests isolation protocols for the affected grid nodes.

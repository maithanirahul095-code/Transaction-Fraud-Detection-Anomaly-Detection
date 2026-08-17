import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# Define the class BEFORE load_artifacts()
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

st.set_page_config(page_title="Transaction Anomaly Detector", layout="wide")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("artifacts/scaler.joblib")
    iso_forest = joblib.load("artifacts/isolation_forest.joblib")
    feature_cols = joblib.load("artifacts/model_features.joblib")

# Must match the bottleneck dimension used during training
    bottleneck_dim = 32

    autoencoder = Autoencoder(
    input_dim=len(feature_cols),
    bottleneck_dim=bottleneck_dim
     )

    autoencoder.load_state_dict(
        torch.load("artifacts/autoencoder_weights.pt", map_location="cpu")
    )

    autoencoder.eval()
    threshold = joblib.load("artifacts/final_threshold.joblib")
    freq_maps = joblib.load("artifacts/freq_maps.joblib")
    card_amt_stats = joblib.load("artifacts/card_amt_stats.joblib")
    best_model_name = joblib.load("artifacts/best_model_name.joblib")

    return scaler, iso_forest, autoencoder, threshold, feature_cols, freq_maps, card_amt_stats, best_model_name

scaler, iso_forest, autoencoder, threshold, feature_cols, freq_maps, card_amt_stats, best_model_name = load_artifacts()

st.title('🏦 Transaction anomaly detection')
st.caption(f"Scoring with: {best_model_name} (threshold cost-optimized on this model's raw score)")

uploaded = st.file_uploader('Upload CSV', type='csv')

if uploaded:
    raw = pd.read_csv(uploaded)

    X = raw.reindex(columns=feature_cols, fill_value=0)
    X_scaled = scaler.transform(X)

    # Use ONLY the model the threshold was actually optimized on in load.py —
    # mixing in an ensemble rank score here would silently invalidate the
    # cost-optimal threshold (different scale, different distribution).
    if best_model_name == 'Isolation Forest':
        scores = -iso_forest.score_samples(X_scaled)
    else:
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            recon = autoencoder(X_t)
            scores = ((recon - X_t) ** 2).mean(dim=1).numpy()

    results = raw.copy()
    results['anomaly_score'] = scores
    results['flagged'] = (scores >= threshold).astype(int)

    st.write(results[results['flagged'] == 1].head())
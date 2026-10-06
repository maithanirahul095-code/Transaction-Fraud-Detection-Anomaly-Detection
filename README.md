# Transaction Fraud Detection & Anomaly Detection

An end-to-end **unsupervised transaction anomaly detection system** designed to identify potentially fraudulent transactions when reliable fraud labels may not be available at prediction time.

The project uses the **IEEE-CIS Fraud Detection** transaction and identity datasets and combines transaction-level, behavioral, card-level, device, email, and temporal features with two complementary anomaly detection approaches:

- **Isolation Forest**
- **PyTorch Autoencoder**

The models are evaluated using fraud labels only for **post-training validation**, while model selection and unsupervised hyperparameter decisions avoid using the fraud label.

---

## 🚀 Project Overview

Traditional fraud detection often relies on supervised learning, where historical transactions are labeled as fraudulent or legitimate. However, in real-world financial systems:

- Fraud labels may arrive with significant delays.
- New fraud patterns may not resemble historical fraud.
- Some fraudulent behavior may be completely unseen during model training.

This project therefore treats fraud detection primarily as an **anomaly detection problem**.

The system learns patterns of normal transaction behavior and assigns each transaction an **anomaly score**. Transactions with sufficiently high anomaly scores are flagged for further investigation.

### Pipeline

```text
Transaction + Identity Data
            │
            ▼
      Data Integration
            │
            ▼
      Time-Based Split
            │
            ▼
    Missing-Value Analysis
            │
            ▼
      Data Cleaning
            │
            ▼
      Feature Engineering
            │
            ├───────────────┐
            ▼               ▼
    Isolation Forest    Autoencoder
            │               │
            └───────┬───────┘
                    ▼
              Anomaly Scores
                    │
                    ▼
              Model Evaluation
                    │
                    ▼
          Cost-Based Threshold
                    │
                    ▼
          Fraud / Anomaly Flag
                    │
                    ▼
              Streamlit App

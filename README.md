# Transaction Fraud Detection & Anomaly Detection

An end-to-end **unsupervised transaction anomaly detection system** designed to identify potentially fraudulent transactions when reliable fraud labels may not be available at prediction time.

The project uses the **IEEE-CIS Fraud Detection** transaction and identity datasets and combines transaction-level, behavioral, card-level, device, email, and temporal features with two complementary anomaly detection approaches:

* **Isolation Forest**
* **PyTorch Autoencoder**

The models are evaluated using fraud labels only for **post-training validation**, while model selection and unsupervised hyperparameter decisions avoid using the fraud label.

---

## 🚀 Project Overview

Traditional fraud detection often relies on supervised learning, where historical transactions are labeled as fraudulent or legitimate. However, in real-world financial systems:

* Fraud labels may arrive with significant delays.
* New fraud patterns may not resemble historical fraud.
* Some fraudulent behavior may be completely unseen during model training.

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
```

---

## 📊 Dataset

The project uses two IEEE-CIS Fraud Detection datasets:

* `train_transaction.csv`
* `train_identity.csv`

The transaction and identity tables are joined using `TransactionID`.

### Dataset characteristics

The transaction dataset contains hundreds of thousands of transactions and hundreds of transaction-related features, while the identity dataset provides additional device, browser, and identity-related information.

The project first checks the identity match rate before performing a left join.

```python
train = train_transaction.merge(
    train_identity,
    on='TransactionID',
    how='left'
)
```

---

# 🔄 1. Time-Based Train/Validation Split

A major design decision in this project is **not using a random train/validation split**.

Transactions are sorted chronologically using `TransactionDT`, and the first 80% are used for training while the remaining 20% are used for validation.

```python
train = train.sort_values('TransactionDT').reset_index(drop=True)

split_idx = int(len(train) * 0.8)

train_df = train.iloc[:split_idx].copy()
val_df = train.iloc[split_idx:].copy()
```

### Why time-based splitting?

A random split can allow future transaction behavior to appear in the training data while predicting earlier transactions.

That creates **temporal leakage**.

A chronological split better represents the real production scenario:

```text
Past transactions ─────────────► Future transactions
       TRAIN                         VALIDATION
```

The code explicitly verifies that the training period ends before the validation period begins.

---

# 🧹 2. Data Cleaning & Missing-Value Handling

Fraud datasets often contain substantial missingness.

The project performs a missingness audit and calculates the percentage of missing values for every feature.

### High-missingness feature removal

Features with more than **90% missing values** are removed.

```python
DROP_THRESHOLD = 90.0

cols_to_drop = missing_pct[
    missing_pct > DROP_THRESHOLD
].index.tolist()
```

The threshold is determined using the training data only and the same columns are subsequently removed from validation data.

### Numerical features

Missing numerical values are replaced with the **training-set median**.

```python
numeric_medians = X_train[numeric_cols].median()

X_train[numeric_cols] = X_train[numeric_cols].fillna(
    numeric_medians
)

X_val[numeric_cols] = X_val[numeric_cols].fillna(
    numeric_medians
)
```

### Categorical features

Categorical missing values are converted into an explicit:

```text
"missing"
```

category rather than using the mode.

This preserves the possibility that missingness itself contains behavioral information.

---

# 💰 3. Transaction Amount Transformation

`TransactionAmt` is heavily right-skewed.

A logarithmic transformation is therefore introduced:

```python
X_train['TransactionAmt_log'] = np.log1p(
    X_train['TransactionAmt']
)
```

The same transformation is applied to validation data.

### Why `log1p`?

Instead of allowing very large transactions to dominate the numerical scale, the logarithmic transformation compresses extreme values while preserving their ordering.

---

# ⚙️ 4. Feature Engineering

The project creates behavioral features specifically designed for fraud/anomaly detection.

## 4.1 Transaction Velocity

Transaction frequency is calculated for each `card1` entity over multiple time windows:

* 1 hour
* 24 hours
* 7 days

The transaction timestamp is converted into time windows using `TransactionDT`.

Example features:

```text
card1_txn_count_1h
card1_txn_count_24h
card1_txn_count_7d
```

### Why velocity matters

Fraudulent activity can involve unusually rapid transactions.

For example:

```text
Normal:
Card → 1 transaction over several hours

Potential anomaly:
Card → 8 transactions within 10 minutes
```

Such behavior can be difficult to capture using raw transaction amount alone.

Importantly, the velocity calculation is performed independently within each split to avoid using future information.

---

# 4.2 Amount Deviation from Card History

The project calculates how unusual a transaction amount is compared with the historical behavior of the same card.

For each `card1`, the training data is used to calculate:

```text
mean transaction amount
standard deviation
```

Then the transaction receives a standardized deviation:

```text
amount z-score =
(current amount - card historical mean)
/
card historical standard deviation
```

The feature is called:

```text
amt_zscore_vs_card
```

Training statistics are used for both training and validation, preventing validation information from influencing the feature construction.

### Example

Suppose a card normally makes transactions around:

```text
₹1,000 – ₹2,000
```

and suddenly makes a:

```text
₹50,000 transaction
```

The transaction can receive a large positive deviation score.

---

# 4.3 Card-Device Consistency

The project learns the most common device associated with each card from training data.

A binary feature is then created:

```text
device_mismatch
```

It indicates whether the current device differs from the card's usual device.

This can capture behavior such as:

```text
Card → normally Device A
      ↓
Current transaction → Device B
      ↓
Potential anomaly
```

---

# 4.4 Card-Email Consistency

The same concept is applied to email domains.

For each card, the most common email domain is learned from training data and compared with the current transaction.

Feature:

```text
email_mismatch
```

This can help identify unusual combinations between a payment card and email identity.

---

# 4.5 Frequency Encoding

Categorical variables are converted into numerical features using **frequency encoding**.

For each categorical value:

```text
frequency =
number of occurrences of category
/
total number of observations
```

The frequency mappings are learned using training data only.

Previously unseen validation categories receive:

```text
0
```

frequency.

### Why frequency encoding?

The anomaly detection models require numerical input.

Frequency encoding provides a compact representation of categorical rarity without creating potentially huge one-hot encoded matrices.

It also avoids target encoding, which could leak `isFraud` information into the features.

---

# 🧮 5. Final Feature Matrix

The final model input combines:

* Numerical transaction features
* Frequency-encoded categorical features
* Card amount deviation
* Transaction velocity
* Device mismatch
* Email mismatch

The resulting feature matrix is completely numerical and checked for remaining missing or infinite values.

```text
Raw transaction data
        +
Behavioral features
        +
Categorical frequency features
        +
Consistency features
        ↓
Numerical feature matrix
```

---

# 📏 6. Feature Scaling

A `StandardScaler` is fitted on the training data:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_model)
X_val_scaled = scaler.transform(X_val_model)
```

The validation set is transformed using the **training scaler** rather than fitting another scaler.

This prevents validation statistics from leaking into preprocessing.

---

# 🌲 7. Isolation Forest

The first anomaly detection model is **Isolation Forest**.

```python
IsolationForest(
    n_estimators=200,
    max_samples=256,
    contamination='auto',
    random_state=42,
    n_jobs=-1
)
```

### How Isolation Forest works

Instead of learning what fraud looks like, Isolation Forest tries to isolate unusual observations.

Anomalous observations generally require fewer random partitioning steps to isolate.

Conceptually:

```text
Normal transactions:

        ┌─────────────────────────────┐
        │  • • • • • • • • • • •     │
        │   • • • • • • • • • •      │
        │    • • • • • • • •         │
        └─────────────────────────────┘

Anomaly:

        ┌─────────────────────────────┐
        │  • • • • • • • • • •      •│
        │   • • • • • • • • •         │
        │                        X     │
        └─────────────────────────────┘
```

The project flips the `score_samples` output so that:

```text
Higher score = More anomalous
```

---

# 📉 8. PCA + Isolation Forest

A second Isolation Forest experiment is performed after PCA dimensionality reduction.

PCA retains:

```text
95% explained variance
```

and the reduced representation is passed into another Isolation Forest.

This provides a useful comparison of:

```text
Original feature space
vs.
Reduced feature space
```

without replacing the primary pipeline.

---

# 🧠 9. Autoencoder

The second major anomaly detector is a **PyTorch Autoencoder**.

Architecture:

```text
Input
  │
  ▼
64 neurons
  │
  ▼
32 neurons
  │
  ▼
Bottleneck
  │
  ▼
32 neurons
  │
  ▼
64 neurons
  │
  ▼
Output
```

The bottleneck dimension is tested using:

```text
8
16
32
```

and selected using validation reconstruction loss only, without using fraud labels.

### How the Autoencoder detects anomalies

The Autoencoder learns to reconstruct the input.

For normal transactions:

```text
Input → Encoder → Latent representation → Decoder → Similar reconstruction
```

For unusual transactions:

```text
Input → Encoder → Latent representation → Decoder → Poor reconstruction
```

The reconstruction error becomes the anomaly score:

```python
((recon - X_t) ** 2).mean(dim=1)
```

Higher reconstruction error means the transaction is more unusual.

---

# 📊 10. Evaluation

Although the models are trained without fraud labels, the validation labels are used afterward to determine how effectively the anomaly scores separate fraudulent transactions.

## PR-AUC

The primary evaluation metric is **PR-AUC / Average Precision**.

```python
average_precision_score(
    y_val,
    anomaly_scores
)
```

The project intentionally prioritizes PR-AUC over ROC-AUC because fraud is a highly imbalanced classification problem.

### Why PR-AUC?

With highly imbalanced data, ROC-AUC can appear strong even when the model produces many false positives.

Precision-Recall focuses directly on:

```text
Precision → How many flagged transactions are actually fraud?
Recall    → How much fraud did we detect?
```

This is more aligned with a fraud investigation workflow.

---

# 🔀 11. Isolation Forest + Autoencoder Ensemble

The project also experiments with combining both anomaly detectors.

Their scores are converted into ranks:

```python
iso_rank = rankdata(val_iso_scores)
ae_rank = rankdata(val_ae_scores)

ensemble_scores = (
    iso_rank + ae_rank
) / 2
```

The combined rank score is then evaluated using PR-AUC.

### Why rank-based combination?

Isolation Forest and Autoencoder produce scores on different scales.

Directly averaging their raw scores could therefore be misleading.

Ranking puts both models onto a comparable relative scale.

---

# 🎯 12. Precision@K

The project also calculates:

```text
Precision@50
Precision@100
Precision@500
Precision@1000
```

This represents a realistic fraud investigation scenario.

For example:

> If analysts can manually investigate only the top 100 suspicious transactions, how many of those 100 are actually fraudulent?

This is often more operationally meaningful than simply selecting an arbitrary probability threshold.

---

# 🌳 13. Supervised XGBoost Baseline

A supervised XGBoost model is included strictly as a comparison benchmark.

Unlike the anomaly detection models, XGBoost receives `isFraud` labels during training.

The purpose is not to replace the unsupervised system.

Instead, it answers:

> **How much performance is sacrificed when fraud labels are unavailable?**

This makes the comparison more realistic from a production perspective.

---

# 💰 14. Cost-Based Threshold Optimization

The project does not simply select an arbitrary anomaly-score threshold.

Instead, it defines assumed business costs:

```text
False Positive = $5
False Negative = average fraudulent transaction amount
```

For every candidate threshold:

```text
Total Cost =
False Positives × FP Cost
+
False Negatives × FN Cost
```

The threshold producing the lowest total cost is selected.

```python
optimal_row = cost_df.loc[
    cost_df['total_cost'].idxmin()
]
```

### Why cost-based thresholding?

In fraud detection, the two errors do not have equal consequences.

```text
False Positive
    ↓
Legitimate customer flagged
    ↓
Customer friction + analyst review

False Negative
    ↓
Fraud not detected
    ↓
Potential financial loss
```

Therefore, threshold selection should consider business impact rather than only statistical metrics.

**Important:** the dollar values used here are explicit modeling assumptions, not externally established business costs.

---

# 📉 15. Default vs Optimized Threshold

The project compares the default Isolation Forest decision boundary with the optimized threshold.

It reports:

* False positives
* False negatives
* False-positive reduction

This demonstrates whether cost-aware threshold optimization improves operational behavior.

---

# 💾 16. Model Artifact Management

All required preprocessing and model artifacts are saved for deployment.

```text
artifacts/
│
├── scaler.joblib
├── model_features.joblib
├── freq_maps.joblib
├── card_amt_stats.joblib
├── isolation_forest.joblib
├── autoencoder_weights.pt
├── autoencoder_bottleneck_dim.joblib
├── best_model_name.joblib
└── final_threshold.joblib
```

These artifacts allow the Streamlit application to reproduce the same preprocessing and scoring pipeline used during training.

---

# 🖥️ 17. Streamlit Deployment

The project includes a Streamlit application for inference.

The application loads:

* Scaler
* Isolation Forest
* Autoencoder
* Feature columns
* Frequency maps
* Card statistics
* Selected model
* Optimized threshold

from the saved artifacts.

The user can upload a CSV file:

```text
Upload CSV
    ↓
Feature alignment
    ↓
Scaling
    ↓
Selected anomaly detector
    ↓
Anomaly score
    ↓
Threshold
    ↓
Flagged transactions
```

The application returns the original transaction data together with:

```text
anomaly_score
flagged
```

where:

```text
flagged = 1 → transaction is flagged
flagged = 0 → transaction is not flagged
```

---

# ⚠️ Important Deployment Note

The Streamlit application intentionally uses **only the model whose threshold was optimized during training**.

It does not combine Isolation Forest and Autoencoder scores at inference time because the ensemble has a different score distribution from the individual models. Using the optimized threshold on a different score distribution would invalidate the threshold.

---

# 📁 Project Structure

```text
transaction-fraud-detection/
│
├── load.py
├── app.py
│
├── train_transaction.csv
├── train_identity.csv
│
├── artifacts/
│   ├── scaler.joblib
│   ├── model_features.joblib
│   ├── freq_maps.joblib
│   ├── card_amt_stats.joblib
│   ├── isolation_forest.joblib
│   ├── autoencoder_weights.pt
│   ├── autoencoder_bottleneck_dim.joblib
│   ├── best_model_name.joblib
│   └── final_threshold.joblib
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Isolation Forest
* PCA
* XGBoost

### Deep Learning

* PyTorch
* Autoencoder

### Statistical Evaluation

* SciPy
* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1-score

### Deployment

* Streamlit

### Model Persistence

* Joblib
* PyTorch model weights

---

# ▶️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd transaction-fraud-detection
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost torch streamlit joblib
```

---

# 🚀 Running the Project

## 1. Train the models

Make sure the dataset files are available:

```text
train_transaction.csv
train_identity.csv
```

Run the training pipeline:

```bash
python load.py
```

This performs:

```text
Data loading
→ Data merging
→ Time split
→ Cleaning
→ Feature engineering
→ Scaling
→ Isolation Forest
→ PCA experiment
→ Autoencoder
→ Evaluation
→ Threshold optimization
→ Artifact saving
```

---

## 2. Launch the Streamlit application

After the artifacts have been generated:

```bash
streamlit run app.py
```

Then upload a CSV containing the required model features.

The application returns anomaly scores and flags suspicious transactions.

---

# 🔐 Leakage Prevention

A major focus of the project is preventing information leakage.

### Measures implemented

**1. Chronological split**

Training data always precedes validation data.

**2. Training-only preprocessing**

Median values are calculated using training data only.

**3. Training-only scaling**

`StandardScaler` is fitted only on training data.

**4. Training-only frequency encoding**

Categorical frequency mappings are created only from training data.

**5. Training-only card statistics**

Card transaction amount statistics are learned from training data.

**6. No target encoding**

`isFraud` is deliberately not used to construct model features.

**7. Unsupervised model selection**

Autoencoder bottleneck selection uses reconstruction loss rather than fraud labels.

These design choices make the validation process substantially closer to a real-world future-transaction prediction scenario.

---

# 🎯 Key Project Highlights

* Built an **unsupervised fraud/anomaly detection pipeline**
* Combined transaction and identity information
* Used **time-based validation** instead of random splitting
* Performed systematic missingness analysis
* Engineered **transaction velocity features**
* Created **card-level amount deviation features**
* Added card-device and card-email consistency features
* Used frequency encoding for categorical variables
* Compared **Isolation Forest and Autoencoder**
* Tested PCA-based anomaly detection
* Evaluated using **PR-AUC and Precision@K**
* Added a supervised XGBoost benchmark
* Optimized the final threshold using **business-cost assumptions**
* Saved complete preprocessing/model artifacts
* Deployed inference through **Streamlit**

---

# ⚠️ Limitations

### 1. Velocity features reset at the train/validation boundary

Validation velocity features are calculated independently within the validation timeline.

Therefore, transactions near the beginning of validation do not use historical transactions from the training period.

This is a conservative implementation and slightly underestimates real-world historical context.

### 2. Cost assumptions are illustrative

The false-positive cost and false-negative cost are modeling assumptions:

```text
FP = $5
FN = average fraud transaction amount
```

A production system should replace these with actual business costs.

### 3. Unsupervised detection may miss subtle fraud

Isolation Forest and Autoencoder models identify unusual behavior, but some fraudulent transactions may look very similar to legitimate transactions.

### 4. Dataset-dependent feature behavior

Card, device, email, and other behavioral relationships may behave differently in another financial dataset.

---

# 🔮 Future Improvements

Potential improvements include:

* Online/streaming velocity features
* Rolling historical features across train/production boundaries
* Real-time feature stores
* SHAP-based explainability for supervised models
* More sophisticated categorical encoders
* Graph-based fraud detection
* Entity-level behavioral profiles
* Online model updating
* Concept-drift monitoring
* Human-in-the-loop fraud review
* Calibrated supervised + unsupervised hybrid models
* Production monitoring for false-positive rates and fraud capture

---

# 📌 Interview Summary

### One-line explanation

> **Built an unsupervised transaction anomaly detection system using Isolation Forest and a PyTorch Autoencoder, with leakage-safe time-based validation, behavioral feature engineering, PR-AUC/Precision@K evaluation, and cost-based threshold optimization, then deployed the scoring pipeline using Streamlit.**

### If asked "Why unsupervised?"

> Fraud labels can be delayed or unavailable when new fraud patterns emerge. Anomaly detection allows the system to identify transactions that deviate significantly from learned behavioral patterns without requiring every transaction to have a fraud label.

### If asked "Why time-based split?"

> Fraud detection is inherently temporal. A random split can allow future behavior to leak into training, so I sorted transactions by `TransactionDT` and trained on the earlier 80% while validating on the later 20%.

### If asked "Why PR-AUC?"

> Fraud is highly imbalanced, so ROC-AUC can be overly optimistic. PR-AUC focuses directly on precision and recall for the minority fraud class and is more relevant when investigating flagged transactions.

### If asked "Why both Isolation Forest and Autoencoder?"

> They identify anomalies using different mechanisms. Isolation Forest detects observations that are easy to isolate through random partitioning, while the Autoencoder detects observations that have high reconstruction error. Comparing both gives a more robust view of anomaly detection performance.

### If asked "Why cost-based threshold?"

> The cost of missing fraud is usually much higher than the cost of reviewing a legitimate transaction. Instead of selecting an arbitrary threshold, I explicitly modeled false-positive and false-negative costs and selected the threshold that minimized estimated total cost.

---

## 📜 Disclaimer

This project is intended for **educational and research purposes**. The cost assumptions, threshold, and anomaly scores should not be treated as production financial decision rules without validation using real business costs, operational constraints, and regulatory requirements.

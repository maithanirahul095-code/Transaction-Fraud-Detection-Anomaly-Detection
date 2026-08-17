import pandas as pd
import numpy as np

# ---- Load ----
train_transaction = pd.read_csv('train_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')

print(f"Transactions: {train_transaction.shape}, Identity: {train_identity.shape}")
print(f"Identity match rate: {train_transaction['TransactionID'].isin(train_identity['TransactionID']).mean():.2%}")

# ---- Merge ----
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
print(f"Merged shape: {train.shape}")
print(f"Fraud rate: {train['isFraud'].mean():.4%}")

# ---- Time-based split (NOT random) ----
train = train.sort_values('TransactionDT').reset_index(drop=True)

split_idx = int(len(train) * 0.8)
train_df = train.iloc[:split_idx].copy()
val_df = train.iloc[split_idx:].copy()

print(f"\nTrain: {train_df.shape}, fraud rate: {train_df['isFraud'].mean():.4%}")
print(f"Val:   {val_df.shape}, fraud rate: {val_df['isFraud'].mean():.4%}")

# Sanity check: no time leakage
assert train_df['TransactionDT'].max() <= val_df['TransactionDT'].min()
print("\nNo time leakage confirmed — train ends before val begins")

# ---- Separate labels — set aside, not used in training ----
y_train = train_df['isFraud'].copy()
y_val = val_df['isFraud'].copy()

X_train = train_df.drop(columns=['isFraud'])
X_val = val_df.drop(columns=['isFraud'])

print(f"\nX_train shape: {X_train.shape} (isFraud removed — model never sees this)")





# eda +cleaning


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. MISSINGNESS AUDIT
# ============================================================
missing_pct = X_train.isnull().mean().sort_values(ascending=False) * 100
print("Top 30 columns by missingness:")
print(missing_pct.head(30))

print(f"\nColumns with >90% missing: {(missing_pct > 90).sum()}")
print(f"Columns with >50% missing: {(missing_pct > 50).sum()}")
print(f"Columns with 0% missing:   {(missing_pct == 0).sum()}")

# Visualize missingness by column group (identity vs transaction vs V-columns)
def col_group(c):
    if c.startswith('id_'): return 'identity'
    if c.startswith('V'): return 'V_columns'
    if c.startswith('D'): return 'D_columns'  # time deltas
    if c.startswith('C'): return 'C_columns'  # counts
    if c.startswith('M'): return 'M_columns'  # match flags
    return 'core'

missing_df = pd.DataFrame({'col': missing_pct.index, 'missing_pct': missing_pct.values})
missing_df['group'] = missing_df['col'].apply(col_group)
print("\nMean missingness by column group:")
print(missing_df.groupby('group')['missing_pct'].mean().sort_values(ascending=False))

# ============================================================
# 2. DROP HIGH-MISSINGNESS COLUMNS (threshold decided from train only)
# ============================================================
DROP_THRESHOLD = 90.0
cols_to_drop = missing_pct[missing_pct > DROP_THRESHOLD].index.tolist()
print(f"\nDropping {len(cols_to_drop)} columns with >{DROP_THRESHOLD}% missing")

X_train = X_train.drop(columns=cols_to_drop)
X_val = X_val.drop(columns=cols_to_drop)  # same columns dropped, no separate decision on val

# ============================================================
# 3. SPLIT REMAINING COLUMNS BY TYPE
# ============================================================
# Categorical columns (object/string/category)
categorical_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

# Numeric columns only
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Remove ID/time columns if you don't want to impute them
numeric_cols = [c for c in numeric_cols if c not in ['TransactionID', 'TransactionDT']]

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numeric columns: {len(numeric_cols)}")


# ============================================================
# 4. IMPUTE — fit stats on train, apply to both
# ============================================================
# Numeric: median imputation
numeric_medians = X_train[numeric_cols].median()

X_train[numeric_cols] = X_train[numeric_cols].fillna(numeric_medians)
X_val[numeric_cols] = X_val[numeric_cols].fillna(numeric_medians)  # train medians, not val's own

# Categorical: explicit "missing" category (missingness itself can be signal —
# e.g. no device info could correlate with certain fraud patterns, so we don't
# want to silently impute it away with a mode)
for c in categorical_cols:
    X_train[c] = X_train[c].astype('object').fillna('missing')
    X_val[c] = X_val[c].astype('object').fillna('missing')

# Sanity check — no NaNs left
assert X_train[numeric_cols + categorical_cols].isnull().sum().sum() == 0
assert X_val[numeric_cols + categorical_cols].isnull().sum().sum() == 0
print("\nNo missing values remain in train or val")

# ============================================================
# 5. AMOUNT DISTRIBUTION — log transform (heavily right-skewed)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(X_train['TransactionAmt'], bins=100)
axes[0].set_title('TransactionAmt — raw')
axes[1].hist(np.log1p(X_train['TransactionAmt']), bins=100)
axes[1].set_title('TransactionAmt — log1p')
plt.tight_layout()
plt.show()

X_train['TransactionAmt_log'] = np.log1p(X_train['TransactionAmt'])
X_val['TransactionAmt_log'] = np.log1p(X_val['TransactionAmt'])

# ============================================================
# 6. FRAUD RATE BY SEGMENT (using y_train — labels only for inspection,
#    NOT joined back into X_train/X_val)
# ============================================================
inspect = X_train.copy()
inspect['isFraud'] = y_train.values

for col in ['ProductCD', 'card4', 'card6', 'P_emaildomain']:
    if col in inspect.columns:
        rate = inspect.groupby(col)['isFraud'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(f"\nFraud rate by {col} (top 10):")
        print(rate.head(10))

del inspect  # don't keep a labeled copy floating around by accident

# ============================================================
# 7. CARDINALITY CHECK (matters for encoding choice in Phase 3)
# ============================================================
print("\nCardinality of categorical columns:")
for c in categorical_cols:
    print(f"  {c}: {X_train[c].nunique()} unique values")


# feature enginerring



import pandas as pd
import numpy as np

# ============================================================
# 1. VELOCITY FEATURES — transaction frequency per card/email
# ============================================================
# TransactionDT is seconds from a reference point — convert to hours/days for windows
for df in [X_train, X_val]:
    df['Transaction_hour_idx'] = df['TransactionDT'] // 3600
    df['Transaction_day_idx'] = df['TransactionDT'] // (3600 * 24)

# IMPORTANT: velocity features must only look BACKWARD in time (no future leakage
# within a transaction's own window), and train/val windows must not cross into
# each other's data. We compute counts using expanding/rolling windows per entity,
# sorted by time, entirely within each split.

def add_velocity_features(df, entity_col, time_col='TransactionDT', windows_sec={'1h': 3600, '24h': 86400, '7d': 604800}):
    df = df.sort_values(time_col).copy()
    for label, window in windows_sec.items():
        counts = []
        # Group by entity, use rolling count within time window — O(n log n) via merge_asof
        # would be ideal at scale; for clarity, groupby + rolling on a time-indexed series:
        for entity, group in df.groupby(entity_col):
            g = group.set_index(pd.to_timedelta(group[time_col], unit='s'))
            roll_count = g[time_col].rolling(f'{window}s').count()
            counts.append(pd.Series(roll_count.values, index=group.index))
        col_name = f'{entity_col}_txn_count_{label}'
        df[col_name] = pd.concat(counts).sort_index()
    return df

# card1 is the most granular card identifier available
X_train = add_velocity_features(X_train, 'card1')
X_val = add_velocity_features(X_val, 'card1')  # computed independently within val's own timeline
                                                  # (val doesn't see train's transaction history —
                                                  # this slightly underestimates val velocity for the
                                                  # first 7 days of val, which is an honest limitation
                                                  # to note in the README rather than hide)

print("Velocity features added:", [c for c in X_train.columns if 'txn_count' in c])
print(X_train[[c for c in X_train.columns if 'txn_count' in c]].describe())

# ============================================================
# 2. AMOUNT DEVIATION — z-score of this txn vs card's historical mean/std
# ============================================================
# Fit per-card mean/std on TRAIN ONLY, apply to both train and val
card_amt_stats = X_train.groupby('card1')['TransactionAmt_log'].agg(['mean', 'std']).rename(
    columns={'mean': 'card_amt_mean', 'std': 'card_amt_std'}
)
# Cards with only 1 transaction get std=NaN — fill with global std as fallback
global_std = X_train['TransactionAmt_log'].std()
card_amt_stats['card_amt_std'] = card_amt_stats['card_amt_std'].fillna(global_std)

for df in [X_train, X_val]:
    df_stats = df.merge(card_amt_stats, on='card1', how='left')
    # Cards seen in val but NOT in train → fallback to global stats (realistic: new card = no history)
    df_stats['card_amt_mean'] = df_stats['card_amt_mean'].fillna(X_train['TransactionAmt_log'].mean())
    df_stats['card_amt_std'] = df_stats['card_amt_std'].fillna(global_std)
    df_stats['amt_zscore_vs_card'] = (
        (df_stats['TransactionAmt_log'] - df_stats['card_amt_mean']) / df_stats['card_amt_std'].replace(0, global_std)
    )
    if df is X_train:
        X_train = df_stats
    else:
        X_val = df_stats

print("\nAmount z-score vs card history — train stats:")
print(X_train['amt_zscore_vs_card'].describe())

# ============================================================
# 3. CARD-DEVICE / CARD-EMAIL CONSISTENCY — does this pairing normally occur?
# ============================================================
# Build card1 -> most common DeviceType and P_emaildomain mapping from TRAIN ONLY
if 'DeviceType' in X_train.columns:
    card_device_mode = X_train.groupby('card1')['DeviceType'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'missing'
    ).rename('card_usual_device')

    for df_name, df in [('train', X_train), ('val', X_val)]:
        merged = df.merge(card_device_mode, on='card1', how='left')
        merged['card_usual_device'] = merged['card_usual_device'].fillna('unknown_card')
        merged['device_mismatch'] = (
            (merged['DeviceType'] != merged['card_usual_device']) &
            (merged['card_usual_device'] != 'unknown_card')
        ).astype(int)
        if df_name == 'train':
            X_train = merged
        else:
            X_val = merged

    print(f"\nDevice mismatch rate — train: {X_train['device_mismatch'].mean():.2%}")
    print(f"Device mismatch rate — val:   {X_val['device_mismatch'].mean():.2%}")
else:
    print("\nDeviceType not in columns (may have been dropped in Phase 2 missingness cut) — skipping device mismatch feature")

# Same pattern for email domain consistency
if 'P_emaildomain' in X_train.columns:
    card_email_mode = X_train.groupby('card1')['P_emaildomain'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'missing'
    ).rename('card_usual_email')

    for df_name, df in [('train', X_train), ('val', X_val)]:
        merged = df.merge(card_email_mode, on='card1', how='left')
        merged['card_usual_email'] = merged['card_usual_email'].fillna('unknown_card')
        merged['email_mismatch'] = (
            (merged['P_emaildomain'] != merged['card_usual_email']) &
            (merged['card_usual_email'] != 'unknown_card')
        ).astype(int)
        if df_name == 'train':
            X_train = merged
        else:
            X_val = merged

    print(f"\nEmail mismatch rate — train: {X_train['email_mismatch'].mean():.2%}")
    print(f"Email mismatch rate — val:   {X_val['email_mismatch'].mean():.2%}")

# ============================================================
# 4. ENCODING CATEGORICALS — frequency encoding (fit on train only)
#    (target encoding is skipped deliberately — using isFraud to encode
#    categoricals is leakage even if labels aren't used in the model itself,
#    since it smuggles label information into the features)
# ============================================================
freq_maps = {}
for c in categorical_cols:
    if c in X_train.columns:
        freq = X_train[c].value_counts(normalize=True)
        freq_maps[c] = freq
        X_train[c + '_freq'] = X_train[c].map(freq)
        # unseen categories in val → 0 (never seen in train = rare by definition)
        X_val[c + '_freq'] = X_val[c].map(freq).fillna(0)

print(f"\nFrequency-encoded {len(freq_maps)} categorical columns")

# ============================================================
# 5. FINAL FEATURE MATRIX — numeric only, ready for isolation forest / autoencoder
# ============================================================
model_features = (
    numeric_cols +
    [c + '_freq' for c in categorical_cols if c in X_train.columns] +
    ['amt_zscore_vs_card'] +
    [c for c in X_train.columns if 'txn_count' in c] +
    (['device_mismatch'] if 'device_mismatch' in X_train.columns else []) +
    (['email_mismatch'] if 'email_mismatch' in X_train.columns else [])
)
model_features = [c for c in model_features if c in X_train.columns]

X_train_model = X_train[model_features].copy()
X_val_model = X_val[model_features].copy()

# Guard against any leftover inf/nan from the z-score division
X_train_model = X_train_model.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val_model = X_val_model.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"\nFinal model feature matrix: train {X_train_model.shape}, val {X_val_model.shape}")
assert X_train_model.isnull().sum().sum() == 0
assert X_val_model.isnull().sum().sum() == 0    


# modeling


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 0. SCALE — fit on train only
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_model)
X_val_scaled = scaler.transform(X_val_model)  # train's scaler applied to val, not refit

print(f"Train scaled: {X_train_scaled.shape}, Val scaled: {X_val_scaled.shape}")

# ============================================================
# 1. ISOLATION FOREST
# ============================================================
# Hyperparameter choices justified WITHOUT touching isFraud:
# - n_estimators=200: original Liu et al. paper shows path-length averages
#   converge well before 200 trees; going higher gives diminishing returns
#   at real cost (this dataset is large)
# - max_samples=256: paper's own recommendation — small subsamples work
#   better for isolation forest than large ones (large samples make normal
#   points harder to isolate, hurting anomaly separation)
# - contamination='auto': do NOT set this to the real fraud rate — that
#   would be using isFraud indirectly. 'auto' uses the original paper's
#   offset heuristic instead.
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples=256,
    contamination='auto',
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

# score_samples: higher = more normal. Flip sign so higher = more anomalous
# (more intuitive for everything downstream)
train_iso_scores = -iso_forest.score_samples(X_train_scaled)
val_iso_scores = -iso_forest.score_samples(X_val_scaled)

print("\nIsolation Forest anomaly score distribution (train, unlabeled):")
print(pd.Series(train_iso_scores).describe())




# ============================================================
# PCA + ISOLATION FOREST (dimensionality reduction test)
# ============================================================
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

print(f"\nPCA: reduced from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} components")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

iso_forest_pca = IsolationForest(
    n_estimators=200, max_samples=256, contamination='auto',
    random_state=42, n_jobs=-1
)
iso_forest_pca.fit(X_train_pca)
val_iso_pca_scores = -iso_forest_pca.score_samples(X_val_pca)

# ============================================================
# 2. AUTOENCODER (PyTorch)
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train_scaled.shape[1]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, bottleneck_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)

# Bottleneck size selection: compare a few sizes using VALIDATION
# RECONSTRUCTION LOSS ONLY (no labels involved — this is legitimate
# unsupervised model selection, distinct from tuning against isFraud)
def train_autoencoder(bottleneck_dim, epochs=30, lr=1e-3, patience=5):
    model = Autoencoder(input_dim, bottleneck_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon = model(X_train_t)
        loss = criterion(recon, X_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss

results = {}
for bdim in [8, 16, 32]:
    model, val_loss = train_autoencoder(bottleneck_dim=bdim)
    results[bdim] = val_loss
    print(f"Bottleneck dim {bdim}: val reconstruction loss = {val_loss:.5f}")

best_bdim = min(results, key=results.get)
print(f"\nSelected bottleneck dim: {best_bdim} (lowest val reconstruction loss, chosen without labels)")

autoencoder, _ = train_autoencoder(bottleneck_dim=best_bdim, epochs=100, patience=10)

# ============================================================
# 3. RECONSTRUCTION ERROR AS ANOMALY SCORE
# ============================================================
autoencoder.eval()
with torch.no_grad():
    train_recon = autoencoder(X_train_t)
    val_recon = autoencoder(X_val_t)
    train_ae_scores = ((train_recon - X_train_t) ** 2).mean(dim=1).cpu().numpy()
    val_ae_scores = ((val_recon - X_val_t) ** 2).mean(dim=1).cpu().numpy()

print("\nAutoencoder reconstruction error distribution (train, unlabeled):")
print(pd.Series(train_ae_scores).describe())

# ============================================================
# 4. SANITY CHECK ONLY — peek at label separation, don't use it to pick anything
#    (This is diagnostic, purely to confirm both models learned *something*
#    before moving to Phase 5's proper evaluation. No thresholds or
#    hyperparameters are touched based on this.)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, scores, name in zip(axes, [val_iso_scores, val_ae_scores], ['Isolation Forest', 'Autoencoder']):
    ax.hist(scores[y_val == 0], bins=50, alpha=0.5, label='Normal', density=True)
    ax.hist(scores[y_val == 1], bins=50, alpha=0.5, label='Fraud', density=True)
    ax.set_title(f'{name} — anomaly score by class (val)')
    ax.legend()
plt.tight_layout()
plt.show()









# eval

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

# ============================================================
# 1. PR-AUC, NOT ROC-AUC — imbalance makes ROC-AUC misleading here
#    (state this reasoning explicitly if asked in interviews)
# ============================================================
iso_pr_auc = average_precision_score(y_val, val_iso_scores)
ae_pr_auc = average_precision_score(y_val, val_ae_scores)

print(f"Isolation Forest — PR-AUC: {iso_pr_auc:.4f}")
print(f"Autoencoder      — PR-AUC: {ae_pr_auc:.4f}")

iso_pca_pr_auc = average_precision_score(y_val, val_iso_pca_scores)
print(f"Isolation Forest (PCA)      — PR-AUC: {iso_pca_pr_auc:.4f}")

# ============================================================
# ENSEMBLE — rank-based combination of Isolation Forest + Autoencoder
# ============================================================
from scipy.stats import rankdata

iso_rank = rankdata(val_iso_scores)
ae_rank = rankdata(val_ae_scores)
ensemble_scores = (iso_rank + ae_rank) / 2
ensemble_pr_auc = average_precision_score(y_val, ensemble_scores)

print(f"Ensemble (IF+AE ranks) — PR-AUC: {ensemble_pr_auc:.4f}")

# Report ROC-AUC too, but only as a secondary/contextual number
print(f"\n(For context) Isolation Forest ROC-AUC: {roc_auc_score(y_val, val_iso_scores):.4f}")
print(f"(For context) Autoencoder ROC-AUC:       {roc_auc_score(y_val, val_ae_scores):.4f}")

# ============================================================
# 2. PRECISION@K — realistic for a fraud team with limited review capacity
#    (bank framing: "analysts can only manually review top-K flagged txns/day")
# ============================================================
def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]  # highest anomaly score first
    top_k_idx = order[:k]
    return y_true.values[top_k_idx].mean() if hasattr(y_true, 'values') else y_true[top_k_idx].mean()

for k in [50, 100, 500, 1000]:
    p_iso = precision_at_k(y_val, val_iso_scores, k)
    p_ae = precision_at_k(y_val, val_ae_scores, k)
    print(f"Precision@{k:>4} — Isolation Forest: {p_iso:.3f} | Autoencoder: {p_ae:.3f}")

# ============================================================
# 3. PR CURVES — visual comparison
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
for scores, name in [(val_iso_scores, 'Isolation Forest'), (val_ae_scores, 'Autoencoder')]:
    precision, recall, _ = precision_recall_curve(y_val, scores)
    ap = average_precision_score(y_val, scores)
    ax.plot(recall, precision, label=f'{name} (AP={ap:.3f})')
baseline = y_val.mean()
ax.axhline(baseline, linestyle='--', color='gray', label=f'Random baseline ({baseline:.4f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall: Unsupervised Anomaly Detectors')
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 4. SUPERVISED XGBOOST BASELINE — quantifies the cost of going unsupervised
#    (trained WITH labels, purely for comparison — not part of the deployed model)
# ============================================================
from xgboost import XGBClassifier

xgb_baseline = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='aucpr', random_state=42
)
xgb_baseline.fit(X_train_model, y_train)
xgb_scores = xgb_baseline.predict_proba(X_val_model)[:, 1]
xgb_pr_auc = average_precision_score(y_val, xgb_scores)

print(f"\n--- Comparison ---")
print(f"Isolation Forest (unsupervised) PR-AUC: {iso_pr_auc:.4f}")
print(f"Autoencoder (unsupervised)      PR-AUC: {ae_pr_auc:.4f}")
print(f"XGBoost (supervised, labels used) PR-AUC: {xgb_pr_auc:.4f}")
print("\nGap between supervised and unsupervised = the real cost of not having")
print("timely fraud labels in production. This is expected and is the story,")
print("not a failure of the unsupervised approach.")

# ============================================================
# 5. COST-BASED THRESHOLD SELECTION
# ============================================================
# Assumed costs (state these explicitly — they're a modeling assumption, not fact):
#   False Positive (flag a normal txn) = customer friction + analyst review time  -> ~$5
#   False Negative (miss actual fraud) = average fraud loss on this dataset       -> ~mean fraud TransactionAmt
avg_fraud_amt = train_df.loc[y_train == 1, 'TransactionAmt'].mean()
COST_FP = 5
COST_FN = avg_fraud_amt

print(f"\nAssumed cost per false positive: ${COST_FP}")
print(f"Assumed cost per false negative (avg fraud amount): ${COST_FN:.2f}")

best_model_scores = val_ae_scores if ae_pr_auc > iso_pr_auc else val_iso_scores
best_model_name = 'Autoencoder' if ae_pr_auc > iso_pr_auc else 'Isolation Forest'
print(f"\nBest unsupervised model: {best_model_name}")

thresholds = np.percentile(best_model_scores, np.arange(90, 100, 0.5))
cost_results = []
for t in thresholds:
    preds = (best_model_scores >= t).astype(int)
    fp = ((preds == 1) & (y_val == 0)).sum()
    fn = ((preds == 0) & (y_val == 1)).sum()
    total_cost = fp * COST_FP + fn * COST_FN
    cost_results.append({'threshold': t, 'fp': fp, 'fn': fn, 'total_cost': total_cost})

cost_df = pd.DataFrame(cost_results)
optimal_row = cost_df.loc[cost_df['total_cost'].idxmin()]
print(f"\nOptimal threshold (min cost): {optimal_row['threshold']:.4f}")
print(f"  → FP: {int(optimal_row['fp'])}, FN: {int(optimal_row['fn'])}, Total cost: ${optimal_row['total_cost']:.2f}")

FINAL_THRESHOLD = optimal_row['threshold']

final_preds = (best_model_scores >= FINAL_THRESHOLD).astype(int)
print(f"\nAt this threshold:")
print(f"  Precision: {precision_score(y_val, final_preds):.4f}")
print(f"  Recall:    {recall_score(y_val, final_preds):.4f}")
print(f"  F1:        {f1_score(y_val, final_preds):.4f}")


# ============================================================
# 6. BASELINE THRESHOLD COMPARISON (default contamination='auto' cutoff)
# ============================================================
# iso_forest.predict() returns -1 (anomaly) / 1 (normal) using the model's
# own default decision boundary — this is the "before optimization" baseline
default_preds_raw = iso_forest.predict(X_val_scaled)
default_preds = (default_preds_raw == -1).astype(int)  # 1 = flagged as fraud

fp_default = ((default_preds == 1) & (y_val == 0)).sum()
fn_default = ((default_preds == 0) & (y_val == 1)).sum()

fp_optimized = int(optimal_row['fp'])
fp_reduction_pct = (fp_default - fp_optimized) / fp_default * 100

print(f"\n--- Baseline vs Optimized Threshold ---")
print(f"Default threshold  → FP: {fp_default}, FN: {fn_default}")
print(f"Optimized threshold → FP: {fp_optimized}, FN: {int(optimal_row['fn'])}")
print(f"False positive reduction: {fp_reduction_pct:.1f}%")


import os
import joblib
import torch

os.makedirs('artifacts', exist_ok=True)

joblib.dump(scaler, 'artifacts/scaler.joblib')
joblib.dump(model_features, 'artifacts/model_features.joblib')
joblib.dump(freq_maps, 'artifacts/freq_maps.joblib')
joblib.dump(card_amt_stats, 'artifacts/card_amt_stats.joblib')
joblib.dump(iso_forest, 'artifacts/isolation_forest.joblib')

# Autoencoder artifacts
torch.save(autoencoder.state_dict(), 'artifacts/autoencoder_weights.pt')
joblib.dump(best_bdim, 'artifacts/autoencoder_bottleneck_dim.joblib')

# Model selection + threshold
joblib.dump(best_model_name, 'artifacts/best_model_name.joblib')
joblib.dump(FINAL_THRESHOLD, 'artifacts/final_threshold.joblib')

print('Artifacts saved successfully.')
print(f'Best model: {best_model_name}')
print(f'Autoencoder bottleneck: {best_bdim}')
print(f'Final threshold: {FINAL_THRESHOLD:.6f}')
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Configurare 
DATA_PATH    = 'telecomunicatii.csv'
TARGET_COL   = 'payment_delay'
THRESHOLD    = 0.35
TEST_SIZE    = 0.20
VAL_SIZE     = 0.25
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    'account_length', 'number_vmail_messages',
    'total_day_minutes', 'total_day_calls', 'total_day_charge',
    'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
    'total_night_minutes', 'total_night_calls', 'total_night_charge',
    'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
    'number_customer_service_calls'
]

CATEGORICAL_FEATURES = ['state', 'area_code', 'international_plan', 'voice_mail_plan']

# ── Date ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
y  = df[TARGET_COL].map({'yes': 1, 'no': 0})
X  = df.drop(columns=[TARGET_COL])

# ── Split: 60% train | 20% validation | 20% test ─────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train:      {len(X_train)} obs ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation: {len(X_val)} obs ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test:       {len(X_test)} obs ({len(X_test)/len(X)*100:.0f}%)")
print(f"Rată delay  train: {y_train.mean():.3f} | val: {y_val.mean():.3f} | test: {y_test.mean():.3f}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=RANDOM_STATE
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), NUMERIC_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# ── Antrenare ─────────────────────────────────────────────────────────────────
print("\nAntrenăm modelul Random Forest...")
model_pipeline.fit(X_train, y_train)

# ── Evaluare ──────────────────────────────────────────────────────────────────
def evaluate(y_true, y_proba, threshold, label):
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\n{'=' * 55}")
    print(f"EVALUARE {label} — prag {threshold}")
    print(f"{'=' * 55}")
    print(f"Acuratețe: {accuracy_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}")
    print(classification_report(y_true, y_pred,
          target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))

y_val_proba  = model_pipeline.predict_proba(X_val)[:, 1]
y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]

evaluate(y_val,  y_val_proba,  THRESHOLD, "VALIDATION")
evaluate(y_test, y_test_proba, THRESHOLD, "TEST")

# ── Salvare ───────────────────────────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telecom_rf_model.pkl')
joblib.dump({'pipeline': model_pipeline, 'threshold': THRESHOLD}, save_path)
print(f"\nModel salvat: {save_path}")
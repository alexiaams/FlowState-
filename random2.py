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

df = pd.read_csv('telecomunicatii.csv')
target_col = 'payment_delay'

y = df[target_col].map({'yes': 1, 'no': 0})
X = df.drop(columns=[target_col])

numeric_features = [
    'account_length', 'number_vmail_messages',
    'total_day_minutes', 'total_day_calls', 'total_day_charge',
    'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
    'total_night_minutes', 'total_night_calls', 'total_night_charge',
    'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
    'number_customer_service_calls'
]
categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']

# ── Split: 60% train | 20% validation | 20% test ─────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Train:      {len(X_train)} obs ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation: {len(X_val)} obs ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test:       {len(X_test)} obs ({len(X_test)/len(X)*100:.0f}%)")
print(f"\nRată delay — train: {y_train.mean():.3f} | val: {y_val.mean():.3f} | test: {y_test.mean():.3f}")

# ── Preprocessor + model ──────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ))
])

# ── Antrenare ─────────────────────────────────────────────────────────────────
print("\nAntrenăm modelul Random Forest...")
model_pipeline.fit(X_train, y_train)

# ── Probabilități pe validation ───────────────────────────────────────────────
y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]

# ── Alegem pragul optim pe VALIDATION ────────────────────────────────────────
print("\n" + "=" * 55)
print("OPTIMIZARE PRAG PE VALIDATION SET")
print("=" * 55)
print(f"{'Prag':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>10}")
print("-" * 55)

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] # best 0.35
results = []

for t in thresholds:
    y_pred_t = (y_val_proba >= t).astype(int)
    report = classification_report(y_val, y_pred_t, output_dict=True, zero_division=0)
    prec = report['1']['precision']
    rec  = report['1']['recall']
    f1   = report['1']['f1-score']
    auc  = roc_auc_score(y_val, y_val_proba)
    results.append({'threshold': t, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc})
    print(f"{t:<8.2f} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f} {auc:>10.3f}")

# Pragul optim = cel mai bun F1 pe clasa minoritară
best = max(results, key=lambda x: x['f1'])
best_threshold = best['threshold']
print(f"\nPrag optim ales (max F1 clasa 'yes'): {best_threshold}")

# ── Evaluare pe VALIDATION cu pragul optim ───────────────────────────────────
print("\n" + "=" * 55)
print(f"EVALUARE VALIDATION — prag {best_threshold}")
print("=" * 55)
y_val_pred_opt = (y_val_proba >= best_threshold).astype(int)
print(f"Acuratețe: {accuracy_score(y_val, y_val_pred_opt):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_val_proba):.3f}")
print(classification_report(y_val, y_val_pred_opt,
      target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))

# ── Evaluare finală pe TEST cu pragul ales pe validation ─────────────────────
print("=" * 55)
print(f"EVALUARE TEST — prag {best_threshold} (rezultat final)")
print("=" * 55)
y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]
y_test_pred_opt = (y_test_proba >= best_threshold).astype(int)
print(f"Acuratețe: {accuracy_score(y_test, y_test_pred_opt):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.3f}")
print(classification_report(y_test, y_test_pred_opt,
      target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))

# ── Salvare model + prag ──────────────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telecom_rf_model.pkl')
joblib.dump({'pipeline': model_pipeline, 'threshold': best_threshold}, save_path)
print(f"Model + prag salvate: {save_path}")
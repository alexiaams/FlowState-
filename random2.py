import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix
)

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

# ── Matrice de corelație (doar variabilele numerice) ──────────────────────────
print(f"\n{'=' * 55}")
print("MATRICE DE CORELAȚIE – VARIABILE NUMERICE")
print(f"{'=' * 55}")

corr_matrix = X[NUMERIC_FEATURES].corr()

# Afișăm perechile cu corelație > 0.85 (foarte corelate)
CORR_THRESHOLD = 0.85
print(f"\nPerechi cu |corelație| > {CORR_THRESHOLD}:\n")

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > CORR_THRESHOLD:
            feat_i = corr_matrix.columns[i]
            feat_j = corr_matrix.columns[j]
            high_corr_pairs.append((feat_i, feat_j, corr_val))
            print(f"  {feat_i:30s} ↔ {feat_j:30s}  r = {corr_val:.4f}")

if not high_corr_pairs:
    print("  (nicio pereche cu corelație > threshold)")

# Eliminăm features-urile redundante (păstrăm prima din fiecare pereche)
# Logica: din fiecare pereche corelată, eliminăm variabila "charge" (derivată din minutes)
features_to_drop = set()
for feat_i, feat_j, corr_val in high_corr_pairs:
    # Preferăm să eliminăm "charge" deoarece e calculat direct din "minutes"
    # (charge = minutes * tarif_per_minut, deci e redundant)
    if 'charge' in feat_j:
        features_to_drop.add(feat_j)
    elif 'charge' in feat_i:
        features_to_drop.add(feat_i)
    else:
        # Dacă niciuna nu e "charge", eliminăm a doua
        features_to_drop.add(feat_j)

if features_to_drop:
    print(f"\n🗑️  Features eliminate (redundante): {sorted(features_to_drop)}")
    NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f not in features_to_drop]
    print(f"✅ Features numerice rămase ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
else:
    print("\n✅ Nicio variabilă eliminată – nu există corelații excesive.")

# Afișăm matricea completă (rezumat)
print(f"\nMatricea de corelație completă (features numerice rămase):")
corr_final = X[NUMERIC_FEATURES].corr()
print(corr_final.round(2).to_string())

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

# ── Funcție evaluare + matrice de confuzie ────────────────────────────────────
def evaluate(y_true, y_proba, threshold, label):
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n{'=' * 55}")
    print(f"EVALUARE {label} — prag {threshold}")
    print(f"{'=' * 55}")
    print(f"Acuratețe: {accuracy_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}")
    print(classification_report(y_true, y_pred,
          target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))

    # ── Matrice de confuzie (terminal) ─────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nMATRICEA DE CONFUZIE — {label}")
    print(f"┌{'─'*26}┬{'─'*14}┬{'─'*14}┐")
    print(f"│{'':26s}│{'Prezis: 0':^14s}│{'Prezis: 1':^14s}│")
    print(f"├{'─'*26}┼{'─'*14}┼{'─'*14}┤")
    print(f"│{'Actual: 0 (fără întârziere)':26s}│{'TN = ' + str(tn):^14s}│{'FP = ' + str(fp):^14s}│")
    print(f"│{'Actual: 1 (cu întârziere)':26s}│{'FN = ' + str(fn):^14s}│{'TP = ' + str(tp):^14s}│")
    print(f"└{'─'*26}┴{'─'*14}┴{'─'*14}┘")
    print(f"  Sensibilitate (Recall cls 1): {tp / (tp + fn):.3f}")
    print(f"  Specificitate (Recall cls 0): {tn / (tn + fp):.3f}")
    if (tp + fp) > 0:
        print(f"  Precizie cls 1:               {tp / (tp + fp):.3f}")

    return y_pred

# ── Evaluare ──────────────────────────────────────────────────────────────────
y_val_proba  = model_pipeline.predict_proba(X_val)[:, 1]
y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]

y_val_pred  = evaluate(y_val,  y_val_proba,  THRESHOLD, "VALIDATION")
y_test_pred = evaluate(y_test, y_test_proba, THRESHOLD, "TEST")

# ── Feature Importances ───────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("IMPORTANȚA VARIABILELOR (Feature Importances)")
print(f"{'=' * 55}")

feature_names = []
feature_names.extend(NUMERIC_FEATURES)
ohe = model_pipeline.named_steps['preprocessor'].transformers_[1][1]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
feature_names.extend(cat_feature_names)

importances = model_pipeline.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 20 variabile care influențează payment_delay:\n")
print(importance_df.head(20).to_string(index=False))

print(f"\n{'─' * 55}")
print("IMPORTANȚA AGREGATĂ PE GRUPE DE VARIABILE:")
print(f"{'─' * 55}")

groups = {
    'Day usage (minutes+calls+charge)': ['total_day_minutes', 'total_day_calls', 'total_day_charge'],
    'Evening usage': ['total_eve_minutes', 'total_eve_calls', 'total_eve_charge'],
    'Night usage': ['total_night_minutes', 'total_night_calls', 'total_night_charge'],
    'International usage': ['total_intl_minutes', 'total_intl_calls', 'total_intl_charge'],
    'Customer service calls': ['number_customer_service_calls'],
    'Account length': ['account_length'],
    'Voicemail': ['number_vmail_messages'],
    'State (toate one-hot)': [f for f in feature_names if f.startswith('state_')],
    'Area code (one-hot)': [f for f in feature_names if f.startswith('area_code_')],
    'International plan': [f for f in feature_names if f.startswith('international_plan_')],
    'Voice mail plan': [f for f in feature_names if f.startswith('voice_mail_plan_')],
}

group_importance = {}
for group_name, features in groups.items():
    mask = importance_df['Feature'].isin(features)
    group_importance[group_name] = importance_df.loc[mask, 'Importance'].sum()

group_df = pd.DataFrame({
    'Grup variabile': list(group_importance.keys()),
    'Importanță totală': list(group_importance.values())
}).sort_values('Importanță totală', ascending=False)

print()
print(group_df.to_string(index=False))
print(f"\nInterpretare: Cu cât importanța e mai mare, cu atât variabila")
print(f"contribuie mai mult la decizia modelului privind payment_delay.")

# ── Rata de delay per stat ────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("RATA DE PAYMENT DELAY PER STAT")
print(f"{'=' * 55}")

# Calculam pe intregul dataset (df original)
state_stats = df.groupby('state').agg(
    total_clienti=('payment_delay', 'count'),
    clienti_cu_delay=('payment_delay', lambda x: (x == 'yes').sum())
).reset_index()

state_stats['rata_delay_%'] = (state_stats['clienti_cu_delay'] / state_stats['total_clienti'] * 100).round(1)
state_stats = state_stats.sort_values('rata_delay_%', ascending=False)

print(f"\n{'Stat':<6} {'Total clienti':>14} {'Cu delay':>10} {'Rata delay %':>13}")
print(f"{'─' * 45}")
for _, row in state_stats.iterrows():
    print(f"{row['state']:<6} {row['total_clienti']:>14} {row['clienti_cu_delay']:>10} {row['rata_delay_%']:>12.1f}%")

print(f"\n{'─' * 45}")
print(f"{'TOTAL':<6} {state_stats['total_clienti'].sum():>14} {state_stats['clienti_cu_delay'].sum():>10} "
      f"{state_stats['clienti_cu_delay'].sum() / state_stats['total_clienti'].sum() * 100:>12.1f}%")

# Top 5 si Bottom 5
print(f"\n🔴 Top 5 state cu cel mai mare risc de delay:")
for _, row in state_stats.head(5).iterrows():
    print(f"   {row['state']}: {row['rata_delay_%']:.1f}% ({row['clienti_cu_delay']}/{row['total_clienti']} clienti)")

print(f"\n🟢 Top 5 state cu cel mai mic risc de delay:")
for _, row in state_stats.tail(5).iterrows():
    print(f"   {row['state']}: {row['rata_delay_%']:.1f}% ({row['clienti_cu_delay']}/{row['total_clienti']} clienti)")

# ── Salvare ───────────────────────────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telecom_rf_model.pkl')
joblib.dump({'pipeline': model_pipeline, 'threshold': THRESHOLD}, save_path)
print(f"\nModel salvat: {save_path}")
import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ── Configurare ───────────────────────────────────────────────────────────────
DATA_PATH    = 'telecomunicatii.csv'
TARGET_COL   = 'payment_delay'
THRESHOLD    = 0.35
TEST_SIZE    = 0.20
VAL_SIZE     = 0.25
RANDOM_STATE = 42

# ── Coloane eliminate pe baza Feature Importance ──────────────────────────────
# total_*_charge  → redundante (charge = minutes × tarif fix, informație dublată)
# area_code       → importanță 0.016, informație deja capturată de 'state'
# number_vmail_messages → importanță 0.020, irelevant pentru riscul de plată
COLOANE_ELIMINATE = [
    'total_day_charge',
    'total_eve_charge',
    'total_night_charge',
    'total_intl_charge',
    'area_code',
    'number_vmail_messages',
]

# ── Features rămase după eliminare ───────────────────────────────────────────
NUMERIC_FEATURES = [
    'account_length',
    'total_day_minutes',   'total_day_calls',
    'total_eve_minutes',   'total_eve_calls',
    'total_night_minutes', 'total_night_calls',
    'total_intl_minutes',  'total_intl_calls',
    'number_customer_service_calls'
]

CATEGORICAL_FEATURES = ['state', 'international_plan', 'voice_mail_plan']

# ── Date ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Eliminăm coloanele redundante/irelevante identificate anterior
df = df.drop(columns=COLOANE_ELIMINATE)
print(f"Coloane eliminate: {COLOANE_ELIMINATE}")
print(f"Features rămase:   {df.shape[1] - 1} (din {df.shape[1] - 1 + len(COLOANE_ELIMINATE)} originale)\n")

y = df[TARGET_COL].map({'yes': 1, 'no': 0})
X = df.drop(columns=[TARGET_COL])

# ── Matrice de Corelație ──────────────────────────────────────────────────────
print("\nGenerăm matricea de corelație...")

# Includem și target-ul în matrice ca să vedem care features corelează cu el
numeric_cols_for_corr = NUMERIC_FEATURES + [TARGET_COL]
df_corr = df[numeric_cols_for_corr].copy()
df_corr[TARGET_COL] = y  # folosim valorile 0/1 în loc de yes/no

corr_matrix = df_corr.corr()

# Grafic heatmap
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,          # afișează valorile în fiecare celulă
    fmt='.2f',           # 2 zecimale
    cmap='RdYlGn',       # roșu (corelație negativă) → galben (0) → verde (pozitivă)
    center=0,            # centrul scalei de culori la 0
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    annot_kws={'size': 8}
)
ax.set_title('Matricea de Corelație — Features Numerice + Target (payment_delay)',
             fontsize=13, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.close()
print("✅ Matricea de corelație salvată ca 'correlation_matrix.png'")

# Afișăm și în text corelațiile cu target-ul (cele mai relevante)
print("\nCorelații cu target-ul 'payment_delay' (sortat descrescător după valoare absolută):")
target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
for feat, val in target_corr.items():
    bar = '█' * int(abs(val) * 30)
    direction = '↑' if val > 0 else '↓'
    print(f"  {feat:<35} {val:+.3f} {direction} {bar}")


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

# ── Pipeline ──────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), NUMERIC_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

# ── Antrenare ─────────────────────────────────────────────────────────────────
print("\nAntrenăm modelul Random Forest (features selectate)...")
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

# ── Feature Importances ───────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print("IMPORTANȚA VARIABILELOR (Feature Importances)")
print(f"{'=' * 55}")

ohe = model_pipeline.named_steps['preprocessor'].transformers_[1][1]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
feature_names = NUMERIC_FEATURES + cat_feature_names

importances = model_pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 15 variabile care influențează payment_delay:\n")
print(importance_df.head(15).to_string(index=False))

# ── Importanță agregată pe grupe ──────────────────────────────────────────────
print(f"\n{'─' * 55}")
print("IMPORTANȚA AGREGATĂ PE GRUPE DE VARIABILE:")
print(f"{'─' * 55}")

groups = {
    'Day usage (minutes+calls)':   ['total_day_minutes', 'total_day_calls'],
    'Evening usage':               ['total_eve_minutes', 'total_eve_calls'],
    'Night usage':                 ['total_night_minutes', 'total_night_calls'],
    'International usage':         ['total_intl_minutes', 'total_intl_calls'],
    'Customer service calls':      ['number_customer_service_calls'],
    'Account length':              ['account_length'],
    'State (toate one-hot)':       [f for f in feature_names if f.startswith('state_')],
    'International plan':          [f for f in feature_names if f.startswith('international_plan_')],
    'Voice mail plan':             [f for f in feature_names if f.startswith('voice_mail_plan_')],
}

group_importance = {
    name: importance_df.loc[importance_df['Feature'].isin(feats), 'Importance'].sum()
    for name, feats in groups.items()
}

group_df = pd.DataFrame({
    'Grup variabile': list(group_importance.keys()),
    'Importanță totală': list(group_importance.values())
}).sort_values('Importanță totală', ascending=False)

print()
print(group_df.to_string(index=False))

# ── Grafic Feature Importance ─────────────────────────────────────────────────
importances_series = pd.Series(
    model_pipeline.named_steps['classifier'].feature_importances_,
    index=feature_names
).sort_values(ascending=False).head(15)

importances_series.plot(kind='barh', figsize=(10, 6), color='steelblue')
plt.title('Top 15 Feature Importances - Random Forest (features selectate)')
plt.xlabel('Importanță')
plt.tight_layout()
plt.savefig('feature_importance_v2.png')
print("\n✅ Graficul actualizat salvat ca 'feature_importance_v2.png'")

# ── Salvare model ─────────────────────────────────────────────────────────────
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telecom_rf_model.pkl')
joblib.dump({'pipeline': model_pipeline, 'threshold': THRESHOLD}, save_path)
print(f"✅ Model salvat: {save_path}")
print("\nGata! Compară rezultatele cu versiunea anterioară (F1 clasa 1: 0.83).")

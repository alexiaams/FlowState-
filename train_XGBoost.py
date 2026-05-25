
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
# ==========================================
# 1. ÎNCĂRCAREA DATELOR
# ==========================================
df=pd.read_csv('Telecomunicatii.csv')
# ==========================================
# 2. PREGĂTIREA DATELOR
# ==========================================
target_col = 'payment_delay'
# Transformăm 'yes' / 'no' în 1 și 0
y = df[target_col].map({'yes': 1, 'no': 0})
X = df.drop(columns=[target_col])
# Coloane numerice și categorice (conform cerinței temei)
numeric_features = [
    'account_length', 'number_vmail_messages',
    'total_day_minutes', 'total_day_calls', 'total_day_charge',
    'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
    'total_night_minutes', 'total_night_calls', 'total_night_charge',
    'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
    'number_customer_service_calls'
]
categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
# ==========================================
# 3. SPLIT 70% TRAIN / 15% VALIDATION / 15% TEST
# ==========================================
print("Împărțim datele: 70% Train | 15% Validation | 15% Test...")
# Pasul 1: Scoatem setul de Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
# Pasul 2: Din restul de 85%, scoatem Validarea (~17.6% din rest = 15% din total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)
print(f"  Train:      {X_train.shape[0]} exemple")
print(f"  Validation: {X_val.shape[0]} exemple")
print(f"  Test:       {X_test.shape[0]} exemple\n")
# ==========================================
# 4. PIPELINE CU XGBOOST
# ==========================================
# Preprocesare identică (OneHotEncoder pentru text, StandardScaler pentru numere)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
# XGBClassifier:
#   n_estimators=200     -> 200 de arbori secvenţiali (mai mulţi = mai precis, mai lent)
#   max_depth=4          -> Adâncimea maximă a fiecărui arbore (limităm ca să evităm overfitting)
#   learning_rate=0.1    -> Cât de "agresiv" corectează fiecare arbore greşelile precedentului
#   scale_pos_weight     -> Echivalentul lui class_weight='balanced' din Random Forest
#                          = nr_clienți_OK / nr_clienți_cu_risc (compensăm dezechilibrul de clase)
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0         # Suprimăm log-urile interne ale XGBoost
    ))
])
# ==========================================
# 5. ANTRENARE
# ==========================================
print("Antrenăm modelul XGBoost... ⚡⚡⚡")
model_pipeline.fit(X_train, y_train)
print("Antrenare completă!\n")
# ==========================================
# 6. GĂSIREA PRAGULUI OPTIM PE VALIDARE
# ==========================================
print("=" * 50)
print("🔍 CĂUTAREA PRAGULUI OPTIM pe Setul de VALIDARE")
print("=" * 50)
# Obținem probabilitățile brute (nu predicțiile binare)
# predict_proba returnează [prob_clasa_0, prob_clasa_1] pentru fiecare client
y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]
# Testăm toate pragurile între 0.1 și 0.9 cu pas de 0.01
# și îl alegem pe cel care maximizează F1-score pe clasa cu risc (1)

best_threshold = 0.5
best_f1 = 0.0
thresholds = np.arange(0.1, 0.9, 0.01)
for threshold in thresholds:
    # Aplicăm pragul manual: dacă probabilitatea > threshold => predicție 1 (risc)
    y_val_pred_thresh = (y_val_proba >= threshold).astype(int)
    # Calculăm F1-score DOAR pentru clasa cu risc (clasa 1) - pos_label=1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, y_val_pred_thresh, pos_label=1, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print(f"Prag implicit (default): 0.50")
print(f"Prag optim găsit:        {best_threshold:.2f}")
print(f"F1-score pe Validare cu pragul optim: {best_f1:.3f}\n")
# Evaluare cu pragul default (0.5) pe validare
y_val_pred_default = (y_val_proba >= 0.5).astype(int)
print("[VALIDARE] Rezultate cu prag DEFAULT (0.50):")
print(classification_report(y_val, y_val_pred_default,
    target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))
# Evaluare cu pragul optim pe validare
y_val_pred_optimal = (y_val_proba >= best_threshold).astype(int)
print(f"[VALIDARE] Rezultate cu prag OPTIM ({best_threshold:.2f}):")
print(classification_report(y_val, y_val_pred_optimal,
    target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))
# ==========================================
# 7. EVALUARE FINALĂ PE TEST cu pragul optim
# ==========================================
print("=" * 50)
print("📋 EVALUARE FINALĂ pe Setul de TEST")
print("=" * 50)
# Obținem probabilitățile brute pentru setul de test
y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]
# Aplicăm pragul optim găsit pe validare
y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
print(f"[TEST] Rezultate cu prag OPTIM ({best_threshold:.2f}):")
print(f"Acuratețe Test: {accuracy_score(y_test, y_test_pred_optimal):.2f}\n")
print(classification_report(y_test, y_test_pred_optimal,
    target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))
# ==========================================
# 8. SALVAREA MODELULUI
# ==========================================
save_path = 'telecom_xgb_model.pkl'
joblib.dump(model_pipeline, save_path)
# Salvăm și pragul optim (API-ul va avea nevoie de el!)
threshold_path = 'optimal_threshold.pkl'
joblib.dump(best_threshold, threshold_path)



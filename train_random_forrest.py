import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv('telecomunicatii.csv')
target_col='payment_delay'

y=df[target_col].map({'yes': 1, 'no':0})

X=df.drop(columns=[target_col])

numeric_features= [
        'account_length', 'number_vmail_messages', 
        'total_day_minutes', 'total_day_calls', 'total_day_charge',
        'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
        'total_night_minutes', 'total_night_calls', 'total_night_charge',
        'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
        'number_customer_service_calls'
    ]

categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

print("Antrenăm modelul Random Forest... ")
model_pipeline.fit(X_train, y_train)
print("Evaluăm modelul pe datele de test...")
y_pred = model_pipeline.predict(X_test)

print(f"\n Acuratețe generală: {accuracy_score(y_test, y_pred):.2f}\n")
print(" Raport de clasificare:")
print(classification_report(y_test, y_pred, target_names=['Fără întârziere (0)', 'Cu întârziere (1)']))

save_path = os.path.join(os.path.dirname(__file__), 'telecom_rf_model.pkl')
joblib.dump(model_pipeline, save_path)

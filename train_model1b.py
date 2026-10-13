import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

bands = ['B02_center','B03_center','B04_center',
         'B08_center','B11_center','B12_center']

# ── Train Turbidity Model ─────────────────────────────────────────────────────
print("=== TURBIDITY MODEL ===")
df_turb = pd.read_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\turbidity_clean.csv')

X_turb = df_turb[bands].values
y_turb = df_turb['turbidity_FNU'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_turb, y_turb, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples...")
model_turb = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)
model_turb.fit(X_train, y_train)

y_pred = model_turb.predict(X_test)
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} FNU")

# Feature importance
for band, imp in zip(bands, model_turb.feature_importances_):
    print(f"  {band}: {imp:.3f}")

joblib.dump(model_turb, 'C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\model_turbidity.pkl')
print("Saved: model_turbidity.pkl")

# ── Train Chlorophyll Model ───────────────────────────────────────────────────
print("\n=== CHLOROPHYLL MODEL ===")
df_chl = pd.read_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\chlorophyll_clean.csv')

X_chl = df_chl[bands].values
y_chl = df_chl['chlorophyll_ugL'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_chl, y_chl, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples...")
model_chl = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)
model_chl.fit(X_train, y_train)

y_pred = model_chl.predict(X_test)
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} µg/L")

for band, imp in zip(bands, model_chl.feature_importances_):
    print(f"  {band}: {imp:.3f}")

joblib.dump(model_chl, 'C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\model_chlorophyll.pkl')
print("Saved: model_chlorophyll.pkl")
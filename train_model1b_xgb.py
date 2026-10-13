import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

bands = ['B02_center','B03_center','B04_center',
         'B08_center','B11_center','B12_center']

base = 'C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\'

# ── Turbidity Model ───────────────────────────────────────────────────────────
print("=== TURBIDITY MODEL (XGBoost) ===")
df_turb = pd.read_csv(base + 'turbidity_clean.csv')

X_turb = df_turb[bands].values
y_turb = df_turb['turbidity_FNU'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_turb, y_turb, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples...")
model_turb = XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist'
)
model_turb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

y_pred = model_turb.predict(X_test)
print(f"\nR²:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} FNU")

for band, imp in zip(bands, model_turb.feature_importances_):
    print(f"  {band}: {imp:.3f}")

joblib.dump(model_turb, base + 'model_turbidity_xgb.pkl')
print("Saved: model_turbidity_xgb.pkl")

# ── Chlorophyll Model ─────────────────────────────────────────────────────────
print("\n=== CHLOROPHYLL MODEL (XGBoost) ===")
df_chl = pd.read_csv(base + 'chlorophyll_clean.csv')

X_chl = df_chl[bands].values
y_chl = df_chl['chlorophyll_ugL'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_chl, y_chl, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples...")
model_chl = XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    tree_method='hist'
)
model_chl.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

y_pred = model_chl.predict(X_test)
print(f"\nR²:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} µg/L")

for band, imp in zip(bands, model_chl.feature_importances_):
    print(f"  {band}: {imp:.3f}")

joblib.dump(model_chl, base + 'model_chlorophyll_xgb.pkl')
print("Saved: model_chlorophyll_xgb.pkl")

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n=== RANDOM FOREST vs XGBOOST ===")
print("Turbidity  RF:  R²=0.7469  MAE=7.5352 FNU")
print("Chlorophyll RF: R²=0.8999  MAE=2.3728 µg/L")
print("(compare with XGBoost scores above)")
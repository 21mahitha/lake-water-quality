import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# ── Load Dataset 2 (108 months) ───────────────────────────────────────────────
df = pd.read_excel('HussainSagar_WaterQuality_WithClimate.xlsx',
                   sheet_name='Monthly Data')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── Select features ───────────────────────────────────────────────────────────
feature_cols = [
    'NDCI_chlorophyll_final',
    'turbidity_proxy_final',
    'rainfall_mm',
    'avg_temp_C',
    'months_since_2016_01'
]

target_cols = [
    'NDCI_chlorophyll_final',
    'turbidity_proxy_final'
]

# Drop monsoon gap rows (no water quality data)
df_clean = df[df['data_status'] != 'missing_monsoon_gap'].copy()
df_clean = df_clean[feature_cols].dropna().reset_index(drop=True)

print(f"\nRows after dropping monsoon gaps: {len(df_clean)}")

# ── Scale features ────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_clean[feature_cols])
joblib.dump(scaler, 'C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\lstm_scaler.pkl')

# ── Create sliding window sequences ──────────────────────────────────────────
WINDOW = 6   # look back 6 months
TARGET_STEPS = 1  # predict 1 month ahead

X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i-WINDOW:i])          # 6 months of all features
    y.append(scaled[i, :2])               # next month's chlorophyll + turbidity

X = np.array(X)
y = np.array(y)

print(f"\nSequences created: {len(X)}")
print(f"X shape: {X.shape}  (samples, timesteps, features)")
print(f"y shape: {y.shape}  (samples, targets)")

# ── Train/test split ──────────────────────────────────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTrain sequences: {len(X_train)}")
print(f"Test sequences:  {len(X_test)}")

# ── Build LSTM ────────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(64, input_shape=(WINDOW, len(feature_cols)), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(2)   # turbidity + chlorophyll
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# Inverse scale for real values
# Create dummy arrays to inverse transform
def inverse_targets(scaled_vals, scaler, n_features=5):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, 0] = scaled_vals[:, 0]  # chlorophyll
    dummy[:, 1] = scaled_vals[:, 1]  # turbidity
    return scaler.inverse_transform(dummy)[:, :2]

y_test_real = inverse_targets(y_test, scaler)
y_pred_real = inverse_targets(y_pred, scaler)

print(f"\n=== LSTM RESULTS ===")
print(f"Chlorophyll R²:  {r2_score(y_test_real[:,0], y_pred_real[:,0]):.4f}")
print(f"Chlorophyll MAE: {mean_absolute_error(y_test_real[:,0], y_pred_real[:,0]):.4f}")
print(f"Turbidity R²:    {r2_score(y_test_real[:,1], y_pred_real[:,1]):.4f}")
print(f"Turbidity MAE:   {mean_absolute_error(y_test_real[:,1], y_pred_real[:,1]):.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
model.save('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\lstm_forecaster.keras')
print("\nSaved: lstm_forecaster.keras")

# ── Plot training loss ────────────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training History')
plt.legend()
plt.savefig('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\lstm_training.png')
plt.show()
print("Saved: lstm_training.png")
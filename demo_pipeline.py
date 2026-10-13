"""
Lake Water Quality Monitor + Forecaster
Hussain Sagar, Hyderabad
End-to-end demo pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import joblib
import rasterio
import warnings
import os
from skimage.filters import threshold_otsu
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Base path ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  LAKE WATER QUALITY MONITOR + FORECASTER")
print("  Case Study: Hussain Sagar, Hyderabad")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# STAGE 1 — WATER BODY SEGMENTATION (NDWI + Otsu)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 1] Water Body Segmentation — NDWI + Otsu Threshold")

with rasterio.open(os.path.join(BASE, 'hussain_sagar_s2.tif')) as src:
    bands = src.read().astype(np.float32)

B2, B3, B4, B8, B11, B12 = bands[0], bands[1], bands[2], bands[3], bands[4], bands[5]

ndwi = np.where((B3 + B8) == 0, 0, (B3 - B8) / (B3 + B8))
threshold = threshold_otsu(ndwi)
water_mask = ndwi > threshold

print(f"  NDWI range:       {ndwi.min():.3f} to {ndwi.max():.3f}")
print(f"  Otsu threshold:   {threshold:.3f}")
print(f"  Water pixels:     {water_mask.sum():,}")
print(f"  Water coverage:   {water_mask.sum()/water_mask.size*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# STAGE 2 — WATER QUALITY PREDICTION (XGBoost)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 2] Water Quality Prediction — XGBoost Models")

water_pixels = np.stack([
    B2[water_mask], B3[water_mask], B4[water_mask],
    B8[water_mask], B11[water_mask], B12[water_mask]
], axis=1)

model_turb = joblib.load(os.path.join(BASE, 'model_turbidity_xgb.pkl'))
model_chl  = joblib.load(os.path.join(BASE, 'model_chlorophyll_xgb.pkl'))

turb_pred = model_turb.predict(water_pixels)
chl_pred  = model_chl.predict(water_pixels)

turb_mean   = turb_pred.mean()
turb_median = np.median(turb_pred)
chl_mean    = chl_pred.mean()
chl_median  = np.median(chl_pred)

print(f"  Turbidity  — Mean: {turb_mean:.2f} FNU  | Median: {turb_median:.2f} FNU")
print(f"  Chlorophyll — Mean: {chl_mean:.2f} µg/L | Median: {chl_median:.2f} µg/L")

# Water quality labels
def turb_label(v):
    if v < 10:   return "LOW", "green"
    elif v < 50: return "MODERATE", "orange"
    else:        return "HIGH", "red"

def chl_label(v):
    if v < 10:   return "LOW", "green"
    elif v < 30: return "MODERATE", "orange"
    else:        return "HIGH", "red"

turb_status, turb_color = turb_label(turb_mean)
chl_status,  chl_color  = chl_label(chl_mean)

print(f"  Turbidity status:   {turb_status}")
print(f"  Chlorophyll status: {chl_status}")

# Spatial maps
turb_map = np.full(water_mask.shape, np.nan)
turb_map[water_mask] = turb_pred

chl_map = np.full(water_mask.shape, np.nan)
chl_map[water_mask] = chl_pred

# ══════════════════════════════════════════════════════════════
# STAGE 3 — TIME SERIES + FORECASTING (LSTM)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 3] Time Series Analysis + LSTM Forecasting")

df = pd.read_excel(os.path.join(BASE, 'HussainSagar_WaterQuality_WithClimate.xlsx'),
                   sheet_name='Monthly Data')

feature_cols = ['NDCI_chlorophyll_final', 'turbidity_proxy_final',
                'rainfall_mm', 'avg_temp_C', 'months_since_2016_01']

df_clean = df[df['data_status'] != 'missing_monsoon_gap'].copy()
df_clean = df_clean[['year_month'] + feature_cols].dropna().reset_index(drop=True)

print(f"  Observed months:  {len(df_clean)}")
print(f"  Date range:       {df_clean['year_month'].iloc[0]} to {df_clean['year_month'].iloc[-1]}")

# Load LSTM and forecast
scaler = joblib.load(os.path.join(BASE, 'lstm_scaler.pkl'))
lstm_model = tf.keras.models.load_model(os.path.join(BASE, 'lstm_forecaster.keras'))

scaled = scaler.transform(df_clean[feature_cols])
WINDOW = 6

# Predict on test portion
X_all = []
for i in range(WINDOW, len(scaled)):
    X_all.append(scaled[i-WINDOW:i])
X_all = np.array(X_all)

y_all_pred = lstm_model.predict(X_all, verbose=0)

# Inverse transform predictions
def inverse_targets(scaled_vals, scaler, n_features=5):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, 0] = scaled_vals[:, 0]
    dummy[:, 1] = scaled_vals[:, 1]
    return scaler.inverse_transform(dummy)[:, :2]

preds_real = inverse_targets(y_all_pred, scaler)
pred_dates = df_clean['year_month'].iloc[WINDOW:].values

# Forecast future — extend 36 months ahead
last_window = scaled[-WINDOW:].copy()
future_preds = []
future_months = []

last_month_num = df_clean['months_since_2016_01'].iloc[-1]
last_rain_mean = df_clean['rainfall_mm'].mean()
last_temp_mean = df_clean['avg_temp_C'].mean()

for step in range(1, 37):
    x_input = last_window.reshape(1, WINDOW, len(feature_cols))
    pred = lstm_model.predict(x_input, verbose=0)[0]
    future_preds.append(pred)
    future_months.append(f"M+{step}")

    # Roll window
    new_row = last_window[-1].copy()
    new_row[0] = pred[0]
    new_row[1] = pred[1]
    new_row[4] = last_month_num + step
    last_window = np.vstack([last_window[1:], new_row])

future_preds = np.array(future_preds)
future_real = inverse_targets(future_preds, scaler)

print(f"  LSTM forecast horizon: 36 months")
print(f"  Predicted turbidity in 3 years:   {future_real[-1,1]:.4f} (proxy index)")
print(f"  Predicted chlorophyll in 3 years: {future_real[-1,0]:.4f} (proxy index)")

# ══════════════════════════════════════════════════════════════
# VISUALIZATION — Full Dashboard
# ══════════════════════════════════════════════════════════════
print("\n[PLOTTING] Generating dashboard...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0f1117')

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

title_color    = '#ffffff'
label_color    = '#cccccc'
grid_color     = '#2a2a3e'
accent_blue    = '#4fc3f7'
accent_green   = '#66bb6a'
accent_orange  = '#ffa726'
accent_red     = '#ef5350'
bg_panel       = '#1a1a2e'

# ── Title ─────────────────────────────────────────────────────
fig.text(0.5, 0.97, 'Lake Water Quality Monitor + Forecaster',
         ha='center', va='top', fontsize=20, fontweight='bold',
         color=title_color)
fig.text(0.5, 0.945, 'Hussain Sagar, Hyderabad  |  Sentinel-2 Analysis  |  2016–2024',
         ha='center', va='top', fontsize=11, color=label_color)

# ── Panel 1: RGB Image ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(bg_panel)
rgb = np.stack([B4, B3, B2], axis=-1)
rgb = np.clip(rgb / 3000, 0, 1)
ax1.imshow(rgb)
ax1.set_title('Sentinel-2 RGB Image', color=title_color, fontsize=10, pad=8)
ax1.axis('off')

# ── Panel 2: NDWI map ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(bg_panel)
im2 = ax2.imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
ax2.set_title(f'NDWI Water Index\n(threshold={threshold:.2f})', 
              color=title_color, fontsize=10, pad=8)
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=label_color)

# ── Panel 3: Turbidity map ────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(bg_panel)
im3 = ax3.imshow(turb_map, cmap='YlOrRd', vmin=0, vmax=100)
ax3.set_title(f'Turbidity Map\nMean: {turb_mean:.1f} FNU ({turb_status})',
              color=title_color, fontsize=10, pad=8)
ax3.axis('off')
cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cb3.set_label('FNU', color=label_color, fontsize=8)

# ── Panel 4: Chlorophyll map ──────────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
ax4.set_facecolor(bg_panel)
im4 = ax4.imshow(chl_map, cmap='YlGn', vmin=0, vmax=30)
ax4.set_title(f'Chlorophyll-a Map\nMean: {chl_mean:.1f} µg/L ({chl_status})',
              color=title_color, fontsize=10, pad=8)
ax4.axis('off')
cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cb4.set_label('µg/L', color=label_color, fontsize=8)

# ── Panel 5: Turbidity time series ───────────────────────────
ax5 = fig.add_subplot(gs[1, :2])
ax5.set_facecolor(bg_panel)

observed_turb = df_clean['turbidity_proxy_final'].values
observed_dates = df_clean['year_month'].values
x_obs = range(len(observed_dates))
x_pred = range(WINDOW, len(observed_dates))
x_future = range(len(observed_dates), len(observed_dates) + 36)

ax5.plot(x_obs, observed_turb, color=accent_blue, linewidth=1.5,
         label='Observed', alpha=0.9)
ax5.plot(x_pred, preds_real[:, 1], color=accent_orange, linewidth=1.5,
         linestyle='--', label='LSTM fitted', alpha=0.9)
ax5.plot(x_future, future_real[:, 1], color=accent_red, linewidth=2,
         linestyle=':', label='Forecast (3yr)', alpha=0.9)
ax5.axvline(x=len(observed_dates)-1, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5)
ax5.text(len(observed_dates)+1, ax5.get_ylim()[0] if ax5.get_ylim()[0] else 0,
         'Forecast →', color='white', fontsize=7, alpha=0.7)

ax5.set_title('Turbidity Trend + Forecast', color=title_color, fontsize=10, pad=8)
ax5.set_ylabel('Turbidity Proxy Index', color=label_color, fontsize=8)
ax5.tick_params(colors=label_color, labelsize=7)
ax5.grid(color=grid_color, linewidth=0.5)
ax5.set_facecolor(bg_panel)
ax5.spines[:].set_color(grid_color)
ax5.legend(fontsize=7, facecolor=bg_panel, labelcolor=label_color)

# Add monsoon shading
for i, (_, row) in enumerate(df.iterrows()):
    if row.get('is_monsoon') == True:
        idx = df_clean.index.get_loc(i) if i in df_clean.index else None

# ── Panel 6: Chlorophyll time series ─────────────────────────
ax6 = fig.add_subplot(gs[1, 2:])
ax6.set_facecolor(bg_panel)

observed_chl = df_clean['NDCI_chlorophyll_final'].values

ax6.plot(x_obs, observed_chl, color=accent_green, linewidth=1.5,
         label='Observed', alpha=0.9)
ax6.plot(x_pred, preds_real[:, 0], color=accent_orange, linewidth=1.5,
         linestyle='--', label='LSTM fitted', alpha=0.9)
ax6.plot(x_future, future_real[:, 0], color=accent_red, linewidth=2,
         linestyle=':', label='Forecast (3yr)', alpha=0.9)
ax6.axvline(x=len(observed_dates)-1, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5)

ax6.set_title('Chlorophyll-a Trend + Forecast', color=title_color, fontsize=10, pad=8)
ax6.set_ylabel('NDCI Chlorophyll Index', color=label_color, fontsize=8)
ax6.tick_params(colors=label_color, labelsize=7)
ax6.grid(color=grid_color, linewidth=0.5)
ax6.set_facecolor(bg_panel)
ax6.spines[:].set_color(grid_color)
ax6.legend(fontsize=7, facecolor=bg_panel, labelcolor=label_color)

# ── Panel 7: Climate data ─────────────────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
ax7.set_facecolor(bg_panel)

df_all = df[['year_month', 'rainfall_mm', 'avg_temp_C']].dropna()
x_climate = range(len(df_all))

ax7_twin = ax7.twinx()
ax7.bar(x_climate, df_all['rainfall_mm'], color=accent_blue,
        alpha=0.6, label='Rainfall (mm)', width=0.8)
ax7_twin.plot(x_climate, df_all['avg_temp_C'], color=accent_red,
              linewidth=1.5, label='Temp (°C)')

ax7.set_title('Hyderabad Climate Data (2016–2024)', color=title_color, fontsize=10, pad=8)
ax7.set_ylabel('Rainfall (mm)', color=accent_blue, fontsize=8)
ax7_twin.set_ylabel('Temperature (°C)', color=accent_red, fontsize=8)
ax7.tick_params(colors=label_color, labelsize=7)
ax7_twin.tick_params(colors=label_color, labelsize=7)
ax7.set_facecolor(bg_panel)
ax7.spines[:].set_color(grid_color)
ax7_twin.spines[:].set_color(grid_color)
ax7.legend(loc='upper left', fontsize=7, facecolor=bg_panel, labelcolor=label_color)
ax7_twin.legend(loc='upper right', fontsize=7, facecolor=bg_panel, labelcolor=label_color)

# ── Panel 8: Model performance summary ───────────────────────
ax8 = fig.add_subplot(gs[2, 2:])
ax8.set_facecolor(bg_panel)
ax8.axis('off')

summary_data = [
    ['Component', 'Model', 'Metric', 'Score'],
    ['Segmentation', 'NDWI + Otsu', 'Water Coverage', '27.5%'],
    ['Turbidity', 'XGBoost', 'R²', '0.8334'],
    ['Turbidity', 'XGBoost', 'MAE', '6.85 FNU'],
    ['Chlorophyll', 'XGBoost', 'R²', '0.9159'],
    ['Chlorophyll', 'XGBoost', 'MAE', '2.22 µg/L'],
    ['Forecaster', 'LSTM', 'Chl R²', '0.6065'],
    ['Forecaster', 'LSTM', 'Turb R²', '0.4805'],
    ['', '', '', ''],
    ['Current State', '', '', ''],
    ['Turbidity', '', f'{turb_mean:.1f} FNU', turb_status],
    ['Chlorophyll', '', f'{chl_mean:.1f} µg/L', chl_status],
]

table = ax8.table(
    cellText=summary_data[1:],
    colLabels=summary_data[0],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(8)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor(bg_panel)
    cell.set_edgecolor(grid_color)
    cell.set_text_props(color=label_color)
    if row == 0:
        cell.set_facecolor('#16213e')
        cell.set_text_props(color=title_color, fontweight='bold')

ax8.set_title('Model Performance Summary', color=title_color, fontsize=10, pad=8)

# ── Save ──────────────────────────────────────────────────────
out_path = os.path.join(BASE, 'dashboard.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print(f"\n✅ Dashboard saved to: {out_path}")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"  Stage 1 — Segmentation:  {water_mask.sum():,} water pixels detected")
print(f"  Stage 2 — Quality:       Turbidity={turb_mean:.1f} FNU, Chl={chl_mean:.1f} µg/L")
print(f"  Stage 3 — Forecast:      36 months ahead generated")
print("=" * 60)
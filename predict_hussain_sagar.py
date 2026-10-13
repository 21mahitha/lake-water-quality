import numpy as np
import rasterio
from skimage.filters import threshold_otsu
import joblib

# ── Load image ────────────────────────────────────────────────────────────────
with rasterio.open('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\hussain_sagar_s2.tif') as src:
    bands = src.read().astype(np.float32)

# Band order: B2, B3, B4, B8, B11, B12
B2  = bands[0]
B3  = bands[1]
B4  = bands[2]
B8  = bands[3]
B11 = bands[4]
B12 = bands[5]

# ── NDWI water mask ───────────────────────────────────────────────────────────
ndwi = np.where((B3 + B8) == 0, 0, (B3 - B8) / (B3 + B8))
threshold = threshold_otsu(ndwi)
water_mask = ndwi > threshold

print(f"Water pixels found: {water_mask.sum()}")

# ── Extract water pixels (6 bands) ───────────────────────────────────────────
water_pixels = np.stack([
    B2[water_mask],
    B3[water_mask],
    B4[water_mask],
    B8[water_mask],
    B11[water_mask],
    B12[water_mask]
], axis=1)

print(f"Water pixel array shape: {water_pixels.shape}")

# ── Load models ───────────────────────────────────────────────────────────────
model_turb = joblib.load('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\model_turbidity.pkl')
model_chl  = joblib.load('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\model_chlorophyll.pkl')

# ── Predict ───────────────────────────────────────────────────────────────────
turb_predictions = model_turb.predict(water_pixels)
chl_predictions  = model_chl.predict(water_pixels)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"HUSSAIN SAGAR — CURRENT WATER QUALITY")
print(f"{'='*40}")
print(f"Turbidity:")
print(f"  Mean:   {turb_predictions.mean():.2f} FNU")
print(f"  Median: {np.median(turb_predictions):.2f} FNU")
print(f"  Range:  {turb_predictions.min():.2f} – {turb_predictions.max():.2f} FNU")
print(f"\nChlorophyll-a:")
print(f"  Mean:   {chl_predictions.mean():.2f} µg/L")
print(f"  Median: {np.median(chl_predictions):.2f} µg/L")
print(f"  Range:  {chl_predictions.min():.2f} – {chl_predictions.max():.2f} µg/L")

# ── Water quality interpretation ──────────────────────────────────────────────
turb_mean = turb_predictions.mean()
chl_mean  = chl_predictions.mean()

print(f"\n{'='*40}")
print(f"INTERPRETATION")
print(f"{'='*40}")

if turb_mean < 10:
    print(f"Turbidity: LOW — relatively clear water")
elif turb_mean < 50:
    print(f"Turbidity: MODERATE — some sediment/pollution")
else:
    print(f"Turbidity: HIGH — heavily turbid water")

if chl_mean < 10:
    print(f"Chlorophyll: LOW — minimal algae")
elif chl_mean < 30:
    print(f"Chlorophyll: MODERATE — some algal activity")
else:
    print(f"Chlorophyll: HIGH — significant algal bloom risk")
    
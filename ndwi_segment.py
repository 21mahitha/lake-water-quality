import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# ── Load image ────────────────────────────────────────────────────────────────
with rasterio.open('data/test-demo/hussain_sagar_s2.tif') as src:
    bands = src.read().astype(np.float32)  # shape: (6, height, width)
    profile = src.profile

# Band order from GEE export: B2, B3, B4, B8, B11, B12
B2  = bands[0]  # Blue
B3  = bands[1]  # Green
B4  = bands[2]  # Red
B8  = bands[3]  # NIR
B11 = bands[4]  # SWIR1
B12 = bands[5]  # SWIR2

# ── NDWI water detection ──────────────────────────────────────────────────────
# Avoid division by zero
ndwi = np.where(
    (B3 + B8) == 0, 0,
    (B3 - B8) / (B3 + B8)
)

# Otsu threshold — automatically finds best cutoff for this image
threshold = threshold_otsu(ndwi)
water_mask = (ndwi > threshold).astype(np.uint8)

print(f"NDWI range:     {ndwi.min():.3f} to {ndwi.max():.3f}")
print(f"Otsu threshold: {threshold:.3f}")
print(f"Water pixels:   {water_mask.sum()}")
print(f"Total pixels:   {water_mask.size}")
print(f"Water coverage: {water_mask.sum()/water_mask.size*100:.1f}%")

# ── Save water mask as GeoTIFF ────────────────────────────────────────────────
mask_profile = profile.copy()
mask_profile.update(count=1, dtype=rasterio.uint8)

with rasterio.open('data/test-demo/hussain_sagar_water_mask.tif', 'w', **mask_profile) as dst:
    dst.write(water_mask[np.newaxis, :, :])

print("\nWater mask saved to: data/test-demo/hussain_sagar_water_mask.tif")

# ── Visualize ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# RGB image
rgb = np.stack([B4, B3, B2], axis=-1)
rgb = np.clip(rgb / 3000, 0, 1)  # normalize Sentinel-2 values
axes[0].imshow(rgb)
axes[0].set_title('Hussain Sagar — RGB')
axes[0].axis('off')

# NDWI heatmap
ndwi_plot = axes[1].imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
axes[1].set_title(f'NDWI (threshold={threshold:.2f})')
axes[1].axis('off')
plt.colorbar(ndwi_plot, ax=axes[1], fraction=0.046)

# Water mask
axes[2].imshow(water_mask, cmap='Blues')
axes[2].set_title('Water Mask (blue = water)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('data/test-demo/hussain_sagar_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualization saved to: data/test-demo/hussain_sagar_segmentation.png")
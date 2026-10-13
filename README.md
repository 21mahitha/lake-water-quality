# lake-water-quality
# Lake Water Quality Monitor + Forecaster
**Case Study: Hussain Sagar, Hyderabad**

A two-stage deep learning pipeline that reads satellite imagery to assess 
current lake water quality and forecasts future conditions using historical 
trends and climate data.

## Pipeline Overview

Sentinel-2 Image
↓
NDWI + Otsu Threshold → Water Mask
↓
XGBoost Models → Turbidity (FNU) + Chlorophyll-a (µg/L)
↓
LSTM Forecaster → Predicted quality t years ahead


## Results

| Model | Task | R² | MAE |
|---|---|---|---|
| XGBoost | Turbidity prediction | 0.8334 | 6.85 FNU |
| XGBoost | Chlorophyll-a prediction | 0.9159 | 2.22 µg/L |
| LSTM | Turbidity forecasting | 0.4805 | — |
| LSTM | Chlorophyll forecasting | 0.6065 | — |

## Data Sources

- **Sentinel-2 imagery** — Google Earth Engine (2016–2024)
- **Water quality labels** — USGS Sentinel-2 Matched Dataset (1.7M samples)
- **Climate data** — ERA5 reanalysis (Hyderabad rainfall + temperature)

## Project Structure

lake_quality/
├── ndwi_segment.py # Stage 1: Water body segmentation
├── prepare_data.py # Data preparation from USGS dataset
├── train_model1b.py # Stage 2: Random Forest baseline
├── train_model1b_xgb.py # Stage 2: XGBoost water quality models
├── predict_hussain_sagar.py # Run prediction on lake image
├── train_lstm.py # Stage 3: LSTM forecaster
├── demo_pipeline.py # Full end-to-end dashboard
├── HussainSagar_WaterQuality_WithClimate.xlsx # 108-month dataset
└── hussain_sagar_s2.tif # Sentinel-2 image


## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install rasterio numpy matplotlib scikit-image
pip install xgboost scikit-learn joblib
pip install tensorflow pandas openpyxl
```

## Run

```bash
# Full pipeline demo
python demo_pipeline.py

# Individual stages
python ndwi_segment.py          # segmentation only
python predict_hussain_sagar.py # quality prediction only
python train_lstm.py            # forecaster training
```

## Study Area

Hussain Sagar is an urban lake in Hyderabad, India, known for significant 
water quality challenges due to urbanization, industrial discharge, and 
seasonal festival activities (Ganesh Chaturthi).

## Citation

> [Paper in preparation] — Lake Water Quality Monitoring and Forecasting 
> Using Sentinel-2 Imagery and Machine Learning, 2026.

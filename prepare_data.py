import pandas as pd

# Load both files
df1 = pd.read_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\Matched_WQ_S2\\Matched_WQ_S2_1.csv', low_memory=False)
df2 = pd.read_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\Matched_WQ_S2\\Matched_WQ_S2_2.csv', low_memory=False)

df = pd.concat([df1, df2], ignore_index=True)
print(f"Total rows combined: {len(df)}")

# ── Columns we need ───────────────────────────────────────────────────────────
bands = ['B02_center','B03_center','B04_center',
         'B08_center','B11_center','B12_center']

# ── Filter turbidity ──────────────────────────────────────────────────────────
turbidity_mask = df['parm_nm'].str.contains('Turbidity', case=False, na=False)
df_turb = df[turbidity_mask][bands + ['MeasurementValue']].copy()
df_turb.rename(columns={'MeasurementValue': 'turbidity_FNU'}, inplace=True)

# ── Filter chlorophyll (µg/L only, not RFU) ───────────────────────────────────
chl_mask = (
    df['parm_nm'].str.contains('Chlorophyll', case=False, na=False) &
    df['parm_nm'].str.contains('micrograms per liter', case=False, na=False)
)
df_chl = df[chl_mask][bands + ['MeasurementValue']].copy()
df_chl.rename(columns={'MeasurementValue': 'chlorophyll_ugL'}, inplace=True)

# ── Drop missing band values ──────────────────────────────────────────────────
df_turb = df_turb.dropna()
df_chl  = df_chl.dropna()

# ── Drop negative or zero measurements ───────────────────────────────────────
df_turb = df_turb[df_turb['turbidity_FNU'] > 0]
df_chl  = df_chl[df_chl['chlorophyll_ugL'] > 0]

# ── Drop extreme outliers (top 1%) ────────────────────────────────────────────
turb_99 = df_turb['turbidity_FNU'].quantile(0.99)
chl_99  = df_chl['chlorophyll_ugL'].quantile(0.99)
df_turb = df_turb[df_turb['turbidity_FNU'] <= turb_99]
df_chl  = df_chl[df_chl['chlorophyll_ugL'] <= chl_99]

print(f"\nTurbidity samples ready:    {len(df_turb)}")
print(f"Chlorophyll samples ready:  {len(df_chl)}")

print(f"\nTurbidity range:   {df_turb['turbidity_FNU'].min():.2f} – {df_turb['turbidity_FNU'].max():.2f} FNU")
print(f"Chlorophyll range: {df_chl['chlorophyll_ugL'].min():.2f} – {df_chl['chlorophyll_ugL'].max():.2f} µg/L")

# ── Save cleaned datasets ─────────────────────────────────────────────────────
df_turb.to_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\turbidity_clean.csv', index=False)
df_chl.to_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\chlorophyll_clean.csv', index=False)

print("\nSaved: turbidity_clean.csv")
print("Saved: chlorophyll_clean.csv")
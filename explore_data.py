import pandas as pd

df1 = pd.read_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\Matched_WQ_S2\\Matched_WQ_S2_1.csv')
df2 = pd.read_csv('C:\\Users\\mahit\\OneDrive - Mahindra University\\SEM7\\sem7_project\\lake_quality\\Matched_WQ_S2\\Matched_WQ_S2_2.csv')

print("=== FILE 1 ===")
print(f"Rows: {len(df1)}")
print(f"Measurement types:\n{df1['parm_nm'].value_counts()}")

print("\n=== FILE 2 ===")
print(f"Rows: {len(df2)}")
print(f"Measurement types:\n{df2['parm_nm'].value_counts()}")
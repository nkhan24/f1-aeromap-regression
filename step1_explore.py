"""Step 1 - pandas: load, check and summarise the data."""
from aeromap import load_sweep, add_aero_loads

df = load_sweep("data/ride_height_sweep.csv")

print("First 5 rows:")
print(df.head())

print("\nRows and columns:", df.shape)
print("\nMissing values per column:")
print(df.isna().sum())

df = add_aero_loads(df)
print("\nSummary statistics:")
print(df.describe().round(2))

df.to_csv("results/ride_height_sweep_with_aero.csv", index=False)
print("\nSaved results/ride_height_sweep_with_aero.csv")
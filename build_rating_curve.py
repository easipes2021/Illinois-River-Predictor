import json, pandas as pd, numpy as np, matplotlib.pyplot as plt
import scipy.stats as st

data = json.load(open("data/usgs_07195430.json"))["value"]["timeSeries"]

series = {}
for ts in data:
    code = ts["variable"]["variableCode"][0]["value"]
    vals = ts["values"][0]["value"]
    df = pd.DataFrame([{"dt": v["dateTime"], "value": float(v["value"])} for v in vals])
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    series[code] = df.set_index("dt")

merged = series["00065"].join(series["00060"], lsuffix="_H", rsuffix="_Q").dropna()
merged = merged[(merged["value_Q"] > 0) & (merged["value_H"] > 0)]

bp = merged["value_H"].median()
low = merged[merged["value_H"] <= bp]
high = merged[merged["value_H"] > bp]

# Fit power-law segments
slope_l, intercept_l, *_ = st.linregress(np.log(low["value_H"]), np.log(low["value_Q"]))
slope_h, intercept_h, *_ = st.linregress(np.log(high["value_H"]), np.log(high["value_Q"]))

A_l, B_l = np.exp(intercept_l), slope_l
A_h, B_h = np.exp(intercept_h), slope_h

# Save outputs
merged.to_csv("paired_stage_flow.csv")

H = np.linspace(merged["value_H"].min(), merged["value_H"].max(), 200)
plt.figure(figsize=(7,5))
plt.scatter(merged["value_H"], merged["value_Q"], s=1)
plt.plot(H, A_l * H**B_l, "r", label="Low-flow segment")
plt.plot(H, A_h * H**B_h, "g", label="High-flow segment")
plt.legend()
plt.xlabel("Stage (ft)")
plt.ylabel("Discharge (cfs)")
plt.title("Illinois River Rating Curve")
plt.savefig("rating_curve.png")

meta = {
    "piecewise_breakpoint": float(bp),
    "low_flow": {"A": float(A_l), "B": float(B_l)},
    "high_flow": {"A": float(A_h), "B": float(B_h)}
}
json.dump(meta, open("rating_curve_metadata.json", "w"), indent=2)

print("✅ Rating curve updated.")
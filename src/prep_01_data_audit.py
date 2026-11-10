"""Step 01: Data audit of the Norfolk micromobility trip records (Lime e-scooters and e-bikes).

Reads raw data (never modifies it), reports structure, missing values,
time coverage, vehicle-type mix and implausible records, and saves
tables to outputs/tables and a summary report to outputs/reports.

Run from the project root:
    python src/prep_01_data_audit.py
"""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
P = {k: ROOT / v for k, v in CFG["paths"].items()}
for k in ("tables", "reports", "interim"):
    P[k].mkdir(parents=True, exist_ok=True)
A = CFG["audit"]
report = []


def log(line=""):
    print(line)
    report.append(line)


def missing_table(df, name):
    t = pd.DataFrame({"n_missing": df.isna().sum(), "pct_missing": (df.isna().mean() * 100).round(3),
                      "dtype": df.dtypes.astype(str), "n_unique": df.nunique()})
    t.to_csv(P["tables"] / f"audit_{name}_missing.csv")
    return t


# ---------------------------------------------------------------- Norfolk
log("# Data audit report\n")
log("## 1. Norfolk micromobility trips (Lime e-scooters and e-bikes)\n")
nf = pd.read_csv(P["raw_norfolk"], dtype={"Trip ID": str, "Starting Tract": "Int64", "Ending Tract": "Int64"})
nf.columns = ["trip_id", "vehicle_type", "duration_min", "distance_mi", "start_date",
              "start_hour", "end_date", "end_hour", "start_tract", "end_tract"]
nf["start_dt"] = pd.to_datetime(nf["start_date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
log(f"- Rows: {len(nf):,}; columns: {nf.shape[1] - 1}")
log(f"- Duplicate trip IDs: {nf['trip_id'].duplicated().sum():,}")
log(f"- Date range: {nf['start_dt'].min().date()} to {nf['start_dt'].max().date()}")
log(f"- Unparseable dates: {nf['start_dt'].isna().sum():,}")
mt = missing_table(nf.drop(columns=["start_dt"]), "norfolk")
log("\nMissing values (%):\n")
log(mt[["n_missing", "pct_missing", "n_unique"]].to_markdown())

vt = nf["vehicle_type"].value_counts(dropna=False)
log("\nVehicle type mix:\n")
log(vt.to_frame("trips").assign(pct=lambda d: (d.trips / d.trips.sum() * 100).round(2)).to_markdown())

nf["ym"] = nf["start_dt"].dt.to_period("M").astype(str)
monthly = nf.pivot_table(index="ym", columns="vehicle_type", values="trip_id", aggfunc="count", fill_value=0)
monthly.to_csv(P["tables"] / "audit_norfolk_monthly_trips_by_type.csv")
yearly = nf.assign(year=nf["start_dt"].dt.year).pivot_table(
    index="year", columns="vehicle_type", values="trip_id", aggfunc="count", fill_value=0)
log("\nTrips per year by vehicle type:\n")
log(yearly.to_markdown())

desc = nf.groupby("vehicle_type")[["duration_min", "distance_mi"]].describe(
    percentiles=[.01, .25, .5, .75, .99]).T.round(2)
desc.to_csv(P["tables"] / "audit_norfolk_duration_distance_by_type.csv")
log("\nDuration and distance by vehicle type:\n")
log(desc.to_markdown())

flags = pd.DataFrame({
    "duration_lt_min": nf["duration_min"] < A["min_duration_min"],
    "duration_gt_max": nf["duration_min"] > A["max_duration_min"],
    "distance_le_0": nf["distance_mi"] <= 0,
    "distance_gt_max": nf["distance_mi"] > A["max_distance_miles"],
    "speed_gt_20mph": (nf["distance_mi"] / (nf["duration_min"] / 60)).replace(np.inf, np.nan) > 20,
    "tract_missing": nf["start_tract"].isna() | nf["end_tract"].isna(),
})
ft = flags.agg(["sum", "mean"]).T.rename(columns={"sum": "n", "mean": "share"})
ft["share"] = (ft["share"] * 100).round(3)
ft.to_csv(P["tables"] / "audit_norfolk_flags.csv")
log("\nPlausibility flags (not removed yet):\n")
log(ft.to_markdown())
log(f"\n- Any flag: {flags.any(axis=1).mean() * 100:.2f}% of trips")
log(f"- Distinct start tracts: {nf['start_tract'].nunique()}; same-tract trips: "
    f"{(nf['start_tract'] == nf['end_tract']).mean() * 100:.1f}%")

nf.drop(columns=["start_date", "end_date"]).to_parquet(P["interim"] / "norfolk_trips.parquet", index=False)
log("- Saved typed copy: data/interim/norfolk_trips.parquet")
del nf

(P["reports"] / "01_data_audit_report.md").write_text("\n".join(report))
print("\nReport saved: outputs/reports/01_data_audit_report.md")

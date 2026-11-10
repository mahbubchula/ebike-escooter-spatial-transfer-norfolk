"""Step 03: Clean trips and build the trip-level modelling dataset.

Target: e_bike = 1 if the trip used a Lime e-bike, 0 if it used an e-scooter.

Rules applied here
  * Only trips from June 2021 onward (e-bikes introduced in June 2021).
  * Duration, recorded distance and speed are kept for description only; they are never features.
  * No statistic of the target is computed here.

Outputs
  data/processed/norfolk_model_data.parquet
  outputs/tables/cleaning_flow_norfolk.csv

Run:  python src/prep_prep_03_clean_and_features.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
A = CFG["audit"]
EXT, INT, PRO, TAB = (ROOT / "data" / "external", ROOT / "data" / "interim",
                      ROOT / "data" / "processed", ROOT / "outputs" / "tables")
PRO.mkdir(parents=True, exist_ok=True)

CBD = {"norfolk": (36.8468, -76.2852)}  # downtown Norfolk
TRACT_VARS = ["pop_density_km2", "job_density_km2", "median_income", "median_age", "pct_age_18_29",
              "pct_degree", "pct_hh_no_car", "pct_commute_transit", "pct_commute_bicycle",
              "pct_commute_walk", "pct_commute_wfh", "pct_jobs_retail_food"]
WEATHER_VARS = {"temperature_2m": "temp_c", "apparent_temperature": "feels_like_c",
                "precipitation": "precip_mm", "snowfall": "snow_cm", "wind_speed_10m": "wind_kmh",
                "relative_humidity_2m": "humidity_pct", "cloud_cover": "cloud_pct", "is_day": "is_daylight"}
HOLIDAYS = set(USFederalHolidayCalendar().holidays("2019-01-01", "2026-12-31").date)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class Flow:
    """Keeps a CONSORT-style record of how many trips remain after each rule."""
    def __init__(self, city, df):
        self.city, self.rows = city, [("raw records", len(df), 0)]

    def step(self, label, df, mask):
        before = len(df)
        df = df[mask].copy()
        self.rows.append((label, len(df), before - len(df)))
        print(f"  {label:<55} kept {len(df):>11,}  removed {before - len(df):>10,}")
        return df

    def save(self):
        pd.DataFrame(self.rows, columns=["step", "remaining", "removed"]).to_csv(
            TAB / f"cleaning_flow_{self.city}.csv", index=False)


def add_time(df, dt):
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["is_holiday"] = dt.dt.date.isin(HOLIDAYS).astype(int)
    df["year"] = dt.dt.year
    df["datetime_hour"] = dt.dt.floor("h")
    return df


def add_weather(df, city):
    w = pd.read_parquet(EXT / "weather" / f"weather_hourly_{city}.parquet").rename(
        columns={"datetime_local": "datetime_hour", **WEATHER_VARS})
    w = w[["datetime_hour", *WEATHER_VARS.values()]]
    # precipitation in the previous 3 hours (wet surfaces)
    w = w.sort_values("datetime_hour")
    w["precip_prev3h_mm"] = w["precip_mm"].shift(1).rolling(3, min_periods=1).sum()
    return df.merge(w, on="datetime_hour", how="left")


def add_tracts(df, city, prefix, geoid_col):
    t = pd.read_parquet(EXT / f"tract_features_{city}.parquet")
    t = t[["GEOID", *TRACT_VARS]].rename(columns={v: f"{prefix}_{v}" for v in TRACT_VARS})
    return df.merge(t, left_on=geoid_col, right_on="GEOID", how="left").drop(columns="GEOID")



# =========================================================== Norfolk (Lime)
print("--- Norfolk")
nf = pd.read_parquet(INT / "norfolk_trips.parquet")
f = Flow("norfolk", nf)
nf = f.step("from 2021-06-01 (both vehicle types offered)", nf, nf["start_dt"] >= "2021-06-01")
nf = f.step(f"duration {A['min_duration_min']}-{A['max_duration_min']} min", nf,
            nf["duration_min"].between(A["min_duration_min"], A["max_duration_min"]))
nf = f.step(f"distance > 0 and <= {A['max_distance_miles']} miles", nf,
            (nf["distance_mi"] > 0) & (nf["distance_mi"] <= A["max_distance_miles"]))
speed = nf["distance_mi"] / (nf["duration_min"] / 60)
nf = f.step("average speed <= 20 mph", nf, speed <= 20)
nf = f.step("start and end tract present", nf, nf["start_tract"].notna() & nf["end_tract"].notna())
f.save()

nf["e_bike"] = (nf["vehicle_type"] == "bicycle").astype(int)
nf["o_tract"] = nf["start_tract"].astype("int64").astype(str)
nf["d_tract"] = nf["end_tract"].astype("int64").astype(str)
nf["same_tract"] = (nf["o_tract"] == nf["d_tract"]).astype(int)
nf["dist_km"] = nf["distance_mi"] * 1.609344  # reported trip distance
cent = pd.read_parquet(EXT / "tract_features_norfolk.parquet", columns=["GEOID", "INTPTLAT", "INTPTLON"])
cent[["INTPTLAT", "INTPTLON"]] = cent[["INTPTLAT", "INTPTLON"]].astype(float)
c = cent.set_index("GEOID")
nf["o_dist_cbd_km"] = haversine_km(nf["o_tract"].map(c["INTPTLAT"]), nf["o_tract"].map(c["INTPTLON"]), *CBD["norfolk"])
nf["d_dist_cbd_km"] = haversine_km(nf["d_tract"].map(c["INTPTLAT"]), nf["d_tract"].map(c["INTPTLON"]), *CBD["norfolk"])
nf = add_time(nf, nf["start_dt"])
nf = add_weather(nf, "norfolk")
nf = add_tracts(nf, "norfolk", "o", "o_tract")
nf = add_tracts(nf, "norfolk", "d", "d_tract")
nf = nf.drop(columns=["vehicle_type", "start_hour", "end_hour", "start_tract", "end_tract", "ym"])
nf.to_parquet(PRO / "norfolk_model_data.parquet", index=False)
print(f"  saved {len(nf):,} trips; e-bike share {nf['e_bike'].mean():.3f}")

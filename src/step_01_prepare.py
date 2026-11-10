"""Step 01: leakage-safe modelling data, fleet-supply proxies, spatial blocks and temporal split.

Pre-trip features only. Recorded trip distance and duration are post-choice and are never features; trip length
enters as the straight-line distance between origin and destination tract centroids.
Fleet supply (unobserved) is proxied by strictly lagged quantities (days t-7..t-1, citywide).
Origin-tract local history (days t-28..t-1) is built for an extension model only.

Outputs:
  data/processed/splits/{inner_train,valid,final_train,test}.parquet
  configs/features.json
  outputs/tables/T01_split.csv, T02_correlation_filter.csv, T03_spatial_blocks.csv
  outputs/reports/01_prepare.json

Run:  python src/step_01_prepare.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
PRO, EXT = ROOT / "data" / "processed", ROOT / "data" / "external"
OUT = ROOT / "outputs"
SPL = PRO / "splits"
for p in (SPL, OUT / "tables", OUT / "reports"):
    p.mkdir(parents=True, exist_ok=True)
SEED, K_BLOCKS = 42, 6
START, END = pd.Timestamp("2021-06-01"), pd.Timestamp("2026-11-01")
SPLITS = {"inner_train": ("2021-06-01", "2024-12-24"), "embargo": ("2024-12-25", "2024-12-31"),
          "valid": ("2025-01-01", "2025-12-31"), "final_train": ("2021-06-01", "2025-12-31"),
          "test": ("2026-01-01", "2026-10-31")}
FORBIDDEN = {"distance_mi", "dist_km", "duration_min", "end_hour", "month", "year", "o_ebike_share_28d_in_main"}

d = pd.read_parquet(PRO / "norfolk_model_data.parquet")
d = d[(d.start_dt >= START) & (d.start_dt < END) & d.o_tract.notna() & d.d_tract.notna()].copy()
assert d["trip_id"].is_unique
d["day"] = d["start_dt"].dt.normalize()
print(f"sample: {len(d):,} trips, e-bike share {d.e_bike.mean():.4f}")

# ------------------------------------------------------------------ OD centroid distance (pre-trip)
tf = pd.read_parquet(EXT / "tract_features_norfolk.parquet").set_index("GEOID")
lat, lon = tf["INTPTLAT"].astype(float), tf["INTPTLON"].astype(float)


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


d["od_dist_km"] = hav(d.o_tract.map(lat).values, d.o_tract.map(lon).values,
                      d.d_tract.map(lat).values, d.d_tract.map(lon).values)
assert d["od_dist_km"].notna().all()
assert (d.loc[d.same_tract == 1, "od_dist_km"] == 0).all()

# ------------------------------------------------------------------ strictly lagged supply proxies
days = pd.date_range(START, END - pd.Timedelta(days=1), freq="D")
cd = d.groupby("day")["e_bike"].agg(n="size", k="sum").reindex(days, fill_value=0)
cd["n7"] = cd["n"].rolling(7, min_periods=1).sum().shift(1)
cd["k7"] = cd["k"].rolling(7, min_periods=1).sum().shift(1)
cd["city_ebike_share_7d"] = np.where(cd["n7"] >= 50, cd["k7"] / cd["n7"].where(cd["n7"] > 0), np.nan)
cd["city_trips_7d"] = cd["n7"]
d = d.merge(cd[["city_ebike_share_7d", "city_trips_7d"]], left_on="day", right_index=True, how="left")

td = d.groupby(["o_tract", "day"])["e_bike"].agg(n="size", k="sum")
full = td.reindex(pd.MultiIndex.from_product([d.o_tract.unique(), days], names=["o_tract", "day"]), fill_value=0)
full = full.reset_index().sort_values(["o_tract", "day"])
g = full.groupby("o_tract", sort=False)
full["n28"] = g["n"].transform(lambda s: s.rolling(28, min_periods=1).sum().shift(1))
full["k28"] = g["k"].transform(lambda s: s.rolling(28, min_periods=1).sum().shift(1))
full["o_ebike_share_28d"] = np.where(full["n28"] >= 30, full["k28"] / full["n28"].where(full["n28"] > 0), np.nan)
d = d.merge(full[["o_tract", "day", "o_ebike_share_28d"]], on=["o_tract", "day"], how="left")

# brute-force self-checks: only earlier days are used
rng = np.random.default_rng(SEED)
for t in rng.choice(days[10:], 60, replace=False):
    w = d[(d.day < t) & (d.day >= t - pd.Timedelta(days=7))]
    exp = w.e_bike.mean() if len(w) >= 50 else np.nan
    got = cd.loc[t, "city_ebike_share_7d"]
    assert (np.isnan(exp) and np.isnan(got)) or abs(exp - got) < 1e-12
chk = full[full.n28 >= 30].sample(60, random_state=SEED)
for _, r in chk.iterrows():
    w = d[(d.o_tract == r.o_tract) & (d.day < r.day) & (d.day >= r.day - pd.Timedelta(days=28))]
    assert len(w) == r.n28 and abs(w.e_bike.mean() - r.o_ebike_share_28d) < 1e-12
print("self-checks passed: supply proxies use only earlier days")

# ------------------------------------------------------------------ spatial blocks (from centroids only)
tr = gpd.read_file(EXT / "tiger" / "tracts_norfolk.gpkg").to_crs(32618)
tr = tr[tr.GEOID.isin(set(d.o_tract) | set(d.d_tract))].copy()
xy = np.column_stack([tr.geometry.centroid.x, tr.geometry.centroid.y])
km = KMeans(K_BLOCKS, random_state=SEED, n_init=50).fit(xy)
order = np.argsort(km.cluster_centers_[:, 1])[::-1]                      # number blocks north -> south
relabel = {old: new + 1 for new, old in enumerate(order)}
tr["block"] = [relabel[c] for c in km.labels_]
blk = tr.set_index("GEOID")["block"]
d["o_block"] = d.o_tract.map(blk)
assert d["o_block"].notna().all()
bt = d.groupby("o_block").agg(trips=("e_bike", "size"), ebike_share=("e_bike", "mean"), tracts=("o_tract", "nunique"))
bt["trip_pct"] = 100 * bt["trips"] / bt["trips"].sum()
bt.to_csv(OUT / "tables" / "T03_spatial_blocks.csv")
tr[["GEOID", "block"]].to_csv(OUT / "tables" / "T03b_tract_blocks.csv", index=False)
print(bt.round(3).to_string())

# ------------------------------------------------------------------ feature groups
TIME = ["hour", "day_of_week", "is_weekend", "is_holiday", "is_daylight"]
WEATHER = ["temp_c", "feels_like_c", "precip_mm", "precip_prev3h_mm", "snow_cm", "wind_kmh", "humidity_pct", "cloud_pct"]
OD = ["od_dist_km", "same_tract", "o_dist_cbd_km", "d_dist_cbd_km"]
CTX = ["pop_density_km2", "job_density_km2", "median_income", "median_age", "pct_age_18_29", "pct_degree",
       "pct_hh_no_car", "pct_commute_transit", "pct_commute_bicycle", "pct_commute_walk", "pct_commute_wfh",
       "pct_jobs_retail_food"]
ORIG, DEST = [f"o_{c}" for c in CTX], [f"d_{c}" for c in CTX]
SUPPLY = ["city_ebike_share_7d", "city_trips_7d"]
CANDIDATES = TIME + WEATHER + OD + ORIG + DEST + SUPPLY
assert not (set(CANDIDATES) & FORBIDDEN)

# ------------------------------------------------------------------ split
d = d.drop(columns=["distance_mi", "dist_km", "duration_min"])                 # post-choice: removed from files
parts = {}
for name, (a, b) in SPLITS.items():
    parts[name] = d[(d.start_dt >= a) & (d.start_dt < pd.Timestamp(b) + pd.Timedelta(days=1))]
it, va, te = parts["inner_train"], parts["valid"], parts["test"]
assert it.start_dt.max() < pd.Timestamp("2024-12-25") and va.start_dt.min() >= pd.Timestamp("2025-01-01")
assert va.start_dt.max() < te.start_dt.min()
assert not (set(it.trip_id) & set(va.trip_id)) and not (set(parts["final_train"].trip_id) & set(te.trip_id))

# ------------------------------------------------------------------ correlation filter (inner train only)
PRIORITY = ["od_dist_km", "temp_c", "is_weekend", "o_dist_cbd_km", "d_dist_cbd_km", "o_pct_age_18_29",
            "d_pct_age_18_29", "o_pct_hh_no_car", "d_pct_hh_no_car", "o_median_income", "d_median_income",
            "o_pop_density_km2", "d_pop_density_km2", "city_ebike_share_7d", "same_tract", "hour"]
rank = {f: (PRIORITY.index(f) if f in PRIORITY else len(PRIORITY) + CANDIDATES.index(f)) for f in CANDIDATES}
cs = it[CANDIDATES].sample(min(300_000, len(it)), random_state=SEED)
C = cs.corr(method="spearman").abs()
pairs = sorted(((C.loc[a, b], a, b) for i, a in enumerate(CANDIDATES) for b in CANDIDATES[i + 1:]
                if C.loc[a, b] >= 0.7), reverse=True)
dropped, rows = set(), []
for r, a, b in pairs:
    if a in dropped or b in dropped:
        rows.append(dict(f1=a, f2=b, abs_spearman=round(r, 3), decision="already resolved"))
        continue
    drop = b if rank[a] < rank[b] else a
    dropped.add(drop)
    rows.append(dict(f1=a, f2=b, abs_spearman=round(r, 3), decision=f"drop {drop}"))
pd.DataFrame(rows).to_csv(OUT / "tables" / "T02_correlation_filter.csv", index=False)
MAIN = [f for f in CANDIDATES if f not in dropped]
FS = {"main": MAIN, "main_plus_history": MAIN + ["o_ebike_share_28d"],
      "origin_only": [f for f in MAIN if not (f.startswith("d_") or f in ("od_dist_km", "same_tract"))]}
(ROOT / "configs" / "features.json").write_text(json.dumps(FS, indent=2))
print(f"dropped by correlation filter ({len(dropped)}): {sorted(dropped)}")
print({k: len(v) for k, v in FS.items()})

keep = ["trip_id", "start_dt", "day", "e_bike", "o_tract", "d_tract", "o_block", "o_ebike_share_28d", *CANDIDATES]
rows = []
for name, part in parts.items():
    if name != "embargo":
        part[keep].to_parquet(SPL / f"{name}.parquet", index=False)
    rows.append(dict(set=name, start=SPLITS[name][0], end=SPLITS[name][1], trips=len(part),
                     ebike_pct=round(100 * part.e_bike.mean(), 2)))
pd.DataFrame(rows).to_csv(OUT / "tables" / "T01_split.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
(OUT / "reports" / "01_prepare.json").write_text(json.dumps({
    "n_trips": int(len(d)), "ebike_share": float(d.e_bike.mean()), "dropped": sorted(dropped),
    "feature_sets": {k: len(v) for k, v in FS.items()}, "k_blocks": K_BLOCKS,
    "missing_city_share_7d": float(d.city_ebike_share_7d.isna().mean()),
    "missing_o_share_28d": float(d.o_ebike_share_28d.isna().mean())}, indent=2))

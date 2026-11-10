"""Step 02: Build external context data at census-tract and hourly level.

Creates:
  data/external/tract_features_norfolk.parquet   (ACS 2019-2023 5-yr + LODES 2022 jobs + area)
  data/external/weather/weather_hourly_{city}.parquet  (Open-Meteo ERA5 archive)

Run:  python src/prep_02_external_data.py
"""
from pathlib import Path
import time

import geopandas as gpd
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
EXT = ROOT / "data" / "external"
(EXT / "weather").mkdir(parents=True, exist_ok=True)

CITIES = {
    # state FIPS, county FIPS, LODES state, lat, lon, weather start, end, local timezone
    "norfolk": dict(state="51", county="710", lodes="va", lat=36.8508, lon=-76.2859,
                    start="2019-10-01", end="2026-11-08", tz="America/New_York"),
}

# ACS 2023 5-year variables (detailed tables)
ACS = {
    "B01003_001E": "pop_total", "B01002_001E": "median_age", "B19013_001E": "median_income",
    "B08301_001E": "commuters_total", "B08301_003E": "commute_drove_alone", "B08301_010E": "commute_transit",
    "B08301_018E": "commute_bicycle", "B08301_019E": "commute_walk", "B08301_021E": "commute_wfh",
    "B25044_001E": "hh_total", "B25044_003E": "hh_owner_no_car", "B25044_010E": "hh_renter_no_car",
    "B15003_001E": "edu_pop25", "B15003_022E": "edu_ba", "B15003_023E": "edu_ma",
    "B15003_024E": "edu_prof", "B15003_025E": "edu_phd",
    "B01001_001E": "age_total", "B01001_007E": "m18_19", "B01001_008E": "m20", "B01001_009E": "m21",
    "B01001_010E": "m22_24", "B01001_011E": "m25_29", "B01001_031E": "f18_19", "B01001_032E": "f20",
    "B01001_033E": "f21", "B01001_034E": "f22_24", "B01001_035E": "f25_29",
}


def get_acs(state, county):
    """Read ACS 2023 5-year table-based summary files (filtered to study tracts).

    Source: https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/
    The Census API now requires a key, so the bulk files are used instead.
    """
    d = None
    for table in sorted({k.split("_")[0] for k in ACS}):
        t = pd.read_csv(EXT / "acs" / f"acs2023_{table.lower()}.dat", sep="|", dtype=str)
        d = t if d is None else d.merge(t, on="GEO_ID", how="outer")
    d["GEOID"] = d["GEO_ID"].str.replace("1400000US", "", regex=False)
    d = d[d["GEOID"].str.startswith(state + county)]
    # API style "B01003_001E" -> summary-file style "B01003_E001"
    d = d.rename(columns={f"{k[:6]}_E{k[7:10]}": v for k, v in ACS.items()})[["GEOID", *ACS.values()]]
    num = list(ACS.values())
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    d[d[num] < 0] = pd.NA  # Census uses -666666666 etc. for unavailable values
    out = pd.DataFrame({"GEOID": d["GEOID"], "pop_total": d["pop_total"],
                        "median_age": d["median_age"], "median_income": d["median_income"]})
    c = d["commuters_total"].where(d["commuters_total"] > 0)
    for k in ("drove_alone", "transit", "bicycle", "walk", "wfh"):
        out[f"pct_commute_{k}"] = 100 * d[f"commute_{k}"] / c
    out["pct_hh_no_car"] = 100 * (d["hh_owner_no_car"] + d["hh_renter_no_car"]) / d["hh_total"].where(d["hh_total"] > 0)
    out["pct_degree"] = 100 * d[["edu_ba", "edu_ma", "edu_prof", "edu_phd"]].sum(axis=1) / d["edu_pop25"].where(d["edu_pop25"] > 0)
    young = d[[c for c in ACS.values() if c[0] in "mf" and c[1].isdigit()]].sum(axis=1)
    out["pct_age_18_29"] = 100 * young / d["age_total"].where(d["age_total"] > 0)
    return out


def get_jobs(lodes):
    w = pd.read_csv(EXT / "lodes" / f"{lodes}_wac_S000_JT00_2022.csv.gz", dtype={"w_geocode": str},
                    usecols=["w_geocode", "C000", "CNS07", "CNS18"])
    w["GEOID"] = w["w_geocode"].str[:11]
    j = w.groupby("GEOID")[["C000", "CNS07", "CNS18"]].sum()
    # CNS07 = retail trade, CNS18 = accommodation and food services
    return j.rename(columns={"C000": "jobs_total", "CNS07": "jobs_retail", "CNS18": "jobs_food_accom"}).reset_index()


def get_weather(c):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = dict(latitude=c["lat"], longitude=c["lon"], start_date=c["start"], end_date=c["end"],
                  hourly="temperature_2m,apparent_temperature,precipitation,rain,snowfall,"
                         "wind_speed_10m,relative_humidity_2m,cloud_cover,is_day",
                  timezone=c["tz"], wind_speed_unit="kmh")
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=300)
        if r.ok:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.rename(columns={"time": "datetime_local"})


for city, c in CITIES.items():
    print(f"--- {city}")
    tr = gpd.read_file(EXT / "tiger" / f"tl_2023_{c['state']}_tract.zip")
    tr = tr[tr["COUNTYFP"] == c["county"]][["GEOID", "ALAND", "AWATER", "INTPTLAT", "INTPTLON", "geometry"]]
    tr.to_file(EXT / "tiger" / f"tracts_{city}.gpkg", driver="GPKG")
    feat = (tr.drop(columns="geometry").merge(get_acs(c["state"], c["county"]), on="GEOID", how="left")
              .merge(get_jobs(c["lodes"]), on="GEOID", how="left"))
    feat[["jobs_total", "jobs_retail", "jobs_food_accom"]] = feat[["jobs_total", "jobs_retail", "jobs_food_accom"]].fillna(0)
    km2 = feat["ALAND"] / 1e6
    feat["pop_density_km2"] = feat["pop_total"] / km2.where(km2 > 0)
    feat["job_density_km2"] = feat["jobs_total"] / km2.where(km2 > 0)
    feat["pct_jobs_retail_food"] = 100 * (feat["jobs_retail"] + feat["jobs_food_accom"]) / feat["jobs_total"].where(feat["jobs_total"] > 0)
    feat.to_parquet(EXT / f"tract_features_{city}.parquet", index=False)
    print(f"tracts: {len(feat)}; missing income: {feat['median_income'].isna().sum()}")
    w = get_weather(c)
    w.to_parquet(EXT / "weather" / f"weather_hourly_{city}.parquet", index=False)
    print(f"weather hours: {len(w):,}; {w['datetime_local'].min()} to {w['datetime_local'].max()}; "
          f"missing temp: {w['temperature_2m'].isna().sum()}")

# Check that Norfolk trip tracts match 2020-vintage TIGER tracts
nf = pd.read_parquet(ROOT / "data" / "interim" / "norfolk_trips.parquet", columns=["start_tract", "end_tract"])
ids = set(nf["start_tract"].dropna().astype("int64").astype(str)) | set(nf["end_tract"].dropna().astype("int64").astype(str))
tiger = set(pd.read_parquet(EXT / "tract_features_norfolk.parquet")["GEOID"])
print(f"Norfolk trip tracts: {len(ids)}; matched to TIGER 2023: {len(ids & tiger)}; unmatched: {sorted(ids - tiger)}")

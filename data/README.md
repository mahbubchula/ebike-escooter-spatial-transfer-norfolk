# Data

Raw data are not stored in this repository because of their size, but every input is public.
Place the files below at the paths shown, then run the pipeline from the repository root.

### Trip records

| Input | Save as | Source |
|---|---|---|
| Norfolk micromobility trips (Lime e-scooters and e-bikes), full CSV export | `data/raw/norfolk/norfolk_micromobility_trips_raw.csv` | https://data.norfolk.gov/Government/Micromobility-Electric-Scooters-and-Bikes-/wqxq-hhe6 (CSV export: https://data.norfolk.gov/api/views/wqxq-hhe6/rows.csv?accessType=DOWNLOAD) |

The dataset is updated weekly. The analysis used the export of November 7 to 9, 2026, and keeps trips from June 1, 2021, to October 31, 2026.

### Census and weather inputs

| Input | Save as | Source |
|---|---|---|
| TIGER/Line 2023 census tracts, Virginia | `data/external/tiger/tl_2023_51_tract.zip` | https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_51_tract.zip |
| ACS 2019 to 2023 5-year, tables B01001, B01002, B01003, B08301, B15003, B19013, B25044 | `data/external/acs/acs2023_<table>.dat` (for example `acs2023_b01001.dat`) | https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/ (files `acsdt5y2023-<table>.dat`) |
| LEHD LODES 8, Workplace Area Characteristics, 2022 | `data/external/lodes/va_wac_S000_JT00_2022.csv.gz` | https://lehd.ces.census.gov/data/lodes/LODES8/va/wac/va_wac_S000_JT00_2022.csv.gz |
| Hourly weather (ERA5 reanalysis) | downloaded automatically by `prep_02_external_data.py` | Open-Meteo historical API, https://open-meteo.com/en/docs/historical-weather-api |

The Census API requires a key, so the table-based summary files are used instead. All census inputs were released before the study period they describe ends, and they are treated as time-invariant tract attributes.

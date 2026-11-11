"""Turn NHS England's daily GP appointment extracts into tidy daily series.

Input : data/raw/daily/*/SUB_ICB_LOCATION_CSV_*.csv  (one file per month, Mar 2024 - Aug 2026)
        data/raw/daily/*/APPOINTMENTS_GP_COVERAGE.csv
Output: data/processed/daily_region.parquet   date x NHS region, with totals by status and mode
        data/processed/coverage_region.csv    registered patients by region and month
"""
from pathlib import Path
import glob

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# NHS England region ONS codes
REGIONS = {
    "E40000003": "London",
    "E40000005": "South East",
    "E40000006": "South West",
    "E40000013": "East of England",
    "E40000014": "North West",
    "E40000011": "Midlands",
    "E40000012": "North East and Yorkshire",
}


def load_daily() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "daily" / "*" / "SUB_ICB_LOCATION_CSV_*.csv")))
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=["REGION_ONS_CODE", "Appointment_Date", "APPT_STATUS", "APPT_MODE",
                                    "HCP_TYPE", "COUNT_OF_APPOINTMENTS"])
        d["date"] = pd.to_datetime(d["Appointment_Date"], format="%d%b%Y")
        d["region"] = d["REGION_ONS_CODE"].map(REGIONS).fillna("UNMAPPED " + d["REGION_ONS_CODE"])
        mode = d["APPT_MODE"].str.lower()
        d["face_to_face"] = d.COUNT_OF_APPOINTMENTS.where(mode.str.contains("face"), 0)
        d["telephone"] = d.COUNT_OF_APPOINTMENTS.where(mode.str.contains("telephone"), 0)
        d["attended"] = d.COUNT_OF_APPOINTMENTS.where(d.APPT_STATUS == "Attended", 0)
        d["dna"] = d.COUNT_OF_APPOINTMENTS.where(d.APPT_STATUS == "DNA", 0)
        d["gp"] = d.COUNT_OF_APPOINTMENTS.where(d.HCP_TYPE == "GP", 0)
        parts.append(d.groupby(["date", "region"], as_index=False)[
            ["COUNT_OF_APPOINTMENTS", "attended", "dna", "face_to_face", "telephone", "gp"]].sum())
        print(f"{Path(f).name}: {len(d):,} rows")
    out = pd.concat(parts)
    missing = out[out.region.str.startswith("UNMAPPED")]
    assert missing.empty, f"unmapped region codes: {missing}"
    out = out.rename(columns={"COUNT_OF_APPOINTMENTS": "appointments"})
    return out.groupby(["date", "region"], as_index=False).sum().sort_values(["region", "date"])


def load_coverage() -> pd.DataFrame:
    f = glob.glob(str(RAW / "daily" / "*" / "APPOINTMENTS_GP_COVERAGE.csv"))[0]
    c = pd.read_csv(f)
    c["month"] = pd.to_datetime(c["Appointment_Month"], format="%d%b%Y")
    c["region"] = c["REGION_CODE"].map(REGIONS)
    return c.groupby(["month", "region"], as_index=False)[
        ["Included Practices", "Patients registered at included practices"]].sum()


if __name__ == "__main__":
    daily = load_daily()
    assert daily.region.notna().all(), "unmapped region code"
    daily.to_parquet(OUT / "daily_region.parquet", index=False)
    cov = load_coverage()
    cov.to_csv(OUT / "coverage_region.csv", index=False)
    print(daily.groupby("region").appointments.sum().sort_values(ascending=False))
    print("Dates:", daily.date.min().date(), "to", daily.date.max().date(), "| rows:", len(daily))

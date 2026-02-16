"""Analytics functions for Fixture Yield Analytics.

Phase 2: core pandas-based functions used by the API and dashboard.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None


def load_data(data_dir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """Safely load CSV data from the project `data/` directory.

    - Resolves `data_dir` relative to the project root (backend/..).
    - If any required CSV is missing or empty, attempts to run the
      `data/generate_sample_data.py` script once (without --force) to populate
      placeholder files.
    - Returns a dict with keys `fixtures`, `production`, `quality` and an
      extra `__meta__` dict with warnings and `sample_generated` flag.
    """
    project_root = Path(__file__).resolve().parents[1]
    # default to project_root / data if not provided
    if data_dir is None:
        p = project_root / "data"
    else:
        p = Path(data_dir)
        if not p.is_absolute():
            p = project_root / p

    # ensure data dir exists
    p.mkdir(parents=True, exist_ok=True)

    required = {
        "fixtures": (p / "fixtures.csv", {"parse_dates": ["installed_date", "last_maintenance_date"]}),
        "production": (p / "production.csv", {"parse_dates": ["production_datetime"]}),
        "quality": (p / "quality.csv", {"parse_dates": ["inspection_datetime"]}),
    }

    warnings: list[str] = []
    need_generate = False

    # quick existence / non-empty check (file size)
    for name, (path, _) in required.items():
        if not path.exists() or path.stat().st_size == 0:
            warnings.append(f"Missing or empty CSV: {path}")
            need_generate = True

    sample_generated = False
    if need_generate:
        # attempt to run generator script once (do not force overwrite existing files)
        gen_script = project_root / "data" / "generate_sample_data.py"
        try:
            import sys
            import subprocess

            if gen_script.exists():
                subprocess.run([sys.executable, str(gen_script), "--out-dir", str(p)], check=True)
                sample_generated = True
                warnings.append("Sample data generated automatically.")
            else:
                warnings.append(f"Generator script not found: {gen_script}")
        except Exception as e:
            warnings.append(f"Failed to generate sample data: {e}")

    # load CSVs, catching read errors and treating failures as empty DataFrames
    results: Dict[str, pd.DataFrame] = {}
    for name, (path, opts) in required.items():
        df = pd.DataFrame()
        if path.exists() and path.stat().st_size > 0:
            try:
                df = pd.read_csv(path, **opts)
                # ensure DataFrame is not None
                if df is None:
                    df = pd.DataFrame()
                # if empty after read, add warning
                if df.empty:
                    warnings.append(f"Loaded but empty: {path}")
            except Exception as e:
                warnings.append(f"Error reading {path}: {e}")
                df = pd.DataFrame()
        else:
            # already warned above when file missing/empty
            df = pd.DataFrame()
        results[name] = df

    results["__meta__"] = {"warnings": warnings, "sample_generated": sample_generated}
    return results


def aggregate_production(df_prod: pd.DataFrame) -> pd.DataFrame:
    if df_prod.empty:
        return pd.DataFrame()
    grp = df_prod.groupby(["fixture_id"]).agg(produced_count=("produced_count", "sum"), good_count=("good_count", "sum"), reject_count=("reject_count", "sum"))
    grp = grp.reset_index()
    grp["yield"] = grp.apply(lambda r: (r["good_count"] / r["produced_count"] * 100) if r["produced_count"] > 0 else float("nan"), axis=1)
    return grp


def compute_yield_trend(df_prod: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    if df_prod.empty:
        return pd.DataFrame()
    df = df_prod.copy()
    df["date"] = pd.to_datetime(df["production_datetime"]).dt.date
    daily = df.groupby(["fixture_id", "date"]).agg(produced_count=("produced_count", "sum"), good_count=("good_count", "sum")).reset_index()
    daily["yield_pct"] = daily.apply(lambda r: (r["good_count"] / r["produced_count"] * 100) if r["produced_count"] > 0 else float("nan"), axis=1)
    daily = daily.sort_values(["fixture_id", "date"]) .groupby("fixture_id") .apply(lambda g: g.assign(yield_ma=g["yield_pct"].rolling(window, min_periods=1).mean())).reset_index(drop=True)
    return daily


def compute_dimensional_deviation(df_quality: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if df_quality.empty or fixtures.empty:
        return pd.DataFrame()
    df = df_quality.merge(fixtures[["fixture_id", "spec_a", "spec_b"]], on="fixture_id", how="left")
    df["dev_a"] = df["measurement_a"] - df["spec_a"]
    df["dev_b"] = df["measurement_b"] - df["spec_b"]

    results = []
    for fid, group in df.groupby("fixture_id"):
        group = group.sort_values("inspection_datetime")
        times = (group["inspection_datetime"].astype("int64") // 10 ** 9).values.astype(float)
        if len(times) >= 2:
            slope_a = np.polyfit(times, group["dev_a"].values, 1)[0]
            slope_b = np.polyfit(times, group["dev_b"].values, 1)[0]
        else:
            slope_a = float("nan")
            slope_b = float("nan")
        results.append({
            "fixture_id": fid,
            "mean_dev_a": group["dev_a"].mean(),
            "mean_dev_b": group["dev_b"].mean(),
            "slope_dev_a": slope_a,
            "slope_dev_b": slope_b,
        })

    return pd.DataFrame(results)


def compute_fixture_timeseries(fixture_id: str, df_prod: pd.DataFrame, df_qual: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Return a timeseries DataFrame with daily yield and mean deviations for a fixture.

    Columns: date, produced_count, good_count, yield_pct, mean_dev_a, mean_dev_b
    """
    if df_prod.empty:
        return pd.DataFrame()
    prod = df_prod[df_prod["fixture_id"] == fixture_id].copy()
    if prod.empty:
        return pd.DataFrame()
    prod["date"] = pd.to_datetime(prod["production_datetime"]).dt.floor(freq)
    daily = prod.groupby("date").agg(produced_count=("produced_count", "sum"), good_count=("good_count", "sum")).reset_index()
    daily["yield_pct"] = daily.apply(lambda r: (r["good_count"] / r["produced_count"] * 100) if r["produced_count"] > 0 else float("nan"), axis=1)

    # attach mean deviations from quality inspections (if available)
    if not df_qual.empty:
        q = df_qual[df_qual["fixture_id"] == fixture_id].copy()
        if not q.empty:
            q["date"] = pd.to_datetime(q["inspection_datetime"]).dt.floor(freq)
            qagg = q.groupby("date").agg(mean_dev_a=("measurement_a", "mean"), mean_dev_b=("measurement_b", "mean")).reset_index()
            daily = daily.merge(qagg, on="date", how="left")
        else:
            daily["mean_dev_a"] = float("nan")
            daily["mean_dev_b"] = float("nan")
    else:
        daily["mean_dev_a"] = float("nan")
        daily["mean_dev_b"] = float("nan")

    return daily.sort_values("date")


def compute_vendor_performance(fixtures: pd.DataFrame, df_prod: pd.DataFrame, df_qual: pd.DataFrame) -> pd.DataFrame:
    """Compute vendor-level metrics: total produced, good, yield_pct, avg_dev_a, defect_rate.

    Returns a DataFrame keyed by `manufacturer`.
    """
    if fixtures.empty or df_prod.empty:
        return pd.DataFrame()
    merged = df_prod.merge(fixtures[["fixture_id", "manufacturer"]], on="fixture_id", how="left")
    grouped = merged.groupby("manufacturer").agg(total_produced=("produced_count", "sum"), total_good=("good_count", "sum")).reset_index()
    grouped["yield_pct"] = grouped.apply(lambda r: (r["total_good"] / r["total_produced"] * 100) if r["total_produced"] > 0 else float("nan"), axis=1)

    # average deviation per vendor from quality (measurement minus spec)
    if not df_qual.empty:
        qual = df_qual.merge(fixtures[["fixture_id", "manufacturer", "spec_a", "spec_b"]], on="fixture_id", how="left")
        qual["dev_a"] = qual["measurement_a"] - qual["spec_a"]
        qagg = qual.groupby("manufacturer").agg(avg_dev_a=("dev_a", "mean"), avg_dev_b=("measurement_b", "mean"), inspections=("inspection_id", "count")).reset_index()
        res = grouped.merge(qagg, on="manufacturer", how="left")
    else:
        res = grouped.copy()
        res["avg_dev_a"] = float("nan")
        res["avg_dev_b"] = float("nan")
        res["inspections"] = 0

    return res


def _normalize(x: float, lo: float, hi: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if hi == lo:
        return 0.0
    return float(min(max((x - lo) / (hi - lo), 0.0), 1.0))


def compute_health_score(fixture_id: str, data: Dict[str, pd.DataFrame], weights: Dict[str, float] | None = None) -> Dict[str, object]:
    if weights is None:
        weights = {"yield": 0.35, "dev": 0.35, "maint": 0.2, "vendor": 0.1}

    fixtures = data.get("fixtures", pd.DataFrame())
    prod = data.get("production", pd.DataFrame())
    qual = data.get("quality", pd.DataFrame())

    if fixtures.empty or fixture_id not in fixtures["fixture_id"].values:
        return {"fixture_id": fixture_id, "score": None, "components": {}}

    now = pd.Timestamp.utcnow()
    cutoff = now - pd.Timedelta(days=30)
    recent = prod[(prod["fixture_id"] == fixture_id) & (pd.to_datetime(prod["production_datetime"]) >= cutoff)]
    recent_yield = None
    if not recent.empty:
        total_prod = recent["produced_count"].sum()
        total_good = recent["good_count"].sum()
        recent_yield = (total_good / total_prod * 100) if total_prod > 0 else float("nan")

    trend_df = compute_yield_trend(prod, window=7)
    slope = float("nan")
    tdf = trend_df[trend_df["fixture_id"] == fixture_id]
    if not tdf.empty and len(tdf) >= 2:
        x = np.arange(len(tdf))
        y = tdf["yield_ma"].fillna(method="ffill").fillna(0).values
        slope = np.polyfit(x, y, 1)[0]

    dev_df = compute_dimensional_deviation(qual, fixtures)
    dev_row = dev_df[dev_df["fixture_id"] == fixture_id]
    slope_dev_a = float(dev_row["slope_dev_a"]) if not dev_row.empty else float("nan")

    frow = fixtures[fixtures["fixture_id"] == fixture_id].iloc[0]
    last_maint = pd.to_datetime(frow.get("last_maintenance_date", pd.NaT))
    if pd.isna(last_maint):
        days_since_maint = 9999
    else:
        # normalize both timestamps to UTC to avoid tz-aware vs tz-naive subtraction
        last_maint = pd.to_datetime(last_maint, utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        days_since_maint = (now_utc - last_maint).days

    vendor = frow.get("manufacturer")
    vendor_yield = float("nan")
    if vendor and not prod.empty:
        merged = prod.merge(fixtures[["fixture_id", "manufacturer"]], on="fixture_id", how="left")
        v = merged[merged["manufacturer"] == vendor]
        if not v.empty:
            tp = v["produced_count"].sum()
            tg = v["good_count"].sum()
            vendor_yield = (tg / tp * 100) if tp > 0 else float("nan")

    comp_yield = _normalize(recent_yield if recent_yield is not None else float("nan"), 60.0, 100.0)
    abs_slope = abs(slope_dev_a) if not math.isnan(slope_dev_a) else float("nan")
    comp_dev = 1.0 - _normalize(abs_slope, 0.0, 1e-3)
    comp_maint = 1.0 - _normalize(days_since_maint, 0.0, 365.0)
    comp_vendor = _normalize(vendor_yield if not math.isnan(vendor_yield) else float("nan"), 60.0, 100.0)

    score_raw = (weights["yield"] * comp_yield + weights["dev"] * comp_dev + weights["maint"] * comp_maint + weights["vendor"] * comp_vendor)
    score = float(round(score_raw * 100, 2))

    components = {
        "recent_yield_pct": recent_yield,
        "yield_trend_slope": slope,
        "slope_dev_a": slope_dev_a,
        "days_since_maintenance": int(days_since_maint) if not math.isnan(days_since_maint) else None,
        "vendor_yield_pct": vendor_yield,
        "normalized": {"yield": comp_yield, "dev": comp_dev, "maint": comp_maint, "vendor": comp_vendor},
    }

    return {"fixture_id": fixture_id, "score": score, "components": components}


def recommend_maintenance(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 40:
        return "urgent"
    if score < 70:
        return "schedule"
    return "monitor"


def anomaly_detector(series: pd.Series) -> pd.Series:
    if series.dropna().shape[0] >= 50 and IsolationForest is not None:
        clf = IsolationForest(random_state=42, contamination=0.05)
        vals = series.fillna(method="ffill").fillna(0).values.reshape(-1, 1)
        preds = clf.fit_predict(vals)
        return pd.Series(preds == -1, index=series.index)
    else:
        s = series.dropna()
        if s.empty:
            return pd.Series([False] * len(series), index=series.index)
        z = (series - series.mean()) / (series.std() if series.std() != 0 else 1)
        return (z.abs() > 3).fillna(False)

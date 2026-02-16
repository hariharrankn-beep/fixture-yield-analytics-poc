import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from backend import analytics


def load_sample_data():
    base = Path(__file__).resolve().parents[1] / "data"
    fixtures = pd.read_csv(base / "fixtures.csv", parse_dates=["installed_date", "last_maintenance_date"]) 
    production = pd.read_csv(base / "production.csv", parse_dates=["production_datetime"]) 
    quality = pd.read_csv(base / "quality.csv", parse_dates=["inspection_datetime"]) 
    return {"fixtures": fixtures, "production": production, "quality": quality}


def test_aggregate_production():
    data = load_sample_data()
    agg = analytics.aggregate_production(data["production"])
    assert not agg.empty
    row = agg[agg["fixture_id"] == "F0001"].iloc[0]
    assert row["produced_count"] == 100
    assert row["good_count"] == 98
    assert abs(row["yield"] - 98.0) < 1e-6


def test_dimensional_deviation():
    data = load_sample_data()
    dev = analytics.compute_dimensional_deviation(data["quality"], data["fixtures"])
    assert not dev.empty
    row = dev[dev["fixture_id"] == "F0001"].iloc[0]
    assert abs(row["mean_dev_a"] - 0.12) < 1e-6


def test_health_score():
    data = load_sample_data()
    res = analytics.compute_health_score("F0001", data)
    assert res["fixture_id"] == "F0001"
    assert isinstance(res["score"], float)

"""FastAPI app for Fixture Yield Analytics (minimal endpoints).

Provides endpoints to list fixtures and compute basic metrics using `backend.analytics`.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
import math
import numpy as np
from typing import Optional

from . import schemas
from fastapi.middleware.cors import CORSMiddleware

from backend import analytics

app = FastAPI(title="Fixture Yield Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    global DATA
    DATA = analytics.load_data("fixture-analytics/data")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fixtures", response_model=list[schemas.Fixture])
def list_fixtures():
    df = DATA.get("fixtures")
    if df is None or df.empty:
        return []
    # convert datetimes to python types via to_dict
    return df.to_dict(orient="records")


@app.get("/fixtures/{fixture_id}/metrics", response_model=schemas.Metrics)
def fixture_metrics(fixture_id: str):
    df_prod = DATA.get("production")
    df_qual = DATA.get("quality")
    if df_prod is None:
        raise HTTPException(status_code=404, detail="Production data not found")

    # Basic aggregates
    prod_agg = analytics.aggregate_production(df_prod)
    row = prod_agg[prod_agg["fixture_id"] == fixture_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Fixture not found in production data")
    metrics = row.to_dict(orient="records")[0]
    # include a simple deviation summary if quality available
    if not df_qual.empty:
        dev = analytics.compute_dimensional_deviation(df_qual, DATA.get("fixtures"))
        dev_row = dev[dev["fixture_id"] == fixture_id]
        if not dev_row.empty:
            metrics.update(dev_row.to_dict(orient="records")[0])

    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_sanitize(v) for v in o]
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    # create model instance to validate types
    # map 'yield' key to 'yield_pct' via alias in Metrics
    metrics.setdefault("yield", metrics.pop("yield", None))
    sanitized = _sanitize(metrics)
    return schemas.Metrics.parse_obj(sanitized)


@app.get("/fixtures/{fixture_id}/health", response_model=schemas.HealthResponse)
def fixture_health(fixture_id: str, yield_w: Optional[float] = Query(None), dev_w: Optional[float] = Query(None), maint_w: Optional[float] = Query(None), vendor_w: Optional[float] = Query(None)):
    # If weights provided, normalize and pass them to compute_health_score
    weights = None
    if any(v is not None for v in (yield_w, dev_w, maint_w, vendor_w)):
        # default missing to 0 and normalize
        yw = float(yield_w or 0.0)
        dw = float(dev_w or 0.0)
        mw = float(maint_w or 0.0)
        vw = float(vendor_w or 0.0)
        total = yw + dw + mw + vw
        if total <= 0:
            raise HTTPException(status_code=400, detail="At least one weight must be > 0")
        weights = {"yield": yw / total, "dev": dw / total, "maint": mw / total, "vendor": vw / total}

    res = analytics.compute_health_score(fixture_id, DATA, weights=weights)
    if res.get("score") is None:
        raise HTTPException(status_code=404, detail="Fixture metadata not found")
    res["recommendation"] = analytics.recommend_maintenance(res.get("score"))
    # Build HealthResponse model
    comp = res.get("components", {})
    components = schemas.HealthComponents.parse_obj(comp)
    return schemas.HealthResponse(fixture_id=res.get("fixture_id"), score=res.get("score"), components=components, recommendation=res.get("recommendation"))


@app.get("/analytics/fixture/{fixture_id}/timeseries", response_model=schemas.TimeseriesResponse)
def fixture_timeseries(fixture_id: str):
    prod = DATA.get("production")
    qual = DATA.get("quality")
    if prod is None:
        raise HTTPException(status_code=404, detail="Production data not found")
    ts = analytics.compute_fixture_timeseries(fixture_id, prod, qual)

    # sanitize NaNs and numpy types
    def _sanitize_df(df):
        if df is None or df.empty:
            return []
        out = df.copy()
        for c in out.columns:
            out[c] = out[c].apply(lambda v: None if (isinstance(v, float) and (math.isnan(v))) else (v.item() if hasattr(v, 'item') else v))
        return out.to_dict(orient="records")

    # convert sanitized records into TimeseriesPoint objects
    records = _sanitize_df(ts)
    pts = [schemas.TimeseriesPoint.parse_obj(r) for r in records]
    return schemas.TimeseriesResponse(fixture_id=fixture_id, timeseries=pts)


@app.post("/analytics/reload")
def reload_data():
    global DATA
    DATA = analytics.load_data("fixture-analytics/data")
    return {"status": "reloaded"}


@app.get("/fixtures/compare", response_model=schemas.VendorList)
def compare_vendors(vendor: Optional[str] = None):
    fixtures = DATA.get("fixtures")
    prod = DATA.get("production")
    if fixtures is None or fixtures.empty or prod is None:
        return {"vendors": []}
    merged = prod.merge(fixtures[["fixture_id", "manufacturer"]], on="fixture_id", how="left")
    grouped = merged.groupby("manufacturer").agg(produced_count=("produced_count", "sum"), good_count=("good_count", "sum")).reset_index()
    grouped["yield_pct"] = grouped.apply(lambda r: (r["good_count"] / r["produced_count"] * 100) if r["produced_count"] > 0 else float("nan"), axis=1)
    records = res.to_dict(orient="records")
    if vendor:
        row = [r for r in records if r.get("manufacturer") == vendor]
        if not row:
            raise HTTPException(status_code=404, detail="Vendor not found")
        return schemas.VendorList(vendors=[schemas.VendorMetric.parse_obj(row[0])])
    return schemas.VendorList(vendors=[schemas.VendorMetric.parse_obj(r) for r in records])

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Fixture(BaseModel):
    fixture_id: str
    model: Optional[str]
    serial_number: Optional[str]
    manufacturer: Optional[str]
    installed_date: Optional[datetime]
    location: Optional[str]
    machine_stage: Optional[str]
    spec_a: Optional[float]
    spec_b: Optional[float]
    expected_life_days: Optional[int]
    last_maintenance_date: Optional[datetime]
    status: Optional[str]


class Metrics(BaseModel):
    fixture_id: str
    produced_count: Optional[int]
    good_count: Optional[int]
    reject_count: Optional[int]
    yield_pct: Optional[float] = Field(None, alias="yield")
    mean_dev_a: Optional[float]
    mean_dev_b: Optional[float]
    slope_dev_a: Optional[float]
    slope_dev_b: Optional[float]


class HealthComponents(BaseModel):
    recent_yield_pct: Optional[float]
    yield_trend_slope: Optional[float]
    slope_dev_a: Optional[float]
    days_since_maintenance: Optional[int]
    vendor_yield_pct: Optional[float]
    normalized: Optional[dict]


class HealthResponse(BaseModel):
    fixture_id: str
    score: Optional[float]
    components: HealthComponents
    recommendation: Optional[str]


class TimeseriesPoint(BaseModel):
    date: datetime
    produced_count: Optional[int]
    good_count: Optional[int]
    yield_pct: Optional[float]
    mean_dev_a: Optional[float]
    mean_dev_b: Optional[float]


class TimeseriesResponse(BaseModel):
    fixture_id: str
    timeseries: List[TimeseriesPoint]


class VendorMetric(BaseModel):
    manufacturer: Optional[str]
    total_produced: Optional[int]
    total_good: Optional[int]
    yield_pct: Optional[float]
    avg_dev_a: Optional[float]
    avg_dev_b: Optional[float]
    inspections: Optional[int]


class VendorList(BaseModel):
    vendors: List[VendorMetric]

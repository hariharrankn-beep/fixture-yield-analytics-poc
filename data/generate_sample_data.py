"""Generate synthetic CNC fixture data with simple wear and maintenance simulation.

Produces: `fixtures.csv`, `production.csv`, `quality.csv`, and `maintenance_log.csv`.

This generator aims to create realistic variation across vendors, machines,
and fixtures so the dashboard analytics show healthy / monitor / maintenance
required cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from datetime import datetime, timedelta
import csv
import math

import numpy as np

FIXTURES_HEADER = [
    "fixture_id",
    "model",
    "serial_number",
    "manufacturer",
    "installed_date",
    "location",
    "machine_stage",
    "spec_a",
    "spec_b",
    "expected_life_days",
    "last_maintenance_date",
    "status",
]

PRODUCTION_HEADER = [
    "event_id",
    "fixture_id",
    "production_datetime",
    "shift",
    "machine_id",
    "line_id",
    "machine_stage",
    "produced_count",
    "good_count",
    "reject_count",
    "downtime_minutes",
    "lot_number",
    "operator_id",
]

QUALITY_HEADER = [
    "inspection_id",
    "fixture_id",
    "inspection_datetime",
    "production_event_id",
    "inspector_id",
    "defect_code",
    "defect_count",
    "pass",
    "measurement_a",
    "measurement_b",
    "severity",
    "comments",
]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate(out_dir: Path, n_fixtures: int = 9, days: int = 30, events_per_day: int = 1, seed: int = 42, force: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # Vendors with different quality characteristics
    vendors = ["VendorA", "VendorB", "VendorC", "VendorD"]
    # Vendor behavior: (base_reject_rate, drift_multiplier)
    vendor_profile = {
        "VendorA": {"base_reject": 0.005, "drift_mult": 0.6},  # stable
        "VendorB": {"base_reject": 0.01, "drift_mult": 1.0},   # moderate
        "VendorC": {"base_reject": 0.02, "drift_mult": 1.6},   # faster wear
        "VendorD": {"base_reject": 0.008, "drift_mult": 0.9},  # mid
    }

    # Machines, with some machines harsher
    machines = [f"M{str(i).zfill(2)}" for i in range(1, 8)]
    machine_influence = {m: (1.0 + (0.05 if m == "M03" else 0.0) + (0.03 if m == "M06" else 0.0)) for m in machines}

    # Create fixtures
    fixtures_rows = []
    fixture_meta = {}
    today = datetime.utcnow()

    for i in range(1, n_fixtures + 1):
        fid = f"F{str(i).zfill(4)}"
        vendor = vendors[i % len(vendors)]
        model = f"FX-{random.choice(['AL','BT','CN','XR'])}-{random.randint(1,5)}"
        serial = f"SN{random.randint(10000,99999)}"
        installed_date = today - timedelta(days=random.randint(30, 400))
        location = f"Cell-{random.randint(1,6)}"
        machine_stage = random.choice(["finish", "drill", "mill"])
        # specs around nominal 12.0, 0.5 with small variation
        spec_a = round(12.0 + random.uniform(-0.2, 0.2), 3)
        spec_b = round(0.5 + random.uniform(-0.05, 0.05), 3)
        expected_life_days = random.randint(90, 720)

        # per-fixture degradation slope and variability
        base_slope = random.uniform(1e-5, 6e-4)  # mm per produced part cumulative
        # vendor multiplier
        slope = base_slope * vendor_profile[vendor]["drift_mult"]
        # some fixtures are longer-lived (lower slope)
        slope *= random.uniform(0.7, 1.3)

        last_maint = installed_date + timedelta(days=random.randint(10, int(expected_life_days / 4)))
        status = "active"

        fixtures_rows.append([
            fid,
            model,
            serial,
            vendor,
            installed_date.date().isoformat(),
            location,
            machine_stage,
            f"{spec_a:.3f}",
            f"{spec_b:.3f}",
            expected_life_days,
            last_maint.date().isoformat(),
            status,
        ])

        fixture_meta[fid] = {
            "vendor": vendor,
            "spec_a": spec_a,
            "spec_b": spec_b,
            "slope": slope,
            "produced": 0,
            "last_maint": last_maint,
            "expected_life_days": expected_life_days,
        }

    # Optionally write fixtures only if missing or force
    if (out_dir / "fixtures.csv").exists() and not force:
        print(f"Skipping existing file: {out_dir / 'fixtures.csv'}")
    else:
        _write_csv(out_dir / "fixtures.csv", FIXTURES_HEADER, fixtures_rows)
        print(f"Wrote: {out_dir / 'fixtures.csv'}")

    # Generate production events and quality inspections
    prod_rows = []
    qual_rows = []
    maint_rows = []

    event_counter = 1
    insp_counter = 1
    maint_counter = 1

    start_date = today - timedelta(days=days)

    # simulate per-day production events
    for day in range(days):
        day_dt = start_date + timedelta(days=day)
        for fid, meta in fixture_meta.items():
            # randomize events per fixture per day around events_per_day
            n_events = max(0, int(np.random.poisson(events_per_day)))
            for e in range(n_events):
                mach = random.choice(machines)
                machine_factor = machine_influence.get(mach, 1.0)
                # produced count between 20-200 per event
                produced = max(1, int(random.gauss(100, 30)))

                # cumulative produced influences drift
                cum = meta["produced"]
                # drift in mm = slope * cumulative_parts + small noise
                drift = meta["slope"] * (cum + produced) + random.gauss(0, 0.005)

                # base reject prob from vendor + drift effect
                base_reject = vendor_profile[meta["vendor"]]["base_reject"]
                # drift effect increases reject probability (scaled)
                drift_effect = min(0.2, abs(drift) * 0.5)
                reject_prob = base_reject + drift_effect
                # machine adds to reject prob
                if mach == "M03":
                    reject_prob *= 1.25
                if mach == "M06":
                    reject_prob *= 1.15

                # occasional maintenance resets part of drift
                days_since_maint = (day_dt - meta["last_maint"]).days
                # if produced exceeds expected life threshold, schedule maintenance
                if cum > meta["expected_life_days"] * 10 and random.random() < 0.01:
                    # record maintenance event
                    maint_date = day_dt
                    maint_rows.append([
                        f"MT{str(maint_counter).zfill(5)}",
                        fid,
                        maint_date.date().isoformat(),
                        "scheduled",
                        "auto",
                        "generated",
                    ])
                    # partially reset drift and update last_maint
                    meta["last_maint"] = maint_date
                    # reduce cumulative-produced effective drift by 30%
                    meta["slope"] *= 0.9
                    maint_counter += 1

                # compute rejects
                expected_rejects = int(round(produced * reject_prob))
                # random additional rejects for severe vendor/machine combos
                if meta["vendor"] == "VendorC" and mach == "M03" and random.random() < 0.02:
                    expected_rejects += int(produced * 0.05)

                expected_rejects = min(expected_rejects, produced)
                good = produced - expected_rejects

                # downtime random small probability
                downtime = int(max(0, random.gauss(2, 5))) if random.random() < 0.02 else 0

                event_dt = day_dt + timedelta(minutes=random.randint(60, 16 * 60))
                eid = f"E{str(event_counter).zfill(6)}"
                lot = f"LOT{random.randint(1000,9999)}"
                shift = random.choice(["A", "B", "C"])
                prod_rows.append([
                    eid,
                    fid,
                    _iso(event_dt),
                    shift,
                    mach,
                    f"LINE-{random.randint(1,3)}",
                    meta.get("machine_stage", "finish"),
                    produced,
                    good,
                    expected_rejects,
                    downtime,
                    lot,
                    f"OP{random.randint(100,999)}",
                ])

                # chance of an inspection for this event
                if random.random() < 0.12:
                    # measurement influenced by drift and machine
                    meas_a = meta["spec_a"] + drift * machine_factor + random.gauss(0, 0.01)
                    meas_b = meta["spec_b"] + random.gauss(0, 0.005)
                    inspector = f"INSP{random.randint(1,9):02d}"
                    passed = "PASS" if abs(meas_a - meta["spec_a"]) < 0.5 and expected_rejects == 0 else "FAIL"
                    severity = "minor" if passed == "PASS" else random.choice(["major", "critical"]) if random.random() < 0.2 else "major"
                    defect_code = "D01" if passed == "FAIL" else ""
                    comments = "simulated"
                    insp_id = f"I{str(insp_counter).zfill(7)}"
                    qual_rows.append([
                        insp_id,
                        fid,
                        _iso(event_dt + timedelta(minutes=5)),
                        eid,
                        inspector,
                        defect_code,
                        expected_rejects,
                        passed,
                        f"{meas_a:.3f}",
                        f"{meas_b:.3f}",
                        severity,
                        comments,
                    ])
                    insp_counter += 1

                # update counters
                meta["produced"] += produced
                event_counter += 1

    # Post-process to ensure demo coverage: at least 1 healthy, 1 monitor, 1 maintenance-required
    fids_all = list(fixture_meta.keys())
    if len(fids_all) >= 3:
        healthy_fid = fids_all[0]
        monitor_fid = fids_all[1]
        maint_fid = fids_all[2]

        # Healthy fixture: make inspections pass and minimize rejects
        for pr in prod_rows:
            if pr[1] == healthy_fid:
                produced = int(pr[7])
                pr[8] = produced  # good_count = produced
                pr[9] = 0  # reject_count
        for qr in qual_rows:
            if qr[1] == healthy_fid:
                # set measurement close to spec and PASS
                spec_a = fixture_meta[healthy_fid]["spec_a"]
                qr[8] = f"{spec_a + random.gauss(0, 0.002):.3f}"
                qr[7] = "PASS"
                qr[5] = ""
                qr[6] = 0

        # Monitor fixture: moderate rejects/measurements
        for pr in prod_rows:
            if pr[1] == monitor_fid:
                produced = int(pr[7])
                # set a small reject fraction
                rej = max(0, int(produced * 0.03))
                pr[8] = produced - rej
                pr[9] = rej
        for qr in qual_rows:
            if qr[1] == monitor_fid:
                # slightly off spec measurements
                spec_a = fixture_meta[monitor_fid]["spec_a"]
                qr[8] = f"{spec_a + random.gauss(0.01, 0.01):.3f}"
                qr[7] = "PASS" if random.random() > 0.2 else "FAIL"

        # Maintenance-required fixture: many rejects and old maintenance date
        for pr in prod_rows:
            if pr[1] == maint_fid:
                produced = int(pr[7])
                rej = max(1, int(produced * 0.25))
                pr[8] = produced - rej
                pr[9] = rej
        for qr in qual_rows:
            if qr[1] == maint_fid:
                # set measurements showing upward drift
                spec_a = fixture_meta[maint_fid]["spec_a"]
                # make measurement noticeably off
                qr[8] = f"{spec_a + abs(random.gauss(0.2, 0.05)):.3f}"
                qr[7] = "FAIL"
                qr[5] = "D99"
                qr[6] = max(1, int(qr[6] or 1))
        # push last_maintenance_date far in the past for maint_fid
        for fr in fixtures_rows:
            if fr[0] == maint_fid:
                old_date = (today - timedelta(days=400)).date().isoformat()
                fr[10] = old_date

    # write production and quality files
    if (out_dir / "production.csv").exists() and not force:
        print(f"Skipping existing file: {out_dir / 'production.csv'}")
    else:
        _write_csv(out_dir / "production.csv", PRODUCTION_HEADER, prod_rows)
        print(f"Wrote: {out_dir / 'production.csv'}")

    if (out_dir / "quality.csv").exists() and not force:
        print(f"Skipping existing file: {out_dir / 'quality.csv'}")
    else:
        _write_csv(out_dir / "quality.csv", QUALITY_HEADER, qual_rows)
        print(f"Wrote: {out_dir / 'quality.csv'}")

    # maintenance log
    if maint_rows:
        _write_csv(out_dir / "maintenance_log.csv", ["maintenance_id", "fixture_id", "maintenance_date", "maintenance_type", "performed_by", "notes"], maint_rows)
        print(f"Wrote: {out_dir / 'maintenance_log.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CSVs for fixture analytics.")
    parser.add_argument("--out-dir", default="fixture-analytics/data", help="Output directory for CSV files")
    parser.add_argument("--n-fixtures", type=int, default=9, help="Number of fixtures to generate (POC: 8-10)")
    parser.add_argument("--days", type=int, default=30, help="Days of production history (POC: 30)")
    parser.add_argument("--events-per-day", type=int, default=1, help="Average events per day per fixture (POC: moderate)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    generate(out_dir, n_fixtures=args.n_fixtures, days=args.days, events_per_day=args.events_per_day, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()

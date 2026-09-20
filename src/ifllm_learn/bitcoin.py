"""Verified Binance hourly data -> leakage-aware daily decision examples."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import io
import math
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

import numpy as np

from .schema import file_sha256, write_json, write_jsonl

UTC = timezone.utc
HOUR = 3600
OPTIONS = [
    {"id": "up", "description": "The next 24-hour return is greater than +1%."},
    {"id": "flat", "description": "The next 24-hour return is between -1% and +1%, inclusive."},
    {"id": "down", "description": "The next 24-hour return is less than -1%."},
]
QUESTION = "Based only on these completed Bitcoin hourly candles, classify the next 24-hour return relative to the latest observed close."
BOUNDS = [
    ("train", datetime(2025, 1, 1, tzinfo=UTC)),
    ("valid", datetime(2025, 4, 1, tzinfo=UTC)),
    ("calibration", datetime(2025, 7, 1, tzinfo=UTC)),
    ("test", datetime(2026, 1, 1, tzinfo=UTC)),
]


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    complete: bool = True


def month_bounds(month):
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("Month must have YYYY-MM format")
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=UTC)
    return start, end


def months_between(start_month, end_month):
    start, _ = month_bounds(start_month)
    end, _ = month_bounds(end_month)
    if start > end:
        raise ValueError("Start month must not be after end month")
    while start <= end:
        yield start.strftime("%Y-%m")
        _, start = month_bounds(start.strftime("%Y-%m"))


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                data = response.read(16 * 1024 * 1024 + 1)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large monthly artifact: {url}")
            return data
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def verify_archive(data, checksum_text, name):
    fields = checksum_text.strip().split()
    if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) or fields[1].lstrip("*") != name:
        raise ValueError(f"Invalid checksum document for {name}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != fields[0].lower():
        raise ValueError(f"SHA256 mismatch for {name}")
    return actual


def download_month(month, cache_dir):
    month_bounds(month)
    name = f"BTCUSDT-1h-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/{name}"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive, checksum = cache_dir / name, cache_dir / (name + ".CHECKSUM")
    checksum_data = checksum.read_bytes() if checksum.exists() else fetch(url + ".CHECKSUM")
    data = archive.read_bytes() if archive.exists() else fetch(url)
    digest = verify_archive(data, checksum_data.decode("ascii"), name)
    for path, content in ((archive, data), (checksum, checksum_data)):
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
    return data, {"month": month, "url": url, "sha256": digest, "bytes": len(data)}


def parse_month(data, month):
    start, end = month_bounds(month)
    scale = 1_000_000 if start.year >= 2025 else 1_000
    expected_name = f"BTCUSDT-1h-{month}.csv"
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.namelist() != [expected_name]:
            raise ValueError(f"Unexpected ZIP members for {month}")
        if archive.getinfo(expected_name).file_size > 16 * 1024 * 1024:
            raise ValueError("Monthly CSV exceeds size limit")
        text = archive.read(expected_name).decode("utf-8")
    candles = []
    for number, fields in enumerate(csv.reader(io.StringIO(text)), 1):
        if len(fields) != 12:
            raise ValueError(f"{month}:{number}: expected 12 fields")
        raw_time, raw_close = int(fields[0]), int(fields[6])
        if raw_time % (HOUR * scale) or not raw_time <= raw_close <= raw_time + HOUR * scale - 1:
            raise ValueError(f"{month}:{number}: invalid timestamp units or close time")
        timestamp = raw_time // scale
        o, h, l, c, v = map(float, fields[1:6])
        if not all(math.isfinite(value) for value in (o, h, l, c, v)):
            raise ValueError("Non-finite OHLCV")
        if min(o, h, l, c) <= 0 or v < 0 or not l <= min(o, c) <= max(o, c) <= h:
            raise ValueError("Invalid OHLCV bounds")
        candles.append(Candle(timestamp, o, h, l, c, v, raw_close == raw_time + HOUR * scale - 1))
    actual = [c.timestamp for c in candles]
    if not actual or any(not int(start.timestamp()) <= value < int(end.timestamp()) for value in actual):
        raise ValueError(f"{month}: empty or out-of-month candles")
    if any(right <= left for left, right in zip(actual, actual[1:])):
        raise ValueError(f"{month}: duplicate or unordered candles")
    return candles


def quality_report(candles, month):
    start, end = month_bounds(month)
    expected = set(range(int(start.timestamp()), int(end.timestamp()), HOUR))
    missing = sorted(expected - {c.timestamp for c in candles})
    partial = [c.timestamp for c in candles if not c.complete]
    return {
        "expected_hours": len(expected), "observed_hours": len(candles),
        "missing_hours": [datetime.fromtimestamp(t, UTC).isoformat() for t in missing],
        "incomplete_hours": [datetime.fromtimestamp(t, UTC).isoformat() for t in partial],
        "policy": "No imputation; exclude every example whose lookback or label horizon intersects a missing/incomplete hour.",
    }


def features(history):
    if len(history) != 168:
        raise ValueError("Features require exactly 168 completed hourly candles")
    closes = np.asarray([c.close for c in history])
    volumes = np.asarray([c.volume for c in history])
    log_returns = np.diff(np.log(np.r_[history[0].open, closes]))
    volume_mean = float(volumes.mean())
    return {
        "asset": "BTCUSDT spot", "completed_hours": 168,
        "returns": {f"{h}h": round(float(closes[-1] / history[-h].open - 1), 6) for h in (1, 6, 24, 168)},
        "hourly_return_std": {f"{h}h": round(float(log_returns[-h:].std()), 6) for h in (24, 168)},
        "close_to_mean": {f"{h}h": round(float(closes[-1] / closes[-h:].mean() - 1), 6) for h in (24, 168)},
        "last24h_volume_to_weekly_hourly_mean": round(float(volumes[-24:].mean() / volume_mean), 6) if volume_mean else 0.0,
        "last24h_high_low_range": round(max(c.high for c in history[-24:]) / min(c.low for c in history[-24:]) - 1, 6),
    }


def make_examples(candles):
    if not candles:
        raise ValueError("No candles")
    timestamps = [c.timestamp for c in candles]
    if any(t % HOUR for t in timestamps) or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Combined candles must be ordered unique hourly timestamps")
    by_time = {c.timestamp: c for c in candles}
    rows, omissions = [], {"lookback": 0, "horizon": 0, "outage_window": 0}
    first_day = timestamps[0] - timestamps[0] % (24 * HOUR)
    for decision in range(first_day, timestamps[-1] + 1, 24 * HOUR):
        observed = datetime.fromtimestamp(decision, UTC)
        if decision - 168 * HOUR < timestamps[0]:
            omissions["lookback"] += 1
            continue
        if decision + 23 * HOUR > timestamps[-1]:
            omissions["horizon"] += 1
            continue
        history = [by_time.get(t) for t in range(decision - 168 * HOUR, decision, HOUR)]
        future = [by_time.get(t) for t in range(decision, decision + 24 * HOUR, HOUR)]
        if any(c is None or not c.complete for c in [*history, *future]):
            omissions["outage_window"] += 1
            continue
        forward = future[-1].close / history[-1].close - 1
        label = "up" if future[-1].close > history[-1].close * 1.01 else "down" if future[-1].close < history[-1].close * 0.99 else "flat"
        rows.append({
            "id": f"BTCUSDT-{observed:%Y-%m-%d}",
            "prompt": features(history), "question": QUESTION,
            "options": OPTIONS, "label": label,
            "observed_at": observed.isoformat(),
            "label_available_at": (observed + timedelta(hours=24)).isoformat(),
            "metadata": {"forward_return": forward, "last_input_open_at": datetime.fromtimestamp(history[-1].timestamp, UTC).isoformat()},
        })
    return rows, omissions


def split_examples(rows):
    splits = {name: [] for name, _ in BOUNDS}
    purged = {name: 0 for name, _ in BOUNDS}
    for row in rows:
        observed = datetime.fromisoformat(row["observed_at"])
        available = datetime.fromisoformat(row["label_available_at"])
        for name, upper in BOUNDS:
            if observed < upper:
                if available >= upper:
                    purged[name] += 1
                else:
                    splits[name].append(row)
                break
        else:
            raise ValueError("Observation is outside the fixed experiment period")
    return splits, purged


def prepare_bitcoin(output_dir, cache_dir, start_month="2023-01", end_month="2025-12"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Dataset output must be new: {output_dir}")
    all_candles, sources = [], []
    for month in months_between(start_month, end_month):
        data, source = download_month(month, cache_dir)
        parsed = parse_month(data, month)
        source["quality"] = quality_report(parsed, month)
        all_candles.extend(parsed)
        sources.append(source)
        print(f"Verified BTCUSDT {month}", flush=True)
    rows, omissions = make_examples(all_candles)
    splits, purged = split_examples(rows)
    if any(not values for values in splits.values()):
        raise ValueError("The fixed workflow needs nonempty train/valid/calibration/test splits; use 2023-2025")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", values)
    manifest = {
        "format": "ifllm-learn.bitcoin.v1", "sources": sources, "candle_count": len(all_candles),
        "lookback_hours": 168, "horizon_hours": 24, "decision_interval_hours": 24,
        "label_threshold": 0.01, "omitted": omissions, "boundary_purged": purged,
        "split_upper_bounds_exclusive": {name: upper.isoformat() for name, upper in BOUNDS},
        "counts": {name: len(values) for name, values in splits.items()},
        "class_counts": {name: {label: sum(row["label"] == label for row in values) for label in ("up", "flat", "down")} for name, values in splits.items()},
        "sha256": {f"{name}.jsonl": file_sha256(output_dir / f"{name}.jsonl") for name in splits},
        "limitations": ["Historical BTC information may have occurred in base-model pretraining; this is not a prospective forecast trial.", "Classification only, no trading, execution-price assumptions or profitability claims.", "Calibrated confidence is not guaranteed under future distribution shift."],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest

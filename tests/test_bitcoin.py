from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import zipfile

import pytest

from ifllm_learn.bitcoin import (
    Candle, HOUR, features, make_examples, month_bounds, months_between,
    parse_month, quality_report, split_examples, verify_archive,
)

UTC = timezone.utc


def archive_for(month, mutate=None):
    start, end = month_bounds(month)
    scale = 1_000_000 if start.year >= 2025 else 1000
    lines = []
    for second in range(int(start.timestamp()), int(end.timestamp()), HOUR):
        fields = [second * scale, 100, 102, 99, 101, 12, (second + HOUR) * scale - 1, 1212, 5, 6, 606, 0]
        lines.append(fields)
    if mutate:
        mutate(lines)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(f"BTCUSDT-1h-{month}.csv", "\n".join(",".join(map(str, fields)) for fields in lines))
    return output.getvalue()


@pytest.mark.parametrize("month,hours", [("2024-02", 696), ("2025-02", 672)])
def test_parse_both_timestamp_units_and_leap_year(month, hours):
    candles = parse_month(archive_for(month), month)
    assert len(candles) == hours
    assert candles[1].timestamp - candles[0].timestamp == HOUR
    assert candles[0].close == 101


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.insert(3, rows[2]),
    lambda rows: rows[0].__setitem__(2, 90),
    lambda rows: rows[0].__setitem__(4, "nan"),
    lambda rows: rows[0].__setitem__(5, -1),
    lambda rows: rows[0].__setitem__(6, rows[0][6] + 1),
    lambda rows: rows[0].__setitem__(0, rows[0][0] * 1000),
])
def test_reject_corrupt_candles(mutate):
    with pytest.raises(ValueError):
        parse_month(archive_for("2024-01", mutate), "2024-01")


def test_checksum_requires_matching_filename_and_content():
    data, name = b"test data", "month.zip"
    digest = hashlib.sha256(data).hexdigest()
    assert verify_archive(data, f"{digest}  {name}\n", name) == digest
    with pytest.raises(ValueError, match="mismatch"):
        verify_archive(b"corrupt", f"{digest}  {name}", name)
    with pytest.raises(ValueError, match="Invalid checksum"):
        verify_archive(data, f"{digest}  other.zip", name)


def make_candles(days=20):
    start = int(datetime(2024, 12, 20, tzinfo=UTC).timestamp())
    return [Candle(start + i * HOUR, 100, 102, 99, 100, 10) for i in range(days * 24)]


def test_future_changes_label_not_input_features():
    candles = make_candles()
    original, omissions = make_examples(candles)
    assert omissions["lookback"] == 7
    assert original[0]["label"] == "flat"
    modified = candles.copy()
    modified[168 + 23] = replace(modified[168 + 23], close=110, high=110)
    changed, _ = make_examples(modified)
    assert changed[0]["label"] == "up"
    assert changed[0]["prompt"] == original[0]["prompt"]
    assert changed[0]["question"] == original[0]["question"]
    modified[168 + 23] = replace(modified[168 + 23], close=90, low=90)
    assert make_examples(modified)[0][0]["label"] == "down"
    assert "forward_return" not in original[0]["prompt"]
    assert datetime.fromisoformat(original[0]["label_available_at"]) - datetime.fromisoformat(original[0]["observed_at"]) == timedelta(hours=24)


def test_purge_equal_boundary_label_and_keep_chronology():
    rows, _ = make_examples(make_candles())
    splits, purged = split_examples(rows)
    assert purged["train"] == 1
    assert all(row["observed_at"] < "2025-01-01" and row["label_available_at"] < "2025-01-01" for row in splits["train"])
    assert splits["valid"][0]["observed_at"].startswith("2025-01-01")
    assert not ({row["id"] for row in splits["train"]} & {row["id"] for row in splits["valid"]})


def test_missing_and_incomplete_candles_exclude_affected_examples():
    candles = make_candles()
    intact, _ = make_examples(candles)
    missing, omissions = make_examples(candles[:180] + candles[181:])
    assert omissions["outage_window"] == 8
    assert len(missing) == len(intact) - 8
    partial = candles.copy()
    partial[180] = replace(partial[180], complete=False)
    assert make_examples(partial)[0] == missing
    with pytest.raises(ValueError, match="unique hourly"):
        make_examples(candles[:20] + [candles[19]] + candles[20:])
    with pytest.raises(ValueError, match="168"):
        features(candles[:12])


def test_monthly_outage_audit_preserves_exact_missing_and_partial_hours():
    def outage(rows):
        rows[100][6] -= 120000
        rows.pop(101)
    candles = parse_month(archive_for("2023-03", outage), "2023-03")
    audit = quality_report(candles, "2023-03")
    start, _ = month_bounds("2023-03")
    assert audit["missing_hours"] == [(start + timedelta(hours=101)).isoformat()]
    assert audit["incomplete_hours"] == [(start + timedelta(hours=100)).isoformat()]
    assert audit["expected_hours"] - audit["observed_hours"] == 1


@pytest.mark.parametrize("close", [99.0, 101.0])
def test_exact_one_percent_returns_are_flat(close):
    candles = make_candles()
    candles[191] = replace(candles[191], close=close)
    assert make_examples(candles)[0][0]["label"] == "flat"


def test_month_range_validation():
    assert list(months_between("2024-12", "2025-02")) == ["2024-12", "2025-01", "2025-02"]
    with pytest.raises(ValueError):
        list(months_between("2025-02", "2025-01"))

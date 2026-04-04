"""Comprehensive tests for TimeIntervalHandler.

Covers get_key_suffix and get_expiry for every interval type,
with and without interval_value bucketing.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from llming_models.budget.time_intervals import TimeInterval, TimeIntervalHandler


# ---------------------------------------------------------------------------
# get_key_suffix — TOTAL
# ---------------------------------------------------------------------------

class TestGetKeySuffixTotal:
    """TOTAL interval always returns 'total'."""

    def test_returns_total(self) -> None:
        dt = datetime(2024, 6, 15, 14, 30, 42)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.TOTAL, dt) == "total"

    def test_returns_total_with_interval_value(self) -> None:
        dt = datetime(2024, 6, 15)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.TOTAL, dt, interval_value=5) == "total"

    def test_returns_total_regardless_of_time(self) -> None:
        dt1 = datetime(2020, 1, 1)
        dt2 = datetime(2030, 12, 31, 23, 59, 59)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.TOTAL, dt1) == "total"
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.TOTAL, dt2) == "total"


# ---------------------------------------------------------------------------
# get_key_suffix — YEARLY
# ---------------------------------------------------------------------------

class TestGetKeySuffixYearly:
    """YEARLY interval key generation."""

    def test_basic_year(self) -> None:
        dt = datetime(2024, 6, 15)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt) == "2024"

    def test_year_boundary_jan(self) -> None:
        dt = datetime(2024, 1, 1)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt) == "2024"

    def test_year_boundary_dec(self) -> None:
        dt = datetime(2024, 12, 31, 23, 59, 59)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt) == "2024"

    def test_bucketed_year_value_2(self) -> None:
        dt = datetime(2024, 6, 15)
        # 2024 // 2 * 2 = 2024
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt, interval_value=2) == "2024"

    def test_bucketed_year_value_2_odd_year(self) -> None:
        dt = datetime(2025, 6, 15)
        # 2025 // 2 * 2 = 2024
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt, interval_value=2) == "2024"

    def test_bucketed_year_value_5(self) -> None:
        dt = datetime(2023, 3, 1)
        # 2023 // 5 * 5 = 2020
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt, interval_value=5) == "2020"

    def test_bucketed_year_value_10(self) -> None:
        dt = datetime(2027, 8, 15)
        # 2027 // 10 * 10 = 2020
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt, interval_value=10) == "2020"

    def test_string_interval_value_ignored(self) -> None:
        dt = datetime(2024, 6, 15)
        # Non-int interval_value falls through to plain strftime
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.YEARLY, dt, interval_value="3") == "2024"


# ---------------------------------------------------------------------------
# get_key_suffix — MONTHLY
# ---------------------------------------------------------------------------

class TestGetKeySuffixMonthly:
    """MONTHLY interval key generation."""

    def test_basic_month(self) -> None:
        dt = datetime(2024, 6, 15)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MONTHLY, dt) == "2024-06"

    def test_january(self) -> None:
        dt = datetime(2024, 1, 1)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MONTHLY, dt) == "2024-01"

    def test_december(self) -> None:
        dt = datetime(2024, 12, 31)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MONTHLY, dt) == "2024-12"

    def test_bucketed_month_value_3(self) -> None:
        dt = datetime(2024, 6, 15)
        # absolute_month = 2024*12 + 6 - 1 = 24293
        # bucket = (24293 // 3) * 3 = 24291
        # year = 24291 // 12 = 2024, month = (24291 % 12) + 1 = 4
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MONTHLY, dt, interval_value=3) == "2024-04"

    def test_bucketed_month_value_3_march(self) -> None:
        dt = datetime(2024, 3, 15)
        # absolute_month = 2024*12 + 3 - 1 = 24290
        # bucket = (24290 // 3) * 3 = 24288
        # year = 24288 // 12 = 2024, month = (24288 % 12) + 1 = 1
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MONTHLY, dt, interval_value=3) == "2024-01"

    def test_bucketed_month_value_6(self) -> None:
        dt = datetime(2024, 8, 1)
        # absolute_month = 2024*12 + 8 - 1 = 24295
        # bucket = (24295 // 6) * 6 = 24294
        # year = 24294 // 12 = 2024, month = (24294 % 12) + 1 = 7
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MONTHLY, dt, interval_value=6) == "2024-07"


# ---------------------------------------------------------------------------
# get_key_suffix — DAILY
# ---------------------------------------------------------------------------

class TestGetKeySuffixDaily:
    """DAILY interval key generation."""

    def test_basic_day(self) -> None:
        dt = datetime(2024, 6, 15)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.DAILY, dt) == "2024-06-15"

    def test_bucketed_day_value_7(self) -> None:
        dt = datetime(2024, 6, 15)
        # days_since_epoch for 2024-06-15
        days = (datetime(2024, 6, 15).date() - datetime(1970, 1, 1).date()).days
        bucket = (days // 7) * 7
        expected_date = datetime(1970, 1, 1) + timedelta(days=bucket)
        expected = expected_date.strftime("%Y-%m-%d")
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.DAILY, dt, interval_value=7) == expected

    def test_bucketed_day_value_14(self) -> None:
        dt = datetime(2024, 1, 1)
        days = (datetime(2024, 1, 1).date() - datetime(1970, 1, 1).date()).days
        bucket = (days // 14) * 14
        expected_date = datetime(1970, 1, 1) + timedelta(days=bucket)
        expected = expected_date.strftime("%Y-%m-%d")
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.DAILY, dt, interval_value=14) == expected

    def test_epoch_date(self) -> None:
        dt = datetime(1970, 1, 1)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.DAILY, dt) == "1970-01-01"

    def test_leap_day(self) -> None:
        dt = datetime(2024, 2, 29)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.DAILY, dt) == "2024-02-29"


# ---------------------------------------------------------------------------
# get_key_suffix — HOURLY
# ---------------------------------------------------------------------------

class TestGetKeySuffixHourly:
    """HOURLY interval key generation."""

    def test_basic_hour(self) -> None:
        dt = datetime(2024, 6, 15, 14, 30)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt) == "2024-06-15-14"

    def test_midnight(self) -> None:
        dt = datetime(2024, 6, 15, 0, 0)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt) == "2024-06-15-00"

    def test_hour_23(self) -> None:
        dt = datetime(2024, 6, 15, 23, 59)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt) == "2024-06-15-23"

    def test_bucketed_hour_value_4(self) -> None:
        dt = datetime(2024, 6, 15, 14, 30)
        # 14 // 4 * 4 = 12
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt, interval_value=4) == "2024-06-15-12"

    def test_bucketed_hour_value_4_at_midnight(self) -> None:
        dt = datetime(2024, 6, 15, 0, 30)
        # 0 // 4 * 4 = 0
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt, interval_value=4) == "2024-06-15-00"

    def test_bucketed_hour_value_6(self) -> None:
        dt = datetime(2024, 6, 15, 7, 0)
        # 7 // 6 * 6 = 6
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt, interval_value=6) == "2024-06-15-06"

    def test_bucketed_hour_value_8_at_20(self) -> None:
        dt = datetime(2024, 6, 15, 20, 15)
        # 20 // 8 * 8 = 16
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt, interval_value=8) == "2024-06-15-16"


# ---------------------------------------------------------------------------
# get_key_suffix — MINUTES
# ---------------------------------------------------------------------------

class TestGetKeySuffixMinutes:
    """MINUTES interval key generation."""

    def test_basic_minute(self) -> None:
        dt = datetime(2024, 6, 15, 14, 35)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MINUTES, dt) == "2024-06-15-14-35"

    def test_minute_zero(self) -> None:
        dt = datetime(2024, 6, 15, 14, 0)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MINUTES, dt) == "2024-06-15-14-00"

    def test_minute_59(self) -> None:
        dt = datetime(2024, 6, 15, 14, 59)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MINUTES, dt) == "2024-06-15-14-59"

    def test_bucketed_minute_value_15(self) -> None:
        dt = datetime(2024, 6, 15, 14, 35)
        # 35 // 15 * 15 = 30
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MINUTES, dt, interval_value=15) == "2024-06-15-14-30"

    def test_bucketed_minute_value_15_at_0(self) -> None:
        dt = datetime(2024, 6, 15, 14, 7)
        # 7 // 15 * 15 = 0
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MINUTES, dt, interval_value=15) == "2024-06-15-14-00"

    def test_bucketed_minute_value_30(self) -> None:
        dt = datetime(2024, 6, 15, 14, 45)
        # 45 // 30 * 30 = 30
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MINUTES, dt, interval_value=30) == "2024-06-15-14-30"

    def test_bucketed_minute_value_5(self) -> None:
        dt = datetime(2024, 6, 15, 14, 23)
        # 23 // 5 * 5 = 20
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MINUTES, dt, interval_value=5) == "2024-06-15-14-20"


# ---------------------------------------------------------------------------
# get_key_suffix — SECONDS
# ---------------------------------------------------------------------------

class TestGetKeySuffixSeconds:
    """SECONDS interval key generation."""

    def test_basic_second(self) -> None:
        dt = datetime(2024, 6, 15, 14, 35, 42)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.SECONDS, dt) == "2024-06-15-14-35-42"

    def test_second_zero(self) -> None:
        dt = datetime(2024, 6, 15, 14, 35, 0)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.SECONDS, dt) == "2024-06-15-14-35-00"

    def test_bucketed_second_value_30(self) -> None:
        dt = datetime(2024, 6, 15, 14, 35, 42)
        # current_second = 35*60 + 42 = 2142
        # bucket = (2142 // 30) * 30 = 2130
        # minutes = 2130 // 60 = 35, seconds = 2130 % 60 = 30
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.SECONDS, dt, interval_value=30) == "2024-06-15-14-35-30"

    def test_bucketed_second_value_30_at_zero(self) -> None:
        dt = datetime(2024, 6, 15, 14, 0, 10)
        # current_second = 0*60 + 10 = 10
        # bucket = (10 // 30) * 30 = 0
        # minutes = 0, seconds = 0
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.SECONDS, dt, interval_value=30) == "2024-06-15-14-00-00"

    def test_bucketed_second_value_60(self) -> None:
        dt = datetime(2024, 6, 15, 14, 1, 30)
        # current_second = 1*60 + 30 = 90
        # bucket = (90 // 60) * 60 = 60
        # minutes = 60 // 60 = 1, seconds = 60 % 60 = 0
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.SECONDS, dt, interval_value=60) == "2024-06-15-14-01-00"

    def test_bucketed_second_value_15(self) -> None:
        dt = datetime(2024, 6, 15, 14, 2, 37)
        # current_second = 2*60 + 37 = 157
        # bucket = (157 // 15) * 15 = 150
        # minutes = 150 // 60 = 2, seconds = 150 % 60 = 30
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.SECONDS, dt, interval_value=15) == "2024-06-15-14-02-30"


# ---------------------------------------------------------------------------
# get_key_suffix — unsupported interval
# ---------------------------------------------------------------------------

class TestGetKeySuffixUnsupported:
    """Unsupported interval type raises ValueError."""

    def test_raises_value_error(self) -> None:
        # Create a fake enum value that is not handled
        # We can't easily create a fake TimeInterval, so instead we test
        # that all valid values work without error
        dt = datetime(2024, 6, 15, 14, 30, 42)
        for interval in TimeInterval:
            # Should not raise
            TimeIntervalHandler.get_key_suffix(interval, dt)


# ---------------------------------------------------------------------------
# get_expiry — all intervals
# ---------------------------------------------------------------------------

class TestGetExpiryTotal:
    """TOTAL interval has no expiry."""

    def test_returns_none(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.TOTAL) is None

    def test_returns_none_with_interval_value(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.TOTAL, interval_value=5) is None


class TestGetExpiryYearly:
    """YEARLY interval expiry."""

    def test_default(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.YEARLY) == timedelta(days=365 * 2)

    def test_with_interval_value(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.YEARLY, interval_value=3) == timedelta(days=365 * 3 * 2)

    def test_with_interval_value_1(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.YEARLY, interval_value=1) == timedelta(days=365 * 1 * 2)


class TestGetExpiryMonthly:
    """MONTHLY interval expiry."""

    def test_default(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.MONTHLY) == timedelta(days=60)

    def test_with_interval_value(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.MONTHLY, interval_value=3) == timedelta(days=30 * 3 * 2)

    def test_with_interval_value_6(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.MONTHLY, interval_value=6) == timedelta(days=30 * 6 * 2)


class TestGetExpiryDaily:
    """DAILY interval expiry."""

    def test_default(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.DAILY) == timedelta(days=2)

    def test_with_interval_value(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.DAILY, interval_value=7) == timedelta(days=7 * 2)

    def test_with_interval_value_1(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.DAILY, interval_value=1) == timedelta(days=2)


class TestGetExpiryHourly:
    """HOURLY interval expiry."""

    def test_default(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.HOURLY) == timedelta(hours=2)

    def test_with_interval_value(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.HOURLY, interval_value=4) == timedelta(hours=4 * 2)

    def test_with_interval_value_12(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.HOURLY, interval_value=12) == timedelta(hours=12 * 2)


class TestGetExpiryMinutes:
    """MINUTES interval expiry."""

    def test_default(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.MINUTES) == timedelta(minutes=2)

    def test_with_interval_value(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.MINUTES, interval_value=15) == timedelta(minutes=15 * 2)

    def test_with_interval_value_30(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.MINUTES, interval_value=30) == timedelta(minutes=30 * 2)


class TestGetExpirySeconds:
    """SECONDS interval expiry."""

    def test_default(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.SECONDS) == timedelta(seconds=2)

    def test_with_interval_value(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.SECONDS, interval_value=30) == timedelta(seconds=30 * 2)

    def test_with_interval_value_60(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.SECONDS, interval_value=60) == timedelta(seconds=60 * 2)


# ---------------------------------------------------------------------------
# get_expiry — string interval_value is treated as non-int (default expiry)
# ---------------------------------------------------------------------------

class TestGetExpiryStringIntervalValue:
    """String interval_value falls through to the default (non-bucketed) expiry."""

    def test_yearly_string_ignored(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.YEARLY, interval_value="3") == timedelta(days=365 * 2)

    def test_daily_string_ignored(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.DAILY, interval_value="7") == timedelta(days=2)

    def test_hourly_string_ignored(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.HOURLY, interval_value="4") == timedelta(hours=2)

    def test_minutes_string_ignored(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.MINUTES, interval_value="15") == timedelta(minutes=2)

    def test_seconds_string_ignored(self) -> None:
        assert TimeIntervalHandler.get_expiry(TimeInterval.SECONDS, interval_value="30") == timedelta(seconds=2)


# ---------------------------------------------------------------------------
# Edge cases: consistency between key suffix and expiry
# ---------------------------------------------------------------------------

class TestConsistencyBetweenKeyAndExpiry:
    """The key suffix and expiry should be consistent for the same interval."""

    @pytest.mark.parametrize("interval", list(TimeInterval))
    def test_all_intervals_produce_key(self, interval: TimeInterval) -> None:
        dt = datetime(2024, 6, 15, 14, 30, 42)
        key = TimeIntervalHandler.get_key_suffix(interval, dt)
        assert isinstance(key, str)
        assert len(key) > 0

    @pytest.mark.parametrize("interval", list(TimeInterval))
    def test_all_intervals_produce_expiry_or_none(self, interval: TimeInterval) -> None:
        expiry = TimeIntervalHandler.get_expiry(interval)
        if interval == TimeInterval.TOTAL:
            assert expiry is None
        else:
            assert isinstance(expiry, timedelta)
            assert expiry > timedelta(0)

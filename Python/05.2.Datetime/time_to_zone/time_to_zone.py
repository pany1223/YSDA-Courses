from datetime import datetime
from zoneinfo import ZoneInfo
import os

DEFAULT_TZ_NAME = "Europe/Moscow"


def now() -> datetime:
    """Return now in default timezone"""
    return datetime.now(tz=ZoneInfo(DEFAULT_TZ_NAME))


def strftime(dt: datetime, fmt: str) -> str:
    """Return dt converted to string according to format in default timezone"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ_NAME))
    dt = dt.astimezone(ZoneInfo(DEFAULT_TZ_NAME))
    return datetime.strftime(dt, fmt)


def strptime(dt_str: str, fmt: str) -> datetime:
    """Return dt parsed from string according to format in default timezone"""
    os.environ["TZ"] = DEFAULT_TZ_NAME
    return datetime.strptime(dt_str, fmt).astimezone(ZoneInfo(DEFAULT_TZ_NAME))\
        .replace(tzinfo=ZoneInfo(DEFAULT_TZ_NAME))


def diff(first_dt: datetime, second_dt: datetime) -> int:
    """Return seconds between two datetimes rounded to int"""
    if first_dt.tzinfo is None:
        first_dt = first_dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ_NAME))
    if second_dt.tzinfo is None:
        second_dt = second_dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ_NAME))
    return int(second_dt.timestamp() - first_dt.timestamp())


def timestamp(dt: datetime) -> int:
    """Return timestamp for given datetime rounded to int"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ_NAME))
    return int(dt.timestamp())


def from_timestamp(ts: float) -> datetime:
    """Return datetime from given timestamp"""
    return datetime.fromtimestamp(ts).astimezone(ZoneInfo(DEFAULT_TZ_NAME))

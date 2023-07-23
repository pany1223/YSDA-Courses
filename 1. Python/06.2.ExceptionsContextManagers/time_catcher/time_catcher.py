from __future__ import annotations
import time
from typing import Optional, Type
from types import TracebackType


class TimeoutException(Exception):
    pass


class SoftTimeoutException(TimeoutException):
    pass


class HardTimeoutException(TimeoutException):
    pass


class TimeCatcher:
    def __init__(self,
                 soft_timeout: Optional[float] = 1e10,
                 hard_timeout: Optional[float] = 1e10) -> None:
        self.soft_timeout = soft_timeout
        self.hard_timeout = hard_timeout

    def __float__(self) -> float:
        delta = time.time() - self.start_time
        return float(delta)

    def __str__(self) -> str:
        return 'Time consumed: ' + str(self.delta)

    def __enter__(self) -> TimeCatcher:
        self.start_time = time.time()
        return self

    def __exit__(self,
                 exc_type: Optional[Type[BaseException]],
                 exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> None:
        time_now = time.time()
        delta = time_now - self.start_time
        self.delta = delta

        if self.soft_timeout is not None:
            # soft not None
            if self.hard_timeout is not None:
                # soft not None, hard not None
                if (self.hard_timeout <= 0 or
                        self.soft_timeout <= 0 or
                        self.hard_timeout < self.soft_timeout):
                    raise AssertionError
            else:
                # soft not None, hard None
                if self.soft_timeout <= 0:
                    raise AssertionError
        else:
            # soft None
            if self.hard_timeout is None:
                # soft None, hard None
                pass
            else:
                # soft None, hard not None
                if self.hard_timeout <= 0:
                    raise AssertionError

        if self.soft_timeout is not None and self.delta > self.soft_timeout:
            raise SoftTimeoutException
        if self.hard_timeout is not None and self.delta > self.hard_timeout:
            raise HardTimeoutException

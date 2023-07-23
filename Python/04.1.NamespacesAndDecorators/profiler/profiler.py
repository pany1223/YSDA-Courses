from datetime import datetime
import functools


def profiler(func):  # type: ignore
    """
    Returns profiling decorator, which counts calls of function
    and measure last function execution time.
    Results are stored as function attributes: `calls`, `last_time_taken`
    :param func: function to decorate
    :return: decorator, which wraps any function passed
    """
    profiler.calls = 0

    @functools.wraps(func)  # to keep __name__, __doc__, __module__
    def wrapper(*args, **kwargs):  # type: ignore
        if profiler.calls == 0:
            wrapper.calls = 0   # set 0 for new run (last) of function

        profiler.calls += 1

        start_dt = datetime.now()
        res = func(*args, **kwargs)
        delta = (datetime.now() - start_dt).total_seconds()

        profiler.calls -= 1
        wrapper.calls += 1
        wrapper.last_time_taken = delta
        return res

    return wrapper

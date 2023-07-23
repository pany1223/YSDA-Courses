from typing import Callable, Any, TypeVar, cast
import functools
from collections import OrderedDict


Function = TypeVar('Function', bound=Callable[..., Any])


def cache(max_size: int) -> Callable[[Function], Function]:
    """
    Returns decorator, which stores result of function
    for `max_size` most recent function arguments.
    :param max_size: max amount of unique arguments to store values for
    :return: decorator, which wraps any function passed
    """
    caches: Any = OrderedDict()
    kwd_mark = object()

    def decorator(func: Function) -> Function:

        @functools.wraps(func)  # to keep __name__, __doc__, __module__
        def wrapper(*args, **kwargs):  # type: ignore
            key = args + (kwd_mark,) + tuple(sorted(kwargs.items()))
            hash_key = hash(key)
            if hash_key in caches.keys():
                res = caches[hash_key]
            else:
                res = func(*args, **kwargs)
                if len(caches) == max_size:
                    caches.popitem(last=False)
                caches[hash_key] = res
            return res
        return cast(Function, wrapper)
    return decorator

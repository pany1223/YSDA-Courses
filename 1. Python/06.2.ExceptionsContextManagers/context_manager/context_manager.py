from contextlib import contextmanager
from typing import Iterator, Optional, TextIO, Type
import traceback
import sys


@contextmanager
def supresser(*types_: Type[BaseException]) -> Iterator[None]:
    try:
        yield None
    except Exception as ex:
        type_ex, value, trace = sys.exc_info()
        if type_ex in types_:
            pass
        else:
            raise ex


@contextmanager
def retyper(type_from: Type[BaseException], type_to: Type[BaseException]) -> Iterator[None]:
    try:
        yield None
    except Exception as exc:
        type_ex, value, trace = sys.exc_info()
        if type_ex == type_from:
            if value is not None:
                raise type_to(*value.args)
            else:
                raise type_to
        else:
            raise exc


@contextmanager
def dumper(stream: Optional[TextIO] = None) -> Iterator[None]:
    try:
        yield None
    except Exception as exc:
        type_ex, value, trace = sys.exc_info()
        if stream is None:
            sys.stderr.write(traceback.format_exception_only(type_ex, value)[0])
        else:
            stream.write(traceback.format_exception_only(type_ex, value)[0])
        raise exc

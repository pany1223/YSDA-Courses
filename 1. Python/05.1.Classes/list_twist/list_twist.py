from collections import UserList
import typing as tp


# https://github.com/python/mypy/issues/5264#issuecomment-399407428
if tp.TYPE_CHECKING:
    BaseList = UserList[tp.Optional[tp.Any]]
else:
    BaseList = UserList


class ListTwist(BaseList):
    """
    List-like class with additional attributes:
        * reversed, R - return reversed list
        * first, F - insert or retrieve first element;
                     Undefined for empty list
        * last, L -  insert or retrieve last element;
                     Undefined for empty list
        * size, S -  set or retrieve size of list;
                     If size less than list length - truncate to size;
                     If size greater than list length - pad with Nones
    """
    def __init__(self, data: tp.Any = None) -> None:
        self.data = list(data) if data else []

    def _reversed_getter(self) -> tp.Any:
        return self.data[::-1]

    reversed = property(_reversed_getter)
    R = property(_reversed_getter)

    def _first_getter(self) -> tp.Any:
        return self.data[0]

    def _first_setter(self, x: tp.Any) -> None:
        self.data[0] = x

    first = property(_first_getter, _first_setter)
    F = property(_first_getter, _first_setter)

    def _last_getter(self) -> tp.Any:
        return self.data[-1]

    def _last_setter(self, x: tp.Any) -> None:
        self.data[-1] = x

    last = property(_last_getter, _last_setter)
    L = property(_last_getter, _last_setter)

    def _size_getter(self) -> tp.Any:
        return len(self.data)

    def _size_setter(self, x: tp.Any) -> None:
        if x < len(self.data):
            self.data = self.data[:x]
        else:
            self.data += [None] * (x - len(self.data))

    size = property(_size_getter, _size_setter)
    S = property(_size_getter, _size_setter)

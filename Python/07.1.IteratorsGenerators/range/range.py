from typing import Iterable, Sized


class Range(Sized, Iterable[int]):
    """The range-like type, which represents an immutable sequence of numbers"""
    class RangeIterator:
        def __init__(self,
                     start: int,
                     stop: int,
                     step: int) -> None:
            self.index = start
            self.start = start
            self.stop = stop
            self.step = step

        def __next__(self) -> int:
            if (self.start > self.stop and self.step == 1) or \
                    (self.start < self.stop and self.step < 0):
                raise StopIteration()

            if self.start > self.stop:
                if self.index <= self.stop:
                    raise StopIteration()
                value = self.index
                self.index += self.step
                return value
            else:
                if self.index >= self.stop:
                    raise StopIteration()
                value = self.index
                self.index += self.step
                return value

    def __init__(self, *args: int) -> None:
        """
        :param args: either it's a single `stop` argument
            or sequence of `start, stop[, step]` arguments.
        If the `step` argument is omitted, it defaults to 1.
        If the `start` argument is omitted, it defaults to 0.
        If `step` is zero, ValueError is raised.
        """
        if len(args) == 1:
            self.start = 0
            self.stop = args[0]
            self.step = 1
        elif len(args) == 2:
            self.start = args[0]
            self.stop = args[1]
            self.step = 1
        elif len(args) == 3:
            if args[2] == 0:
                raise ValueError
            else:
                self.start = args[0]
                self.stop = args[1]
                self.step = args[2]
        else:
            raise ValueError

    def __iter__(self) -> RangeIterator:  # type: ignore
        return self.RangeIterator(self.start, self.stop, self.step)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        base = f'range({self.start}, {self.stop}'
        if self.step != 1:
            extra = f', {self.step})'
        else:
            extra = ')'
        return base + extra

    def __contains__(self, key: int) -> bool:
        if self.step == 1:
            return self.start <= key <= self.stop
        else:
            for item in self:
                if item is key or item == key:
                    return True
            return False

    def __getitem__(self, key: int) -> int:
        if key < 0 or key >= self.stop:
            raise IndexError(key)
        i = 0
        for item in self:
            if i == key:
                return item
            i += 1
        raise IndexError(key)

    def __len__(self) -> int:
        i = 0
        for _ in self:
            i += 1
        return i

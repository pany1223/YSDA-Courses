from typing import Iterable, Generator, Any


def flat_it(sequence: Iterable[Any]) -> Generator[Any, None, None]:
    """
    :param sequence: sequence with arbitrary level of nested iterables
    :return: generator producing flatten sequence
    """
    try:
        for subseq in sequence:
            if (isinstance(subseq, str)) and (len(subseq) == 1):
                yield subseq
            else:
                for el in flat_it(subseq):
                    yield el
    except TypeError:
        yield sequence

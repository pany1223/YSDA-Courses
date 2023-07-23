import typing as tp
import heapq


def merge(seq: tp.Sequence[tp.Sequence[int]]) -> list[int]:
    """
    :param seq: sequence of sorted sequences
    :return: merged sorted list
    """
    # the short solution: return list(heapq.merge(*seq))
    heap: tp.List[int] = []
    for lst in seq:
        for elem in lst:
            heapq.heappush(heap, elem)
    sorted_list = []
    while heap:
        sorted_list.append(heapq.heappop(heap))
    return sorted_list

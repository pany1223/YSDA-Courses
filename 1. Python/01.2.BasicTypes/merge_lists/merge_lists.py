def merge_iterative(lst_a: list[int], lst_b: list[int]) -> list[int]:
    """
    Merge two sorted lists in one sorted list
    :param lst_a: first sorted list
    :param lst_b: second sorted list
    :return: merged sorted list
    """
    i, j = 0, 0
    res = []
    for _ in range(len(lst_a) + len(lst_b)):
        if i + 1 > len(lst_a):
            res.append(lst_b[j])
            j += 1
        elif j + 1 > len(lst_b):
            res.append(lst_a[i])
            i += 1
        elif lst_a[i] < lst_b[j]:
            res.append(lst_a[i])
            i += 1
        else:
            res.append(lst_b[j])
            j += 1
    return res


def merge_sorted(lst_a: list[int], lst_b: list[int]) -> list[int]:
    """
    Merge two sorted lists in one sorted list ising `sorted`
    :param lst_a: first sorted list
    :param lst_b: second sorted list
    :return: merged sorted list
    """
    return sorted(lst_a + lst_b)

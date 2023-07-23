import typing as tp


def filter_list_by_list(lst_a: tp.Union[list[int], range], lst_b: tp.Union[list[int], range]) -> list[int]:
    """
    Filter first sorted list by other sorted list
    :param lst_a: first sorted list
    :param lst_b: second sorted list
    :return: filtered sorted list
    """
    if not lst_a:
        return []
    if not lst_b:
        return list(lst_a)
    j = 0
    res = []
    for a in lst_a:
        if a == lst_b[j]:
            continue
        elif a > lst_b[j]:
            while a > lst_b[j] and j + 1 < len(lst_b):
                j += 1
            if a == lst_b[j]:
                continue
            else:  # a < lst_b[j]
                res.append(a)
        else:
            res.append(a)
    return res

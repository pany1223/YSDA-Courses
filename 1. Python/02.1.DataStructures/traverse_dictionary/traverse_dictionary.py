import typing as tp


def traverse_dictionary_immutable(
        dct: tp.Mapping[str, tp.Any],
        prefix: str = "") -> list[tuple[str, int]]:
    """
    :param dct: dictionary of undefined depth with integers or other dicts as leaves with same properties
    :param prefix: prefix for key used for passing total path through recursion
    :return: list with pairs: (full key from root to leaf joined by ".", value)
    """
    res = []
    for k, v in dct.items():
        if isinstance(v, dict):
            res += traverse_dictionary_immutable(v, prefix + k + '.')
        else:
            res.append((prefix + k, v))
    return res


def traverse_dictionary_mutable(
        dct: tp.Mapping[str, tp.Any],
        result: list[tuple[str, int]],
        prefix: str = "") -> None:
    """
    :param dct: dictionary of undefined depth with integers or other dicts as leaves with same properties
    :param result: list with pairs: (full key from root to leaf joined by ".", value)
    :param prefix: prefix for key used for passing total path through recursion
    :return: None
    """
    for k, v in dct.items():
        if isinstance(v, dict):
            traverse_dictionary_mutable(v, result, prefix + k + '.')
        else:
            result.append((prefix + k, v))


def traverse_dictionary_iterative(
        dct: tp.Mapping[str, tp.Any]
        ) -> list[tuple[str, int]]:
    """
    :param dct: dictionary of undefined depth with integers or other dicts as leaves with same properties
    :return: list with pairs: (full key from root to leaf joined by ".", value)
    """
    res = []
    prefix = ""
    stack = [(dct, prefix)]
    while stack:
        curr_dict = stack.pop()
        d, p = curr_dict[0], curr_dict[1]
        for k, v in d.items():
            if isinstance(v, dict):
                stack.append((v, p + k + '.'))
            else:
                res.append((p + k, v))
    return res

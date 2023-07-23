import typing as tp


def convert_to_common_type(data: list[tp.Any]) -> list[tp.Any]:
    """
    Takes list of multiple types' elements and convert each element to common type according to given rules
    :param data: list of multiple types' elements
    :return: list with elements converted to common type
    """
    types = list(map(type, data))
    types_non_empty = list(map(type, [x for x in data if x not in [None, ""]]))
    res: tp.List[tp.Any] = []

    if set(types).intersection({list, tuple}):
        for elem in data:
            if elem in [None, ""]:
                res.append([])
            elif type(elem) in [int, float, bool, str]:
                res.append([elem])
            else:
                res.append(list(elem))
    elif str in set(types_non_empty):
        for elem in data:
            if elem in [None, ""]:
                res.append("")
            elif type(elem) in [int, float, bool]:
                res.append(str(elem))
            else:
                res.append(elem)
    elif all(l := [x == int for x in types_non_empty]) and l:
        for elem in data:
            if elem in [None, ""]:
                res.append(0)
            else:
                res.append(int(elem))
    elif float in types_non_empty:
        for elem in data:
            if elem in [None, ""]:
                res.append(0.0)
            else:
                res.append(float(elem))
    elif {bool, int} == set(types_non_empty):
        for elem in data:
            if elem in [None, ""]:
                res.append(False)
            else:
                res.append(bool(elem))
    elif {bool} == set(types_non_empty):
        for elem in data:
            res.append(elem)
    elif len(set(data) - {0, None, ""}) == 0:
        for _ in data:
            res.append("")
    return res

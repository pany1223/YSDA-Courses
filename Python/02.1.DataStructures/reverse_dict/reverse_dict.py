import typing as tp


def revert(dct: tp.Mapping[str, str]) -> dict[str, list[str]]:
    """
    :param dct: dictionary to revert in format {key: value}
    :return: reverted dictionary {value: [key1, key2, key3]}
    """
    res: tp.Dict[str, tp.List[str]] = dict()
    for k, v in dct.items():
        if v in res:
            res[v] += [k]
        else:
            res.update({v: [k]})
    return res

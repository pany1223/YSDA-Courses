def get_common_type(type1: type, type2: type) -> type:
    """
    Calculate common type according to rule, that it must have the most adequate interpretation after conversion.
    Look in tests for adequacy calibration.
    :param type1: one of [bool, int, float, complex, list, range, tuple, str] types
    :param type2: one of [bool, int, float, complex, list, range, tuple, str] types
    :return: the most concrete common type, which can be used to convert both input values
    """
    if type1 is str or type2 is str:
        return str

    if type1 in [list, tuple, range] or type2 in [list, tuple, range]:
        if (type1 is tuple and type2 is tuple) or (type1 is range and type2 is range):
            return tuple
        elif type1 is list and type2 is list:
            return list
        elif (type1, type2) in [(range, list), (list, range)]:
            return list
        elif (type1, type2) in [(range, tuple), (tuple, range)]:
            return tuple
        elif (type1, type2) in [(list, tuple), (tuple, list)]:
            return list
        else:
            return str

    if (type2 == complex) or (type1 == complex):
        return complex

    if (type2 == float) or (type1 == float):
        return float

    if (type2 == int) or (type1 == int):
        return int

    return bool

def get_middle_value(a: int, b: int, c: int) -> int:
    """
    Takes three values and returns middle value.
    """
    if (a >= b >= c) or (a <= b <= c):
        return b
    elif (b >= a >= c) or (b <= a <= c):
        return a
    else:
        return c

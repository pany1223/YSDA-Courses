import typing as tp


def get_fizz_buzz(n: int) -> list[tp.Union[int, str]]:
    """
    If value divided by 3 - "Fizz",
       value divided by 5 - "Buzz",
       value divided by 15 - "FizzBuzz",
    else - value.
    :param n: size of sequence
    :return: list of values.
    """
    fizz_buzz_list: tp.List[tp.Union[int, str]] = []
    for x in range(1, n+1):
        if x % 15 == 0:
            fizz_buzz_list.append("FizzBuzz")
        elif x % 3 == 0:
            fizz_buzz_list.append("Fizz")
        elif x % 5 == 0:
            fizz_buzz_list.append("Buzz")
        else:
            fizz_buzz_list.append(x)
    return fizz_buzz_list

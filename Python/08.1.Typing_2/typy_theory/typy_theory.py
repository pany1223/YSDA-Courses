def problem01() -> dict[int, str]:
    return {
        5: 'a может быть None, сложение не определено для него',
        7: 'get() может вернуть None, а у нас строго int у переменной a'
    }


def problem02() -> dict[int, str]:
    return {
        5: 'ближайший общий тип элементов листа будет object, а для него сложение не определено'
    }


def problem03() -> dict[int, str]:
    return {
        9: 'ожидается int, подали float',
        13: 'ожидается int, подали bool'
    }


def problem04() -> dict[int, str]:
    return {
        9: 'ожидается int, подали float (а bool в данном случае ок)'
    }


def problem05() -> dict[int, str]:
    return {
        11: 'вернется класс A, а ожидаем B, более общий (потомок)'
    }


def problem06() -> dict[int, str]:
    return {
        15: 'в родителе переменная VAR уже с типом S, а пытаемся присвоить тип T, более узкий (предок)'
    }


def problem07() -> dict[int, str]:
    return {
        25: 'g ожидает функцию с аргументов A и возвращающую B, а подали А-А',
        27: 'g ожидает функцию с аргументов A и возвращающую B, а подали B-А',
        28: 'g ожидает функцию с аргументов A и возвращающую B, а подали B-B'
    }


def problem08() -> dict[int, str]:
    return {
        6: 'Iterable не значит, что от него можно взять len()',
        18: 'по классу А не ясно, что он Iterable[int], нет метода __iter__',
        24: 'класс B - Iterable, но по [int], а не [str]'
    }


def problem09() -> dict[int, str]:
    return {
        32: 'у Fooable не определен ни contains(), ни getitem(), нельзя in делать',
        34: 'у list нет метода (=protocol member) foo, как того требует Fooable',
        37: 'у C не protocol member __len__',
        38: 'ожидали Fooable, подали foo'
    }


def problem10() -> dict[int, str]:
    return {
        18: 'подали str, как typevar T',
        29: 'ожидаем int, дали float'
    }

import typing as tp


def count_util(text: str, flags: tp.Optional[str] = None) -> dict[str, int]:
    """
    :param text: text to count entities
    :param flags: flags in command-like format - can be:
        * -m stands for counting characters
        * -l stands for counting lines
        * -L stands for getting length of the longest line
        * -w stands for counting words
    More than one flag can be passed at the same time, for example:
        * "-l -m"
        * "-lLw"
    Ommiting flags or passing empty string is equivalent to "-mlLw"
    :return: mapping from string keys to corresponding counter, where
    keys are selected according to the received flags:
        * "chars" - amount of characters
        * "lines" - amount of lines
        * "longest_line" - the longest line length
        * "words" - amount of words
    """
    res = dict()
    if flags:
        flags_set = {f for f in flags if f in ['m', 'l', 'L', 'w']}
    else:
        flags_set = {'m', 'l', 'L', 'w'}

    if 'm' in flags_set:
        res['chars'] = len(text)

    if 'l' in flags_set:
        res['lines'] = len(text.split('\n')) - 1

    if 'L' in flags_set:
        max_len = 0
        for line in text.split('\n'):
            if len(line) > max_len:
                max_len = len(line)
        res['longest_line'] = max_len

    if 'w' in flags_set:
        words = 0
        for word_raw in text.split():
            if word_raw:
                words += 1
        res['words'] = words

    return res

import numpy as np


def caesar_encrypt(message: str, n: int) -> str:
    """Encrypt message using caesar cipher

    :param message: message to encrypt
    :param n: shift
    :return: encrypted message
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_shifted = ''.join(np.roll(list(alphabet), shift=n))
    trantab_only_low = message.maketrans(alphabet_shifted, alphabet)
    message_only_low = message.translate(trantab_only_low)
    trantab = message_only_low.maketrans(alphabet_shifted.upper(), alphabet.upper())
    return message_only_low.translate(trantab)

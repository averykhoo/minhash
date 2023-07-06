import hashlib
from functools import lru_cache
from typing import List

HASH_LENGTH_BYTES = 8  # 64-bit hash


def hamming_distance(x: int, y: int) -> int:
    """
    number of bits that are DIFFERENT between two binary integers

    >>> hamming_distance(0b0111, 0b1000)
    4
    >>> hamming_distance(0b0111, 0b0011)
    1
    >>> hamming_distance(0b1111, 0b1101)
    1

    :param x: non-negative 64-bit integer
    :param y: non-negative 64-bit integer
    :return:
    """
    # sanity check
    assert 0 <= x < 0x1_0000_0000_0000_0000
    assert 0 <= y < 0x1_0000_0000_0000_0000

    # https://docs.python.org/3/library/stdtypes.html#int.bit_count
    try:
        return (x ^ y).bit_count()  # only available from python 3.10 onwards

    # https://en.wikipedia.org/wiki/Hamming_weight
    except AttributeError:
        # warning: this code only works for up to uint64
        res = x ^ y
        res -= (res >> 1) & 0x5555_5555_5555_5555
        res = (res & 0x3333_3333_3333_3333) + ((res >> 2) & 0x3333_3333_3333_3333)
        res = (res + (res >> 4)) & 0X0F0F_0F0F_0F0F_0F0F
        return ((res * 0x0101_0101_0101_0101) & 0xFFFF_FFFF_FFFF_FFFF) >> 56


@lru_cache(maxsize=0xFFFF)
def hash_token_to_int(text: str) -> int:
    """
    consistently hash a string to an integer

    :param text:
    :return: a 64-bit non-negative integer
    """
    return int.from_bytes(hashlib.blake2b(text.encode('utf8'), digest_size=8).digest(), 'big')


BITARRAY_LOOKUP = [[1 if bit_value == '1' else 0 for bit_value in f'{byte_value:08b}'] for byte_value in range(256)]


@lru_cache(maxsize=0xFFFF)
def hash_token_to_bitarray(text: str) -> List[int]:
    """
    consistently hash a string to an array of bits
    warning: assumes big endianness

    :param text:
    :return: a 64-bit non-negative integer
    """
    out = []
    for byte in hashlib.blake2b(text.encode('utf8'), digest_size=8).digest():
        out.extend(BITARRAY_LOOKUP[byte])
    return out


def simhash_average_bitarrays(*bitarrays: List[int]) -> List[int]:
    """
    average multiple bitarrays into a single bitarray, per-element
    warning: don't take the average of an average unless you know what you're doing

    >>> simhash_average_bitarrays([1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0])
    [1, 1, 1, 0]
    >>> simhash_average_bitarrays([1, 1], [0, 0])
    [1, 1]
    >>> simhash_average_bitarrays([1, 1], [0, 0], [1, 0])
    [1, 0]
    >>> simhash_average_bitarrays([0, 1, 0, 1, 0])
    [0, 1, 0, 1, 0]
    >>> simhash_average_bitarrays([999], [0], [0], [0], [0])  # never do this please
    [1]

    :param bitarrays:
    :return:
    """
    assert len(bitarrays) > 0
    assert all(len(bitarray) == len(bitarrays[0]) for bitarray in bitarrays)

    min_count = len(bitarrays) / 2
    return [int(sum(column) >= min_count) for column in zip(*bitarrays)]  # in 3.10 onwards set strict=True


def bitarray_to_integer(bitarray: List[int]) -> int:
    """
    >>> bitarray_to_integer([0, 0, 1, 1, 1])
    7
    >>> bitarray_to_integer([0, 1, 0, 0, 0])
    8
    >>> bitarray_to_integer([0])
    0
    >>> bitarray_to_integer(integer_to_bitarray(123))
    123

    :param bitarray:
    :return:
    """
    _constant = len(bitarray) - 1
    return sum(v << (_constant - i) for i, v in enumerate(bitarray))


def integer_to_bitarray(number: int, *, bit_length: int = HASH_LENGTH_BYTES * 8) -> List[int]:
    """
    >>> integer_to_bitarray(7, 5)
    [0, 0, 1, 1, 1]
    >>> integer_to_bitarray(8, 5)
    [0, 1, 0, 0, 0]
    >>> integer_to_bitarray(2, 1)
    [0]
    >>> integer_to_bitarray(bitarray_to_integer([1, 0, 1]), 5)
    [0, 0, 1, 0, 1]

    :param number:
    :param bit_length:
    :return:
    """
    assert bit_length >= 1
    return [(number >> i) & 1 for i in range(bit_length - 1, -1, -1)]


def tokenize(text: str) -> List[str]:
    return text.split()


def simhash(text: str) -> int:
    """
    >>> simhash('hello world')
    18426460281723402111
    >>> simhash('hello $USER')
    18444489699396762749

    :param text:
    :return:
    """
    return bitarray_to_integer(simhash_average_bitarrays(*(hash_token_to_bitarray(token) for token in tokenize(text))))


def integer_to_hex(number: int, *, length: int = HASH_LENGTH_BYTES) -> str:
    return number.to_bytes(length, 'big').hex()


if __name__ == '__main__':
    str_x = 'def int2hex(number: int, len: int = 8) -> str:'
    str_y = 'def integer_to_hex(number: int, *, length: int = HASH_LENGTH_BYTES) -> str:'

    print(bin(simhash(str_x)))
    print(bin(simhash(str_y)))
    print(''.join(['^ '[x == y] for x, y in zip(bin(simhash(str_y)), bin(simhash(str_x)))]))
    print(hamming_distance(simhash(str_x), simhash(str_y)))

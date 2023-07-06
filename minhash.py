import pprint
import random

import tabulate


class SequenceID(dict):
    def __missing__(self, key):
        result = self[key] = len(self) + 1
        return result


# def
dimension = 10  # dimension should be as long as the longest input length
P = 2 ** 31 - 1  # prime field can be bigger for 64-bit cpus(see:https://primes.utm.edu/lists/2small/0bit.html)

# init
A0 = random.randint(1, P)
A1 = random.randint(0, P)
B0 = random.randint(1, P)
B1 = random.randint(0, P)
assert A0 != B0
A = [(A0 * x + A1) % P for x in range(1, dimension + 1)]
B = [(B0 * x + B1) % P for x in range(1, dimension + 1)]
seen = SequenceID()


# feature generator
def n_gram(text, n=2):
    if 0 < n <= len(text):
        temp = text[:n - 1]
        for char in text[n - 1:]:
            temp += char
            yield temp
            temp = temp[1:]


def min_hash(seq):
    hash_val = [P] * dimension
    for item in seq:
        for i in range(dimension):
            hash_val[i] = min(hash_val[i], (A[i] * seen[item] + B[i]) % P)
    return hash_val


def similarity(hash_1, hash_2):
    return sum(a == b for a, b in zip(hash_1, hash_2)) * 1.0 / dimension


if __name__ == '__main__':
    # test data
    test_sequences = ['`1234567890-=',
                      'qwertyuiop[]',
                      'asdfghjkl;',
                      'zxcvbnm,./',
                      'qwert234yasdfg',
                      'ertyuiop',
                      'cvbnm,.',
                      ',./',
                      'qweyuiop[]',
                      'asdfghj',
                      'qwertyasdfg',
                      'eyuiaop',
                      'cvbnwerm,.',
                      ',./sd',
                      '1234567890-=qwertyuiop[]asdfghjkl;zxcvbnm,./',
                      'cvbn,./',
                      ]
    # setup
    hashes = []
    items = []
    covar = []

    # hash the data
    for seq in test_sequences:
        hashes.append([seq] + min_hash(list(seq)
                                       + list(n_gram(seq, n=2))
                                       + list(n_gram(seq, n=3))
                                       + list(n_gram(seq, n=4))
                                       + list(n_gram(seq, n=5))
                                       ))
        items.append(seq)

    # compare the hashes
    for row_1 in hashes:
        item_1 = row_1[0]
        hash_1 = row_1[1:]
        new = [item_1]
        for row_2 in hashes:
            item_2 = row_2[0]
            hash_2 = row_2[1:]
            new.append(similarity(hash_1, hash_2))
        covar.append(new)

    # print
    print(tabulate.tabulate(covar, headers=[''] + items))
    # print(tabulate.tabulate(seen.items(), headers=['item', 'x']))
    print('count of n-grams =', len(seen))
    # print(tabulate.tabulate(hashes, headers=['seq'] + ['h%02d' % i for i in range(1, dimension + 1)]))

    print(seen)
    print(pprint.pformat(hashes))

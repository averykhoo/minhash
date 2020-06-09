import io
import math
import random
import sys
from collections import Counter
from math import log

import tabulate


def harmonic(n):
    """
    Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant (gamma)
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + log(n) + 0.5 / n - 1.0 / (12 * n ** 2) + 1.0 / (120 * n ** 4)


def get_features(string, max_features=None):
    """
    extract features from a string
    for now, just return all n-grams for n=[2,3,4,5]
    (including n=1 returns too many false positives)
    """

    # account for string boundaries by adding start and end tokens
    words = [None]
    words.extend(string.split())
    words.append(None)
    words = tuple(words)
    length = len(words)

    # generate features
    feature_list_of_lists = [
        # list(words),  # unigrams cause too many false positives
        [words[i:i + 2] for i in range(length - 1)],  # bigrams
        [words[i:i + 3] for i in range(length - 2)],  # trigrams
        # [words[i:i + 4] for i in range(length - 3)],  # 4-grams
        # [words[i:i + 5] for i in range(length - 4)],  # 5-grams
    ]
    common_features = Counter(feature for feature_list in feature_list_of_lists for feature in feature_list)
    return set(feature for feature, count in common_features.most_common(max_features))


class SequentialID(dict):
    def __missing__(self, key):
        result = self[key] = len(self) + 1
        return result


class MinHashHammingField(object):
    """
    not sure if it's really a field, since there's no defined + - * / operations over the elements
    it's more like a finite commutative monoid (where the identity element is maxint)
    or maybe it's a Hamming space over a Galois field

    the literature speaks almost exclusively about Hamming space over an N-dimensional binary Galois field (GF-2^N)
    but this is a GF-int64^N field, so I think the name should be different

    time complexity: O(dim) per lookup
    space complexity: O(dim * items) ~~> about 400mb for 2000 items at dimension 1000
    """

    def __init__(self, dimension, threshold=0.05, trim_features=True):

        assert threshold >= 0
        assert threshold <= 1.0

        # self.P = 2 ** 31 - 1  # prime
        self.P = 61
        self.dim = dimension

        self.ALPHABET = '123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        assert len(self.ALPHABET) == self.P

        # init random hashing vals
        A0 = random.randint(1, self.P)
        A1 = random.randint(0, self.P)
        B0 = random.randint(1, self.P)
        B1 = random.randint(0, self.P)
        self.A = [(A0 * x + A1) % self.P for x in range(1, dimension + 1)]
        self.B = [(B0 * x + B1) % self.P for x in range(1, dimension + 1)]

        self.threshold = int(threshold * self.dim)
        self.feature_IDs = SequentialID()
        # self.lookup_tables = [dict() for _ in range(dimension)]
        self.lookup_table = dict()

        # if you have more features than this, it's >50% likely that some will not be represented in the hash
        # but it's okay since this is probabilistic anyway
        self.max_feature_len = max(n for n in range(1, 10000) if n * harmonic(n) < dimension)

        if trim_features:
            self.trim_features = self.max_feature_len
        else:
            self.trim_features = None

    def min_hash(self, feature_set, warn_if_too_long=True):
        """

        :param feature_set:
        :param dimension:
        :return:
        """

        if warn_if_too_long and len(feature_set) > self.max_feature_len:
            warning = f'Too many features (n={feature_set}, max={self.max_feature_len} for dim={self.dim},' \
                      f' some will be lost in hashing'
            print(warning, file=sys.stderr)

        hash_val = [self.P] * self.dim
        for feature in feature_set:
            for i in range(self.dim):
                hash_val[i] = min(hash_val[i], (self.A[i] * self.feature_IDs[feature] + self.B[i]) % self.P)
        # return hash_val
        return ''.join(self.ALPHABET[h] for h in hash_val)

    def insert(self, string):
        """
        hash and index a string for lookup
        :return: minhash of string (list of ints)
        """
        # hash
        string_features = get_features(string, max_features=self.trim_features)
        string_hash = self.min_hash(string_features)

        # save lookup
        for i, val in enumerate(string_hash):
            # self.lookup_tables[i].setdefault(val, set()).add(string)
            self.lookup_table.setdefault((i, val), set()).add(string)

        # done now return
        return string_hash

    def lookup(self, string):
        # hash
        """
        find similar inserted strings
        :return: [(candidate, similarity), ...]
        """
        # hash
        string_features = get_features(string, max_features=self.trim_features)
        string_hash = self.min_hash(string_features)

        # enumerate and score match candidates
        candidates = Counter()
        for i, val in enumerate(string_hash):
            # for candidate in self.lookup_tables[i].get(val, []):
            for candidate in self.lookup_table.get((i, val), []):
                candidates[candidate] += 1

        # filter by threshold
        output = []
        for candidate, count in candidates.most_common():
            if count >= self.threshold:
                output.append((candidate, count * 1.0 / self.dim))
            else:
                break

        # done now return
        return output


def yield_lines(file_path, make_lower=False, threshold_len=0):
    """
    yields all non-empty lines in a file
    :param file_path: file to read
    :param make_lower: force line to lowercase
    :param threshold_len: ignore lines equal <= this length
    """
    for line in io.open(file_path, mode='r', encoding='utf-8'):
        line = line.strip()
        if make_lower:
            line = line.lower()
        if len(line) > threshold_len:
            yield line


if __name__ == '__main__':

    t = [(i, math.ceil((i) * harmonic(i))) for i in range(10, 101, 10)]
    print(tabulate.tabulate(t, headers=('max features', 'dimension')))
    # sys.exit(0)

    # index names using minhash
    lt = MinHashHammingField(160)

    # REPL for testing input/output
    while 1:
        # get input and lowercase
        # input_word = raw_input('\nname to look up:\n')
        input_word = input('\nname to look up:\n')
        input_word = input_word.lower().strip()
        # print or exit
        if not input_word.strip():
            break
        print('searched for:', input_word)
        print('hash:', lt.min_hash(get_features(input_word)))

        # print top-k results
        k = 8
        skipped = 0
        for (candidate, score) in lt.lookup(input_word):
            if k:
                print(score, candidate)
                k -= 1
            else:
                skipped += 1

        # notify how many results were skipped
        if skipped:
            print('(skipped {} other results)'.format(skipped))

        lt.insert(input_word)
        print('inserted:', input_word)

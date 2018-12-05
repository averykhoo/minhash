import csv
import io
import math
import random
import time
import warnings
from collections import Counter, defaultdict
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


def get_n_grams(string, n=2):
    assert n > 0
    for i in xrange(len(string) - n + 1):
        yield string[i:i + n]


def get_features(string):
    """
    extract features from a string
    """

    # account for string boundaries by adding start and end tokens
    string = u'\0' + string + u'\0'.format(string)

    # generate features
    feature_list_of_lists = [
        # get_n_grams(string, n=1),  # bad performance, too many false positives
        # get_n_grams(string, n=2),  # bad performance, too many false positives
        get_n_grams(string, n=3),
        get_n_grams(string, n=4),
        get_n_grams(string, n=5),
        get_n_grams(string, n=6),
        get_n_grams(string, n=7),
    ]
    return set(feature for feature_list in feature_list_of_lists for feature in feature_list)


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

    def __init__(self, dimension, threshold=0.05, rand_init=4):

        assert threshold >= 0
        assert threshold <= 1.0

        self.P = 2 ** 31 - 1  # any prime, coincidentally int32 maxint, but could also be 64-bit
        self.dim = dimension
        self.seed = rand_init

        # init random hashing vals
        A0 = random.randint(1, self.P)
        A1 = random.randint(0, self.P)
        B0 = random.randint(1, self.P)
        B1 = random.randint(0, self.P)
        self.A = [(A0 * x + A1) % self.P for x in range(1, dimension + 1)]
        self.B = [(B0 * x + B1) % self.P for x in range(1, dimension + 1)]

        self.threshold = int(threshold * self.dim)
        self.feature_IDs = SequentialID()
        # self.lookup_tables = [dict() for _ in xrange(dimension)]
        # self.lookup_table = dict()
        self.lookup_table = defaultdict(set)

        # if you have more features than this, it's >50% likely that some will not be represented in the hash
        # but it's okay since this is probabilistic anyway
        self.max_feature_len = max(n for n in range(1, 10000) if n * harmonic(n) < dimension)

        # memoization for speed
        self.hashes = {}

    def min_hash(self, shard_set, warn_if_too_long=True):
        """

        :param shard_set:
        :param dimension:
        :param warn_if_too_long:
        :return:
        """

        if warn_if_too_long and len(shard_set) >= self.max_feature_len:
            warnings.warn('Too many features (n={LEN_F}) for hash dim={DIM}, >50% chance some will be lost'
                          .format(LEN_F=len(shard_set), DIM=self.dim))

        if not shard_set:
            return [self.P] * self.dim

        hashes = []
        for feature in shard_set:
            feature_id = self.feature_IDs[feature]
            if feature_id not in self.hashes:
                self.hashes[feature_id] = list(self._hash(feature_id))
            hashes.append(self.hashes[feature_id])
        return list(map(min, zip(*hashes)))

    def _hash(self, feature_id):
        feature_id += self.seed
        return [(self.A[idx] * feature_id + self.B[idx]) % self.P for idx in range(self.dim)]

    def insert(self, string):
        """
        hash and index a string for lookup
        :return: minhash of string (list of ints)
        """
        # hash
        string_shards = get_features(string)
        string_hash = self.min_hash(string_shards)

        # save lookup
        for i, val in enumerate(string_hash):
            # self.lookup_tables[i].setdefault(val, set()).add(string)
            # self.lookup_table.setdefault((i, val), set()).add(string)
            self.lookup_table[i, val].add(string)

        # done now return
        return string_hash

    def lookup(self, string):
        # hash
        """
        find similar inserted strings
        :return: [(candidate, similarity), ...]
        """
        # hash
        string_features = get_features(string)
        string_hash = self.min_hash(string_features)

        # enumerate and score match candidates
        match_candidates = Counter()
        for i, val in enumerate(string_hash):
            # for candidate in self.lookup_tables[i].get(val, []):
            for match_candidate in self.lookup_table.get((i, val), []):
                match_candidates[match_candidate] += 1

        # filter by threshold
        output = []
        for match_candidate, count in match_candidates.most_common():
            if count >= self.threshold:
                output.append((match_candidate, count * 1.0 / self.dim))
            else:
                break

        # done now return
        return output


class MinHashHammingFieldFibonacci(MinHashHammingField):
    """
    same as above, but using fibonacci hashing to avoid the slow prime modulo
    """

    def __init__(self, dimension, threshold=0.05):
        # uses Fibonacci Hashing, so `P` does not need to be prime
        super(MinHashHammingFieldFibonacci, self).__init__(dimension, threshold)

        phi = 1.61803398874989484820458683436563811772030918  # the golden ratio = (sqrt(5) + 1) / 2

        self.P = 0x100000000  # 2^32 to allow the hash to bit-mask instead of module
        self.phi_P = int((phi - 1) * self.P) | 1  # ideally an odd number less than P

    def _hash(self, feature_id):
        # return [(self.phi_P * (feature_id + self.seed + idx + 1)) & 0xffffffff for idx in range(self.dim)]
        h_val = self.phi_P * (feature_id + self.seed)
        for _ in xrange(self.dim):
            h_val = (h_val + self.phi_P) & 0xffffffff
            yield h_val


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

    print(tabulate.tabulate([(num_feat, math.ceil(num_feat * harmonic(num_feat))) for num_feat in range(10, 201, 10)],
                            headers=('max features', 'dimension')))
    # sys.exit(0)

    # read names from csv
    name_lang = dict()
    with open('name_lang.csv', 'r') as f:
        c = csv.reader(f)
        for name, lang in c:
            if random.random() < .05:
                name_lang.setdefault(name, set()).add(lang)

    # index names using minhash
    print('hashing...')
    t = time.time()
    lt = MinHashHammingField(800)
    for word in name_lang:
        lt.insert(word)
    print('{} names inserted in {} seconds'.format(len(name_lang), time.time() - t))

    # index names using minhash
    print('hashing...')
    t = time.time()
    lt2 = MinHashHammingFieldFibonacci(800)
    for word in name_lang:
        lt2.insert(word)
    print('{} names inserted in {} seconds'.format(len(name_lang), time.time() - t))

    # REPL for testing input/output
    while 0:
        # get input and lowercase
        # input_word = raw_input('\nname to look up:\n')
        input_word = raw_input('\nname to look up:\n')
        input_word = input_word.lower().strip()
        # print or exit
        if not input_word.strip():
            break
        print('searched for:', input_word)

        # print top-k results
        k = 8
        skipped = 0
        for (candidate, score) in lt.lookup(input_word):
            if k:
                print(candidate, name_lang[candidate], score)
                k -= 1
            else:
                skipped += 1
        # notify how many results were skipped
        if skipped:
            print('(skipped {} other results)'.format(skipped))
        print('--')

        # print top-k results
        k = 8
        skipped = 0
        for (candidate, score) in lt2.lookup(input_word):
            if k:
                print(candidate, name_lang[candidate], score)
                k -= 1
            else:
                skipped += 1
        # notify how many results were skipped
        if skipped:
            print('(skipped {} other results)'.format(skipped))
        print('--')

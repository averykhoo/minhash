import csv
import random
import time
from collections import Counter
from math import log


def harmonic(n):
    """
    Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant (gamma)
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)


def get_features(string):
    """
    extract features from a string
    for now, just return all n-grams for n=[2,3,4,5]
    (including n=1 returns too many false positives)
    """

    def n_gram(text, n=2):
        if 0 < n <= len(text):
            temp = text[:n - 1]
            for char in text[n - 1:]:
                temp += char
                yield temp
                temp = temp[1:]

    # account for string boundaries by adding start and end tokens
    string = '\0' + string + '\0'

    # generate features
    feature_list_of_lists = [
        list(n_gram(string, n=2)),
        list(n_gram(string, n=3)),
        list(n_gram(string, n=4)),
        list(n_gram(string, n=5)),
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

    requires about 400mb for 2000 items at dimension 1000
    time complexity: O(dim) per lookup
    space complexity: O(dim * items)
    """

    def __init__(self, dimension, threshold=0.05):

        assert threshold >= 0
        assert threshold <= 1.0

        self.P = 2 ** 31 - 1  # prime
        self.dim = dimension

        # init random hashing vals
        A0 = random.randint(1, self.P)
        A1 = random.randint(0, self.P)
        B0 = random.randint(1, self.P)
        B1 = random.randint(0, self.P)
        self.A = [(A0 * x + A1) % self.P for x in range(1, dimension + 1)]
        self.B = [(B0 * x + B1) % self.P for x in range(1, dimension + 1)]

        self.lookup_table = dict()
        self.threshold = int(threshold * self.dim)
        self.feature_IDs = SequentialID()

        # if you have more features than this, it's >50% likely that some will not be represented in the hash
        # but it's okay since this is probabilistic anyway
        self.max_feature_len = max(n for n in range(1, 10000) if n * harmonic(n) < dimension)

    def min_hash(self, feature_set):
        """

        :param feature_set:
        :param dimension:
        :return:
        """

        hash_val = [self.P] * self.dim
        for feature in feature_set:
            for i in range(self.dim):
                hash_val[i] = min(hash_val[i], (self.A[i] * self.feature_IDs[feature] + self.B[i]) % self.P)
        return hash_val

    def insert(self, string):
        """
        hash and index a string for lookup
        :return: minhash of string (list of ints)
        """
        # hash
        string_features = get_features(string)
        string_hash = self.min_hash(string_features)

        # save lookup
        for i, val in enumerate(string_hash):
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
        string_features = get_features(string)
        string_hash = self.min_hash(string_features)

        # enumerate and score match candidates
        candidates = Counter()
        for i, val in enumerate(string_hash):
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


if __name__ == '__main__':
    # read names from csv
    name_lang = dict()
    with open('name_lang.csv', 'rb') as f:
        c = csv.reader(f)
        for name, lang in c:
            if random.random() < .01:
                name_lang.setdefault(name, set()).add(lang)

    # index names using minhash
    t = time.time()
    lt = MinHashHammingField(300)
    for word in name_lang:
        lt.insert(word)
    print '{} names inserted in {} seconds'.format(len(name_lang), time.time() - t)

    # REPL for testing input/output
    while 1:
        # get input and lowercase
        input_word = raw_input('\nname to look up:\n')
        input_word = input_word.lower().strip()
        # print or exit
        if not input_word.strip():
            break
        print input_word
        # print top-k results (k=8)
        skipped = 0
        for i, (candidate, score) in enumerate(lt.lookup(input_word)):
            if i >= 8:
                skipped += 1
            else:
                print(candidate, name_lang[candidate], score)
        # notify how many results were skipped
        if skipped:
            print '(skipped {} other results)'.format(skipped)

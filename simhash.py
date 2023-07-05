try:
    (123).bit_count()  # available starting in python 3.10


    def hamming_distance(x: int, y: int) -> int:
        """
        number of bits that are DIFFERENT between two binary integers

        see: https://docs.python.org/3/library/stdtypes.html#int.bit_count
        for below implementation, see: https://en.wikipedia.org/wiki/Hamming_weight

        :param x:
        :param y:
        :return:
        """
        return (x ^ y).bit_count()
except AttributeError:

    def hamming_distance(x, y):
        res = x ^ y
        # if res < 0:
        #     res = - res  # should never happen
        if 0 <= res < 0x1_0000_0000_0000_0000:
            res -= (res >> 1) & 0x5555_5555_5555_5555
            res = (res & 0x3333_3333_3333_3333) + ((res >> 2) & 0x3333_3333_3333_3333)
            res = (res + (res >> 4)) & 0X0F0F_0F0F_0F0F_0F0F
            return (res * 0x0101_0101_0101_0101) >> 56
        else:
            return bin(res).count('1')

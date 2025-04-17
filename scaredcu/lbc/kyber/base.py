from .. import modop, base
import cupy as _cp


q = 3329
n = 256
n_2 = 128
root = 17

du   = {512: 10, 768: 10, 1024: 11}
eta1 = {512: 3 , 768: 3 , 1024: 2 }

########################### KYBER SIMPLE NTT #############################

class NTT(base.NTT):

    def __init__(self):
        super().__init__(q=q, n=n, root=root, dtype='uint16')

    def ntt(self, a):
        temp = _cp.empty((256), dtype='uint16')
        temp[ ::2] = super().ntt(a[ ::2])
        temp[1::2] = super().ntt(a[1::2])

    def ntt_inv(self, a):
        temp = _cp.empty((256), dtype='uint16')
        temp[ ::2] = super().ntt_inv(a[ ::2])
        temp[1::2] = super().ntt_inv(a[1::2])
        return temp


####################################### BASE MULTIPLICATION ##################################################


class BaseMul(base.BaseMulIncomplete):

    def __init__(self, central=True, reduce=True):
        if central:
            super().__init__(reduction=modop.ReductionQ2Q2(q, 'int16'), reduce=reduce)
        else:
            super().__init__(reduction=modop.Reduction0Q(q, 'uint16'), reduce=reduce)

####################################### PLANTARD #####################################################

# Plantard arithmetic models the implementation in https://eprint.iacr.org/2022/112.pdf
# with improved plantard reduction implementation https://eprint.iacr.org/2022/956.pdf
# included in pqm4 library, commit 3743a66

class BaseMulPlant(base.BaseMulIncomplete):

    def __init__(self):
        _zetas = [21932846, 4273034450, 3562152210, 732815086, 752167598, 3542799698, 3417653460, 877313836, 2112004045, 2182963251, 932791035, 3362176261, 2951903026, 1343064270, 1419184148, 2875783148, 1817845876, 2477121420, 3434425636, 860541660, 4233039261, 61928035, 300609006, 3994358290, 975366560, 3319600736, 2781600929, 1513366367, 3889854731, 405112565, 3935010590, 359956706, 2197155094, 2097812202, 2130066389, 2164900907, 3598276897, 696690399, 2308109491, 1986857805, 2382939200, 1912028096, 1228239371, 3066727925, 1884934581, 2410032715, 3466679822, 828287474, 1211467195, 3083500101, 2977706375, 1317260921, 3144137970, 1150829326, 3080919767, 1214047529, 945692709, 3349274587, 3015121229, 1279846067, 345764865, 3949202431, 826997308, 3467969988, 2043625172, 2251342124, 2964804700, 1330162596, 2628071007, 1666896289, 4154339049, 140628247, 483812778, 3811154518, 3288636719, 1006330577, 2696449880, 1598517416, 2122325384, 2172641912, 1371447954, 2923519342, 411563403, 3883403893, 3577634219, 717333077, 976656727, 3318310569, 2708061387, 1586905909, 723783916, 3571183380, 3181552825, 1113414471, 3346694253, 948273043, 3617629408, 677337888, 1408862808, 2886104488, 519937465, 3775029831, 1323711759, 2971255537, 1474661346, 2820305950, 2773859924, 1521107372, 3580214553, 714752743, 1143088323, 3151878973, 2221668274, 2073299022, 1563682897, 2731284399, 2417773720, 1877193576, 1327582262, 2967385034, 2722253228, 1572714068, 3786641338, 508325958, 1141798155, 3153169141, 2779020594, 1515946702]
        zetas = _cp.array(_zetas, dtype='uint32').astype('int32')
        super().__init__(reduction=modop.PlantardReduction(q, 0x6ba8f301, 26632, zetas, 'int16'))

    def _basemul_low(self, a, b, frame=range(0,n_2)):
        t0 = self.reduction.reduce_zetas(b[...,1::2].astype(_cp.int16), frame)
        t = a[...,::2].astype(_cp.int16).astype(_cp.int32) * b[...,::2].astype(_cp.int16).astype(_cp.int32) + a[...,1::2].astype(_cp.int16).astype(_cp.int32) * t0.astype(_cp.int32)
        return self.reduction.reduce(t)

    def basemul(self, a, b, frame=range(0,n_2), low=True, high=True):
        return super().basemul(a, b, frame=frame, low=low, high=high)


####################################### MONTGOMERY #####################################################

# based on the implementation in https://github.com/uclcrypto/pqm4_masked
# kyber768
# commit 5fe90ba

class BaseMulMonty(base.BaseMulIncomplete):

    def __init__(self, correction=False):
        _zetas = [2226, -2226, 430, -430, 555, -555, 843, -843, 2078, -2078, 871, -871, 1550, -1550, 105, -105, 422, -422, 587, -587, 177, -177, 3094, -3094, 3038, -3038, 2869, -2869, 1574, -1574, 1653, -1653, 3083, -3083, 778, -778, 1159, -1159, 3182, -3182, 2552, -2552, 1483, -1483, 2727, -2727, 1119, -1119, 1739, -1739, 644, -644, 2457, -2457, 349, -349, 418, -418, 329, -329, 3173, -3173, 3254, -3254, 817, -817, 1097, -1097, 603, -603, 610, -610, 1322, -1322, 2044, -2044, 1864, -1864, 384, -384, 2114, -2114, 3193, -3193, 1218, -1218, 1994, -1994, 2455, -2455, 220, -220, 2142, -2142, 1670, -1670, 2144, -2144, 1799, -1799, 2051, -2051, 794, -794, 1819, -1819, 2475, -2475, 2459, -2459, 478, -478, 3221, -3221, 3021, -3021, 996, -996, 991, -991, 958, -958, 1869, -1869, 1522, -1522, 1628, -1628]
        self.zetas = _cp.array(_zetas, dtype='int16')
        super().__init__(reduction=modop.MontgomeryReduction(q=q, qinv=3327, correction=correction, o_dtype='int16'))

    def _basemul_low(self, a, b, frame=range(0,n_2)):
        t0 = self.reduction.reduce(a[...,1::2].astype(_cp.int16).astype(_cp.int32) * b[...,1::2].astype(_cp.int16).astype(_cp.int32))
        t = a[...,::2].astype(_cp.int16).astype(_cp.int32) * b[...,::2].astype(_cp.int16).astype(_cp.int32) + self.zetas[frame] * t0.astype(_cp.int16).astype(_cp.int32)
        return self.reduction.reduce(t)

    def basemul(self, a, b, frame=range(0,n_2), low=True, high=True):
        return super().basemul(a, b, frame=frame, low=low, high=high)


####################################### CIPHERTEXT AND SK DECODING ############################################



def _bytes_to_bits(input_bytes):
    bit_string = ''.join(format(byte, '08b')[::-1] for byte in input_bytes)
    return list(map(int, list(bit_string)))


def decode(byte_arr, l):
    poly = _cp.zeros(n, dtype='uint16')
    B = _bytes_to_bits(byte_arr)
    for i in range(n):
        poly[i] = sum([B[i * l + j] * 2 ** j for j in range(l)])
    return poly

def decode_kyber512_du(byte_arr):
    return decode(byte_arr, du[512])

def decode_kyber768_du(byte_arr):
    return decode(byte_arr, du[768])

def decode_kyber1024_du(byte_arr):
    return decode(byte_arr, du[1024])


def decode_sk(byte_arr):
    return decode(byte_arr, 12)


def _round_half_up(x):
    out = x.copy()
    mask = (out >= 0)
    out[mask] = _cp.floor(out[mask] + 0.5)
    out[~mask] = _cp.ceil(out[~mask] - 0.5)
    return out


def decompress(poly, d):
    return _round_half_up(poly * (q / (1 << d))).astype('uint16')

def kyber512_decompress_du(poly):
    return decompress(poly, du[512])

def kyber768_decompress_du(poly):
    return decompress(poly, du[768])

def kyber1024_decompress_du(poly):
    return decompress(poly, du[1024])


def poly_unpack_decompress_du(byte_arr, du):
    poly_compressed_bytes = (du*n) // 8
    assert (len(byte_arr)) == poly_compressed_bytes
    poly = decode(byte_arr, du)
    poly_d = decompress(poly, du)
    return poly_d

def kyber512_poly_unpack_decompress_du(byte_arr):
    return poly_unpack_decompress_du(byte_arr, du[512])

def kyber768_poly_unpack_decompress_du(byte_arr):
    return poly_unpack_decompress_du(byte_arr, du[768])

def kyber1024_poly_unpack_decompress_du(byte_arr):
    return poly_unpack_decompress_du(byte_arr, du[1024])
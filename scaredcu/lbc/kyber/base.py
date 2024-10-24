from .. import modop
import cupy as _cu
import math

########################### KYBER SIMPLE NTT #############################


q = 3329
n = 256
n_2 = 128
root = 17
root_inv = pow(root, -1, q)

def _reverse_bits(n, width=7):    
    b = '{:0{width}b}'.format(n, width=width)
    return int(b[::-1], 2)

bit_reverse_table = [_reverse_bits(i) for i in range(n_2)]


__ntt_mat = _cu.zeros((n_2,n_2),dtype='int16')
for i in range(n_2):
    for j in range(n_2):
        __ntt_mat[i,j] = pow(root, (2*bit_reverse_table[i] + 1)*j, q)

__ntt_mat_inv = _cu.zeros((n_2,n_2), dtype='int16')
inv2 = pow(n_2, -1, q)
for i in range(n_2):
    for j in range(n_2):
        __ntt_mat_inv[j,i] = (pow(root_inv, (2*bit_reverse_table[i] + 1)*j, q) * inv2) % q


def ntt(a, central_red=True):
    temp = _cu.empty((256), dtype='int16')
    temp[::2] = (_cu.matmul(__ntt_mat.astype('int64'), a[::2].astype('int64')) % q).astype('int16')
    temp[1::2] = (_cu.matmul(__ntt_mat.astype('int64'), a[1::2].astype('int64')) % q).astype('int16')
    if central_red:
        return temp - (temp > q//2)*q # centralize
    else:
        return temp


def ntt_inv(a, central_red=True):
    temp = _cu.empty((256), dtype='int16')
    temp[::2] = (_cu.matmul(__ntt_mat_inv.astype('int64'), a[::2].astype('int64')) % q).astype('int16')
    temp[1::2] = (_cu.matmul(__ntt_mat_inv.astype('int64'), a[1::2].astype('int64')) % q).astype('int16')
    if central_red:
        return temp - (temp > q//2)*q # centralize
    else:
        return temp


__ntt_mat = _cu.zeros((n_2,n_2),dtype='uint16')
for i in range(n_2):
    for j in range(n_2):
        __ntt_mat[i,j] = pow(root, (2*bit_reverse_table[i] + 1)*j, q)

__ntt_mat_inv = _cu.zeros((n_2,n_2), dtype='uint16')
inv2 = pow(n_2, -1, q)
for i in range(n_2):
    for j in range(n_2):
        __ntt_mat_inv[j,i] = (pow(root_inv, (2*bit_reverse_table[i] + 1)*j, q) * inv2) % q


def ntt(a, central_red=False):
    temp = _cu.empty((256), dtype='uint16')
    temp[::2] = (_cu.matmul(__ntt_mat.astype('uint64'), a[::2].astype('int64')) % q).astype('uint16')
    temp[1::2] = (_cu.matmul(__ntt_mat.astype('uint64'), a[1::2].astype('int64')) % q).astype('uint16')
    if central_red:
        return (temp - (temp > q//2)*q).astype('int16') # centralize
    else:
        return temp


def ntt_inv(a, central_red=False):
    temp = _cu.empty((256), dtype='uint16')
    temp[::2] = (_cu.matmul(__ntt_mat_inv.astype('uint64'), a[::2].astype('int64')) % q).astype('uint16')
    temp[1::2] = (_cu.matmul(__ntt_mat_inv.astype('uint64'), a[1::2].astype('int64')) % q).astype('uint16')
    if central_red:
        return (temp - (temp > q//2)*q).astype('int16') # centralize
    else:
        return temp



####################################### PLANTARD #####################################################



# Plantard arithmetic models the implementation in https://eprint.iacr.org/2022/112.pdf
# with improved plantard reduction implementation https://eprint.iacr.org/2022/956.pdf
# included in pqm4 library, commit 3743a66

_zetas_plant = [21932846, -21932846, 3562152210, -3562152210, 752167598, -752167598, 3417653460, -3417653460, 2112004045, -2112004045, 932791035, -932791035, 2951903026, -2951903026, 1419184148, -1419184148, 1817845876, -1817845876, 3434425636, -3434425636, 4233039261, -4233039261, 300609006, -300609006, 975366560, -975366560, 2781600929, -2781600929, 3889854731, -3889854731, 3935010590, -3935010590, 2197155094, -2197155094, 2130066389, -2130066389, 3598276897, -3598276897, 2308109491, -2308109491, 2382939200, -2382939200, 1228239371, -1228239371, 1884934581, -1884934581, 3466679822, -3466679822, 1211467195, -1211467195, 2977706375, -2977706375, 3144137970, -3144137970, 3080919767, -3080919767, 945692709, -945692709, 3015121229, -3015121229, 345764865, -345764865, 826997308, -826997308, 2043625172, -2043625172, 2964804700, -2964804700, 2628071007, -2628071007, 4154339049, -4154339049, 483812778, -483812778, 3288636719, -3288636719, 2696449880, -2696449880, 2122325384, -2122325384, 1371447954, -1371447954, 411563403, -411563403, 3577634219, -3577634219, 976656727, -976656727, 2708061387, -2708061387, 723783916, -723783916, 3181552825, -3181552825, 3346694253, -3346694253, 3617629408, -3617629408, 1408862808, -1408862808, 519937465, -519937465, 1323711759, -1323711759, 1474661346, -1474661346, 2773859924, -2773859924, 3580214553, -3580214553, 1143088323, -1143088323, 2221668274, -2221668274, 1563682897, -1563682897, 2417773720, -2417773720, 1327582262, -1327582262, 2722253228, -2722253228, 3786641338, -3786641338, 1141798155, -1141798155, 2779020594, -2779020594]
zetas_plant = _cu.array(_zetas_plant, dtype='int32')
kyber_plant = modop.PlantardReduction(q, 0x6ba8f301, 26632, zetas_plant, 'int16')


def basemul_plant_low(a, b, frame=range(0,n_2)):
    t0 = kyber_plant.reduce_zetas(b[...,1::2].astype(_cu.int16), frame)
    t = a[...,::2].astype(_cu.int16).astype(_cu.int32) * b[...,::2].astype(_cu.int16).astype(_cu.int32) + a[...,1::2].astype(_cu.int16).astype(_cu.int32) * t0.astype(_cu.int32)
    return kyber_plant.reduce(t)


def basemul_plant_high(a, b):
    t = a[...,::2].astype(_cu.int16).astype(_cu.int32) * b[...,1::2].astype(_cu.int16).astype(_cu.int32) + a[...,1::2].astype(_cu.int16).astype(_cu.int32) * b[...,::2].astype(_cu.int16).astype(_cu.int32)
    return kyber_plant.reduce(t)


def basemul_plant(a, b, frame=range(0,n_2), low=True, high=True):

    if not low and not high:
        raise ValueError('At least one of low coefficient or high coefficient must be selected')

    if (not low) or (not high):
        if low:
            return basemul_plant_low(a, b, frame=frame)
        else:
            return basemul_plant_high(a, b)
    else:
        r = _cu.empty(shape=a.shape, dtype='int16')
        r[...,::2] = basemul_plant_low(a, b, frame=frame)
        r[...,1::2] = basemul_plant_high(a, b)
        return r



####################################### MONTGOMERY #####################################################

_zetas_monty = [2226, -2226, 430, -430, 555, -555, 843, -843, 2078, -2078, 871, -871, 1550, -1550, 105, -105, 422, -422, 587, -587, 177, -177, 3094, -3094, 3038, -3038, 2869, -2869, 1574, -1574, 1653, -1653, 3083, -3083, 778, -778, 1159, -1159, 3182, -3182, 2552, -2552, 1483, -1483, 2727, -2727, 1119, -1119, 1739, -1739, 644, -644, 2457, -2457, 349, -349, 418, -418, 329, -329, 3173, -3173, 3254, -3254, 817, -817, 1097, -1097, 603, -603, 610, -610, 1322, -1322, 2044, -2044, 1864, -1864, 384, -384, 2114, -2114, 3193, -3193, 1218, -1218, 1994, -1994, 2455, -2455, 220, -220, 2142, -2142, 1670, -1670, 2144, -2144, 1799, -1799, 2051, -2051, 794, -794, 1819, -1819, 2475, -2475, 2459, -2459, 478, -478, 3221, -3221, 3021, -3021, 996, -996, 991, -991, 958, -958, 1869, -1869, 1522, -1522, 1628, -1628]
zetas_monty = _cu.array(_zetas_monty, dtype='int16')
kyber_monty = modop.MontgomeryReduction(q, 3327, 'int16')


def basemul_monty_low(a, b, frame=range(0,n_2)):
    t0 = kyber_monty.reduce(a[...,1::2].astype(_cu.int16).astype(_cu.int32) * b[...,1::2].astype(_cu.int16).astype(_cu.int32))
    t = a[...,::2].astype(_cu.int16).astype(_cu.int32) * b[...,::2].astype(_cu.int16).astype(_cu.int32) + zetas_monty[frame] * t0.astype(_cu.int16).astype(_cu.int32)
    return kyber_monty.reduce(t)


def basemul_monty_high(a, b):
    t = a[...,::2].astype(_cu.int16).astype(_cu.int32) * b[...,1::2].astype(_cu.int16).astype(_cu.int32) + a[...,1::2].astype(_cu.int16).astype(_cu.int32) * b[...,::2].astype(_cu.int16).astype(_cu.int32)
    return kyber_monty.reduce(t)


def basemul_monty(a, b, frame=range(0,n_2), low=True, high=True):

    if not low and not high:
        raise ValueError('At least one of low coefficient or high coefficient must be selected')
    if (not low) or (not high):
        if low:
            return basemul_monty_low(a, b, frame=frame)
        else:
            return basemul_monty_high(a, b)
    else:
        r = _cu.empty(shape=a.shape, dtype='int16')
        r[...,::2] = basemul_monty_low(a, b, frame=frame)
        r[...,1::2] = basemul_monty_high(a, b)
        return r
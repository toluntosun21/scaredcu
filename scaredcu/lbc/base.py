import scaredcu._utils as utils
import cupy as _cp
from . import modop
import math

class NTT:

    @staticmethod
    def _reverse_bits(n, width):    
        b = '{:0{width}b}'.format(n, width=width)
        return int(b[::-1], 2)

    def __init__(self, reduction, n, root):
        self.reduction = reduction
        self.q = reduction.q
        self.n = n
        self.root = root
        self.root_inv = pow(root, -1, self.q)
        self.logn = int(math.log2(n))
        bit_reverse_table = [self._reverse_bits(i, self.logn) for i in range(self.n)]

        self.ntt_mat = _cp.zeros((self.n,self.n),dtype='uint16')
        for i in range(self.n):
            for j in range(self.n):
                self.ntt_mat[i,j] = pow(self.root, (2*bit_reverse_table[i] + 1)*j, self.q)

        self.ntt_mat_inv = _cp.zeros((self.n,self.n), dtype='uint16')
        inv2 = pow(self.n, -1, self.q)
        for i in range(self.n):
            for j in range(self.n):
                self.ntt_mat_inv[j,i] = (pow(self.root_inv, (2*bit_reverse_table[i] + 1)*j, self.q) * inv2) % self.q

    def ntt(self, a):
        t = (_cp.matmul(self.ntt_mat.astype('uint64'), a[::2].astype('int64')) % self.q).astype('uint16')
        return self.reduction.reduce(t)

    def ntt_inv(self, a):
        t = (_cp.matmul(self.ntt_mat_inv.astype('uint64'), a[::2].astype('int64')) % self.q).astype('uint16')
        return self.reduction.reduce(t)



####################################### BASE MULTIPLICATION ##################################################


class BaseMul:

    def __init__(self, reduction, reduce=True):
        self.reduction = reduction
        self.dtype = reduction.o_dtype
        self.mult_dtype = utils.s22s(reduction.o_dtype)
        self.reduce = reduce

    def basemul(self, a, b):
        t = a.astype(self.dtype).astype(self.mult_dtype) * b.astype(self.dtype).astype(self.mult_dtype)
        if self.reduce:
            return self.reduction.reduce(t)
        else:
            return t


class BaseMulIncomplete:

    def __init__(self, reduction, reduce=True):
        self.reduction = reduction
        self.dtype = reduction.o_dtype
        self.mult_dtype = utils.s22s(reduction.o_dtype)
        self.reduce = reduce

    def basemul_low(self, a, b, frame=None):
        raise NotImplementedError()

    def basemul_high(self, a, b):
        t = a[...,::2].astype(self.dtype).astype(self.mult_dtype) * b[...,1::2].astype(self.dtype).astype(self.mult_dtype) + \
            a[...,1::2].astype(self.dtype).astype(self.mult_dtype) * b[...,::2].astype(self.dtype).astype(self.mult_dtype)
        if self.reduce:
            return self.reduction.reduce(t)
        else:
            return t

    def basemul(self, a, b, frame=None, low=True, high=True):
        if not low and not high:
            raise ValueError('At least one of low coefficient or high coefficient must be selected')

        if (not low) or (not high):
            if low:
                return self.basemul_low(a, b, frame=frame)
            else:
                return self.basemul_high(a, b)
        else:
            r = _cp.empty(shape=a.shape, dtype=self.dtype) if self.reduce else _cp.empty(shape=a.shape, dtype=self.mult_dtype)
            r[...,::2] = self.basemul_low(a, b, frame=frame)
            r[...,1::2] = self.basemul_high(a, b)
            return r


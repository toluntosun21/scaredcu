import scaredcu._utils as utils
import cupy as _cp
from . import modop
import math

class NTT:

    @staticmethod
    def _reverse_bits(n, width):    
        b = '{:0{width}b}'.format(n, width=width)
        return int(b[::-1], 2)

    def __init__(self, q, n, root, dtype='uint32'):
        self.q = q
        self.n = n
        self.root = root
        self.root_inv = pow(root, -1, self.q)
        self.logn = int(math.log2(n))
        self.dtype = dtype
        bit_reverse_table = [self._reverse_bits(i, self.logn) for i in range(self.n)]

        self._ntt_mat = _cp.zeros((self.n, self.n), dtype=dtype)
        for i in range(self.n):
            for j in range(self.n):
                self._ntt_mat[i,j] = pow(self.root, (2*bit_reverse_table[i] + 1)*j, self.q)

        self._ntt_mat_inv = _cp.zeros((self.n,self.n), dtype=dtype)
        inv2 = pow(self.n, -1, self.q)
        for i in range(self.n):
            for j in range(self.n):
                self._ntt_mat_inv[j,i] = (pow(self.root_inv, (2*bit_reverse_table[i] + 1)*j, self.q) * inv2) % self.q

    def ntt(self, a):
        a_ = a % self.q
        t = (_cp.matmul(self._ntt_mat.astype('uint64'), a_.astype('uint64')) % self.q).astype(self.dtype)
        return t

    def ntt_inv(self, a):
        a_ = a % self.q
        t = (_cp.matmul(self._ntt_mat_inv.astype('uint64'), a_.astype('uint64')) % self.q).astype(self.dtype)
        return t



####################################### BASE MULTIPLICATION ##################################################


def _basemul_set_params(obj, q, dtype, central, reduce, reduction):

    if reduction is not None and q is not None:
        raise ValueError('Exactly one of reduction or q must be provided.')
    if reduce and reduction is None and q is None:
        raise ValueError('Either reduction or q must be provided if reduce is True.')

    if reduction is None and reduce:
        try:
            _dtype = _cp.dtype(dtype)
        except TypeError:
            raise ValueError(f'{dtype} is not a valid dtype.')
        if central and _dtype == 'u':
            dtype = utils.u2s(dtype)
        reduction = modop.ReductionQ2Q2(q, o_dtype=dtype) if central else modop.Reduction0Q(q, o_dtype=dtype)
    obj.reduction = reduction

    obj.dtype = obj.reduction.o_dtype if reduce else dtype
    try:
        _dtype = _cp.dtype(obj.dtype)
    except TypeError:
        raise ValueError(f'{obj.dtype} is not a valid dtype.')
    if _dtype.kind == 'u':
        obj.mult_dtype = utils.u22u(obj.dtype)
    else:
        obj.mult_dtype = utils.s22s(obj.dtype)


class BaseMul:

    def __init__(self, reduction=None, reduce=True, q=None, dtype='uint32', central=False):
        _basemul_set_params(self, q, dtype, central, reduce, reduction)
        self.reduce = reduce

    def basemul(self, a, b):
        t = a.astype(self.dtype).astype(self.mult_dtype) * b.astype(self.dtype).astype(self.mult_dtype)
        if self.reduce:
            return self.reduction.reduce(t)
        else:
            return t


class BaseMulIncomplete:

    def __init__(self, reduction=None, reduce=True, q=None, dtype='uint32', central=False):
        _basemul_set_params(self, q, dtype, central, reduce, reduction)
        self.reduce = reduce

    def _basemul_low(self, a, b, frame=None):
        raise NotImplementedError()

    def _basemul_high(self, a, b):
        t = a[..., ::2].astype(self.dtype).astype(self.mult_dtype) * b[...,1::2].astype(self.dtype).astype(self.mult_dtype) + \
            a[...,1::2].astype(self.dtype).astype(self.mult_dtype) * b[..., ::2].astype(self.dtype).astype(self.mult_dtype)
        if self.reduce:
            return self.reduction.reduce(t)
        else:
            return t

    def basemul(self, a, b, frame=None, low=True, high=True):
        if not low and not high:
            raise ValueError('At least one of low coefficient or high coefficient must be selected')

        if (not low) or (not high):
            if low:
                return self._basemul_low(a, b, frame=frame)
            else:
                return self._basemul_high(a, b)
        else:
            low_res = self._basemul_low(a, b, frame=frame)
            high_res = self._basemul_high(a, b)
            return _cp.stack((low_res, high_res), axis=-1).reshape(*low_res.shape[:-1], -1)

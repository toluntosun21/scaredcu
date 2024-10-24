from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
import scaredcu._utils as util
import cupy as _cu
import numpy as _np


class _BaseMul:

    def __init__(self, reduction, reduce=True, words=None):
        self.reduction = reduction
        self.reduce = reduce
        self.words = words
        self.dtype_long = util.s22s(self.reduction.o_dtype)

    def __call__(self, c, guesses):
        c_ = c[:,:].astype(self.dtype_long) if self.words is None else c[:, self.words].astype(self.dtype_long)
        if self.reduce:
            res = self.reduction.reduce(c_[:, _cu.newaxis, :] * guesses[_cu.newaxis, :, _cu.newaxis])
        else:
            res = c_[:, _cu.newaxis, :] * guesses[_cu.newaxis, :, _cu.newaxis]
        return res


class BaseMul:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, reduce=True, words=None, ciphertext_tag='c', key_tag='s'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMul(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=reduction.reduce(_cu.arange(reduction.q, dtype=reduction.o_dtype)),
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulLargeQ:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, cu_step, reduce=True, words=None, ciphertext_tag='c', key_tag='s'):
        return _decorated_selection_function(
            _IteratedAttackSelectionFunctionWrapped,
            _BaseMul(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=reduction.reduce(_np.arange(reduction.q, dtype=reduction.o_dtype)),
            cu_step=cu_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class _BaseMulIncomplete(_BaseMul):

    def __call__(self, c, guesses):
        c_ = c[:,:].astype(self.dtype_long) if self.words is None else c[:, self.words].astype(self.dtype_long)
        if self.reduce:
            res = self.reduction.reduce((c_[:, _cu.newaxis, :, :] * guesses[_cu.newaxis, :, _cu.newaxis, :]).sum(axis=3))
        else:
            res = (c_[:, _cu.newaxis, :, :] * guesses[_cu.newaxis, :, _cu.newaxis, :]).sum(axis=3)
        return res


class BaseMulIncomplete:
    """Build an attack selection function which computes output values from the incomplete base multiplication.
    """

    def __new__(cls, reduction, reduce=True, words=None, ciphertext_tag='c', key_tag='s'):
        guesses_coeff = reduction.reduce(_cu.arange(0, reduction.q, dtype=reduction.o_dtype))
        high = _cu.repeat(guesses_coeff, reduction.q)
        low = _cu.tile(guesses_coeff, reduction.q)
        guesses = _cu.concatenate((low[:, _cu.newaxis], high[:, _cu.newaxis]), axis=1)
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMulIncomplete(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)
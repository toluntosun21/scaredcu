from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
import scaredcu._utils as util
import cupy as _cu
import numpy as _np


def _make_basemul(reduction, reduce, words=None):
    dtype_long = util.s22s(reduction.o_dtype)
    def _basemul(c, guesses):
        c_ = c[:,:].astype(dtype_long) if words is None else c[:, words].astype(dtype_long)
        if reduce:
            res = reduction.reduce(c_[:, _cu.newaxis, :] * guesses[_cu.newaxis, :, _cu.newaxis])
        else:
            res = c_[:, _cu.newaxis, :] * guesses[_cu.newaxis, :, _cu.newaxis]
        return res
    return _basemul


class BaseMul:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, reduce=True, words=None, ciphertext_tag='c', key_tag='s'):
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _make_basemul(reduction, reduce, words),
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
            _make_basemul(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=reduction.reduce(_np.arange(reduction.q, dtype=reduction.o_dtype)),
            cu_step=cu_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


def _make_basemul_incomplete(reduction, reduce, words=None):
    dtype_long = util.s22s(reduction.o_dtype)
    def _basemul_incomplete(c, guesses):
        c_ = c[:,:].astype(dtype_long) if words is None else c[:, words].astype(dtype_long)
        if reduce:
            res = reduction.reduce((c_[:, _cu.newaxis, :, :] * guesses[_cu.newaxis, :, _cu.newaxis, :]).sum(axis=3))
        else:
            res = (c_[:, _cu.newaxis, :, :] * guesses[_cu.newaxis, :, _cu.newaxis, :]).sum(axis=3)
        return res
    return _basemul_incomplete


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
            _make_basemul_incomplete(reduction, reduce),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)
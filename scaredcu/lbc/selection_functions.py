from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
import scaredcu._utils as util
import cupy as _cp
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
            res = self.reduction.reduce(c_[:, _cp.newaxis, :] * guesses[_cp.newaxis, :, _cp.newaxis])
        else:
            res = c_[:, _cp.newaxis, :] * guesses[_cp.newaxis, :, _cp.newaxis]
        return res


class BaseMul:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False):
        guesses = reduction.reduce(_cp.arange(reduction.q, dtype=reduction.o_dtype)) if not neg_trick else reduction.reduce(_cp.arange(reduction.q//2 + 1, dtype=reduction.o_dtype))
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMul(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulIterated:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, cp_step, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False):
        guesses = reduction.reduce(_np.arange(reduction.q, dtype=reduction.o_dtype)) if not neg_trick else reduction.reduce(_np.arange(reduction.q//2 + 1, dtype=reduction.o_dtype))
        return _decorated_selection_function(
            _IteratedAttackSelectionFunctionWrapped,
            _BaseMul(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            cp_step=cp_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class _BaseMulIncomplete(_BaseMul):

    def __call__(self, c, guesses):
        c_ = c[:,:].astype(self.dtype_long) if self.words is None else c[:, self.words].astype(self.dtype_long)
        if self.reduce:
            res = self.reduction.reduce((c_[:, _cp.newaxis, :, :] * guesses[_cp.newaxis, :, _cp.newaxis, :]).sum(axis=3))
        else:
            res = (c_[:, _cp.newaxis, :, :] * guesses[_cp.newaxis, :, _cp.newaxis, :]).sum(axis=3)
        return res


class BaseMulIncomplete:
    """Build an attack selection function which computes output values from the incomplete base multiplication.
    """

    def __new__(cls, reduction, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False):
        guesses_low = reduction.reduce(_cp.arange(reduction.q, dtype=reduction.o_dtype))
        guesses_high = guesses_low if not neg_trick else guesses_low[:reduction.q//2 + 1]
        guesses = _cp.concatenate((_cp.tile(guesses_low, len(guesses_high))[:,_cp.newaxis], _cp.repeat(guesses_high, len(guesses_low))[:,_cp.newaxis]), axis=-1)        
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMulIncomplete(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulIncompleteIterated:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, cp_step, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False):
        guesses_low = reduction.reduce(_np.arange(reduction.q, dtype=reduction.o_dtype))
        guesses_high = guesses_low if not neg_trick else guesses_low[:reduction.q//2 + 1]
        guesses = _np.concatenate((_np.tile(guesses_low, len(guesses_high))[:,_np.newaxis], _np.repeat(guesses_high, len(guesses_low))[:,_np.newaxis]), axis=-1)        
        return _decorated_selection_function(
            _IteratedAttackSelectionFunctionWrapped,
            _BaseMulIncomplete(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            cp_step=cp_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)

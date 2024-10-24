from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
from .. import selection_functions
import scaredcu._utils as util
from . import base as kyber
import cupy as _cu
import numpy as _np


MONTGOMERY = 0
PLANTARD = 1

class _KyberBaseMul(selection_functions._BaseMul):

    def __init__(self, reduction=MONTGOMERY, low=False, high=True, reduce=True, words=None):
        self.reduce = reduce
        self.words = words
        self.words_flat = _cu.tile(self.words, 2) * 2
        self.words_flat[1::2] += 1
        self.low = low
        self.high = high
        if reduction == MONTGOMERY:
            self.f = kyber.basemul_monty
        elif reduction == PLANTARD:
            self.f = kyber.basemul_plant
        else:
            raise ValueError(f'Unknown reduction method: {reduction}. Reduction method must be either "MONTGOMERY" or "PLANTARD".')

    def __call__(self, c, guesses):
        c_ = c[:,:].astype('int32') if self.words_flat is None else c[:, self.words_flat].astype('int32')
        k = self.f(c_[:, _cu.newaxis, _cu.newaxis, :], guesses[_cu.newaxis, :, _cu.newaxis], self.words, self.low, self.high)
        return k[:,:,0]

class KyberBaseMul:
    """Build an attack selection function which computes output values from the Kyber's base multiplication.
    """

    def __new__(cls, reduce=True, words=None, low=True, high=False, guesses_low=_cu.arange(kyber.q, dtype='int16'), guesses_high=None, neg_trick=False, ciphertext_tag='c', key_tag='s'):
        if guesses_high is None:
            guesses_high = _np.arange(kyber.q, dtype='int16') if not neg_trick else _np.arange(kyber.q//2 + 1, dtype='int16')
        guesses = _cu.concatenate((_cu.tile(guesses_low, len(guesses_high))[:,_cu.newaxis], _cu.repeat(guesses_high, len(guesses_low))[:,_cu.newaxis]), axis=-1)
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _KyberBaseMul(MONTGOMERY, low, high, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class KyberBaseMulIterated:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, cu_step, reduce=True, words=None, low=True, high=False, guesses_low=_np.arange(kyber.q, dtype='int16'), guesses_high=None, neg_trick=False, ciphertext_tag='c', key_tag='s'):
        if guesses_high is None:
            guesses_high = _np.arange(kyber.q, dtype='int16') if not neg_trick else _np.arange(kyber.q//2 + 1, dtype='int16')
        guesses = _np.concatenate((_np.tile(guesses_low, len(guesses_high))[:,_np.newaxis], _np.repeat(guesses_high, len(guesses_low))[:,_np.newaxis]), axis=-1)        
        return _decorated_selection_function(
            _IteratedAttackSelectionFunctionWrapped,
            _KyberBaseMul(MONTGOMERY, low, high, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            cu_step=cu_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)

# class KyberBaseMulPlant:
#     """Build an attack selection function which computes output values from the Kyber's base multiplication.
#     """

#     def __new__(cls, reduction, reduce=True, words=None, low=True, high=False, ciphertext_tag='c', key_tag='s'):
#         return _decorated_selection_function(
#             _AttackSelectionFunctionWrapped,
#             _KyberBaseMul('plantard', low, high, reduce, words),
#             expected_key_function=None,
#             words=None,
#             guesses=reduction.reduce(_cu.arange(reduction.q, dtype=reduction.o_dtype)),
#             target_tag=ciphertext_tag,
#             key_tag=key_tag,
#             byte_guesses=False)


# class KyberBaseMulBarrett:
#     """Build an attack selection function which computes output values from the Kyber's base multiplication.
#     """

#     def __new__(cls, reduction, reduce=True, words=None, low=True, high=False, ciphertext_tag='c', key_tag='s'):
#         return _decorated_selection_function(
#             _AttackSelectionFunctionWrapped,
#             _BaseMul(reduction, reduce, words),
#             expected_key_function=None,
#             words=None,
#             guesses=reduction.reduce(_cu.arange(reduction.q, dtype=reduction.o_dtype)),
#             target_tag=ciphertext_tag,
#             key_tag=key_tag,
#             byte_guesses=False)
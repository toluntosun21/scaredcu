from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
from .. import selection_functions
from . import base as kyber
import cupy as _cp
import numpy as _np


class _BaseMul(selection_functions._BaseMul):

    def __init__(self, basemul_imp, low=False, high=True, reduction=None, words=None):
        self.reduction = reduction
        self.words = words
        self.words_flat = _cp.repeat(self.words, 2) * 2
        self.words_flat[1::2] += 1
        self.low = low
        self.high = high
        self.basemul_imp = basemul_imp

    def __call__(self, c, guesses):
        c_ = c[:,:] if self.words_flat is None else c[:, self.words_flat]
        k = self.basemul_imp.basemul(c_[:, _cp.newaxis, _cp.newaxis, :], guesses[_cp.newaxis, :, _cp.newaxis], self.words, self.low, self.high)
        if self.reduction is not None:
            k = self.reduction.reduce(k)
        return k[:,:,0]


class BaseMul:
    """Build an attack selection function which computes output values from the Kyber's base multiplication.
    """

    def __new__(cls, basemul_imp=None, reduction=None, words=None, low=True, high=False, guesses_low=_cp.arange(kyber.q, dtype='int16'), guesses_high=None, neg_trick=False, ciphertext_tag='c', key_tag='s'):
        if basemul_imp is None:
            basemul_imp = kyber.BaseMulMonty()
        if guesses_high is None:
            guesses_high = guesses_low if not neg_trick else _cp.arange(kyber.q//2 + 1, dtype='int16')
        guesses = _cp.concatenate((_cp.tile(guesses_low, len(guesses_high))[:,_cp.newaxis], _cp.repeat(guesses_high, len(guesses_low))[:,_cp.newaxis]), axis=-1)
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMul(basemul_imp, low, high, reduction, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulMonty(BaseMul):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, basemul_imp=kyber.BaseMulMonty(), **kwargs)


class BaseMulPlant(BaseMul):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, basemul_imp=kyber.BaseMulPlant(), **kwargs)


class BaseMulIterated:
    """Build an iterated attack selection function which computes output values from the Kyber's base multiplication.
    """

    def __new__(cls, cp_step, basemul_imp=None, reduce=True, words=None, low=True, high=False, guesses_low=_np.arange(kyber.q, dtype='int16'), guesses_high=None, neg_trick=False, ciphertext_tag='c', key_tag='s'):
        if basemul_imp is None:
            basemul_imp = kyber.BaseMulMonty()
        if guesses_high is None:
            guesses_high = guesses_low if not neg_trick else _np.arange(kyber.q//2 + 1, dtype='int16')
        guesses = _np.concatenate((_np.tile(guesses_low, len(guesses_high))[:,_np.newaxis], _np.repeat(guesses_high, len(guesses_low))[:,_np.newaxis]), axis=-1)        
        return _decorated_selection_function(
            _IteratedAttackSelectionFunctionWrapped,
            _BaseMul(basemul_imp, low, high, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            cp_step=cp_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulMontyIterated(BaseMulIterated):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, basemul_imp=kyber.BaseMulMonty(), **kwargs)


class BaseMulPlantIterated(BaseMulIterated):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, basemul_imp=kyber.BaseMulPlant(), **kwargs)
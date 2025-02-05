from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
from .. import selection_functions
import scaredcu._utils as util
from . import base as dilithium
import cupy as _cu
import numpy as _np


class _BaseMul(selection_functions._BaseMul):

    def __init__(self, basemul_imp=dilithium.BaseMulMonty, reduce=True, words=None):
        self.reduce = reduce
        self.words = words
        self.basemul_imp = basemul_imp()

    def __call__(self, c, guesses):
        c_ = c[:,:] if self.words is None else c[:, self.words]
        return self.basemul_imp.basemul(c_[:, _cu.newaxis, _cu.newaxis, :], guesses[_cu.newaxis, :, _cu.newaxis], self.words, self.low, self.high)


class BaseMul:
    """Build an attack selection function which computes output values from the Dilithium's base multiplication.
    """

    def __new__(cls, basemul_imp=dilithium.BaseMulMonty, reduce=True, words=None, guesses=None, neg_trick=False, ciphertext_tag='c', key_tag='s'):
        if guesses is None:
            guesses = _np.arange(dilithium.q, dtype='int32') if not neg_trick else _np.arange(dilithium.q//2 + 1, dtype='int32')
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMul(basemul_imp, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulMonty(BaseMul):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, basemul_imp=dilithium.BaseMulMonty, **kwargs)


class BaseMulIterated:
    """Build an iterated attack selection function which computes output values from the Dilithium's base multiplication.
    """

    def __new__(cls, cu_step, basemul_imp=dilithium.BaseMulMonty, reduce=True, words=None, guesses=None, neg_trick=False, ciphertext_tag='c', key_tag='s'):
        if guesses is None:
            guesses = _np.arange(dilithium.q, dtype='int32') if not neg_trick else _np.arange(dilithium.q//2 + 1, dtype='int32')
        return _decorated_selection_function(
            _IteratedAttackSelectionFunctionWrapped,
            _BaseMul(basemul_imp, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            cu_step=cu_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulMontyIterated(BaseMulIterated):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, basemul_imp=dilithium.BaseMulMonty, **kwargs)
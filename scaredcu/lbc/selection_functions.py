from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
import scaredcu._utils as util
import cupy as _cp
import numpy as _np
from . import base


class _BaseMul:

    def __init__(self, basemul_imp=None, reduction=None, reduce=True, words=None):

        self.reduce = reduce
        self.words = words
        if basemul_imp is None:
            self.basemul_imp = base.BaseMul(reduction, reduce)
        else:
            self.basemul_imp = basemul_imp

    def __call__(self, c, guesses):
        c_ = c[:,:] if self.words is None else c[:, self.words]
        return self.basemul_imp.basemul(c_[:, _cp.newaxis, :], guesses[_cp.newaxis, :, _cp.newaxis])


class _BaseMulBase:
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, basemul_imp=None, reduction=None, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False, guesses=None):
        if basemul_imp is None and (reduction is None and reduce):
            raise ValueError('Either basemul_imp or reduction must be provided.')
        if guesses is None:
            if basemul_imp is None and reduction is None:
                raise ValueError('Either basemul_imp or reduction must be provided.')
            temp_reduction = basemul_imp.reduction if basemul_imp is not None else reduction
            guesses = _cp.arange(temp_reduction.q, dtype=temp_reduction.o_dtype) if not neg_trick else _cp.arange(temp_reduction.q//2 + 1, dtype=temp_reduction.o_dtype)
        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMul(basemul_imp=basemul_imp, reduction=reduction, reduce=reduce, words=words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMul(_BaseMulBase):
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, *args, **kwargs):
        super().__new__(cls, None, reduction, *args, **kwargs)


class _BaseMulIteratedBase:
    """Build an iterated attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, cp_step, basemul_imp=None, reduction=None, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False, guesses=None):
        if guesses is None:
            guesses = reduction.reduce(_np.arange(reduction.q, dtype=reduction.o_dtype)) if not neg_trick else reduction.reduce(_np.arange(reduction.q//2 + 1, dtype=reduction.o_dtype))
        return _decorated_selection_function(
            _IteratedAttackSelectionFunctionWrapped,
            _BaseMul(basemul_imp=basemul_imp, reduction=reduction, reduce=reduce, words=words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            cp_step=cp_step,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulIterated(_BaseMulIteratedBase):
    """Build an iterated attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, cp_step, *args, **kwargs):
        super().__new__(cls, cp_step, None, reduction, *args, **kwargs)


class _BaseMulIncomplete(_BaseMul):

    def __init__(self, reduction, reduce=True, words=None):
        if words is not None:
            words_flat = _cp.repeat(words, 2) * 2
            words_flat[1::2] += 1
        else:
            words_flat = None
        super().__init__(reduction, reduce, words_flat)

    def __call__(self, c, guesses):
        c_ = c[:,:].astype(self.dtype_long) if self.words is None else c[:, self.words].astype(self.dtype_long)
        low = (c_[:, _cp.newaxis, 0::2] * guesses[_cp.newaxis, :, 0::2])
        high = (c_[:, _cp.newaxis, 1::2] * guesses[_cp.newaxis, :, 1::2])
        t = low + high
        if self.reduce:
            return self.reduction.reduce(t)
        return t


class BaseMulIncomplete:
    """Build an attack selection function which computes output values from the incomplete base multiplication.
    """

    @classmethod
    def create_guesses(cls, reduction, neg_trick=False, guesses_low=None, guesses_high=None, mode='full', cpnp=_cp):
        if mode not in ['same', 'full']:
            raise ValueError('Only same or full mode are available for combination preprocesses.')

        if guesses_low is None:
            guesses_low = reduction.reduce(cpnp.arange(reduction.q, dtype=reduction.o_dtype))
        if guesses_high is None:
            guesses_high = guesses_low if not neg_trick else guesses_low[:reduction.q//2 + 1]

        if mode == 'same':
            if len(guesses_low) != len(guesses_high):
                raise ValueError('In same mode, the length of guesses_low and guesses_high must be the same.')
            guesses = cpnp.concatenate((guesses_low[:,cpnp.newaxis], guesses_high[:,cpnp.newaxis]), axis=-1)
        elif mode == 'full':
            guesses = cpnp.concatenate((cpnp.tile(guesses_low, len(guesses_high))[:,cpnp.newaxis], cpnp.repeat(guesses_high, len(guesses_low))[:,cpnp.newaxis]), axis=-1)
        return guesses

    def __new__(cls, reduction, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False, guesses_low=None, guesses_high=None, mode='full'):
        guesses = cls.create_guesses(reduction, neg_trick, guesses_low, guesses_high, mode)

        return _decorated_selection_function(
            _AttackSelectionFunctionWrapped,
            _BaseMulIncomplete(reduction, reduce, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False)


class BaseMulIncompleteIterated(BaseMulIncomplete):
    """Build an attack selection function which computes output values from the base multiplication.
    """

    def __new__(cls, reduction, cp_step, reduce=True, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False, guesses_low=None, guesses_high=None, mode='full'):
        guesses = cls.create_guesses(reduction, neg_trick, guesses_low, guesses_high, mode, cpnp=_np)

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

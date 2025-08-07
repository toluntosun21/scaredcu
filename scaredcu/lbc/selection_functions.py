from abc import ABC, abstractmethod
from scaredcu.selection_functions.base import _decorated_selection_function, _AttackSelectionFunctionWrapped
from scaredcu.selection_functions.iterated import _IteratedAttackSelectionFunctionWrapped
import cupy as _cp
import numpy as _np
from . import base, modop
import scaredcu._utils as utils

class _BaseMul:

    def __init__(self, basemul_imp, words=None):
        self.basemul_imp = basemul_imp
        self.words = words

    def __call__(self, c, guesses):
        c_ = c[:,:] if self.words is None else c[:, self.words]
        return self.basemul_imp.basemul(c_[:, _cp.newaxis, :], guesses[_cp.newaxis, :, _cp.newaxis])


class _BaseMulCpNp(ABC):

    @classmethod
    @abstractmethod
    def cpnp(cls):
        pass

    @classmethod
    @abstractmethod
    def attack_sel_cls(cls):
        pass

    @classmethod
    def get_or_create_guesses(cls, basemul_imp, neg_trick=False, guesses=None):
        if guesses is None:
            cpnp = cls.cpnp()
            temp_reduction = basemul_imp.reduction
            guesses = cpnp.arange(temp_reduction.q, dtype=temp_reduction.o_dtype) if not neg_trick else cpnp.arange(temp_reduction.q//2 + 1, dtype=temp_reduction.o_dtype)
        return guesses

    def __new__(cls, basemul_imp, words=None, ciphertext_tag='c', key_tag='s', neg_trick=False, guesses=None, **kwargs):
        guesses = cls.get_or_create_guesses(basemul_imp, neg_trick, guesses)
        return _decorated_selection_function(
            cls.attack_sel_cls(),
            _BaseMul(basemul_imp, words),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False,
            **kwargs)


class _BaseMulBase(_BaseMulCpNp):

    @classmethod
    def cpnp(cls):
        return _cp

    @classmethod
    def attack_sel_cls(cls):
        return _AttackSelectionFunctionWrapped


class _BaseMulIteratedBase(_BaseMulCpNp):

    @classmethod
    def cpnp(cls):
        return _np

    @classmethod
    def attack_sel_cls(cls):
        return _IteratedAttackSelectionFunctionWrapped



####################################### INCOMPLETE NTT ###################################################################


class _BaseMulIncomplete(_BaseMul):

    def __init__(self, basemul_imp, words=None, low=False, high=True):
        super().__init__(basemul_imp, words)
        self.words_flat = _cp.repeat(self.words, 2) * 2
        self.words_flat[1::2] += 1
        self.low = low
        self.high = high

    def __call__(self, c, guesses):
        c_ = c[:,:] if self.words_flat is None else c[:, self.words_flat]
        t = self.basemul_imp.basemul(c_[:, _cp.newaxis, _cp.newaxis, :], guesses[_cp.newaxis, :, _cp.newaxis], self.words, self.low, self.high)
        return t[:,:,0]


class _BaseMulIncompleteCpNp(_BaseMulCpNp):

    @classmethod
    def create_guesses(cls, basemul_imp=None, neg_trick=False, guesses_low=None, guesses_high=None, mode='full'):
        cpnp = cls.cpnp()
        if mode not in ['same', 'full']:
            raise ValueError('Only same or full mode are available for combination preprocesses.')
        if guesses_low is None:
            guesses_low = super().get_or_create_guesses(basemul_imp, False)
        if guesses_high is None:
            guesses_high = guesses_low if not neg_trick else guesses_low[:basemul_imp.reduction.q//2 + 1]
        if mode == 'same':
            if len(guesses_low) != len(guesses_high):
                raise ValueError('In same mode, the length of guesses_low and guesses_high must be the same.')
            guesses = cpnp.concatenate((guesses_low[:,cpnp.newaxis], guesses_high[:,cpnp.newaxis]), axis=-1)
        elif mode == 'full':
            guesses = cpnp.concatenate((cpnp.tile(guesses_low, len(guesses_high))[:,cpnp.newaxis], cpnp.repeat(guesses_high, len(guesses_low))[:,cpnp.newaxis]), axis=-1)
        return guesses

    def __new__(cls, basemul_imp, words=None, ciphertext_tag='c', key_tag='s',
                neg_trick=False, guesses_low=None, guesses_high=None, mode='full', low=True, high=True, **kwargs):
        guesses = cls.create_guesses(basemul_imp, neg_trick, guesses_low, guesses_high, mode)

        return _decorated_selection_function(
            cls.attack_sel_cls(),
            _BaseMulIncomplete(basemul_imp, words, low, high),
            expected_key_function=None,
            words=None,
            guesses=guesses,
            target_tag=ciphertext_tag,
            key_tag=key_tag,
            byte_guesses=False,
            **kwargs)


class _BaseMulIncompleteBase(_BaseMulIncompleteCpNp):

    @classmethod
    def cpnp(cls):
        return _cp

    @classmethod
    def attack_sel_cls(cls):
        return _AttackSelectionFunctionWrapped


class _BaseMulIncompleteIteratedBase(_BaseMulIncompleteCpNp):

    @classmethod
    def cpnp(cls):
        return _np

    @classmethod
    def attack_sel_cls(cls):
        return _IteratedAttackSelectionFunctionWrapped



####################################### API ###########################


class BaseMul(_BaseMulBase):

    def __new__(cls, *args, reduction=None, q=None, dtype='uint32', central=False, reduce=True, **kwargs):
        basemul_imp = base.BaseMul(reduction, reduce, q, dtype, central)
        return super().__new__(cls, basemul_imp, *args, **kwargs)


class BaseMulIterated(_BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, reduction=None, q=None, dtype='uint32', central=False, reduce=True, **kwargs):
        basemul_imp = base.BaseMul(reduction, reduce, q, dtype, central)
        return super().__new__(cls, basemul_imp, *args, cp_step=cp_step, **kwargs)


class BaseMulIncomplete(_BaseMulIncompleteBase):

    def __new__(cls, *args, reduction=None, q=None, dtype='uint32', central=False, reduce=True, **kwargs):
        basemul_imp = base.BaseMulIncomplete(reduction, reduce, q, dtype, central)
        return super().__new__(cls, basemul_imp, *args, **kwargs)


class BaseMulIncompleteIterated(_BaseMulIncompleteIteratedBase):

    def __new__(cls, cp_step, *args, reduction=None, q=None, dtype='uint32', central=False, reduce=True, **kwargs):
        basemul_imp = base.BaseMulIncomplete(reduction, reduce, q, dtype, central)
        return super().__new__(cls, basemul_imp, *args, cp_step=cp_step, **kwargs)

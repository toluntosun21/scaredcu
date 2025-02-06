from .base import _AttackSelectionFunctionWrapped
import logging
import cupy as _cu
import numpy as _np


logger = logging.getLogger(__name__)


class _IteratedAttackSelectionFunctionWrapped(_AttackSelectionFunctionWrapped):

    def __init__(self, function, guesses, cp_step, *args, **kwargs):
        super().__init__(function=function, guesses=_cu.asarray(guesses[:cp_step]), *args, **kwargs)
        self.cp_step = cp_step
        self.num_steps = ((len(guesses) - 1) // cp_step) + 1 
        self.i = 0
        self.np_guesses = guesses
        self.scores = None

    def is_last(self):
        self.i == (self.num_steps - 1)

    def done(self):
        return self.i == self.num_steps

    def reset(self):
        self.i = 0
        self._base_kwargs['guesses'] = self.guesses = _cu.asarray(self.np_guesses[:self.cp_step])
        self.scores = None

    def next(self):
        self.i += 1
        if self.done():
            return False
        elif self.is_last():
            temp_step = len(self.np_guesses) - (self.num_steps - 1) * self.cp_step
            self._base_kwargs['guesses'] = self.guesses = self.np_guesses[-temp_step:]
            return True
        else:
            self._base_kwargs['guesses'] = self.guesses = _cu.asarray(self.np_guesses[self.i * self.cp_step: (self.i + 1) * self.cp_step])
            return True

    def save_scores(self, scores):
        if self.scores is None:
            self.scores = scores.get()
        else:
            self.scores = _np.concatenate((self.scores, scores.get()), axis=0)
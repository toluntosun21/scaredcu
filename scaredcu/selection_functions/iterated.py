from .base import _AttackSelectionFunctionWrapped
import logging
import cupy as _cu
import numpy as _np


logger = logging.getLogger(__name__)


class _IteratedAttackSelectionFunctionWrapped(_AttackSelectionFunctionWrapped):

    def __init__(self, function, guesses, cu_step, *args, **kwargs):
        super().__init__(function=function, guesses=_cu.asarray(guesses[:cu_step]), *args, **kwargs)
        self.step = cu_step
        self.num_steps = ((len(guesses) - 1) // cu_step) + 1 
        self.i = 0
        self.np_guesses = guesses
        self.scores = None

    def is_last(self):
        self.i == (self.num_steps - 1)

    def done(self):
        return self.i == self.num_steps

    def reset(self):
        self.i = 0
        self.guesses[:] = _cu.asarray(self.np_guesses[:self.step])

    def next(self):
        self.i += 1
        if self.done():
            return False
        elif self.is_last():
            temp_step = len(self.np_guesses) - (self.num_steps - 1) * self.step
            self.guesses[-temp_step:] = self.np_guesses[-temp_step:]
            self.guesses = self.guesses[-temp_step:]
            return True
        else:
            self.guesses[:] = _cu.asarray(self.np_guesses[self.i * self.step: (self.i + 1) * self.step])
            return True

    def save_scores(self, scores):
        if self.scores is None:
            self.scores = scores.get()
        else:
            self.scores = _np.concatenate((self.scores, scores.get()), axis=1)
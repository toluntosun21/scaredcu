from .. import selection_functions, modop
from . import base as dilithium


class BaseMul_Q2Q2(selection_functions._BaseMulBase):

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, dilithium.BaseMul(central=True), None, *args, **kwargs)


class BaseMul_0Q(selection_functions._BaseMulBase):

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, dilithium.BaseMul(central=False), None, *args, **kwargs)


class BaseMulMonty(selection_functions._BaseMulBase):

    def __new__(cls, *args, correction=False, **kwargs):
        return super().__new__(cls, dilithium.BaseMulMonty(correction), None, *args, **kwargs)


class BaseMulIterated_Q2Q2(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        super().__new__(cls, cp_step, dilithium.BaseMul(central=True), None, *args, **kwargs)


class BaseMulIterated_0Q(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        super().__new__(cls, cp_step, dilithium.BaseMul(central=False), None, *args, **kwargs)


class BaseMulMontyIterated(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, correction=False, **kwargs):
        super().__new__(cls, cp_step, dilithium.BaseMulMonty(correction), None, *args, **kwargs)

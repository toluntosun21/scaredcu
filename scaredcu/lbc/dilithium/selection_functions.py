from .. import selection_functions
from . import base as dilithium


class BaseMulQ2Q2(selection_functions._BaseMulBase):

    def __new__(cls, *args, reduce=True, **kwargs):
        return super().__new__(cls, dilithium.BaseMul(central=True, reduce=reduce), *args, **kwargs)


class BaseMul0Q(selection_functions._BaseMulBase):

    def __new__(cls, *args, reduce=True, **kwargs):
        return super().__new__(cls, dilithium.BaseMul(central=False, reduce=reduce), *args, **kwargs)


class BaseMulMonty(selection_functions._BaseMulBase):

    def __new__(cls, *args, correction=False, **kwargs):
        return super().__new__(cls, dilithium.BaseMulMonty(correction), *args, **kwargs)


class BaseMulQ2Q2Iterated(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        return super().__new__(cls, dilithium.BaseMul(central=True), *args, cp_step=cp_step, **kwargs)


class BaseMul0QIterated(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        return super().__new__(cls, dilithium.BaseMul(central=False), *args, cp_step=cp_step, **kwargs)


class BaseMulMontyIterated(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, correction=False, **kwargs):
        return super().__new__(cls, dilithium.BaseMulMonty(correction), *args, cp_step=cp_step, **kwargs)

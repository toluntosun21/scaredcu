from .. import selection_functions
from . import base as kyber


class BaseMulQ2Q2(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, reduce=True, **kwargs):
        return super().__new__(cls, kyber.BaseMul(central=True, reduce=reduce), *args, **kwargs)


class BaseMul0Q(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, reduce=True, **kwargs):
        return super().__new__(cls, kyber.BaseMul(central=False, reduce=reduce), *args, **kwargs)


class BaseMulMonty(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, correction=False, **kwargs):
        return super().__new__(cls, kyber.BaseMulMonty(correction), *args, **kwargs)


class BaseMulPlant(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, kyber.BaseMulPlant(), *args, **kwargs)


class BaseMulQ2Q2Iterated(selection_functions._BaseMulIncompleteIteratedBase):

    def __new__(cls, cp_step, *args, reduce=True, **kwargs):
        return super().__new__(cls, cp_step, kyber.BaseMul(central=True, reduce=reduce), *args, **kwargs)


class BaseMul0QIterated(selection_functions._BaseMulIncompleteIteratedBase):

    def __new__(cls, cp_step, *args, reduce=True, **kwargs):
        return super().__new__(cls, cp_step, kyber.BaseMul(central=False, reduce=reduce), *args, **kwargs)


class BaseMulMontyIterated(selection_functions._BaseMulIncompleteIteratedBase):

    def __new__(cls, cp_step, *args, correction=False, **kwargs):
        return super().__new__(cls, cp_step, kyber.BaseMulMonty(correction), *args, **kwargs)


class BaseMulPlantIterated(selection_functions._BaseMulIncompleteIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        return super().__new__(cls, cp_step, kyber.BaseMulPlant(), *args, **kwargs)

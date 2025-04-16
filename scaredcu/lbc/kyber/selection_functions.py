from .. import selection_functions
from . import base as kyber


class BaseMul_Q2Q2(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, reduce=True, **kwargs):
        super().__new__(cls, kyber.BaseMul(central=True, reduce=reduce), None, *args, **kwargs)


class BaseMul_0Q(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, reduce=True, **kwargs):
        super().__new__(cls, kyber.BaseMul(central=False, reduce=reduce), None, *args, **kwargs)


class BaseMulMonty(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, correction=False, **kwargs):
        return super().__new__(cls, kyber.BaseMulMonty(correction), None, *args, **kwargs)


class BaseMulPlant(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, kyber.BaseMulPlant(), None, *args, **kwargs)


class BaseMulMonty(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, correction=False, **kwargs):
        return super().__new__(cls, kyber.BaseMulMonty(correction), None, *args, **kwargs)


class BaseMulPlant(selection_functions._BaseMulIncompleteBase):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, kyber.BaseMulPlant(), None, *args, **kwargs)

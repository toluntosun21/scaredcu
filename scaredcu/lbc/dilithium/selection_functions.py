from .. import selection_functions, modop
from . import base as dilithium


class BaseMul_Q2Q2(selection_functions._BaseMulBase):

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, None, modop.Reduction_Q2Q2(q=dilithium.q, o_dtype='int32'), *args, **kwargs)


class BaseMul_0Q(selection_functions._BaseMulBase):

    def __new__(cls, *args, **kwargs):
        super().__new__(cls, None, modop.Reduction_0Q(q=dilithium.q, o_dtype='uint32'), *args, **kwargs)


class BaseMulMonty(selection_functions._BaseMulBase):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, basemul_imp=dilithium.BaseMulMonty, **kwargs)


class BaseMulIterated_Q2Q2(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        super().__new__(cls, cp_step, None, modop.Reduction_Q2Q2(q=dilithium.q, o_dtype='int32'), *args, **kwargs)


class BaseMulIterated_0Q(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        super().__new__(cls, cp_step, None, modop.Reduction_0Q(q=dilithium.q, o_dtype='int32'), *args, **kwargs)


class BaseMulMontyIterated(selection_functions._BaseMulIteratedBase):

    def __new__(cls, cp_step, *args, **kwargs):
        super().__new__(cls, cp_step, dilithium.BaseMulMonty, None, *args, **kwargs)

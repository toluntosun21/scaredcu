from .. import modop
import cupy as _cu
import math


q = 8380417
n = 256

root = 1753

####################################### BASE MULTIPLICATION ##################################################


class BaseMul:

    def __init__(self, reduction):
        self.reduction = reduction

    def basemul(self, a, b):
        t = a.astype(_cu.int32).astype(_cu.int64) * b.astype(_cu.int32).astype(_cu.int64)
        return self.reduction.reduce(t)


####################################### MONTGOMERY #####################################################

class BaseMulMonty(BaseMul):

    def __init__(self):
        super().__init__(reduction=modop.MontgomeryReduction(q, -58728449, 'int32'))
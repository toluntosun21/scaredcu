import os
from scaredcu.models import Model
from scaredcu.lbc.modop import Reduction
import cupy as _cp
from tqdm.auto import tqdm


TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")


def _build_opf_table_reduction(reduction, leakage_0, leakage_1, K=None, fix0=True):
    # dedicated to mean free product

    table = _cp.empty(reduction.q)
    if K is None:
        K = reduction.q

    for i in tqdm(range(0, reduction.q, K)):
        if (i + K) > reduction.q:
            K_ = reduction.q - i
        else:
            K_ = K

        dtype = reduction.o_dtype

        if i == 0:
            X0_ = _cp.arange(K, dtype=dtype)
            X0 = _cp.repeat(X0_[:, _cp.newaxis], reduction.q, axis=1)
            X0 = reduction.reduce(X0)
            X1_ = _cp.arange(reduction.q, dtype=dtype)
            X1 = reduction.reduce(_cp.repeat(X1_[_cp.newaxis, :], K, axis=0))
            X0 = reduction.reduce(X0 - X1)
            mean_0 = leakage_0[X1[0]].mean()
            X1 = leakage_1[X1]
            mean_1 = X1[0,:].mean()
            X1 = X1 - mean_1

        else:
            X0 += K
            reduction.reduce(X0)

        X0_L = leakage_0[X0] - mean_1
        X = X0_L * X1
        table[i:i+K] = X.mean(axis=1)[:K_]
    if fix0:
        table[0] = (mean_0*mean_1)


    return table


class OPFTableReductionBuilder:

    def __init__(self, model, reduction, K=None, fix0=True, save=True, load=True):
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        if not isinstance(reduction, Reduction):
            raise ValueError("reduction must be an instance of Reduction")
        if K is not None:
            if not isinstance(K, int):
                raise ValueError("K must be an integer")
            elif K <= 0:
                raise ValueError("K must be greater than 0")
        if not isinstance(fix0, bool):
            raise ValueError("fix0 must be a boolean")
        self.model = model
        self.reduction = reduction
        self.table_map = {}
        self.K = K
        self.fix0 = fix0
        self.save = save
        self.load = load

    def _filepath(self, d):
        if not os.path.exists(TABLES_DIR):
            os.makedirs(TABLES_DIR)
        return os.path.join(TABLES_DIR, f"opf_reduction_d{d}_r{self.reduction.id}_q{self.reduction.q}_B{self.reduction.o_dtype.itemsize*8}")

    def _build(self, d):
        if d not in self.table_map:
            flag = False
            if self.load and os.path.exists(self._filepath(d)+".npy"):
                table = _cp.load(self._filepath(d)+".npy")
                flag = True
            elif d == 1:
                X = self.reduction.reduce(_cp.arange(self.reduction.q, dtype=self.reduction.o_dtype))
                table = self.model._compute(X, 0)
            else:
                d1, d2 = d//2, d - d//2
                self._build(d1)
                self._build(d2)
                table = _build_opf_table_reduction(self.reduction, self.table_map[d1], self.table_map[d2], self.K, self.fix0)
            self.table_map[d] = table
            if self.save and not flag:
                _cp.save(self._filepath(d), table)


    def build(self, d=2, delete=True):
        if d <= 0:
            raise ValueError("d must be greater than 0")
        self._build(d)
        if delete:
            keys_to_delete = [key for key in self.table_map.keys() if key != d]
            for key in keys_to_delete:
                del self.table_map[key]
        return self.table_map[d]

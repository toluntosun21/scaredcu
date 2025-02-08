from .opf.opf import OPFTableReductionBuilder
from scaredcu.models import Model, OPFTable, SignedHammingWeight
from .modop import Reduction

class OPFTableReduction(OPFTable):
    """Optimal Prediction Function for Reduction Algorithms.

    Instances of this class are callables which takes a data numpy array as input and returns it unchanged.

    Args:
        data (numpy.ndarray): numeric numpy ndarray

    Returns:
        (numpy.ndarray): unchanged input data numpy ndarray.

    """
    def __init__(self, reduction, base_model=None, fix0=True, d=2, K=None, save=True, load=True):
        if base_model is None:
            base_model = SignedHammingWeight(expected_dtype=reduction.o_dtype)
        if not isinstance(base_model, Model):
            raise ValueError("base_model must be an instance of Model")
        table_builder = OPFTableReductionBuilder(base_model, reduction, fix0=fix0, save=save, load=load, K=K)
        self.table = table_builder.build(d)

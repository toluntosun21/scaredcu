from .opf.opf import OPFTableReductionBuilder
from scaredcu.models import Model, OPFTableWithBuild, SignedHammingWeight


class OPFTableReduction(OPFTableWithBuild, OPFTableReductionBuilder):
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
        OPFTableReductionBuilder.__init__(self, base_model, reduction, fix0=fix0, save=save, load=load, K=K)
        OPFTableWithBuild.__init__(self, d)
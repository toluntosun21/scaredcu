from .base import BaseAttack, BaseReverse, BasePartitionedAttack, BasePartitionedReverse
from scaredcu import distinguishers, models


class CPAReverse(BaseReverse, distinguishers.CPADistinguisherMixin):
    __doc__ = distinguishers.CPADistinguisherMixin.__doc__ + BaseReverse.__doc__


class CPAAttack(BaseAttack, distinguishers.CPADistinguisherMixin):
    __doc__ = distinguishers.CPADistinguisherMixin.__doc__ + BaseAttack.__doc__


class CPAReverseAlternative(BaseReverse, distinguishers.CPAAlternativeDistinguisherMixin):
    __doc__ = distinguishers.CPAAlternativeDistinguisherMixin.__doc__ + BaseReverse.__doc__


class CPAAttackAlternative(BaseAttack, distinguishers.CPAAlternativeDistinguisherMixin):
    __doc__ = distinguishers.CPAAlternativeDistinguisher.__doc__ + BaseAttack.__doc__

def _check_model_consistency(obj):
    if not isinstance(obj.model, models.Monobit):
        raise distinguishers.DistinguisherError(f'DPA analysis can be processed only with Monobit model, not {type(obj.model)}.')


class DPAReverse(BaseReverse, distinguishers.DPADistinguisherMixin):
    __doc__ = distinguishers.DPADistinguisherMixin.__doc__ + BaseReverse.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_model_consistency(self)


class DPAAttack(BaseAttack, distinguishers.DPADistinguisherMixin):
    __doc__ = distinguishers.DPADistinguisherMixin.__doc__ + BaseAttack.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _check_model_consistency(self)


class KWAttack(BasePartitionedAttack, distinguishers.KWDistinguisherMixin):
    __doc__ = distinguishers.KWDistinguisherMixin.__doc__ + BaseAttack.__doc__


class KWReverse(BasePartitionedReverse, distinguishers.KWDistinguisherMixin):
    __doc__ = distinguishers.KWDistinguisherMixin.__doc__ + BaseReverse.__doc__


class ANOVAAttack(BasePartitionedAttack, distinguishers.ANOVADistinguisherMixin):
    __doc__ = distinguishers.ANOVADistinguisherMixin.__doc__ + BaseAttack.__doc__


class NICVAttack(BasePartitionedAttack, distinguishers.NICVDistinguisherMixin):
    __doc__ = distinguishers.NICVDistinguisherMixin.__doc__ + BaseAttack.__doc__


class SNRAttack(BasePartitionedAttack, distinguishers.SNRDistinguisherMixin):
    __doc__ = distinguishers.SNRDistinguisherMixin.__doc__ + BaseAttack.__doc__


class MIAAttack(BasePartitionedAttack, distinguishers.MIADistinguisherMixin):
    __doc__ = distinguishers.MIADistinguisherMixin.__doc__ + BaseAttack.__doc__

    def __init__(self, bins_number=128, bin_edges=None, place_outliers=False, *args, **kwargs):
        distinguishers.mia._set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges, place_outliers=place_outliers)
        return super().__init__(*args, **kwargs)


class ANOVAReverse(BasePartitionedReverse, distinguishers.ANOVADistinguisherMixin):
    __doc__ = distinguishers.ANOVADistinguisherMixin.__doc__ + BaseReverse.__doc__


class NICVReverse(BasePartitionedReverse, distinguishers.NICVDistinguisherMixin):
    __doc__ = distinguishers.NICVDistinguisherMixin.__doc__ + BaseReverse.__doc__


class SNRReverse(BasePartitionedReverse, distinguishers.SNRDistinguisherMixin):
    __doc__ = distinguishers.SNRDistinguisherMixin.__doc__ + BaseReverse.__doc__


class MIAReverse(BasePartitionedReverse, distinguishers.MIADistinguisherMixin):
    __doc__ = distinguishers.MIADistinguisherMixin.__doc__ + BaseReverse.__doc__

    def __init__(self, bins_number=128, bin_edges=None, place_outliers=False, *args, **kwargs):
        distinguishers.mia._set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges, place_outliers=place_outliers)
        return super().__init__(*args, **kwargs)


class MMIAAttack(BasePartitionedAttack, distinguishers.MMIADistinguisherMixin):
    __doc__ = distinguishers.MMIADistinguisherMixin.__doc__ + BaseAttack.__doc__

    def __init__(self, offset=1, bins_number=128, bins_numberx2=None, same_bins=True, bin_edges=None, bin_edgesx2=None,
                 place_outliers=False, *args, **kwargs):
        distinguishers.mmia._set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges, offset=offset, same_bins=same_bins,
                                  bin_edgesx2=bin_edgesx2, bins_numberx2=bins_numberx2, place_outliers=place_outliers)
        return super().__init__(*args, **kwargs)


class MMIAReverse(BasePartitionedReverse, distinguishers.MMIADistinguisherMixin):
    __doc__ = distinguishers.MMIADistinguisherMixin.__doc__ + BaseReverse.__doc__

    def __init__(self, offset=1, bins_number=128, bins_numberx2=None, same_bins=True, bin_edges=None, bin_edgesx2=None,
                 place_outliers=False, *args, **kwargs):
        distinguishers.mmia._set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges, offset=offset, same_bins=same_bins,
                                  bin_edgesx2=bin_edgesx2, bins_numberx2=bins_numberx2, place_outliers=place_outliers)
        return super().__init__(*args, **kwargs)
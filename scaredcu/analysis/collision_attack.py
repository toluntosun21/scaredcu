from .base import BasePartitionedAttack
from scaredcu.distinguishers import CollisionDistinguisherMixin
import cupy as _cp


def _nanmean(arr, axis=None):
    nan_mask = ~_cp.isnan(arr)
    valid_count = _cp.sum(nan_mask, axis=axis)
    total_sum = _cp.nansum(arr, axis=axis)
    return total_sum / valid_count


class CollisionAttack(BasePartitionedAttack, CollisionDistinguisherMixin):
    """Correlation enhanced collision attack.

    """

    def __init__(self, offset, reverse_selection_function, attack_selection_function, **kwargs):
        super().__init__(selection_function=reverse_selection_function, model=None, **kwargs)
        self.attack_selection_function = attack_selection_function
        self.offset = offset

    def _set_model(self, model):
        pass

    def compute_intermediate_values(self, metadata):
        self.saved_keys = metadata.keys()
        return self.selection_function(**metadata)

    def _compute(self):
        p0_values = self.partitions
        kwargs = {}
        for key in self.saved_keys:
            kwargs[key] = p0_values
        p1_values = self.attack_selection_function(**kwargs)

        key_index = 0 # for now

        mean = self.sum / self.counters
        for j in range(mean.shape[0]):
            mean[j, _cp.isnan(mean[j])] = _nanmean(mean[j], axis=0).mean()

        arr0 = mean[: mean.shape[0] - self.offset, key_index    , _cp.newaxis, :]
        arr1 = mean[self.offset : mean.shape[0]  , key_index + 1, p1_values     ]

        corr = _cp.empty((arr1.shape[1], 1, arr1.shape[0]), dtype=self.precision)
        for i in range(corr.shape[2]):
            corr[:, 0, i] = _cp.corrcoef(arr0[i], arr1[i], rowvar=False)[0, 1:]
        return corr
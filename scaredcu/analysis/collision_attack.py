from .base import BasePartitionedAttack
from scaredcu.distinguishers import CollisionDistinguisherMixin
import cupy as _cp


class CollisionAttack(BasePartitionedAttack, CollisionDistinguisherMixin):
    """Correlation enhanced collision attack.

    """

    def __init__(self, offset, reverse_selection_function, delta_selection_function, **kwargs):
        super().__init__(selection_function=reverse_selection_function, model=None, **kwargs)
        CollisionDistinguisherMixin.__init__(self, offset, delta_selection_function)

    def _set_model(self, model):
        pass

    def compute_intermediate_values(self, metadata):
        self._set_delta_func_args(metadata.keys())
        return self.selection_function(**metadata)

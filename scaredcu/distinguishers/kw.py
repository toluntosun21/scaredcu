from .base import DistinguisherError
from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
import cupy as _cp
from numba import cuda as _cpda
import logging

logger = logging.getLogger(__name__)


def _rankmin(x):
    _, inv = _cp.unique(x, return_inverse=True)
    return inv


def _rankmin_2d(x):
    return _cp.apply_along_axis(_rankmin, 0, x)


def insert_and_rank_columns(data, new_row):
    """ Inserts a new row into a 2D CuPy array and updates ranks column-wise. """
    if data is None:
        data = new_row
    else:
        data = _cp.vstack([data, new_row])  # Append new row
    sorted_indices = _cp.argsort(data, axis=0)  # Get sorted indices per column
    ranks = _cp.argsort(sorted_indices, axis=0)  # Convert indices to ranks
    return data, ranks


class KWDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """This distinguisher mixin applies a Kruskal-Wallis test."""

    def _initialize_accumulators(self):
        self.traces = None
        self.data = None
        self.counters = _cp.zeros((self._data_words, len(self.partitions)), dtype='uint32')
        self.result = _cp.zeros((self._data_words, self._trace_length), dtype=self.precision)
        self.sum = _cp.zeros((self._data_words, self._trace_length, len(self.partitions)), dtype='uint64')

    @staticmethod
    @_cpda.jit(cache=True)
    def _accumulate_counters(data, self_counters):
        start = _cpda.grid(1)
        stride = _cpda.gridsize(1)
        for data_idx in range(start, data.shape[1], stride):
            for trace_idx in range(data.shape[0]):
                data_value = data[trace_idx, data_idx]
                if data_value != (-1):
                    self_counters[data_idx, data_value] += 1

    def _accumulate(self, traces, data):
        # not a real accumulation since KW operates over ranked data.
        self.traces, self.trace_ranks = insert_and_rank_columns(self.traces, traces)
        if (self.data is None):
            self.data = _cp.copy(data.reshape(traces.shape[0], self._data_words))
        else:
            self.data = _cp.concatenate((self.data, data.reshape(traces.shape[0], self._data_words)), axis=0)
        self._accumulate_counters[64,8](data, self.counters)

    @staticmethod
    @_cpda.jit(cache=True)
    def _accumulate_ranks_and_compute(trace_ranks, data, self_counters, self_sum, result):
        start = _cpda.grid(1)
        stride = _cpda.gridsize(1)
        for data_idx in range(start, data.shape[1], stride):
            for trace_idx in range(trace_ranks.shape[0]):
                data_value = data[trace_idx, data_idx]
                if data_value != (-1):
                    for sample_idx in range(trace_ranks.shape[1]):
                        self_sum[data_idx, sample_idx, data_value] += trace_ranks[trace_idx, sample_idx]
            for sample_idx in range(trace_ranks.shape[1]):
                result[data_idx, sample_idx] = 0
                for data_idy in range(self_counters.shape[1]):
                    if self_counters[data_idx, data_idy] > 0:
                        result[data_idx, sample_idx] += ((self_sum[data_idx, sample_idx, data_idy])*(self_sum[data_idx, sample_idx, data_idy])) / self_counters[data_idx, data_idy]

    def _compute(self):
        self.sum[:] = 0
        self._accumulate_ranks_and_compute[128,8](self.trace_ranks, self.data, self.counters, self.sum, self.result)
        return self.result

    @property
    def _distinguisher_str(self):
        return 'KW'


class KWDistinguisher(PartitionedDistinguisherBase, KWDistinguisherMixin):
    """Standalone distinguisher class using KW."""
from .base import DistinguisherError
from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
import cupy as _cu
from numba import cuda as _cuda
import logging

logger = logging.getLogger(__name__)


def _rankmin(x):
    _, inv = _cu.unique(x, return_inverse=True)
    return inv

def _rankmin_2d(x):
    return _cu.apply_along_axis(_rankmin, 0, x)


class KWDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """This distinguisher mixin applies a Kruskal-Wallis test."""

    def _initialize_accumulators(self):
        self.traces = None
        self.data = None
        self.counters = _cu.zeros((self._data_words, len(self.partitions)), dtype='uint32')
        self.result = _cu.zeros((self._data_words, self._trace_length), dtype=self.precision)
        self.sum = _cu.zeros((self._data_words, self._trace_length, len(self.partitions)), dtype='uint64')

    @staticmethod
    @_cuda.jit(cache=True)
    def _accumulate_counters(data, self_counters):
        start = _cuda.grid(1)
        stride = _cuda.gridsize(1)
        for data_idx in range(start, data.shape[1], stride):
            for trace_idx in range(data.shape[0]):
                data_value = data[trace_idx, data_idx]
                if data_value != (-1):
                    self_counters[data_idx, data_value] += 1

    def _accumulate(self, traces, data):
        # not a real accumulation since KW operates over ranked data.
        if (self.traces is None):
            self.traces = _cu.copy(traces)
            self.data =  _cu.copy(data.reshape(traces.shape[0], self._data_words))
        else:
            self.traces = _cu.concatenate((self.traces, traces), axis=0)
            self.data = _cu.concatenate((self.data, data.reshape(traces.shape[0], self._data_words)), axis=0)
        self._accumulate_counters[64,8](data, self.counters)

    @staticmethod
    @_cuda.jit(cache=True)
    def _accumulate_ranks_and_compute(trace_ranks, data, self_counters, self_sum, result):
        start = _cuda.grid(1)
        stride = _cuda.gridsize(1)
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
        self.trace_ranks = _rankmin_2d(self.traces) + 1
        self.sum[:] = 0
        self._accumulate_ranks_and_compute[64,8](self.trace_ranks, self.data, self.counters, self.sum, self.result)
        return self.result

    @property
    def _distinguisher_str(self):
        return 'KW'


class KWDistinguisher(PartitionedDistinguisherBase, KWDistinguisherMixin):
    """Standalone distinguisher class using KW."""
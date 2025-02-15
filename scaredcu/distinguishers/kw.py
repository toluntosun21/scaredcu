from .base import DistinguisherError
from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
import cupy as _cp
from numba import cuda as _cuda
import logging
import time

logger = logging.getLogger(__name__)


_accumulate_ranks_and_compute_kernel = _cp.RawKernel(r'''
extern "C" __global__
void accumulate_ranks_and_compute(
    const float* trace_ranks, 
    const int* data, 
    const float* self_counters, 
    float* self_sum, 
    float* result,
    int num_traces,
    int num_samples,
    int num_data_cols,
    int num_classes
) {

                                                     
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (data_idx >= num_data_cols) return;

    // Accumulate trace ranks
    for (int trace_idx = 0; trace_idx < num_traces; trace_idx++) {
        int data_idy = data[trace_idx * num_data_cols + data_idx];
        if (data_idy != -1) {
            for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
                atomicAdd(&self_sum[(sample_idx * num_classes + data_idy) * num_data_cols + data_idx], 
                          trace_ranks[trace_idx * num_samples + sample_idx]);
            }
        }
    }

    // Compute final result
    for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
        float sum_result = 0;
        for (int data_idy = 0; data_idy < num_classes; data_idy++) {
            int counter = self_counters[data_idx * num_classes + data_idy];
            if (counter > 0) {
                float sum_val = self_sum[(sample_idx * num_classes + data_idy) * num_data_cols + data_idx];
                sum_result += sum_val * (sum_val / counter);
            }
        }
        result[data_idx * num_samples + sample_idx] = sum_result;
    }  
}
''', 'accumulate_ranks_and_compute')


# Function to call the CuPy kernel
def accumulate_ranks_and_compute(trace_ranks, data, self_counters, self_sum, result):
    
    num_traces, num_samples = trace_ranks.shape
    num_data_cols = data.shape[1]
    num_classes = self_counters.shape[1]
    
    block_size = 128  # Tunable based on GPU architecture
    grid_size = (num_data_cols + block_size - 1) // block_size
    # shared_mem_size = block_size * num_classes * (num_samples//50) * 4  # 4 bytes per float
    _accumulate_ranks_and_compute_kernel(
        (grid_size,), (block_size,),
        (
            trace_ranks, data, self_counters, self_sum, result,
            num_traces, num_samples, num_data_cols, num_classes
        )#, shared_mem=shared_mem_size
    )



def insert_and_rank_columns_with_ties(data):
    """ Inserts a new row into a 2D CuPy array and updates ranks column-wise with tie handling. """
    sorted_indices = _cp.argsort(data, axis=0)
    
    ranks = _cp.zeros_like(data, dtype=_cp.float32)
    
    for col in range(data.shape[1]):
        sorted_col = data[sorted_indices[:, col], col]
        unique_vals, inverse_indices, counts = _cp.unique(sorted_col, return_inverse=True, return_counts=True)
        
        cumulative_counts = _cp.cumsum(counts)
        avg_ranks = (cumulative_counts - counts / 2.0).astype(_cp.float32)
        
        ranks[sorted_indices[:, col], col] = avg_ranks[inverse_indices]
    
    return data, ranks


class KWDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """This distinguisher mixin applies a Kruskal-Wallis test."""

    def _initialize_accumulators(self):
        self.max_traces = 200000
        self.data = _cp.empty((self.max_traces, self._data_words), dtype=_cp.int32)
        self.traces = _cp.empty((self.max_traces, self._trace_length), dtype=self.precision)
        self.counters = _cp.zeros((self._data_words, len(self.partitions)), dtype=self.precision)
        self.result = _cp.zeros((self._data_words, self._trace_length), dtype=self.precision)
        self.sum = _cp.zeros((self._trace_length * len(self.partitions) * self._data_words), dtype=self.precision)

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
        self.traces[self.processed_traces - traces.shape[0] : self.processed_traces] = traces.astype(self.precision)
        self.traces[: self.processed_traces], self.trace_ranks = insert_and_rank_columns_with_ties(self.traces[: self.processed_traces])
        self.data[self.processed_traces - traces.shape[0] : self.processed_traces] = data.astype(self.data.dtype)
        self._accumulate_counters[64,8](data, self.counters)

    @staticmethod
    @_cuda.jit(cache=True)
    def _accumulate_ranks_and_compute(trace_ranks, data, self_counters, self_sum, result):
        start =  _cuda.grid(1)
        stride = _cuda.gridsize(1)
        for data_idx in range(start, data.shape[1], stride):
            for trace_idx in range(trace_ranks.shape[0]):
                data_idy = data[trace_idx, data_idx]
                if data_idy != (-1):
                    for sample_idx in range(trace_ranks.shape[1]):
                        self_sum[data_idx, sample_idx, data_idy] += trace_ranks[trace_idx, sample_idx]
            for sample_idx in range(trace_ranks.shape[1]):
                result[data_idx, sample_idx] = 0
                for data_idy in range(self_counters.shape[1]):
                    if self_counters[data_idx, data_idy] > 0:
                        result[data_idx, sample_idx] += ((self_sum[data_idx, sample_idx, data_idy]))*((self_sum[data_idx, sample_idx, data_idy]) / self_counters[data_idx, data_idy])

    def _compute(self):
        self.sum[:] = 0
        accumulate_ranks_and_compute(self.trace_ranks, self.data, self.counters, self.sum, self.result)
        return self.result

    @property
    def _distinguisher_str(self):
        return 'KW'


class KWDistinguisher(PartitionedDistinguisherBase, KWDistinguisherMixin):
    """Standalone distinguisher class using KW."""
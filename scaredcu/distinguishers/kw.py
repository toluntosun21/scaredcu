from .base import DistinguisherError
from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
import cupy as _cp
from numba import cuda as _cuda
import logging
import math

logger = logging.getLogger(__name__)


def get_kernel_code_0(num_samples, num_partitions, num_data_cols, log_samples_div, samples_cache, partition_dtype, result_dtype):
    return f'''
    extern "C" __global__
    void accumulate_ranks_and_compute(
        const float* __restrict__ trace_ranks, 
        const {partition_dtype}* __restrict__ data, 
        const float* __restrict__ self_counters, 
        {result_dtype}* result,
        int num_traces
    ) {{
        extern __shared__ float shared_mem[];
        
        const int num_samples = {num_samples};
        const int num_partitions = {num_partitions};
        const int log_samples_div = {log_samples_div};
        const int samples_cache = {samples_cache};                                                     
        const int num_data_cols = {num_data_cols};
                                                                                                            
        int data_idx = (blockIdx.x * blockDim.x + threadIdx.x) >> log_samples_div;
        int sample_idy = (blockIdx.x * blockDim.x + threadIdx.x) & ((1 << log_samples_div) - 1);                                                     
        if (data_idx >= num_data_cols) return;

        for (int sample_idx = 0; sample_idx < samples_cache; sample_idx++) {{
            for (int data_idy = 0; data_idy < num_partitions; data_idy++) {{                                                     
                shared_mem[(sample_idx * num_partitions + data_idy) * blockDim.x + threadIdx.x] = 0;
            }}
        }}                                                     
        for (int trace_idx = 0; trace_idx < num_traces; trace_idx++) {{
            long int data_index = ((long int) trace_idx) * num_data_cols + data_idx;
            int data_idy = *(data + data_index);
            if (data_idy != -1) {{
                for (int sample_idx = 0; sample_idx < samples_cache; sample_idx++) {{
                    int sample_id = sample_idy * samples_cache + sample_idx;
                    if (sample_id >= num_samples) continue;                                                                                                          
                    shared_mem[(sample_idx * num_partitions + data_idy) * blockDim.x + threadIdx.x] += 
                        trace_ranks[trace_idx * num_samples + sample_id];
                }}
            }}
        }}
        
        float sum_val;                                                     

        for (int sample_idx = 0; sample_idx < samples_cache; sample_idx++) {{
            float sum_result = 0;
            int sample_id = sample_idy * samples_cache + sample_idx;
            if (sample_id >= num_samples) continue;
            for (int data_idy = 0; data_idy < num_partitions; data_idy++) {{
                float counter = self_counters[data_idx * num_partitions + data_idy];
                if (counter > 0) {{
                    sum_val = shared_mem[(sample_idx * num_partitions + data_idy) * blockDim.x + threadIdx.x];
                    sum_result += sum_val * (sum_val / counter);
                }}
            }}
            result[data_idx * num_samples + sample_id] = sum_result;
        }}
    }}
    '''


def get_kernel_code_1(num_samples, num_partitions, num_data_cols, partition_dtype, result_dtype):
    return f'''
    extern "C" __global__
    void accumulate_ranks_and_compute(
        const float* __restrict__ trace_ranks, 
        const {partition_dtype}* __restrict__ data, 
        const float* __restrict__ self_counters, 
        {result_dtype}* self_sum, 
        {result_dtype}* result,
        int num_traces
    ) {{

        const int num_samples = {num_samples};
        const int num_partitions = {num_partitions};
        const int num_data_cols = {num_data_cols};

        int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (data_idx >= num_data_cols) return;

        for (int trace_idx = 0; trace_idx < num_traces; trace_idx++) {{
            long int data_index = ((long int) trace_idx) * num_data_cols + data_idx;
            int data_idy = *(data + data_index);
            if (data_idy != -1) {{
                for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {{
                    atomicAdd(&self_sum[(sample_idx * num_partitions + data_idy) * num_data_cols + data_idx], 
                            trace_ranks[trace_idx * num_samples + sample_idx]);
                }}
            }}
        }}

        for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {{
            {result_dtype} sum_result = 0;
            for (int data_idy = 0; data_idy < num_partitions; data_idy++) {{
                float counter = self_counters[data_idx * num_partitions + data_idy];
                if (counter > 0) {{
                    float sum_val = self_sum[(sample_idx * num_partitions + data_idy) * num_data_cols + data_idx];
                    sum_result += sum_val * (sum_val / counter);
                }}
            }}
            result[data_idx * num_samples + sample_idx] = sum_result;
        }}  
    }}
    '''


class KWDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """This distinguisher mixin applies a Kruskal-Wallis test."""

    def accumulate_ranks_and_compute(self):
        num_traces = self.trace_ranks.shape[0]
        if self.kernel_method == 0:
            self._accumulate_ranks_and_compute_kernel(
                (self.grid_size,), (self.block_size,),
                (
                    self.trace_ranks, self.data, self.counters, self.result,
                    num_traces
                ), shared_mem=self.shared_mem_size
            )
        elif self.kernel_method == 1:
            self.sum[:] = 0
            self._accumulate_ranks_and_compute_kernel(
                (self.grid_size,), (self.block_size,),
                (
                    self.trace_ranks, self.data, self.counters, self.sum, self.result,
                    num_traces
                )
            )

    def _init_kernel(self):
        self.shared_mem_size = 48 * 1024
        self.block_size = 512
        num_partitions = self.counters.shape[1]
        self.samples_cache = self.shared_mem_size // (self.block_size * 4 * num_partitions)
        if self.samples_cache > 0:
            samples_div_init = ((self._trace_length - 1) // self.samples_cache) + 1
            self.log_samples_div = math.ceil(math.log2(samples_div_init))
            samples_div = 1 << self.log_samples_div
            self.grid_size = ((self._data_words * samples_div) + self.block_size - 1) // self.block_size
            assert self.shared_mem_size >= (self.block_size * 4 * num_partitions * self.samples_cache)
            kernel_code = get_kernel_code_0(num_samples=self._trace_length, num_partitions=num_partitions, num_data_cols=self._data_words,
                                            log_samples_div=self.log_samples_div, samples_cache=self.samples_cache,
                                            partition_dtype=self.partition_dtype_c, result_dtype=self.precision_c)
            self._accumulate_ranks_and_compute_kernel = _cp.RawKernel(kernel_code, 'accumulate_ranks_and_compute')
            self.kernel_method = 0
        else:
            self.grid_size = (self._data_words + self.block_size - 1) // self.block_size
            kernel_code = get_kernel_code_1(num_samples=self._trace_length, num_partitions=num_partitions, num_data_cols=self._data_words,
                                            partition_dtype=self.partition_dtype_c, result_dtype=self.precision_c)
            self._accumulate_ranks_and_compute_kernel = _cp.RawKernel(kernel_code, 'accumulate_ranks_and_compute')
            self.kernel_method = 1
            self.sum = _cp.zeros((self._trace_length * len(self.partitions) * self._data_words), dtype=self.precision)


    def _initialize_accumulators(self):
        self.max_traces = 400000
        self.data = None
        self.traces = None
        self.counters = _cp.zeros((self._data_words, len(self.partitions)), dtype='float32')
        self.result = _cp.zeros((self._data_words, self._trace_length), dtype=self.precision)
        if self.counters.shape[1] <= 256:
            self.partition_dtype = _cp.uint8
            self.partition_dtype_c = 'unsigned char'
        elif self.counters.shape[1] <= 65536:
            self.partition_dtype = _cp.uint16
            self.partition_dtype_c = 'unsigned short'
        else:
            self.partition_dtype = _cp.uint32
            self.partition_dtype_c = 'unsigned int'
        if self.precision == 'float32' or self.precision == 'float':
            self.precision_c = 'float'
        elif self.precision == 'float64' or self.precision == 'double':
            self.precision_c = 'double'
        self._init_kernel()


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

    @staticmethod
    def _insert_and_rank_columns_with_ties(traces):
        sorted_indices = _cp.argsort(traces, axis=0)
        
        ranks = _cp.zeros_like(traces, dtype=_cp.float32)
        
        for col in range(traces.shape[1]):
            sorted_col = traces[sorted_indices[:, col], col]
            _, inverse_indices, counts = _cp.unique(sorted_col, return_inverse=True, return_counts=True)
            
            cumulative_counts = _cp.cumsum(counts)
            avg_ranks = (cumulative_counts - counts / 2.0).astype(_cp.float32)
            
            ranks[sorted_indices[:, col], col] = avg_ranks[inverse_indices]
        
        return ranks

    def _accumulate(self, traces, data):
        if self.traces is None:
            self.traces = traces.astype('float32')
        else:
            self.traces = _cp.concatenate((self.traces, traces.astype('float32')), axis=0)
        self.trace_ranks = self._insert_and_rank_columns_with_ties(self.traces[: self.processed_traces])
        if self.data is None:
            self.data = data.astype(self.partition_dtype)
        else:
            self.data = _cp.concatenate((self.data, data.astype(self.partition_dtype)), axis=0)
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
        self.accumulate_ranks_and_compute()
        return self.result

    @property
    def _distinguisher_str(self):
        return 'KW'


class KWDistinguisher(PartitionedDistinguisherBase, KWDistinguisherMixin):
    """Standalone distinguisher class using KW."""
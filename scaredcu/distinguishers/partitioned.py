from .base import DistinguisherMixin, _StandaloneDistinguisher
import cupy as _cp
import numba as _nb
from numba import cuda as _cpda
import time as _time
import logging as _logging

logger = _logging.getLogger(__name__)


class _PartitionnedDistinguisherBaseMixin(DistinguisherMixin):

    def _memory_usage(self, traces, data):
        self._init_partitions(data)
        dtype_size = _cp.dtype(self.precision).itemsize
        return 3 * dtype_size * data.shape[1] * traces.shape[1] * len(self.partitions)

    def _init_partitions(self, data):
        maxdata = _cp.nanmax(data)
        mindata = _cp.nanmin(data)
        if self.partitions is None:
            if maxdata > 255:
                raise ValueError('max value for intermediate data is greater than 255, you need to provide partitions explicitly at init.')
            if mindata < 0:
                raise ValueError('min value for intermediate data is lower than 0, you need to provide partitions explicitly at init.')
            ls = [0, 9, 64, 256]
            for r in ls:
                if maxdata <= r:
                    break
            self.partitions = _cp.arange(r, dtype='int32')

    def _initialize(self, traces, data):
        self._trace_length = traces.shape[1]
        self._data_words = data.shape[1]
        self._data_to_partition_index = _define_lut_func(self.partitions)
        self._initialize_accumulators()

    def _update(self, traces, data):
        if traces.shape[1] != self._trace_length:
            raise ValueError(f'traces has different length {traces.shape[1]} than already processed traces {self._trace_length}.')
        if data.shape[1] != self._data_words:
            raise ValueError(f'data has different number of data words {data.shape[1]} than already processed data {self._data_words}.')
        if not _cp.issubdtype(data.dtype, _cp.integer):
            raise TypeError(f'data dtype for partitioned distinguisher, including MIA and Template, must be an integer dtype, not {data.dtype}.')
        logger.info(f'Update of partitioned distinguisher {self.__class__.__name__} in progress.')
        data = self._data_to_partition_index(data)
        self._accumulate(traces, data)
        logger.info(f'End of accumulations of traces for {self.__class__.__name__}.')


@_cpda.jit()
def _build_lut(partitions, lut):
    start = _cpda.grid(1)
    stride = _cpda.gridsize(1)
    for i in range(start, len(partitions), stride):
        lut[partitions[i]] = i


def _define_lut_func(partitions):
    lut = _cp.zeros(2**17, dtype='int32') - 1
    _build_lut[32,32](partitions, lut)
    #@_cp.vectorize([_cp.int32(_cp.uint8), _cp.int32(_cp.uint16), _cp.int32(_cp.uint32), _cp.int32(_cp.uint64),
    #                _cp.int32(_cp.int8), _cp.int32(_cp.int16), _cp.int32(_cp.int32), _cp.int32(_cp.int64)])
    # @_cp.fuse
    def _lut_function(x):
        return lut[x]

    return _lut_function


class PartitionedDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """Base mixin for various traces partitioning based attacks (ANOVA, NICV, SNR, ...).

    Attacks differs mainly in the metric computation, not in the accumulation process.

    Attributes:
        partitions (numpy.ndarray or range, default=None): partitions used to categorize traces according to intermediate data value.
            if None, it will be automatically estimated at first update of distinguisher.
        sum (numpy.ndarray): sum of traces accumulator with shape (trace_size, data_words, len(partitions))
        sum_square (numpy.ndarray): sum of traces squared accumulator with shape (trace_size, data_words, len(partitions))
        counters (numpy.ndarray): number of traces accumulated by data word and partitions, with shape (data_words, len(partitions)).

    """

    def _initialize_accumulators(self):
        self.sum = _cp.zeros((self._trace_length, self._data_words, len(self.partitions)), dtype=self.precision)
        self.sum_square = _cp.zeros((self._trace_length, self._data_words, len(self.partitions)), dtype=self.precision)
        self.counters = _cp.zeros((self._data_words, len(self.partitions)), dtype=self.precision)

    @staticmethod
    @_nb.njit(parallel=True)
    def _accumulate_core_1(traces, data, self_sum, self_sum_square, self_counters, self_precision):
        for sample_idx in _nb.prange(traces.shape[1]):
            tmp_sum = _cp.zeros((self_counters.shape[0], self_counters.shape[1]), dtype='float64')
            tmp_sum_square = _cp.zeros((self_counters.shape[0], self_counters.shape[1]), dtype='float64')
            for trace_idx in range(traces.shape[0]):
                x = traces[trace_idx, sample_idx]
                xx = x * x
                for data_idx in range(data.shape[1]):
                    data_value = data[trace_idx, data_idx]
                    if data_value != -1:
                        tmp_sum[data_idx, data_value] += x
                        tmp_sum_square[data_idx, data_value] += xx
                        if sample_idx == 0:
                            self_counters[data_idx, data_value] += 1
            self_sum[sample_idx] += tmp_sum
            self_sum_square[sample_idx] += tmp_sum_square

    @staticmethod
    @_nb.njit(parallel=True)
    def _accumulate_core_2(traces, data, self_sum, self_sum_square, self_counters, self_precision):
        """Faster when number of partitions is <=9."""
        ftraces = traces.astype(self_precision)
        bool_mask = _cp.empty((traces.shape[0], data.shape[1] * self_counters.shape[1]), dtype=self_precision)
        for p in range(self_counters.shape[1]):
            tmp_bool = data == p  # Data are already transformed to correspond to partition indexes.
            self_counters[:, p] += tmp_bool.sum(0)
            bool_mask[:, p * data.shape[1]:(p + 1) * data.shape[1]] = tmp_bool
        self_sum += (bool_mask.T @ ftraces).reshape(self_counters.shape[1], data.shape[1], traces.shape[1]).T
        self_sum_square += (bool_mask.T @ (ftraces ** 2)).reshape(self_counters.shape[1], data.shape[1], traces.shape[1]).T

    def _accumulate(self, traces, data):
        """If the number of partitions is >9, the method 1 is selected.

        Otherwise, the fastest method is selected empirically.
        """
        if len(self.partitions) > 9:
            self._accumulate_core_1(traces, data, self.sum, self.sum_square, self.counters, self.precision)
        else:
            if not hasattr(self, '_timings'):
                self._timings = [-2, -1]
            function_idx = _cp.argmin(self._timings)
            function = [self._accumulate_core_1, self._accumulate_core_2][function_idx]
            t0 = _time.process_time()
            function(traces, data, self.sum, self.sum_square, self.counters, self.precision)
            self._timings[function_idx] = _time.process_time() - t0

    def _compute(self):
        self.sum = self.sum.swapaxes(0, 1)
        self.sum_square = self.sum_square.swapaxes(0, 1)

        result = _cp.empty((self._data_words, self._trace_length), dtype=self.precision)

        for i in range(self._data_words):
            non_zero_indices = self.counters[i] > 0
            non_zero_counters = self.counters[i][non_zero_indices]
            sums = self.sum[i][:, non_zero_indices]
            sums_squared = self.sum_square[i][:, non_zero_indices]
            number_non_zero = _cp.sum(non_zero_counters)

            tmp_result = self._compute_metric(
                non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero
            )
            tmp_result[_cp.isinf(tmp_result)] = _cp.nan
            result[i] = tmp_result.astype(self.precision)

        self.sum = self.sum.swapaxes(0, 1)
        self.sum_square = self.sum_square.swapaxes(0, 1)

        return result


def _set_partitions(obj, partitions):
    if partitions is not None:
        if not isinstance(partitions, (_cp.ndarray, list, range)):
            raise TypeError(f'partitions should be a ndarray, list or range instance, not {type(partitions)}.')
        if not isinstance(partitions, _cp.ndarray):
            partitions = _cp.array(partitions, dtype='int32')
        elif partitions.dtype.kind not in 'iu':
            raise ValueError(f'partitions should be an integer array, not {partitions.dtype}.')
        if _cp.max(partitions) >= 2**16:
            raise ValueError(f'partition values must be in ]-2^16, 2^16[, but {_cp.max(partitions)} found.')
        if _cp.min(partitions) <= -2**16:
            raise ValueError(f'partition values must be in ]-2^16, 2^16[, but {_cp.min(partitions)} found.')
    obj.partitions = partitions


class PartitionedDistinguisherBase(_StandaloneDistinguisher):
    def __init__(self, partitions=None, precision='float32'):
        super().__init__(precision=precision)
        _set_partitions(self, partitions=partitions)


class PartitionedDistinguisher(PartitionedDistinguisherBase, PartitionedDistinguisherMixin):
    pass


class ANOVADistinguisherMixin(PartitionedDistinguisherMixin):
    """This standalone partitioned distinguisher applies the ANOVA F-test metric."""

    def _compute_metric(self, non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero):
        total_non_empty_partitions = _cp.count_nonzero(non_zero_indices)

        partitions_means = (sums / non_zero_counters)
        mean = _cp.sum(sums, axis=-1, keepdims=True) / number_non_zero

        numerator = _cp.sum(
            (non_zero_counters * (partitions_means - mean) ** 2),
            axis=-1
        ) / (total_non_empty_partitions - 1)

        denominator = _cp.sum(
            (sums_squared - sums ** 2 / non_zero_counters),
            axis=-1
        ) / (number_non_zero - total_non_empty_partitions)

        return numerator / denominator

    @property
    def _distinguisher_str(self):
        return 'ANOVA'


class ANOVADistinguisher(PartitionedDistinguisherBase, ANOVADistinguisherMixin):
    __doc__ = PartitionedDistinguisherMixin.__doc__ + ANOVADistinguisherMixin.__doc__


class NICVDistinguisherMixin(PartitionedDistinguisherMixin):
    """This standalone partitioned distinguisher applies the NICV (Normalized Inter-Class Variance) metric."""

    def _compute_metric(self, non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero):
        mean = _cp.sum(sums, axis=1) / number_non_zero

        numerator = (((sums / non_zero_counters).T - mean).T)**2
        numerator *= non_zero_counters / number_non_zero
        numerator = _cp.sum(numerator, axis=1)

        denominator = _cp.sum(sums_squared, axis=1) / number_non_zero - (mean)**2

        return numerator / denominator

    @property
    def _distinguisher_str(self):
        return 'NICV'


class NICVDistinguisher(PartitionedDistinguisherBase, NICVDistinguisherMixin):
    __doc__ = PartitionedDistinguisherMixin.__doc__ + NICVDistinguisherMixin.__doc__


class SNRDistinguisherMixin(PartitionedDistinguisherMixin):
    """This standalone partitioned distinguisher applies the SNR (Signal to Noise Ratio) metric."""

    def _compute_metric(self, non_zero_indices, non_zero_counters, sums, sums_squared, number_non_zero):
        mean = _cp.sum(sums, axis=1) / number_non_zero
        numerator = (((sums / non_zero_counters).T - mean).T)**2
        numerator = _cp.sum(numerator, axis=1) / non_zero_indices.shape[0]

        denominator = (sums_squared / non_zero_counters) - (sums / non_zero_counters)**2
        denominator = _cp.sum(denominator, axis=1) / non_zero_indices.shape[0]

        return numerator / denominator

    @property
    def _distinguisher_str(self):
        return 'SNR'


class SNRDistinguisher(PartitionedDistinguisherBase, SNRDistinguisherMixin):
    __doc__ = PartitionedDistinguisherMixin.__doc__ + SNRDistinguisherMixin.__doc__


def _nanmean(arr, axis=None):
    nan_mask = ~_cp.isnan(arr)
    valid_count = _cp.sum(nan_mask, axis=axis)
    total_sum = _cp.nansum(arr, axis=axis)
    return total_sum / valid_count


class CollisionDistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """DistinguisherMixin for collision attacks."""

    def __init__(self, offset, delta_selection_function):
        self.offset = offset
        self.delta_selection_function = delta_selection_function
        self._saved_keys = None

    def _initialize_accumulators(self):
        if self._data_words != 2:
            raise ValueError('Collision distinguisher can only be used with 2 data words.')
        self.sum = _cp.zeros((self._trace_length, self._data_words, len(self.partitions)), dtype=self.precision)
        self.counters = _cp.zeros((self._data_words, len(self.partitions)), dtype=self.precision)

    @staticmethod
    @_cpda.jit(cache=True)
    def _accumulate_core(traces, data, self_sum, self_counters):
        start = _cpda.grid(1)
        stride = _cpda.gridsize(1)
        for sample_idx in range(start, traces.shape[1], stride):
            for trace_idx in range(traces.shape[0]):
                x = traces[trace_idx, sample_idx]
                for data_idx in range(data.shape[1]):
                    data_value = data[trace_idx, data_idx]
                    if data_value != -1:
                        self_sum[sample_idx, data_idx, data_value] += x
                        if sample_idx == 0:
                            self_counters[data_idx, data_value] += 1

    def _accumulate(self, traces, data):
        self._accumulate_core[128,8](traces, data, self.sum, self.counters)

    def _set_delta_func_args(self, keys):
        self._saved_keys = keys

    def _compute(self):

        if self._saved_keys is None:
            raise ValueError('No metadata provided for the delta selection function.')

        p0_values = self.partitions

        kwargs = {}
        for key in self._saved_keys:
            kwargs[key] = p0_values
        p1_values = self.delta_selection_function(**kwargs)

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

    @property
    def _distinguisher_str(self):
        return 'Collision'
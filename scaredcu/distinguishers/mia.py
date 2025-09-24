from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
import cupy as _cp
from numba import cuda as _cpda
import logging

logger = logging.getLogger(__name__)


class MIADistinguisherMixin(_PartitionnedDistinguisherBaseMixin):
    """This partitioned distinguisher mixin applies a mutual information computation."""

    def _memory_usage(self, traces, data):
        self._init_partitions(data)
        self._init_bin_edges(traces)
        dtype_size = _cp.dtype(self.precision).itemsize
        return 3 * dtype_size * data.shape[1] * traces.shape[1] * len(self.partitions) * self.bins_number

    def _init_bin_edges(self, traces):
        if self.bin_edges is None:
            logger.info('Start setting y_window and bin_edges.')
            self.y_window = (_cp.min(traces), _cp.max(traces))
            self.bin_edges = _cp.linspace(*self.y_window, self.bins_number + 1)
            logger.info('Bin edges set.')

    @property
    def bin_edges(self):
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, bin_edges):
        if bin_edges is None or not isinstance(bin_edges, (list, _cp.ndarray, range)):
            raise TypeError(f'bin_edges must be a ndarray, a list or a range, not {type(bin_edges)}.')
        if len(bin_edges) <= 1:
            raise ValueError(f'bin_edges length must be >1, but {len(bin_edges)}, found.')
        if not isinstance(bin_edges, _cp.ndarray):
            bin_edges = _cp.array(bin_edges, dtype='float64')
        for a, b in zip(bin_edges, bin_edges[1:]):
            if not a < b:
                raise ValueError(f'bin_edges must be sorted, but {a} >= {b}.')
        if _cp.sum(_cp.diff(_cp.diff(bin_edges))) > 1e-9:
            raise ValueError('bin_edges must be uniform (i.e with bins equally spaced.')
        self._bin_edges = bin_edges
        self.bins_number = len(bin_edges) - 1

    def _set_precision(self, precision):
        try:
            precision = _cp.dtype(precision)
        except TypeError:
            raise TypeError(f'precision should be a valid dtype, not {precision}.')
        self.precision = precision

    def _initialize_accumulators(self):
        self.accumulators = _cp.zeros((self._trace_length, self.bins_number, len(self.partitions), self._data_words),
                                      dtype=self.precision)

    @staticmethod
    @_cpda.jit(cache=True)
    def _accumulate_core_1(traces, data, self_bin_edges, self_accumulators, self_place_outliers):
        start = _cpda.grid(1)
        stride = _cpda.gridsize(1)

        nbins = len(self_bin_edges) - 1
        min_edge = self_bin_edges[0]
        max_edge = self_bin_edges[-1]
        norm = nbins / (max_edge - min_edge)

        for sample_idx in range(start, traces.shape[1], stride):
            for trace_idx in range(traces.shape[0]):
                x = traces[trace_idx, sample_idx]
                if x >= min_edge and x < max_edge:
                    bin_idx = int((x - min_edge) * norm)
                elif x == max_edge:
                    bin_idx = nbins - 1
                elif self_place_outliers:
                    if x > max_edge:
                        bin_idx = nbins - 1
                    else:
                        bin_idx = 0
                else:
                    continue
                for data_idx in range(data.shape[1]):
                    self_accumulators[sample_idx, bin_idx, data[trace_idx, data_idx], data_idx] += 1

    @staticmethod
    @_cpda.jit(cache=True)
    def _accumulate_core_2(traces, data, self_bin_edges, self_accumulators, self_place_outliers):
        start = _cpda.grid(1)
        stride = _cpda.gridsize(1)

        nbins = len(self_bin_edges) - 1
        min_edge = self_bin_edges[0]
        max_edge = self_bin_edges[-1]
        norm = nbins / (max_edge - min_edge)

        for data_idx in range(start, data.shape[1], stride):
            for trace_idx in range(traces.shape[0]):
                data_idy = data[trace_idx, data_idx]
                for sample_idx in range(traces.shape[1]):
                    x = traces[trace_idx, sample_idx]
                    if x >= min_edge and x < max_edge:
                        bin_idx = int((x - min_edge) * norm)
                    elif x == max_edge:
                        bin_idx = nbins - 1
                    elif self_place_outliers:
                        if x > max_edge:
                            bin_idx = nbins - 1
                        else:
                            bin_idx = 0
                    else:
                        continue
                    self_accumulators[sample_idx, bin_idx, data_idy, data_idx] += 1

    def _accumulate(self, traces, data):
        if self._data_words > 256:
            self.block_size = 256
            self.grid_size = ((self._data_words - 1) // self.block_size) + 1
            self._accumulate_core_2[self.grid_size, self.block_size](traces, data, self.bin_edges, self.accumulators, self.place_outliers)
        else:
            self.block_size = 256
            self.grid_size = ((self._trace_length - 1) // self.block_size) + 1
            self._accumulate_core_1[self.grid_size, self.block_size](traces, data, self.bin_edges, self.accumulators, self.place_outliers)

    def _compute_pdf(self, array, axis):
        s = array.sum(axis=axis)
        s[s == 0] = 1
        return (array.swapaxes(0, 1) / s).swapaxes(0, 1)

    def _compute(self):
        background = self.accumulators.sum(axis=2)

        pdfs_background = self._compute_pdf(background, axis=1) # P(L)
        pdfs_background[pdfs_background == 0] = 1
        print(f'pdfs_background p(L): {pdfs_background.shape}')

        pdfs_of_histos = self._compute_pdf(self.accumulators, axis=1) # P(L | H)
        pdfs_of_histos[pdfs_of_histos == 0] = 1
        print(f'pdfs_of_histos p(L | H): {pdfs_of_histos.shape}')

        histos_sums = self.accumulators.sum(axis=1)
        ratios = (histos_sums.swapaxes(0, 1) / background.sum(axis=1)).swapaxes(0, 1) # P(H)
        print(f'ratios P(H): {ratios.shape}')

        expected = pdfs_background * _cp.log(pdfs_background)  # P(L) * log(P(L))
        print(f'expected P(L) * log(P(L)): {expected.shape}')
        real = pdfs_of_histos * _cp.log(pdfs_of_histos) # P(L | H) * log(P(L | H))
        print(f'real P(L | H) * log(P(L | H)): {real.shape}')
        delta = (real.swapaxes(1, 2).swapaxes(0, 1) - expected).swapaxes(0, 1).swapaxes(1, 2) # (P(L | H) * log(P(L | H)) - P(L) * log(P(L)))
        print(f'delta (P(L | H) * log(P(L | H)) - P(L) * log(P(L)): {delta.shape}')
        res = delta.sum(axis=1) * ratios
        print(f'res (delta * P(H)): {res.shape}')
        return _cp.sum(res, axis=1).swapaxes(0, 1)

    @property
    def _distinguisher_str(self):
        return 'MIA'


def _set_histogram_parameters(obj, bins_number, bin_edges, place_outliers):
    if not isinstance(bins_number, int):
        raise TypeError(f'bins_number must be an integer, not {type(bins_number)}.')
    obj.bins_number = bins_number
    obj._bin_edges = None
    obj.y_window = None
    if bin_edges is not None:
        obj.bin_edges = bin_edges
    obj.place_outliers = place_outliers


class MIADistinguisher(PartitionedDistinguisherBase, MIADistinguisherMixin):

    def __init__(self, bins_number=128, bin_edges=None, partitions=None, place_outliers=False, precision='uint32'):
        _set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges, place_outliers=place_outliers)
        return super().__init__(partitions=partitions, precision=precision)

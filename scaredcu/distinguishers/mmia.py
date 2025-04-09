from .partitioned import PartitionedDistinguisherBase, _PartitionnedDistinguisherBaseMixin
from .mia import MIADistinguisherMixin
from .mia import _set_histogram_parameters as _set_histogram_parameters_mia
from numba import cuda as _cuda
import cupy as _cp
import logging


logger = logging.getLogger(__name__)


class MMIADistinguisherMixin(MIADistinguisherMixin):
    """This partitioned distinguisher mixin applies a multivariate mutual information computation."""

    def _memory_usage(self, traces, data):
        self._init_partitions(data)
        self._init_bin_edges(traces)
        dtype_size = _cp.dtype(self.precision).itemsize
        return 3 * dtype_size * data.shape[1] * traces.shape[1] * len(self.partitions) * self.bins_number * self.bins_numberx2

    def _init_bin_edges(self, traces):
        if self.same_bins:
            if self.bin_edges is None and self.bin_edgesx2 is None:
                self.y_window = (min(_cp.min(traces[:, :self.offset]), _cp.min(traces[:, -self.offset:])),
                                 max(_cp.max(traces[:, :self.offset]), _cp.max(traces[:, -self.offset:])))
                self.bin_edges = _cp.linspace(*self.y_window, self.bins_number + 1)
                self.y_windowx2 = self.y_window
                self.bin_edgesx2 = self.bin_edges
            elif self.bin_edgesx2 is None:
                self.bin_edgesx2 = self.bin_edges
        else:
            if self.bin_edges is None:
                self.y_window = (min(_cp.min(traces[:, :self.offset]), _cp.min(traces[:, -self.offset:])),
                                 max(_cp.max(traces[:, :self.offset]), _cp.max(traces[:, -self.offset:])))
                self.bin_edges = _cp.linspace(*self.y_window, self.bins_number + 1)
            if self.bin_edgesx2 is None:
                self.y_windowx2 = (_cp.min(traces[:, -self.offset:]), _cp.max(traces[:, -self.offset:]))
                self.bin_edgesx2 = _cp.linspace(*self.y_windowx2, self.bins_numberx2 + 1)

    def _initialize_accumulators(self):
        self.accumulators = _cp.zeros((self._trace_length - self.offset, self.bins_number, self.bins_numberx2,
                                       len(self.partitions), self._data_words),
                                      dtype=self.precision)

    @staticmethod
    @_cuda.jit(cache=True)
    def _accumulate_core_1(traces, data, self_bin_edges, self_accumulators, offset, self_bin_edgesx2,
                           self_place_outliers):
        start = _cuda.grid(1)
        stride = _cuda.gridsize(1)

        nbins = len(self_bin_edges) - 1
        min_edgex1 = self_bin_edges[0]
        max_edgex1 = self_bin_edges[-1]
        normx1 = nbins / (max_edgex1 - min_edgex1)

        nbinsx2 = len(self_bin_edgesx2) - 1
        min_edgex2 = self_bin_edgesx2[0]
        max_edgex2 = self_bin_edgesx2[-1]
        normx2 = nbinsx2 / (max_edgex2 - min_edgex2)

        for sample_idx in range(start, traces.shape[1] - offset, stride):
            for trace_idx in range(traces.shape[0]):
                x1 = traces[trace_idx, sample_idx]
                if x1 >= min_edgex1 and x1 < max_edgex1:
                    bin_idx1 = int((x1 - min_edgex1) * normx1)
                elif x1 == max_edgex1:
                    bin_idx1 = nbins - 1
                elif self_place_outliers:
                    if x1 > max_edgex1:
                        bin_idx1 = nbins - 1
                    else:
                        bin_idx1 = 0
                else:
                    continue
                x2 = traces[trace_idx, sample_idx + offset]
                if x2 >= min_edgex2 and x2 < max_edgex2:
                    bin_idx2 = int((x2 - min_edgex2) * normx2)
                elif x2 == max_edgex2:
                    bin_idx2 = nbinsx2 - 1
                elif self_place_outliers:
                    if x2 > max_edgex2:
                        bin_idx2 = nbinsx2 - 1
                    else:
                        bin_idx2 = 0
                else:
                    continue
                for data_idx in range(data.shape[1]):
                    self_accumulators[sample_idx, bin_idx1, bin_idx2, data[trace_idx, data_idx], data_idx] += 1


    @staticmethod
    @_cuda.jit(cache=True)
    def _accumulate_core_2(traces, data, self_bin_edges, self_accumulators, offset, self_bin_edgesx2,
                           self_place_outliers):
        start = _cuda.grid(1)
        stride = _cuda.gridsize(1)

        nbins = len(self_bin_edges) - 1
        min_edgex1 = self_bin_edges[0]
        max_edgex1 = self_bin_edges[-1]
        normx1 = nbins / (max_edgex1 - min_edgex1)

        nbinsx2 = len(self_bin_edgesx2) - 1
        min_edgex2 = self_bin_edgesx2[0]
        max_edgex2 = self_bin_edgesx2[-1]
        normx2 = nbinsx2 / (max_edgex2 - min_edgex2)

        for data_idx in range(start, data.shape[1], stride):
            for trace_idx in range(traces.shape[0]):
                data_idy = data[trace_idx, data_idx]
                for sample_idx in range(traces.shape[1]):
                    x1 = traces[trace_idx, sample_idx]
                    if x1 >= min_edgex1 and x1 < max_edgex1:
                        bin_idx1 = int((x1 - min_edgex1) * normx1)
                    elif x1 == max_edgex1:
                        bin_idx1 = nbins - 1
                    elif self_place_outliers:
                        if x1 > max_edgex1:
                            bin_idx1 = nbins - 1
                        else:
                            bin_idx1 = 0
                    else:
                        continue
                    x2 = traces[trace_idx, sample_idx + offset]
                    if x2 >= min_edgex2 and x2 < max_edgex2:
                        bin_idx2 = int((x2 - min_edgex2) * normx2)
                    elif x2 == max_edgex2:
                        bin_idx2 = nbinsx2 - 1
                    elif self_place_outliers:
                        if x2 > max_edgex2:
                            bin_idx2 = nbinsx2 - 1
                        else:
                            bin_idx2 = 0
                    else:
                        continue
                    for data_idx in range(data.shape[1]):
                        self_accumulators[sample_idx, bin_idx1, bin_idx2, data_idy, data_idx] += 1


    def _accumulate(self, traces, data):
        if self._data_words > 256:
            self.block_size = 256
            self.grid_size = ((self._data_words - 1) // self.block_size) + 1
            self._accumulate_core_2[self.grid_size, self.block_size](traces, data, self.bin_edges, self.accumulators, self.offset, self.bin_edgesx2,
                                self.place_outliers)
        else:
            self.block_size = 256
            self.grid_size = ((self._trace_length - 1) // self.block_size) + 1        
            self._accumulate_core_1[self.grid_size, self.block_size](traces, data, self.bin_edges, self.accumulators, self.offset, self.bin_edgesx2,
                                self.place_outliers)

    def _compute_super(self, accumulators):
        background = accumulators.sum(axis=2)

        pdfs_background = self._compute_pdf(background, axis=1)
        pdfs_background[pdfs_background == 0] = 1

        pdfs_of_histos = self._compute_pdf(accumulators, axis=1)
        pdfs_of_histos[pdfs_of_histos == 0] = 1

        histos_sums = accumulators.sum(axis=1)
        ratios = (histos_sums.swapaxes(0, 1) / background.sum(axis=1)).swapaxes(0, 1)
        ratios[_cp.isnan(ratios)] = 0

        expected = pdfs_background * _cp.log(pdfs_background)

        real = pdfs_of_histos * _cp.log(pdfs_of_histos)

        delta = (real.swapaxes(1, 2).swapaxes(0, 1) - expected).swapaxes(0, 1).swapaxes(1, 2)
        res = delta.sum(axis=1) * ratios
        return _cp.sum(res, axis=1).swapaxes(0, 1)

    def _compute(self):
        sum_x1_y = self.accumulators.sum(axis=1).sum(axis=2)
        x2_pdf = self._compute_pdf(sum_x1_y, axis=1)
        x2_pdf = x2_pdf.swapaxes(0, 2)
        assert x2_pdf.shape[1] == self.accumulators.shape[2]
        for i in range(self.accumulators.shape[2]):
            IX1Y_X2i = self._compute_super(self.accumulators[:, :, i, :, :])
            x2_pdf[:, i, :] *= IX1Y_X2i
        sum_x2 = self.accumulators.sum(axis=2)
        IX1Y = self._compute_super(sum_x2)
        return -1 * (IX1Y - x2_pdf.sum(axis=1))

    @property
    def _distinguisher_str(self):
        return 'MMIA'


def _set_histogram_parameters(obj, bins_number, bin_edges, offset, same_bins, bin_edgesx2, bins_numberx2, place_outliers):
    if bin_edges is not None:
        bins_number = len(bin_edges) - 1
    _set_histogram_parameters_mia(obj, bins_number=bins_number, bin_edges=bin_edges, place_outliers=place_outliers)
    obj.offset = offset
    obj.same_bins = same_bins

    assert not (same_bins and (bin_edgesx2 is not None and bins_numberx2 is not None))

    if same_bins:
        obj.bins_numberx2 = bins_number
    elif bin_edgesx2 is not None:
        obj.bins_numberx2 = len(bin_edgesx2) - 1

    obj.bin_edgesx2 = bin_edgesx2


class MMIADistinguisher(PartitionedDistinguisherBase, MMIADistinguisherMixin):

    def __init__(self, offset=1, bins_number=128, bins_numberx2=None, same_bins=True, bin_edges=None, bin_edgesx2=None,
                 place_outliers=False, *args, **kwargs):
        _set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges, offset=offset, same_bins=same_bins,
                                  bin_edgesx2=bin_edgesx2, bins_numberx2=bins_numberx2, place_outliers=place_outliers)
        return super().__init__(*args, **kwargs)
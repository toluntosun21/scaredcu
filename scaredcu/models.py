from . import _utils as utils
import cupy as _cp
import abc
import numba

_HW_LUT = _cp.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype='uint32')


# @numba.vectorize([numba.uint32(numba.uint8)])
def _fhw8(x):
    return _HW_LUT[x]


# @numba.vectorize([numba.uint32(numba.uint16)])
def _fhw16(x):
    return _HW_LUT[x & 0x00ff] + _HW_LUT[x >> 8]


# @numba.vectorize([numba.uint32(numba.uint32)])
def _fhw32(x):
    r = 0
    for _ in range(4):
        r += _HW_LUT[x & 0x000000ff]
        x >>= 8
    return r


# @numba.vectorize([numba.uint32(numba.uint64)])
def _fhw64(x):
    r = 0
    for _ in range(8):
        r += _HW_LUT[x & 0x00000000000000ff]
        x >>= 8
    return r


_hw_functions_list = dict([(1, _fhw8), (2, _fhw16), (4, _fhw32), (8, _fhw64)])


class Model(abc.ABC):
    """Leakage model callable abstract class.

    Use this abstract class to implement your own leakage function. Subclass it and define a _compute method which
    take a data argument.

    _compute function must returns a cupy array with all dimensions preserved except the last.

    See implementations of Value, HammingWeight or Monobit model for examples.

    """

    def __call__(self, data, axis=-1):
        if not isinstance(data, _cp.ndarray):
            raise TypeError(f'Model should take ndarray as input data, not {type(data)}.')
        if data.dtype.kind not in ('b', 'i', 'u', 'f', 'c'):
            raise ValueError(f'Model should take numerical ndarray as input data, not {data.dtype}).')

        if axis == -1:
            axis = len(data.shape) - 1

        results = self._compute(data, axis=axis)

        check_shape = [d for i, d in enumerate(results.shape) if i != axis]
        origin_shape = [d for i, d in enumerate(data.shape) if i != axis]
        if check_shape != origin_shape:
            raise ValueError(f'Model instance {self.__class__} does not preserve dimensions of data properly on call.')
        return results

    @abc.abstractmethod
    def _compute(self, data, axis):
        pass

    @property
    @abc.abstractmethod
    def max_data_value(self):
        pass


class Value(Model):
    """Value leakage model class.

    Instances of this class are callables which takes a data cupy array as input and returns it unchanged.

    Args:
        data (cupy.ndarray): numeric cupy ndarray

    Returns:
        (cupy.ndarray): unchanged input data cupy ndarray.

    """

    def _compute(self, data, axis):
        return data

    @property
    def max_data_value(self):
        return 256

    def __str__(self):
        return 'Value'


class Monobit(Model):
    """Monobit model leakage class.

    Instances of this class are callables which takes a data cupy array as input and
    returns the monobit model value computed on last dimension of the array.

    Attributes:
        bit (int): number of the bit targeted. Should be between 0 and 8, otherwise raises a ValueError.

    Args:
        data (cupy.ndarray): a ndarray of numeric type

    Returns:
        (cupy.ndarray) an ndarray of the same shape as data, with the result monobit mask applied.

    """

    def __init__(self, bit):
        if not isinstance(bit, int):
            raise TypeError(f'bit target should be an int, not {type(bit)}.')
        if bit < 0 or bit > 8:
            raise ValueError(f'bit should be between 0 and 8, not {bit}.')
        self.bit = bit

    def _compute(self, data, axis):
        return (_cp.bitwise_and(data, 2 ** self.bit) > 0).astype('uint8')

    @property
    def max_data_value(self):
        return 1

    def __str__(self):
        return f'Monobit {self.bit}'


class HammingWeight(Model):
    """Hamming weight leakage model for unsigned integer arrays.

    Instances of this class are callables which takes a data cupy array as input and returns the
    Hamming Weight values computed on the last dimension of the array, and on a number of words
    defined at instantiation.

    Attributes:
        nb_words (int, default=1): number of words on which to compute the hamming weight.
        expected_dtype(cupy.dtype, default='uint8'): expected dtype of input data.

    Args:
        data (cupy.ndarray): a unsigned integer ndarray.

    Returns:
        (cupy.ndarray) an ndarray with hamming weight computed on last dimension.
            Every dimensions but the last are preserved.

    """

    def __init__(self, nb_words=1, expected_dtype='uint8'):
        if not isinstance(nb_words, int):
            raise TypeError(f'nb_words should be an integer, not {nb_words}.')
        if nb_words <= 0:
            raise ValueError(f'nb_words must be strictly greater than 0, not {nb_words}.')
        try:
            expected_dtype = _cp.dtype(expected_dtype)
        except TypeError:
            raise ValueError(f'{expected_dtype} is not a valid dtype.')
        if expected_dtype.kind != 'u':
            raise ValueError(f'`expected_dtype` should be an unsigned integer dtype, not {expected_dtype}).')
        self.nb_words = nb_words
        self.expected_dtype = expected_dtype

    def _compute(self, data, axis):
        if data.dtype.kind != 'u':
            raise ValueError(f'HammingWeight should take unsigned integer data as input, not {data.dtype}).')

        if data.dtype != self.expected_dtype:
            raise ValueError(f'Expected dtype for HammingWeight input data is {self.expected_dtype}, not {data.dtype}.')

        if data.shape[axis] < self.nb_words:
            raise ValueError(f'data should have at least {self.nb_words} as dimension with index {axis}, not {data.shape[axis]}.')

        result_data = _hw_functions_list[data.dtype.itemsize](data)
        if self.nb_words > 1:
            final_w_dimension = data.shape[axis] // self.nb_words
            final_shape = [d if i != axis else final_w_dimension for i, d in enumerate(data.shape)]
            result = _cp.zeros(final_shape, dtype='uint32').swapaxes(0, axis)
            result_data = _cp.swapaxes(result_data, 0, axis)
            for i in range(result.shape[0]):
                slices = result_data[i * self.nb_words: (i + 1) * self.nb_words]
                result[i] = _cp.sum(slices, axis=0)
            result_data = result
            result_data = _cp.swapaxes(result, 0, axis)

        return result_data

    @property
    def max_data_value(self):
        return self.nb_words * self.expected_dtype.itemsize * 8

    def __str__(self):
        return f'Hamming Weight on {self.nb_words} word(s).'


class SignedHammingWeight(HammingWeight):
    """Signed Hamming Weight leakage model class.

    Instances of this class are callables which takes a data cupy array as input and returns it unchanged.

    Args:
        data (cupy.ndarray): numeric cupy ndarray

    Returns:
        (cupy.ndarray): unchanged input data cupy ndarray.

    """
    def __init__(self, expected_dtype='int16'):
        self.s_dtype = expected_dtype
        super().__init__(expected_dtype=utils.s2u(expected_dtype))

    def _compute(self, data, axis):
        if data.dtype != self.s_dtype:
            raise ValueError(f'Expected dtype for SignedHammingWeight input data is {self.s_dtype}, not {data.dtype}.')
        return super()._compute(data.astype(self.expected_dtype), axis)

    def __str__(self):
        return 'Signed Hamming Weight'


class AbsoluteValue(Model):
    """Absolute Value leakage model class.

    Instances of this class are callables which takes a data cupy array as input and returns it unchanged.

    Args:
        data (cupy.ndarray): numeric cupy ndarray

    Returns:
        (cupy.ndarray): unchanged input data cupy ndarray.

    """

    def _compute(self, data, axis):
        return _cp.abs(data)

    @property
    def max_data_value(self):
        raise Exception('Unimplemented method, AbsoluteValue.max_data_value()')

    def __str__(self):
        return 'Absolute Value'


class ShiftRight(Model):
    def __init__(self, numb_bits, abs=False, max=None):
        self.numb_bits = numb_bits
        self.abs = abs
        self.max = max if max is not None else 2**(16 - numb_bits) - 1

    def _compute(self, data, axis):
        if self.abs:
            return _cp.abs(data) >> self.numb_bits
        else:
            return data >> self.numb_bits

    @property
    def max_data_value(self):
        return self.max

    def __str__(self):
        return 'ShiftRight'


class Mask(Model):
    def __init__(self, numb_bits, abs=False):
        self.mask = (1 << numb_bits) - 1
        self.max = self.mask
        self.abs = abs

    def _compute(self, data, axis):
        if self.abs:
            return _cp.abs(data) & self.mask
        else:
            return data & self.mask

    @property
    def max_data_value(self):
        return self.mask

    def __str__(self):
        return 'Mask'


class OPFTable(Model):
    """Optimal Prediction Function Table.

    Instances of this class are callables which takes a data cupy array as input and returns it unchanged.

    Args:
        data (cupy.ndarray): numeric cupy ndarray

    Returns:
        (cupy.ndarray): unchanged input data cupy ndarray.

    """
    def __init__(self, table):
        self.table = table

    @property
    def max_data_value(self):
        return self.table.max()

    def _compute(self, data, axis):
        return self.table[data]


class OPFTableWithBuild(OPFTable):
    """Optimal Prediction Function Table.

    Instances of this class are callables which takes a data cupy array as input and returns it unchanged.

    Args:
        data (cupy.ndarray): numeric cupy ndarray

    Returns:
        (cupy.ndarray): unchanged input data cupy ndarray.

    """
    def __init__(self, d=2):
        self.table = self.build(d)

    @abc.abstractmethod
    def build(self, d):
        pass

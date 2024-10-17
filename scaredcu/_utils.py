import cupy as _cu
import numpy as _np

def _is_bytes_array(array):
    # Note: Integer arrays cannot contain np.nan or np.inf
    if not isinstance(array, _cu.ndarray):
        raise TypeError(f'array should be a Numpy ndarray instance, not {type(array)}.')
    if array.dtype == _cu.uint8:
        return True
    if array.dtype.kind not in 'ui':
        raise ValueError(f'array should be an integer array, not {array.dtype}.')
    if array.dtype.kind == 'i' and _cu.min(array) < 0:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but lowest value {_cu.min(array)} found.')
    if array.dtype != _cu.int8 and _cu.max(array) > 255:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but highest value {_cu.max(array)} found.')
    return True


def unpack_guess(guess, q):
    return guess % q, guess // q


def pack_guess(guess, q):
    return guess[0] + guess[1]*q


def succ_ratio(key, q, conv, convergence_traces, accept_shift=False, accept_neg=True, clear_nan_inf=True, sa=2, incomplete=False, np_flag=False):

    _cunp = _cu if not np_flag else _np

    if len(convergence_traces.shape) == 2:
        convergence_traces = convergence_traces[:, :, _cunp.newaxis]
    elif len(convergence_traces.shape) != 3:
        raise ValueError(f'Invalid shape for Convergence Traces {convergence_traces.shape}')

    if clear_nan_inf:
        convergence_traces[_cunp.isnan(convergence_traces)] = 0
        convergence_traces[_cunp.isinf(convergence_traces)] = 0

    succ_ratios = _cunp.zeros((convergence_traces.shape[1], convergence_traces.shape[2]))
    for key_index in range(convergence_traces.shape[1]):
        N_t = conv
        for j in range(convergence_traces.shape[2]):
            s = (convergence_traces[:,key_index,j].argsort()[::-1][0])
            if incomplete:
                s = _cunp.array(unpack_guess(s, q), dtype=key.dtype)
            s = s % q
            s_ = key[key_index] % q

            if (s_ == s).all() or (accept_neg and ((q - s_) == s).all()):
                succ_ratios[key_index, j] = 1
            elif accept_shift:
                if (is_shift(s_, s, q, sa)):
                    succ_ratios[key_index, j] = 1
                elif (is_shift(q - s_, s, q, sa)):
                    succ_ratios[key_index, j] = 1
        N_t += conv
    
    return succ_ratios.mean(axis=0)


def s2u(dtype):
    if dtype == 'int32':
        return 'uint32'
    elif dtype == 'int16':
        return 'uint16'
    elif dtype == 'int64':
        return 'uint64'
    elif dtype == 'int8':
        return 'uint8'
    elif dtype == 'uint64' or dtype == 'uint32' or dtype == 'uint16' or dtype == 'uint8':
        return dtype
    else: raise ValueError(f'Invalid dtype {dtype}')


def u2s(dtype):
    if dtype == 'uint32':
        return 'int32'
    elif dtype == 'uint16':
        return 'int16'
    elif dtype == 'uint64':
        return 'int64'
    elif dtype == 'uint8':
        return 'int8'
    else: raise ValueError(f'Invalid dtype {dtype}')


def u22u(dtype):
    if dtype == 'uint32':
        return 'uint64'
    elif dtype == 'uint16':
        return 'uint32'
    elif dtype == 'uint8':
        return 'uint16'
    else: raise ValueError(f'Invalid dtype {dtype}')


def s22s(dtype):
    if dtype == 'int32':
        return 'int64'
    elif dtype == 'int16':
        return 'int32'
    elif dtype == 'int8':
        return 'int16'
    else: raise ValueError(f'Invalid dtype {dtype}')


def u2b(dtype):
    if dtype == 'uint32':
        return 32
    elif dtype == 'uint16':
        return 16
    elif dtype == 'uint8':
        return 8
    else: raise ValueError(f'Invalid dtype {dtype}')
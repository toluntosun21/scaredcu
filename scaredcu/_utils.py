import cupy as _cp
import numpy as _np

def _is_bytes_array(array):
    # Note: Integer arrays cannot contain np.nan or np.inf
    if not isinstance(array, _cp.ndarray):
        raise TypeError(f'array should be a Cupy ndarray instance, not {type(array)}.')
    if array.dtype == _cp.uint8:
        return True
    if array.dtype.kind not in 'ui':
        raise ValueError(f'array should be an integer array, not {array.dtype}.')
    if array.dtype.kind == 'i' and _cp.min(array) < 0:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but lowest value {_cp.min(array)} found.')
    if array.dtype != _cp.int8 and _cp.max(array) > 255:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but highest value {_cp.max(array)} found.')
    return True


def unpack_guess(guess, q):
    return guess % q, guess // q


def pack_guess(guess, q):
    return guess[0] + guess[1]*q


def succ_ratio(key, q, conv, convergence_traces, accept_neg=True, clear_nan_inf=True, incomplete=False, np_flag=False, guesses=None):

    _cpnp = _cp if not np_flag else _np

    if len(convergence_traces.shape) == 2:
        convergence_traces = convergence_traces[:, :, _cpnp.newaxis]
    elif len(convergence_traces.shape) != 3:
        raise ValueError(f'Invalid shape for Convergence Traces {convergence_traces.shape}')

    if clear_nan_inf:
        convergence_traces[_cpnp.isnan(convergence_traces)] = 0
        convergence_traces[_cpnp.isinf(convergence_traces)] = 0

    succ_ratios = _cpnp.zeros((convergence_traces.shape[1], convergence_traces.shape[2]))
    for key_index in range(convergence_traces.shape[1]):
        N_t = conv
        for j in range(convergence_traces.shape[2]):
            s = (convergence_traces[:,key_index,j].argmax())
            if guesses is None:
                if incomplete:
                    s = _cpnp.array(unpack_guess(s, q), dtype=key.dtype)
                s = s % q
            else:
                s = guesses[s,:,j] % q
            if incomplete:
                s_ = key[2*key_index:2*key_index+2] % q
            else:                
                s_ = key[key_index] % q

            if (s_ == s).all() or (accept_neg and ((q - s_) == s).all()):
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
from tqdm.auto import tqdm
import cupy as _cu
from . import modop
import scaredcu._utils as utils
import scaredcu.gpu_format as gpu_format


_HW_LUT = _cu.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
                     3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype='uint32')


def hw(a, dtype='uint32', B=None):
    if B is None:
        B = _cu.dtype(dtype).itemsize*8
    b = _cu.zeros(a.shape, dtype=dtype)
    L = 4 if dtype == 'uint32' else 2
    for i in range(L):
        b += _HW_LUT[(a >> (8*i)) & 0xFF]
    return b



def collect_traces_basemult(N, reduction=None, n=256, q=769, alpha=1, beta1=0, beta2=0,sigma=0, 
                            const_seed=False, prng_off=False, boolean_mask=False,
                            two_step=False, mult_hw_dtype=None, leak_reduction=True, hw_dtype=None,
                            incomplete=0, seed=0):
    incomplete_ = 1 << incomplete
    dtype = reduction.o_dtype
    q = reduction.q

    if const_seed:
        _cu.random.seed(seed)

    if hw_dtype == None:
        hw_dtype = utils.s2u(dtype)

    if two_step and mult_hw_dtype is None:
        mult_hw_dtype = utils.u22u(hw_dtype)

    if reduction is None:
        reduction = modop.Reduction_Q2Q2(q, dtype)

    mult_dtype = utils.s22s(dtype)

    samples, cs, ss, s0s, s1s = [], [], [], [], []

    s = reduction.reduce(_cu.random.randint(1, q, (n, incomplete_), dtype=dtype))
    
    for _ in tqdm(range(N)):

        c = reduction.reduce(_cu.random.randint(0, q, (n, incomplete_), dtype=dtype))
        masks = _cu.random.randint(0, q, (n, incomplete_), dtype=dtype) if not prng_off else _cu.zeros((n, incomplete_), dtype=dtype)

        cs.append(c)


        s0 = reduction.reduce((s - masks))
        s1 = reduction.reduce(masks)
        s0s.append(s0)
        s1s.append(s1)
        ss.append(s)

        res0 = (c.astype(mult_dtype) * s.astype(mult_dtype)).sum(axis=1)
        res = reduction.reduce(res0)

        if boolean_mask:
            if two_step:
                res01 = _cu.random.randint(0, q*q, dtype=mult_hw_dtype)
                res00 = res0 ^ res01
            res1 = _cu.random.randint(0, q, dtype=hw_dtype)
            res0 = res ^ res1
        else:
            res00 = ((c.astype(mult_dtype) * s0.astype(mult_dtype))).sum(axis=1)
            res01 = ((c.astype(mult_dtype) * s1.astype(mult_dtype))).sum(axis=1)
            res0 = reduction.reduce(res00)
            res1 = reduction.reduce(res01)
            assert (((res0 + res1) % q) == (res % q)).all()


        if leak_reduction:
            samples_i1 = hw(res0, hw_dtype)*alpha + beta1 + _cu.random.normal(0, sigma, n)
            samples_i2 = hw(res1, hw_dtype)*alpha + beta2 + _cu.random.normal(0, sigma, n)

        if two_step:
            samples_i01 = hw(res00, mult_hw_dtype)*alpha + beta1 + _cu.random.normal(0, sigma, n)
            samples_i02 = hw(res01, mult_hw_dtype)*alpha + beta2 + _cu.random.normal(0, sigma, n)

            if leak_reduction:
                samples_i1_concat = _cu.empty((samples_i1.shape[0]*2))
                samples_i2_concat = _cu.empty((samples_i2.shape[0]*2))
                samples_i1_concat[1::2] = samples_i1
                samples_i1_concat[::2] = samples_i01
                samples_i2_concat[1::2] = samples_i2
                samples_i2_concat[::2] = samples_i02
                samples_i_concat = _cu.concatenate(( samples_i1_concat, samples_i2_concat ))
            else:
                samples_i_concat = _cu.concatenate(( samples_i01, samples_i02 ))
        else:
            samples_i_concat = _cu.concatenate(( samples_i1, samples_i2 )) 

        samples.append(samples_i_concat)
        
    if incomplete_ == 1:
        return gpu_format.read_ths_from_ram(_cu.array(samples)[:,:], c=_cu.array(cs)[:,:,0],
                                          s=_cu.array(ss)[:,:,0], s0=_cu.array(s0s)[:,:,0], s1=_cu.array(s1s)[:,:,0])
    else:
        return gpu_format.read_ths_from_ram(_cu.array(samples)[:,:], c=_cu.array(cs)[:,:],
                                      s=_cu.array(ss)[:,:], s0=_cu.array(s0s), s1=_cu.array(s1s))
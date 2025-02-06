from tqdm.auto import tqdm
import cupy as _cu
from . import modop
import scaredcu._utils as utils
import scaredcu.gpu_format as gpu_format
from  scaredcu.models import SignedHammingWeight as SignedHammingWeight

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



def collect_traces_basemult(N, d=2, reduction=None, n=256, q=769, alpha=1, beta=0,sigma=0, 
                            const_seed=False, prng_off=False,
                            incomplete=0, seed=0, model=None):


    incomplete_ = 1 << incomplete
    dtype = reduction.o_dtype
    q = reduction.q

    if const_seed:
        _cu.random.seed(seed)

    if model is None:
        model = SignedHammingWeight(expected_dtype=dtype)

    if reduction is None:
        reduction = modop.Reduction_Q2Q2(q, dtype)

    mult_dtype = utils.s22s(dtype)

    samples, cs, ss, masks = [], [], [], []

    s = reduction.reduce(_cu.random.randint(1, q, (n, incomplete_), dtype=dtype))
    
    for _ in tqdm(range(N)):

        c = reduction.reduce(_cu.random.randint(0, q, (n, incomplete_), dtype=dtype))
        s_ = _cu.random.randint(0, q, (d, n, incomplete_), dtype=dtype) if not prng_off else _cu.zeros((d, n, incomplete_), dtype=dtype)
        s_ = reduction.reduce(s_)


        masks_sum = s_[1:].astype(mult_dtype).sum(axis=0).astype(dtype)

        s_[0] = reduction.reduce(s - masks_sum) if d > 1 else s

        cs.append(c)
        masks.append(s_)
        ss.append(s)

        res_mult_unmasked = (c.astype(mult_dtype) * s.astype(mult_dtype)).sum(axis=1)
        res_red_unmasked = reduction.reduce(res_mult_unmasked)

        res_mult_masked = ((c.astype(mult_dtype) * s_.astype(mult_dtype))).sum(axis=2)
        res_red_masked = reduction.reduce(res_mult_masked)

        assert (((res_red_masked.sum(axis=0)) % q) == (res_red_unmasked % q)).all()


        samples_i = model._compute(res_red_masked,1)*alpha + beta + _cu.random.normal(0, sigma, n)

        samples.append(samples_i.ravel())
        
    if incomplete_ == 1:
        return gpu_format.read_ths_from_ram(_cu.array(samples)[:,:], c=_cu.array(cs)[:,:,0],
                                          s=_cu.array(ss)[:,:,0], masks=_cu.array(masks)[:,:,:,0])
    else:
        return gpu_format.read_ths_from_ram(_cu.array(samples)[:,:], c=_cu.array(cs)[:,:],
                                      s=_cu.array(ss)[:,:], masks=_cu.array(masks)[:,:,:])
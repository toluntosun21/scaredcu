from tqdm.auto import tqdm
import cupy as _cp
import numpy as _np
from . import modop
import scaredcu._utils as utils
import scaredcu.estraces.gpu_format as gpu_format
from  scaredcu.models import SignedHammingWeight as SignedHammingWeight
import estraces





def collect_traces_basemult(N, d=2, reduction=None, n=256, q=769, alpha=1, beta=0,sigma=0, 
                            const_seed=False, prng_off=False,
                            incomplete=0, seed=0, model=None, filename=None, get_masks=False):


    _npcp = _cp if filename is None else _np

    incomplete_ = 1 << incomplete
    dtype = reduction.o_dtype
    q = reduction.q

    if const_seed:
        _cp.random.seed(seed)

    if model is None:
        model = SignedHammingWeight(expected_dtype=dtype)

    if reduction is None:
        reduction = modop.Reduction_Q2Q2(q, dtype)

    mult_dtype = utils.s22s(dtype)

    samples, cs, ss, masks = [], [], [], []

    s = reduction.reduce(_cp.random.randint(1, q, (n, incomplete_), dtype=dtype))

    if filename is not None:
        es_writer = estraces.ETSWriter(filename=filename, overwrite=False)

    for _ in tqdm(range(N)):

        c = reduction.reduce(_cp.random.randint(0, q, (n, incomplete_), dtype=dtype))
        s_ = _cp.random.randint(0, q, (d, n, incomplete_), dtype=dtype) if not prng_off else _cp.zeros((d, n, incomplete_), dtype=dtype)
        s_ = reduction.reduce(s_)


        masks_sum = s_[1:].astype(mult_dtype).sum(axis=0).astype(dtype)

        s_[0] = reduction.reduce(s - masks_sum) if d > 1 else s


        res_mult_unmasked = (c.astype(mult_dtype) * s.astype(mult_dtype)).sum(axis=1)
        res_red_unmasked = reduction.reduce(res_mult_unmasked)

        res_mult_masked = ((c.astype(mult_dtype) * s_.astype(mult_dtype))).sum(axis=2)
        res_red_masked = reduction.reduce(res_mult_masked)

        assert (((res_red_masked.sum(axis=0)) % q) == (res_red_unmasked % q)).all()


        samples_i = model._compute(res_red_masked,1)*alpha + beta + _cp.random.normal(0, sigma, res_red_masked.shape)

        c = c.flatten()
        s_flat = s.flatten()
        if get_masks:
            s_ = masks.reshape((s_.shape[0],s_.shape[1]*s_.shape[2]))

        if filename is None:
            cs.append(c)
            if get_masks:
                masks.append(s_)
            ss.append(s_flat)
            samples.append(samples_i.ravel())
        else:
            es_writer.write_samples(samples_i.ravel().get())
            es_writer.write_metadata('c', c.get())
            es_writer.write_metadata('s', s_flat.get())
        
    if filename is not None:
        return
    else:
        metas = {}
        if incomplete_ == 1:
            metas['c'] = _cp.array(cs)[:,:]
            metas['s'] = _cp.array(ss)[:,:]
            if get_masks:
                metas['masks'] =_cp.array(masks)[:,:,:]
        else:
            metas['c'] = _cp.array(cs)[:,:]
            metas['s'] = _cp.array(ss)[:,:]
            if get_masks:
                metas['masks'] =_cp.array(masks)[:,:,:]
        

    return gpu_format.read_ths_from_ram(_cp.array(samples)[:,:], **metas)
import cupy as _cu

RR_Q2Q2 = 0
RR_0Q = 2


class Reduction():

    def __init__(self, q, o_dtype=None):
        if o_dtype is not None:
            try:
                o_dtype = _cu.dtype(o_dtype)
            except TypeError:
                raise ValueError(f'{o_dtype} is not a valid dtype.')
        
        self.q = q
        self.o_dtype = o_dtype

    def _reduce(self, data):
        pass

    def reduce(self, data):
        if self.o_dtype is None:
            return self._reduce(data)
        else:
            return self._reduce(data).astype(self.o_dtype)

    def _id(self):
        pass

    @property
    def id(self):
        return self._id()

    def __str__(self):
        pass

class Reduction_Q2Q2(Reduction):
    
    def _reduce(self, data):
        data = data % self.q
        return data - (data > self.q//2)*self.q

    def _id(self):
        return RR_Q2Q2

    def __str__(self):
        return 'Basic [-q/2,q/2]'


class Reduction_0Q(Reduction):
    
    def _reduce(self, data):
        return data % self.q

    def _id(self):
        return RR_0Q

    def __str__(self):
        return 'Basic [0,q)'


class PlantardReduction(Reduction):

    def __init__(self, q, qa, qinv, zetas=None, o_dtype='int16'):
        if o_dtype != 'int16':
            raise ValueError('Plantard reduction only supports int16 output.') 
        self.qa = _cu.int32(qa)
        self.qinv = _cu.uint32(qinv)
        self.zetas = zetas
        super().__init__(q, o_dtype)

    def _reduce_core(self, data, qinv):
        a1 = data.astype('int32')
        t0 = (((qinv.astype('int64') * a1.astype('int64')).astype('uint64') >> _cu.uint64(16))).astype('uint16').astype('int16').astype('int32')
        t1 = ((self.qa + t0 * _cu.int32(self.q)).astype('uint32') >> 16)#.astype('int16')
        return t1

    def _reduce(self, data):
        return self.reduce_core(data, self.qinv)

    def reduce_zetas(self, data, frame):
        return self.reduce(self.reduce_core(data, self.zetas[frame]))

    def _id(self):
        return RR_0Q

    def __str__(self):
        return 'Plantard [-q/2,q/2]'


class MontgomeryReduction(Reduction):

    def __init__(self, q, qinv, o_dtype='int16'):
        if o_dtype != 'int16' :
            raise ValueError('Montgomery reduction only supports int16 output.') 
        self.qinv = qinv
        super().__init__(q, o_dtype)

    def _reduce(self, data):
        t = data.astype(_cu.int32)
        t1 = t.astype(_cu.int16).astype(_cu.int32) * _cu.int32(self.qinv)
        t1 = t1.astype(_cu.int16).astype(_cu.int32) * _cu.int32(self.q)
        t += t1
        return (t >> 16)

    def _id(self):
        return RR_0Q

    def __str__(self):
        return 'Montgomery (-q,q)'
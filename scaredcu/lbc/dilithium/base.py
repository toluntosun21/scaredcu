from .. import modop, base
from xoflib import shake256
import cupy as _cp


q    = 8380417
n    = 256
root = 1753

tau = {2: 39, 3:49, 5:60}

####################################### DILITHIUM SIMPLE NTT #######################################

class NTT(base.NTT):

    def __init__(self):
        super().__init__(q=q, n=n, root=root, dtype='int32')


####################################### BASE MULTIPLICATION ##################################################


class BaseMul(base.BaseMul):

    def __init__(self, central=True):
        if central:
            super().__init__(reduction=modop.Reduction_Q2Q2(q, 'int32'), reduce=True)
        else:
            super().__init__(reduction=modop.Reduction_0Q(q, 'int32'), reduce=True)


####################################### MONTGOMERY #####################################################

class BaseMulMonty(base.BaseMul):

    def __init__(self):
        super().__init__(reduction=modop.MontgomeryReduction(q, -58728449, 'int32'))



####################################### CHALLENGE SAMPLING #####################################################


# from https://github.com/GiacomoPope/dilithium-py/blob/main/src/dilithium_py/polynomials/polynomials.py
def _sample_in_ball(seed, tau):
    """
    Figure 2 (Sample in Ball)
        https://pq-crystals.org/dilithium/data/dilithium-specification-round3-20210208.pdf

    Create a random 256-element array with τ ±1’s and (256 − τ) 0′s using
    the input seed ρ (and an SHAKE256) to generate the randomness needed
    """

    def rejection_sample(i, xof):
        """
        Sample random bytes from `xof_bytes` and
        interpret them as integers in {0, ..., 255}

        Rejects values until a value j <= i is found
        """
        while True:
            j = xof.read(1)[0]
            if j <= i:
                return j

    # Initialise the XOF
    xof = shake256(seed)

    # Set the first 8 bytes for the sign, and leave the rest for
    # sampling.
    sign_bytes = xof.read(8)
    sign_int = int.from_bytes(sign_bytes, "little")

    # Set the list of coeffs to be 0
    coeffs = [0 for _ in range(256)]

    # Now set tau values of coeffs to be ±1
    for i in range(256 - tau, 256):
        j = rejection_sample(i, xof)
        coeffs[i] = coeffs[j]
        coeffs[j] = 1 - 2 * (sign_int & 1)
        sign_int >>= 1

    return _cp.array(coeffs, dtype=_cp.int32)


def dilithium2_sample_in_ball(seed):
    return _sample_in_ball(seed, tau[2])

def dilithium3_sample_in_ball(seed):
    return _sample_in_ball(seed, tau[3])

def dilithium5_sample_in_ball(seed):
    return _sample_in_ball(seed, tau[5])

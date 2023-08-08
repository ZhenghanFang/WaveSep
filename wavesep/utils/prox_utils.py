# utils for proximal operator for wavelet l1 penalty
import numpy as np
import pywt
import copy

def prox(x, Lambda, alpha, wavelet):
    """
    Inputs:
        x: (w,h,d,c), [xp, xn]
        Lambda: float or (c)
        alpha: float
        wavelet: wavelet type, str or pywt.Wavelet. Must be orthogonal.
    Outputs:
        1. output of proximal operator, (w,h,d,c)
        2. value of L1 penalty \sum_i Lambda[i] ||W^{-1} x[i]||_1
    """
    if type(Lambda) == float:
        Lambda = np.ones(x.shape[-1]) * Lambda
    x_new = np.zeros(x.shape, dtype='float32')
    L1 = 0
    for i in range(x.shape[-1]):
        # get single volume
        xx = x[:,:,:,i]
        # compute wavelet coefficients
        coeffs = pywt.wavedecn(xx, wavelet, mode='periodization')
        # compute value of L1 penalty
        L1 = L1 + np.sum(np.abs(flatten_coeffs(coeffs))) * Lambda[i]
        # soft thresholding
        coeffs_soft = coeffs_soft_thresholding(coeffs, alpha*Lambda[i])
        # reconstruct
        xx_new = pywt.waverecn(coeffs_soft, wavelet, mode='periodization')
        x_new[:,:,:,i] = xx_new

    return x_new, L1

def flatten_coeffs(coeffs):
    """
    Flatten the wavelet coefficients computed by pywt
    Inputs:
        coeffs: wavelet coefficients from pywt.wavedecn
    Output:
        flattened coeffs, numpy array
    """
    o = []
    for i in range(len(coeffs)):
        if type(coeffs[i]) == np.ndarray:
            o.append(coeffs[i])
        elif type(coeffs[i]) == dict:
            for k, v in coeffs[i].items():
                o.append(v)
        else:
            raise
    o = np.concatenate(o, axis=None)
    return o

def coeffs_soft_thresholding(coeffs, th):
    """
    Soft thresholding for wavelet coefficients computed by pywt
    Inputs:
        coeffs: wavelet coefficients. output from pywt.wavedecn
        th: threshold
    Output:
        new coeffs
    """
    def Sfunc(z, th):
        return (z > th) * (z - th) + (z < -th) * (z + th)

    coeffs = copy.deepcopy(coeffs)
    for i in range(len(coeffs)):
        if type(coeffs[i]) == np.ndarray:
            coeffs[i] = Sfunc(coeffs[i], th)
        elif type(coeffs[i]) == dict:
            for k, v in coeffs[i].items():
                coeffs[i][k] = Sfunc(v, th)
        else:
            raise
    return coeffs

import os
import numpy as np
import pywt
import nibabel as nib
from tqdm import tqdm
import copy
import json
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

from .prox_utils import flatten_coeffs, coeffs_soft_thresholding
from .StiEvaluationToolkit import transform_matrix

oj = os.path.join


# Define gradient computations
def proj_tensor(t, h):
    """
    Projec tensor image to field direction, ie, compute h^T x h.
    Inputs:
        t: (w,h,d,6), tensor image
        h: (3), magnetic field direction
    Return:
        (w,h,d)
    """
    mtx = transform_matrix(t)
    proj = np.matmul(
        np.matmul(h[None, None, None, None, :], mtx), h[None, None, None, :, None]
    )
    proj = np.squeeze(proj)  # (w,h,d)
    return proj


def get_grad_R2p(x, h, R2p, Dr):
    """
    Compute gradient of the R2p fidelity term: 1/2 ||xp - h^T xn h - R2p/Dr||_2^2
    h: (3)
    Return: (w,h,d,7), scalar
    """
    xp = x[:, :, :, 0]
    xn = x[:, :, :, 1:]
    b = R2p / Dr
    proj_xn = proj_tensor(xn, h)

    # gradient w.r.t. xp
    grad_xp = xp - proj_xn - b

    # gradient w.r.t. xn
    r = proj_xn + b - xp
    grad_xn = np.stack(
        (
            r * h[0] * h[0],
            r * h[0] * h[1],
            r * h[0] * h[2],
            r * h[1] * h[1],
            r * h[1] * h[2],
            r * h[2] * h[2],
        ),
        axis=-1,
    )

    # gradient for all
    grad = np.concatenate((grad_xp[:, :, :, None], grad_xn), axis=-1)

    # function value
    F = np.linalg.norm(r) ** 2 / 2

    return grad, F


def xp2tensor(xp):
    """
    Compute xp * I
    Return: (w,h,d,6)
    """
    return np.stack((xp, xp * 0, xp * 0, xp, xp * 0, xp), axis=-1)


def get_grad_sti(x, sti):
    """
    Compute gradient of the sti fidelity term: 1/2 ||xp * I + xn - sti||_2^2
    Return: (w,h,d,7), scalar
    """
    xp = x[:, :, :, 0]
    xn = x[:, :, :, 1:]
    b = sti

    # gradient w.r.t. xn
    grad_xn = xp2tensor(xp) + xn - b

    # gradient w.r.t. xp
    r = xp2tensor(xp) + xn - b
    grad_xp = r[:, :, :, 0] + r[:, :, :, 3] + r[:, :, :, 5]

    # gradient for all
    grad = np.concatenate((grad_xp[:, :, :, None], grad_xn), axis=-1)

    # function value
    F = np.linalg.norm(r) ** 2 / 2

    return grad, F


def get_grad_fidelity(x, h, R2p, sti, Dr, beta=1, gamma=1):
    """
    Compute gradient of all the fidelity term:
        beta * 1/nori \sum_{i \in [nori]} 1/2 ||xp - h^T xn h - R2p/Dr||_2^2
        + gamma * 1/2 ||xp * I + xn - sti||_2^2.
    Inputs:
        h: (3, nori)
        R2p: (w,h,d, nori)
        beta: weight on R2p
        gamma: weight on sti
    Return:
        (w,h,d,7), scalar
    """
    img_sz = x.shape[:3]
    nori = h.shape[1]
    grad = np.zeros(x.shape, dtype="float32")
    F = 0

    # R2p terms
    for i in range(nori):
        g, f = get_grad_R2p(x, h[:, i], R2p[:, :, :, i], Dr)
        grad = grad + g / nori * beta
        F = F + f / nori * beta

    # sti term
    g, f = get_grad_sti(x, sti)
    grad = grad + g * gamma
    F = F + f * gamma

    return grad, F


class SolverOperators(object):
    def __init__(self, wavelet, shape, level):
        self.wavelet = wavelet
        self.shape = shape
        self.level = level  # wavelet level

    def prox_step(self, x, Lambda, alpha):
        """
        x: (w,h,d,7), [xp, xn]; or (w,h,d,1), xp
        Lambda: float or (7)
        alpha: float
        """
        assert x.ndim == 4
        if type(Lambda) == float:
            Lambda = np.ones(x.shape[-1]) * Lambda
        x_new = np.zeros(x.shape, dtype="float32")
        L1 = 0
        for i in range(x.shape[-1]):
            # get single volume
            xx = x[:, :, :, i]
            # compute wavelet coefficients
            coeffs = pywt.wavedecn(
                xx, self.wavelet, mode="periodization", level=self.level
            )
            # compute value of L1 penalty
            L1 = L1 + np.sum(np.abs(flatten_coeffs(coeffs))) * Lambda[i]
            # soft thresholding
            coeffs_soft = coeffs_soft_thresholding(coeffs, alpha * Lambda[i])
            # reconstruct
            xx_new = pywt.waverecn(coeffs_soft, self.wavelet, mode="periodization")
            x_new[:, :, :, i] = xx_new

        return x_new, L1

    def projection(self, x):
        """
        x: (w,h,d,7), [xp, xn]
        Project xp to positive, and xn to negative semi-definite
        """
        xp = x[:, :, :, 0]
        xn = x[:, :, :, 1:]

        xp = np.maximum(xp, 0)

        matrix_data = transform_matrix(xn)
        L, V = np.linalg.eigh(matrix_data)
        L = np.minimum(L, 0)
        xn_mtx = eig2mtx(L, V)
        xn = np.stack(
            (
                xn_mtx[:, :, :, 0, 0],
                xn_mtx[:, :, :, 0, 1],
                xn_mtx[:, :, :, 0, 2],
                xn_mtx[:, :, :, 1, 1],
                xn_mtx[:, :, :, 1, 2],
                xn_mtx[:, :, :, 2, 2],
            ),
            axis=-1,
        )

        return np.concatenate((xp[:, :, :, None], xn), axis=-1)


def eig2mtx(L, V):
    """
    Construct matrix from its eigendecomposition.
    Assumes matrix is real symmetric.
    Inputs:
        L: eigenvalues, (*, n)
        V: eigenvectors, (*, n, n), last dim is evect index (same order as L)
    Return:
        (*, n, n)
    """
    mtx = np.zeros(V.shape, dtype="float32")
    for i in range(L.shape[-1]):
        LL = L[..., i]  # (*)
        VV = V[..., i]  # (*, n)
        mtx = mtx + LL[..., None, None] * np.matmul(VV[..., None], VV[..., None, :])
    return mtx


class Solver(object):
    def __init__(self):
        pass

    def solve(
        self,
        sti,
        R2p,
        mask,
        h,
        Dr_pos,
        alpha,
        Lambda,
        beta=1,
        gamma=1,
        wavelet="db4",
        level=None,
        maxit=100,
        evaluator=None,
    ):
        """
        sti: (w,h,d,6)
        R2p: (w,h,d,nori), in Hz
        mask: (w,h,d)
        h: (3,nori)
        Dr_pos: float
        alpha: step size
        Lambda: weight on L1. float (same weight for xp and xn) or tuple of float (weight on xp, weight on xn).
        beta: weight on R2p fidelity term
        gamma: weight on sti fidelity term
        wavelet: wavelet type, str or pywt.Wavelet
        level: wavelet level
        evaluator: instance of Evaluator class. Calculates accuracy of reconstruction wrt. reference.

        Return: (w,h,d,7) [xp, xn]
        """
        self.init()

        img_sz = sti.shape[:3]

        x = np.zeros(img_sz + (7,), dtype="float32")
        Sop = SolverOperators(wavelet=wavelet, shape=img_sz, level=level)

        for it in tqdm(range(maxit)):
            xold = x

            # gradient step
            g, F = get_grad_fidelity(x, h, R2p, sti, Dr_pos, beta=beta, gamma=gamma)
            x = x - alpha * g

            # proximal step
            if type(Lambda) == float:
                if Lambda > 0:
                    x, L1 = Sop.prox_step(x, Lambda, alpha)
                else:
                    L1 = 0
            else:
                assert len(Lambda) == 2
                Lambda_array = np.array(
                    [
                        Lambda[0],
                        Lambda[1],
                        Lambda[1],
                        Lambda[1],
                        Lambda[1],
                        Lambda[1],
                        Lambda[1],
                    ]
                )
                x, L1 = Sop.prox_step(x, Lambda_array, alpha)

            # projection step
            x = x * mask[:, :, :, None]  # outside brain mask = 0
            x = Sop.projection(x)

            # record progress
            self.record(x, xold, F, L1, evaluator)
            if self.metrics:
                print(self.metrics[-1])
            print(self.relative_change[-1])
            if self.relative_change[-1] < 5e-3:
                break

        print("done")

        return x

    def init(self):
        """
        Initialize
        """
        self.x = None
        self.change = []
        self.relative_change = []
        self.metrics = []
        self.F = []
        self.L1 = []
        self.obj = []

    def record(self, x, xold, F, L1, evaluator):
        """
        Record progress
        """
        self.x = x

        # relative change in x
        change = np.linalg.norm(x - xold)
        relative_change = change / np.linalg.norm(x)
        self.change.append(change)
        self.relative_change.append(relative_change)

        # evaluate
        if evaluator is not None:
            self.metrics.append(evaluator.evaluate(x))

        # data fidelity term
        self.F.append(F)

        # L1 term
        self.L1.append(L1)

        # total objective value
        self.obj.append(F + L1)


# helper functions
def plot(solver):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 5), tight_layout=True)

    axes[0, 0].plot(solver.change)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("change")

    axes[0, 1].plot(solver.relative_change)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("relative change")

    if len(solver.metrics) != 0:
        keys = solver.metrics[0].keys()
        for k in keys:
            axes[0, 2].plot([m[k] for m in solver.metrics])
        axes[0, 2].legend(keys)
        #         axes[0,2].set_yticks(np.arange(np.floor(min(solver.psnr)),np.ceil(max(solver.psnr))+0.5,0.5))
        axes[0, 2].grid("on")
        axes[0, 2].set_title("metrics")

    axes[1, 0].plot(solver.F)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Data fidelity")

    axes[1, 1].plot(solver.L1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("L1 penalty")

    axes[1, 2].plot(solver.obj)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Total objective")

    return fig


def save_fig(solver, fn):
    fig = plot(solver)
    fig.savefig(fn)
    plt.close(fig)


def save_nii(x, affine, folder):
    x = x.astype("float32")
    os.makedirs(folder, exist_ok=True)
    nib.Nifti1Image(x[:, :, :, 0], affine).to_filename(folder + "/xp.nii.gz")
    nib.Nifti1Image(-x[:, :, :, 1:], affine).to_filename(folder + "/xn.nii.gz")

import numpy as np
import copy
import pywt
import os
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

from .evaluators import QsmSepEvaluator

oj = os.path.join


class ChiSepOperators(object):
    def __init__(self):
        self.beta = 1.0

    def A_forward(self, x):
        """
        x: (# voxels * 2), [flattened xp, flattened xn]
        """
        xp = x[: x.shape[0] // 2]
        xn = x[x.shape[0] // 2 :]
        b1 = xp + xn
        b2 = self.beta * (xp - xn)
        return np.concatenate((b1, b2))

    def A_transpose(self, b):
        """
        b: (# voxels * 2), [observed qsm, observed R2p (scaled by Dr)]
        """
        b1 = b[: b.shape[0] // 2]
        b2 = b[b.shape[0] // 2 :]
        xp = b1 + self.beta * b2
        xn = b1 - self.beta * b2
        return np.concatenate((xp, xn))


class SolverOperators(object):
    def __init__(self, wavelet, shape, level):
        self.wavelet = wavelet
        self.shape = shape
        self.level = level  # wavelet level

    def soft_thresholding(self, coeffs, th):
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

    def wavedec(self, x):
        """
        x: (2 * # voxels), [xp, xn]
        """
        xp = x[: x.shape[0] // 2]
        xn = x[x.shape[0] // 2 :]
        xp3d = xp.reshape(self.shape)
        xn3d = xn.reshape(self.shape)
        coeffsp = pywt.wavedecn(
            xp3d, self.wavelet, mode="periodization", level=self.level
        )
        coeffsn = pywt.wavedecn(
            xn3d, self.wavelet, mode="periodization", level=self.level
        )
        return coeffsp, coeffsn

    def flatten_coeffs(self, coeffs):
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

    def wavedec_flat(self, x):
        """
        x: (2 * # voxels), [xp, xn]
        """
        coeffsp, coeffsn = self.wavedec(x)
        coeffsp = self.flatten_coeffs(coeffsp)
        coeffsn = self.flatten_coeffs(coeffsn)
        return np.concatenate((coeffsp, coeffsn))

    def prox_step(self, x, th):
        """
        x: (2 * # voxels), [xp, xn]
        th: float
        """
        xp = x[: x.shape[0] // 2]
        xn = x[x.shape[0] // 2 :]
        xp3d = xp.reshape(self.shape)
        xn3d = xn.reshape(self.shape)
        coeffsp = pywt.wavedecn(
            xp3d, self.wavelet, mode="periodization", level=self.level
        )
        coeffsn = pywt.wavedecn(
            xn3d, self.wavelet, mode="periodization", level=self.level
        )

        # soft thresholding
        coeffsp_soft = self.soft_thresholding(coeffsp, th)
        coeffsn_soft = self.soft_thresholding(coeffsn, th)

        # reconstruct
        xp3d_new = pywt.waverecn(coeffsp_soft, self.wavelet, mode="periodization")
        xn3d_new = pywt.waverecn(coeffsn_soft, self.wavelet, mode="periodization")

        # flatten
        x_new = np.concatenate((xp3d_new.reshape(-1), xn3d_new.reshape(-1)))
        return x_new

    def projection(self, x):
        """
        x: (2 * # voxels), [xp, xn]
        """
        xp = x[: x.shape[0] // 2]
        xn = x[x.shape[0] // 2 :]

        xp[xp < 0] = 0
        xn[xn > 0] = 0

        return np.concatenate((xp, xn))


def reshape(x, shape):
    """
    x: (2 * #_voxels), [xp, xn]
    """
    xp = x[: x.shape[0] // 2]
    xn = x[x.shape[0] // 2 :]
    xp3d = xp.reshape(shape)
    xn3d = xn.reshape(shape)

    return xp3d, xn3d


def get_grad_qsm(x, qsm, img_sz):
    xp, xn = reshape(x, img_sz)
    grad = xp + xn - qsm
    grad = grad.flatten()
    grad = np.concatenate((grad, grad))
    return grad


def get_grad_R2p(x, R2p, Dr_pos, img_sz):
    xp, xn = reshape(x, img_sz)
    r = xp - xn - R2p / Dr_pos
    r = r.flatten()
    grad = np.concatenate((r, -r))
    return grad


class Solver(object):
    def __init__(self, qsm, R2p, Dr_pos, mask, gt=None):
        self.qsm = qsm
        self.R2p = R2p
        self.Dr_pos = Dr_pos
        self.mask = mask
        self.evaluator = QsmSepEvaluator(gt) if gt is not None else None

    def solve(self, alpha, Lambda, wavelet, level, maxit=100):
        """
        alpha: step size
        Lambda: weight on L1
        wavelet: wavelet type, str or pywt.Wavelet
        level: wavelet level

        Return: 3D images, (xp, xn)
        """
        self.init()
        qsm, R2p, Dr_pos, mask = self.qsm, self.R2p, self.Dr_pos, self.mask
        #         b = np.stack([qsm, R2p / Dr_pos])
        #         b = b.reshape(-1)

        img_sz = qsm.shape

        x = np.zeros(2 * np.prod(img_sz))
        Sop = SolverOperators(wavelet=wavelet, shape=img_sz, level=level)
        Aop = ChiSepOperators()

        for it in tqdm(range(maxit)):
            xold = x

            # gradient step
            # g = Aop.A_transpose(Aop.A_forward(x)) - Aop.A_transpose(b)
            g = get_grad_qsm(x, qsm, img_sz)
            nori = R2p.shape[3]
            for kori in range(nori):
                g = g + get_grad_R2p(x, R2p[:, :, :, kori], Dr_pos, img_sz) / nori
            x = x - alpha * g

            # proximal step
            x = Sop.prox_step(x, alpha * Lambda)
            z = Sop.wavedec_flat(x)
            #             print(np.sum(z!=0) / z.shape[0])

            # projection step
            x = Sop.projection(x)
            x = x * np.concatenate((mask.flatten(), mask.flatten()))
            z = Sop.wavedec_flat(x)
            #             print(np.sum(z!=0) / z.shape[0])

            # record progress
            xp, xn = reshape(x, img_sz)
            self.record(
                x,
                xold,
                F=None,
                L1=None,
                evaluator=self.evaluator,
                x3d=np.stack((xp, xn), axis=-1),
            )
            print(self.metrics[-1])
            print(self.relative_change[-1])
            if self.relative_change[-1] < 1e-3:
                break

        xp, xn = reshape(x, img_sz)
        print("done")

        return xp, xn

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

    def record(self, x, xold, F, L1, evaluator, x3d):
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
            self.metrics.append(evaluator.evaluate(x3d))

        # data fidelity term
        if F is not None:
            self.F.append(F)

        # L1 term
        if L1 is not None:
            self.L1.append(L1)

        # total objective value
        if F is not None and L1 is not None:
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


def save_nii(xp, xn, affine, folder):
    os.makedirs(folder, exist_ok=True)
    nib.Nifti1Image(xp, affine).to_filename(folder + "/xp.nii.gz")
    nib.Nifti1Image(-xn, affine).to_filename(folder + "/xn.nii.gz")

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

from .metrics import psnr_qsmsep
from .StiEvaluationToolkit import transform_matrix


# evaluate
class StiSepEvaluator(object):
    def __init__(self, gt_xp, gt_xn_mms):
        """
        Evaluator for sti separation
        gt_xp: (w,h,d)
        gt_xn_mms: (w,h,d)
        """
        self.gt_xp = gt_xp
        self.gt_xn_mms = gt_xn_mms

    def evaluate(self, x):
        """
        Inputs:
            x: (w,h,d,7)
        Outputs:
            dict
        """
        xp = x[:, :, :, 0]
        xn = x[:, :, :, 1:]

        matrix_data = transform_matrix(xn)
        L, V = np.linalg.eigh(matrix_data)
        xn_mms = np.mean(L, axis=-1)

        psnr_xp = skimage_psnr(
            self.gt_xp, xp, data_range=self.gt_xp.max() - self.gt_xp.min()
        )
        psnr_xn_mms = skimage_psnr(
            self.gt_xn_mms,
            xn_mms,
            data_range=self.gt_xn_mms.max() - self.gt_xn_mms.min(),
        )

        return {"psnr_xp": psnr_xp, "psnr_xn_mms": psnr_xn_mms}


class QsmSepEvaluator(object):
    def __init__(self, gt):
        """
        Evaluator for qsm separation
        gt: (w,h,d,2), [xp,xn]
        """
        self.gt = gt

    def evaluate(self, x):
        """
        Inputs:
            x: (w,h,d,2), [xp,xn]
        Outputs:
            dict
        """
        psnr = psnr_qsmsep(self.gt, x)
        return {"psnr_xp": psnr["xp"], "psnr_xn": psnr["xn"]}

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

def psnr_qsmsep(gt, x):
    """
    gt: (w,h,d,2), [xp,xn]
    x: (w,h,d,2), [xp,xn]
    """
    gt_xp = gt[:,:,:,0]
    gt_xn = gt[:,:,:,1]
    xp = x[:,:,:,0]
    xn = x[:,:,:,1]
    psnr_combined = skimage_psnr(gt, x, data_range=gt.max()-gt.min())
    psnr_xp = skimage_psnr(gt_xp, xp, data_range=gt_xp.max()-gt_xp.min())
    psnr_xn = skimage_psnr(gt_xn, xn, data_range=gt_xn.max()-gt_xn.min())
    
    return {'combined': psnr_combined, 'xp': psnr_xp, 'xn': psnr_xn}

def ssim_qsmsep(gt, x):
    """
    gt: (w,h,d,2), [xp,xn]
    x: (w,h,d,2), [xp,xn]
    """
    gt_xp = gt[:,:,:,0]
    gt_xn = gt[:,:,:,1]
    xp = x[:,:,:,0]
    xn = x[:,:,:,1]
    ssim = skimage_ssim(gt, x, data_range=gt.max()-gt.min(), channel_axis=-1)
    ssim_xp = skimage_ssim(gt_xp, xp, data_range=gt_xp.max()-gt_xp.min(), channel_axis=-1)
    ssim_xn = skimage_ssim(gt_xn, xn, data_range=gt_xn.max()-gt_xn.min(), channel_axis=-1)
    
    return {'combined': ssim, 'xp': ssim_xp, 'xn': ssim_xn}

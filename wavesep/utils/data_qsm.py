import json
import nibabel as nib
import numpy as np


# data loading function
def load_data(data_dict):
    R2p = [nib.load(fn).get_fdata() for fn in data_dict["R2p_fn_list"]]
    R2p = np.stack(R2p, axis=-1)

    qsm = nib.load(data_dict["qsm_fn"]).get_fdata().squeeze()
    mask = nib.load(data_dict["mask_fn"]).get_fdata()
    affine = nib.load(data_dict["qsm_fn"]).affine

    R2p = R2p * mask[:, :, :, None]
    qsm = qsm * mask

    with open(data_dict["params_fn"], "r") as f:
        params = json.load(f)

    out = {
        "R2p": R2p,
        "qsm": qsm,
        "mask": mask,
        "affine": affine,
        "params": params,
    }

    if "gt_xp_fn" in data_dict and "gt_xn_fn" in data_dict:
        gt_xp = nib.load(data_dict["gt_xp_fn"]).get_fdata()
        gt_xn = -nib.load(data_dict["gt_xn_fn"]).get_fdata()
        out["gt"] = np.stack((gt_xp, gt_xn), axis=-1)
    else:
        out["gt"] = None

    return out

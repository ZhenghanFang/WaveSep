import os
import json
import nibabel as nib
import numpy as np

from .StiEvaluationToolkit import StiEvaluationToolkit as stet


oj = os.path.join


def load_data(data_dict):
    R2p = [nib.load(fn).get_fdata() for fn in data_dict["R2p_fn_list"]]
    R2p = np.stack(R2p, axis=-1)
    sti = nib.load(data_dict["sti_fn"]).get_fdata()
    mask = nib.load(data_dict["mask_fn"]).get_fdata()
    affine = nib.load(data_dict["sti_fn"]).affine
    h = []
    for fn in data_dict["H0_fn_list"]:
        with open(fn, "r") as f:
            h.append([float(_) for _ in f.readlines()])
    h = np.array(h).T
    H0 = h

    R2p = R2p * mask[:, :, :, None]
    sti = sti * mask[:, :, :, None]

    with open(data_dict["params_fn"], "r") as f:
        params = json.load(f)

    if "gt_xp_fn" in data_dict:
        gt_xp = nib.load(data_dict["gt_xp_fn"]).get_fdata()
    else:
        gt_xp = None
    if "gt_xn_mms_fn" in data_dict:
        gt_xn_mms = -nib.load(data_dict["gt_xn_mms_fn"]).get_fdata()
    elif "gt_xn_sti_fn" in data_dict:
        gt_xn = -nib.load(data_dict["gt_xn_sti_fn"]).get_fdata()
        L, V, avg, ani, V1, modpev = stet.tensor2misc(gt_xn)
        gt_xn_mms = avg
    else:
        gt_xn_mms = None

    return {
        "R2p": R2p,
        "sti": sti,
        "mask": mask,
        "H0": H0,
        "affine": affine,
        "params": params,
        "gt_xp": gt_xp,
        "gt_xn_mms": gt_xn_mms,
    }

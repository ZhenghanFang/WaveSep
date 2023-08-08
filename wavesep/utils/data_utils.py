# utils for data loading
import os
oj = os.path.join

def get_filenames(Sub, Oris, data_dir):
    """
    Get file names in susceptibility source separation dataset.
    Inputs:
        Sub: str, e.g. 'SUB001'
        Oris: list, e.g. [1,2,...]
        data_dir: str
    Output:
        dict
    """
    
    data_dict = {}
    data_dict['Oris'] = Oris
    data_dict['R2p_fn_list'] = [
        oj(data_dir, Sub, 'Generated', 'Ori{:03d}'.format(i), '{}_Ori{:03d}_R2p.nii.gz'.format(Sub, i)) for i in Oris
    ]
    data_dict['phase_fn_list'] = [
        oj(data_dir, Sub, 'Generated', 'Ori{:03d}'.format(i), '{}_Ori{:03d}_phase.nii.gz'.format(Sub, i)) for i in Oris
    ]
    data_dict['params_fn'] = oj(data_dir, Sub, 'Generated', 'sti', Sub+'_params.json')
    data_dict['mask_fn'] = oj(data_dir, Sub, 'Generated', 'sti', Sub+'_mask.nii.gz')
    data_dict['H0_fn_list'] = [
        oj(data_dir, Sub, 'Generated', 'Ori{:03d}'.format(i), '{}_Ori{:03d}_ang.txt'.format(Sub, i)) for i in Oris
    ]
    data_dict['gt_xp_fn'] = oj(data_dir, Sub, 'Generated', 'sti', Sub+'_xp_cosmos_nnls.nii.gz')
    data_dict['gt_xn_fn'] = oj(data_dir, Sub, 'Generated', 'sti', Sub+'_xn_cosmos_nnls.nii.gz')
    
    return data_dict

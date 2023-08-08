from .data_sti import load_data
from .solver_wavesep_sti import Solver, save_nii, save_fig
from .evaluators import StiSepEvaluator


def run_wavesep_sti(data_info, alg_config):
    folder = data_info["output_folder"]

    data = load_data(data_info)
    R2p, sti, mask, H0, affine, params, gt_xp, gt_xn_mms = (
        data["R2p"],
        data["sti"],
        data["mask"],
        data["H0"],
        data["affine"],
        data["params"],
        data["gt_xp"],
        data["gt_xn_mms"],
    )
    Dr_pos, Dr_neg = params["Dr_pos"], params["Dr_neg"]
    assert Dr_pos == Dr_neg  # only support Dr_pos == Dr_neg

    if gt_xp is not None and gt_xn_mms is not None:
        evaluator = StiSepEvaluator(gt_xp, gt_xn_mms)
    else:
        evaluator = None

    solver = Solver()
    alpha, wavelet, Lambda = (
        alg_config["alpha"],
        alg_config["wavelet"],
        alg_config["Lambda"],
    )

    x = solver.solve(
        sti,
        R2p,
        mask,
        H0,
        Dr_pos,
        alpha,
        Lambda,
        wavelet=wavelet,
        maxit=100,
        evaluator=evaluator,
    )
    save_nii(x, affine, folder)
    save_fig(solver, f"{folder}/plot.png")
    if evaluator is not None:
        with open(f"{folder}/metrics.txt", "w") as f:
            print(solver.metrics[-1], file=f)

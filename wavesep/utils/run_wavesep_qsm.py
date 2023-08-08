from .data_qsm import load_data
from .solver_wavesep_qsm import Solver, save_nii, save_fig


def run_wavesep_qsm(data_info, alg_config):
    folder = data_info["output_folder"]
    data = load_data(data_info)
    R2p, qsm, mask, affine, params, gt = (
        data["R2p"],
        data["qsm"],
        data["mask"],
        data["affine"],
        data["params"],
        data["gt"],
    )
    Dr_pos, Dr_neg = params["Dr_pos"], params["Dr_neg"]
    assert Dr_pos == Dr_neg  # only support Dr_pos == Dr_neg

    solver = Solver(qsm, R2p, Dr_pos, mask, gt)

    alpha, wavelet, level, Lambda = (
        alg_config["alpha"],
        alg_config["wavelet"],
        alg_config["level"],
        alg_config["Lambda"],
    )
    xp, xn = solver.solve(alpha, Lambda, wavelet, level, maxit=100)

    xp, xn = xp.astype("float32") * mask, xn.astype("float32") * mask
    save_nii(xp, xn, affine, folder)
    save_fig(solver, f"{folder}/plot.png")
    if gt is not None:
        with open(f"{folder}/metrics.txt", "w") as f:
            print(solver.metrics[-1], file=f)

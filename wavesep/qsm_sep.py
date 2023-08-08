import argparse
import yaml

from utils.run_wavesep_qsm import run_wavesep_qsm


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/yml/example_qsm.yml")
args = parser.parse_args()
with open(args.data, "r") as f:
    data_list = yaml.safe_load(f)


alg_config = {}
alg_config["alpha"] = 0.2
alg_config["wavelet"] = "db4"
alg_config["level"] = None
alg_config["Lambda"] = 0.02


for data_info in data_list:
    run_wavesep_qsm(data_info, alg_config)

import argparse
import yaml

from utils.run_wavesep_sti import run_wavesep_sti


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/yml/example_sti.yml")
args = parser.parse_args()
with open(args.data, "r") as f:
    data_list = yaml.safe_load(f)

alg_config = {}
alg_config["wavelet"] = "db4"
alg_config["alpha"] = 0.2
alg_config["Lambda"] = (1e-2, 1e-3)  # xp, xn

for data_info in data_list:
    run_wavesep_sti(data_info, alg_config)

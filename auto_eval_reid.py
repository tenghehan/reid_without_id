#!/usr/bin/env python
from typing import List
from attr import dataclass
import attr
import argparse
from marshmallow_attrs import class_schema
import yaml
from pprint import pprint
import subprocess
import os.path as osp

@dataclass
class Dataset:
    name: str
    fps: int
    sampling_rate: float = 1.0


@dataclass
class Config:
    model_config: str
    datasets: List[Dataset]


def read_config(path: str) -> Config:
    with open(path) as f:
        j = yaml.safe_load(f)
    schema = class_schema(Config)()
    return schema.load(j)


g_dry_run = False


def run(cmd: List[str]):
    print("+ %s" % cmd)
    if not g_dry_run:
        subprocess.check_call(cmd)

def evaluate_dukemtmc(index: int, dataset: Dataset):
    # python train_net.py --eval-only 
    # --config-file fastreid_configs/MOT/bagtricks_R50.yml 
    # DATASETS.TESTS ("DukeMTMC",) 
    # MODEL.WEIGHTS logs/mot/bagtricks_R50/MOT16-06/model_final.pth
    # OUTPUT_DIR "logs/mot/bagtricks_R50/MOT16-06/dukemtmc"
    cmd = ["python", "train_net.py", "--eval-only"]
    cmd.extend(["--config-file", config.model_config])
    cmd.extend(["DATASETS.TESTS", "(\"DukeMTMC\",)"])
    cmd.extend(["MODEL.WEIGHTS", osp.join("logs/mot/bagtricks_R50", dataset.name, "model_final.pth")])
    cmd.extend(["OUTPUT_DIR", osp.join("logs/mot/bagtricks_R50", dataset.name, "dukemtmc")])

    run(cmd)

def evaluate_market1501(index: int, dataset: Dataset):
    # python train_net.py --eval-only 
    # --config-file fastreid_configs/MOT/bagtricks_R50.yml 
    # DATASETS.TESTS ("Market1501",) 
    # MODEL.WEIGHTS logs/mot/bagtricks_R50/MOT16-06/model_final.pth
    # OUTPUT_DIR "logs/mot/bagtricks_R50/MOT1-16-06/market1501"
    cmd = ["python", "train_net.py", "--eval-only"]
    cmd.extend(["--config-file", config.model_config])
    cmd.extend(["DATASETS.TESTS", "(\"Market1501\",)"])
    cmd.extend(["MODEL.WEIGHTS", osp.join("logs/mot/bagtricks_R50", dataset.name, "model_final.pth")])
    cmd.extend(["OUTPUT_DIR", osp.join("logs/mot/bagtricks_R50", dataset.name, "market1501")])

    run(cmd)

def step(index: int, dataset: Dataset):
    evaluate_dukemtmc(index, dataset)
    evaluate_market1501(index, dataset)


def main():

    for i in range(args.start_index, len(config.datasets)):
        step(i, config.datasets[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)

    args = parser.parse_args()
    g_dry_run = args.dry_run

    config = read_config(args.config_file)

    pprint(attr.asdict(config))

    main()
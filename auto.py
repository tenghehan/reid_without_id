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


def invoke_yolov3_deepsort_ims(index: int, dataset: Dataset):
    cmd = ["python", "yolov3_deepsort_ims.py"]
    cmd.extend(["--config_file", config.model_config])
    cmd.extend([osp.join("image_sequence", dataset.name)])
    cmd.extend(["--fps", "%d" % dataset.fps])
    if index > 0:
        last_dataset_name = config.datasets[index - 1].name
        cmd.extend([
            "--model_path",
            osp.join("logs/mot/bagtricks_R50", last_dataset_name, "model_final.pth"),
        ])
    run(cmd)

def invoke_track2reid(dataset: Dataset):
    cmd = ["python", "track2reid.py"]
    cmd.extend(["--dataset_name", dataset.name])
    cmd.extend(["--sampling_rate", "%.2f" % dataset.sampling_rate])
    run(cmd)

def invoke_train_net(index: int, dataset: Dataset):
    cmd = ["python", "train_net.py"]
    cmd.extend(["--config-file", config.model_config])
    cmd.extend(["--specific_dataset", dataset.name])
    if index > 0:
        last_dataset_name = config.datasets[index - 1].name
        cmd.extend(["--finetune"])
        cmd.extend([
            "--model_path",
            osp.join("logs/mot/bagtricks_R50", last_dataset_name, "model_final.pth"),
        ])
    run(cmd)


def step(index: int, dataset: Dataset):
    invoke_yolov3_deepsort_ims(index, dataset)
    invoke_track2reid(dataset)
    invoke_train_net(index, dataset)


def main():
    for i, dataset in enumerate(config.datasets):
        step(i, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("-n", "--dry-run", action="store_true")

    args = parser.parse_args()
    g_dry_run = args.dry_run

    config = read_config(args.config_file)

    pprint(attr.asdict(config))

    main()

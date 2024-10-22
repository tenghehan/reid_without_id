"""
1. train reid model
2. evaluate reid model weights (model_path) on specific datasets, e.g. Market1501, DukeMTMC
"""
#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys
import os.path as osp
import random

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SPECIFIC_DATASET = args.specific_dataset
    if args.specific_dataset is not None and not args.eval_only:
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, args.specific_dataset)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = args.imageNet
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)
    if args.finetune: 
        C = Checkpointer(trainer.model)
        C.load(cfg.MODEL.WEIGHTS)  # load trained model to funetune

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    rand_seed = 50
    random.seed(rand_seed)

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AccumAMPTrainer,
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import TensorboardXWriter,WandbWriter
import os

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    if cfg.train.amp.enabled:
        trainer = AccumAMPTrainer(model, train_loader, optim, accum_iter=cfg.accum_iter)
    else:
        trainer = SimpleTrainer(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir, 
        trainer=trainer,
    )
    writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
    # if cfg.log.use_tensorboad:
    #     writers.append(TensorboardXWriter(cfg.train.output_dir))
    if cfg.log.use_wandb:
        writers.append(WandbWriter(cfg, cfg.train.output_dir, entity=cfg.log.wandb_entity, project=cfg.log.wandb_project, job_name=cfg.job_name))
    
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)#cgf.opt
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def cfg_overrides(cfg):
    # cfg.dataloader.train.total_batch_size
    num_images = 117266
    cfg.dataloader.train.total_batch_size=cfg.bs*cfg.ngpus
    num_images_per_iter = num_images//cfg.dataloader.train.total_batch_size
    cfg.train.num_images_per_iter = num_images_per_iter
    cfg.train.checkpointer=dict(period=num_images_per_iter, max_to_keep=100) # checkpoint every epoch
    cfg.train.eval_period=num_images_per_iter
    
    cfg.train.max_iter=cfg.train.num_images_per_iter * cfg.epochs
    
    cfg.lr_multiplier.warmup_length = cfg.warmup_iters / cfg.train.max_iter
    
    if cfg.pretrain_job_name.endswith(".pth"):
        cfg.train.init_checkpoint = os.path.join(cfg.modelzoo_dir,cfg.pretrain_job_name)
    else:
        cfg.train.init_checkpoint = os.path.join(cfg.ckpt_dir,cfg.pretrain_job_name,"checkpoints",f"checkpoint_{cfg.ckpt_epoch}_detectron2.pth")
        
    cfg.job_name=f"det_bs{cfg.bs}x{cfg.ngpus}x{cfg.accum_iter}_blr{cfg.optimizer.lr}_wd{cfg.optimizer.weight_decay}_dp{cfg.model.backbone.net.drop_path_rate}_{cfg.pretrain_job_name}_{cfg.ckpt_epoch}_ep{cfg.epochs}_im{cfg.model.backbone.net.img_size}"
    cfg.train.output_dir = f"{cfg.ckpt_dir}/{cfg.job_name}"
    
    return cfg

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    cfg = cfg_overrides(cfg)
    
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

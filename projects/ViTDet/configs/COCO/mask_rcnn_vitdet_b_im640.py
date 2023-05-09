from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

import detectron2.data.transforms as T

image_size = 640 #480 bs6 per gpu #640 4 per gpu

## MDOEL
model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model #configs/common/models/mask_rcnn_vitdet.py
model.backbone.net.img_size = image_size
model.backbone.square_pad = image_size

## DATA
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]

## TRAIN
# num_images = 117266
train = model_zoo.get_config("common/train.py").train
# num_images_per_iter = num_images//dataloader.train.total_batch_size
# train.num_images_per_iter = num_images_per_iter
train.amp.enabled = True
train.ddp.fp16_compression = True
# train.checkpointer=dict(period=num_images_per_iter, max_to_keep=100) # checkpoint every epoch
# train.eval_period=num_images_per_iter
train.log_period=50

# ckpt_dir = "/srv/home/pmorgado/workspace/mae2cl/checkpoints"
# pretrain_job_name = "path1_maefeat_d3t12_compl[0, 0.25]_m0.9_c0.2_epeintrinsic_dpeglobal_blr0.0005_infonce_patches_in100_vitb_bs128x1_ep100_id3"
# pretrain_ckpt = os.path.join(ckpt_dir,pretrain_job_name,"checkpoints","checkpoint_latest_detectron2.pth")
# train.init_checkpoint = pretrain_ckpt
# train.init_checkpoint = (
#    "/srv/home/pmorgado/workspace/mae2cl/checkpoints/mae_in100_vitb_bs128x1_ep100_id3/checkpoints/checkpoint_latest_detectron2.pth"# "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True" # 
# )

# SCHEDULE
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(0.89*train.max_iter), int(0.96*train.max_iter)],#[163889, 177546], #
        num_updates=train.max_iter,
    ),
    warmup_length=1000 / train.max_iter, # 250 / train.max_iter
    warmup_factor=0.001,
)

# OPTIMIZER
optimizer = model_zoo.get_config("common/optim.py").AdamW
# optimizer.lr = 1e-4 # ImageNet 8e-5 | None 1.6e-4 | MAE 1e-4
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

# LOG
# train.output_dir = f"/srv/home/pmorgado/workspace/mae2cl/checkpoints/det__{pretrain_job_name}__bs{dataloader.train.total_batch_size}_blr{optimizer.lr}_im{image_size}_debug"
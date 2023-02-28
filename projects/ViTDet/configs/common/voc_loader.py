from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import PascalVOCDetectionEvaluator

dataloader = OmegaConf.create()
image_size = 800 #1024
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="voc_2007_trainval"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations = [
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeScale)(
                min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
            ),
            L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
        ],
        image_format="RGB",
        use_instance_mask=True,
        recompute_boxes = True # recompute boxes due to cropping
    ),
    total_batch_size=1,#16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="voc_2007_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations = [
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(PascalVOCDetectionEvaluator)(
    dataset_name="${..test.dataset.names}",
)
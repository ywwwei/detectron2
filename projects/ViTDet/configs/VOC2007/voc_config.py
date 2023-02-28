from detectron2.config import CfgNode as CN

# Define the configuration for training ViTDet on VOC2007
cfg = CN()

# Dataset configuration
cfg.DATASETS.TRAIN = ("voc_2007_trainval",)
cfg.DATASETS.TEST = ("voc_2007_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.INPUT.MIN_SIZE_TRAIN = (224,)
cfg.INPUT.MAX_SIZE_TRAIN = 224
cfg.INPUT.MIN_SIZE_TEST = 224
cfg.INPUT.MAX_SIZE_TEST = 224

# Model configuration
cfg.MODEL.BACKBONE.NAME = "vit_base_patch16_224"
# cfg.MODEL.BACKBONE.PRETRAINED = "path/to/pretrained/checkpoint.pth"
cfg.MODEL.POSITION_EMBEDDING.NAME = "learned"
cfg.MODEL.RESNETS.DEPTH = 50
cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
# cfg.MODEL.WEIGHTS = "path/to/initial/weights.pth"

# Solver configuration
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (8000,)
cfg.SOLVER.GAMMA = 0.1

# Test configuration
cfg.TEST.EVAL_PERIOD = 500

# Create a new directory to store the output of the training run
cfg.OUTPUT_DIR = "path/to/output/directory"


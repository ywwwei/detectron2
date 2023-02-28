# ViTDet
# COCO | Mask RCNN | ViT
tools/lazyconfig_train_net.py \
--config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py \
--num-gpus 4

srun -N 1 -G 4 --job-name=detection --partition=morgadolab --time=24:00:00 --mem=64G --cpus-per-task=64 \
tools/lazyconfig_train_net.py \
--config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py \
--num-gpus 4


nohup srun -N 1 -G 4 --job-name=detection_lr1 --partition=morgadolab --time=24:00:00 --mem=64G --cpus-per-task=64 --nodelist=euler23 \
tools/lazyconfig_train_net.py \
--config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py \
--num-gpus 4 &

# VOC2007
tools/lazyconfig_train_net.py \
--config-file projects/ViTDet/configs/VOC2007/mask_rcnn_vitdet_b_100ep.py
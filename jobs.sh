# ViTDet
# COCO | Mask RCNN | ViT
srun -N 1 -G 4 --job-name=det_mae --partition=morgadolab --time=24:00:00 --mem=128G --cpus-per-task=63 --nodelist=euler23 \  
tools/lazyconfig_train_net.py \
--config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
--resume --num-gpus 4 \
pretrain_job_name=mae_in100_vitb_bs128x1_ep100_id2 \
ckpt_dir=/srv/home/pmorgado/workspace/mae2cl/checkpoints \
epochs=10 warmup_iters=1000 \
dataloader.train.total_batch_size=24 optimizer.lr=1e-4

# Using sbatch
# bash sbatch_det_ep10.sh euler22 \
# lr dec_depth tgt_depth mask_ratio epe dpe loss postfix

# Target depth 
bash sbatch_det_ep10.sh euler22 \
0.0001 3 12 0.9 intrinsic global infonce_patches _id3

# Mask Ratio
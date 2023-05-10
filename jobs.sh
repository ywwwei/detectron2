# ViTDet
# COCO | Mask RCNN | ViT
# MAE
    # original mae
    srun -N 1 -G 4 --job-name=det_mae --partition=morgadolab --time=24:00:00 --mem=128G --cpus-per-task=63 --nodelist=euler10 \  
    tools/lazyconfig_train_net.py \
    --config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
    --resume --num-gpus 1 \
    modelzoo_dir=/srv/home/wei96/modelzoo \
    pretrain_job_name=mae_pretrain_vit_base.pth \
    ckpt_dir=/srv/home/pmorgado/workspace/mae2cl/checkpoints \
    epochs=10 warmup_iters=1000 \
    dataloader.train.total_batch_size=24 optimizer.lr=5e-4

    # small scale mae
    srun -N 1 -G 4 --job-name=det_mae --partition=morgadolab --time=24:00:00 --mem=128G --cpus-per-task=63 --nodelist=euler23 \  
    tools/lazyconfig_train_net.py \
    --config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
    --resume --num-gpus 4 \
    pretrain_job_name=mae_in100_vitb_bs128x1_ep100_id2 \
    ckpt_dir=/srv/home/pmorgado/workspace/mae2cl/checkpoints \
    epochs=10 warmup_iters=1000 \
    dataloader.train.total_batch_size=24 optimizer.lr=5e-4

    # debug
    srun -N 1 -G 4 --job-name=vitdet_mae_lr5e-4 --partition=morgadolab --time=24:00:00 --mem=64G --cpus-per-task=32 \
    tools/lazyconfig_train_net.py \
    --config-file /srv/home/wei96/project/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
    --num-gpus 4 \
    modelzoo_dir=/srv/home/wei96/modelzoo \
    pretrain_job_name=mae_pretrain_vit_base.pth \
    ckpt_dir=/srv/home/wei96/checkpoints/mae2cl \
    epochs=10 warmup_iters=1000 \
    dataloader.train.total_batch_size=16 optimizer.lr=5e-4

# Using sbatch
# path1_maefeat
    # bash sbatch_path1_maefeat_det_ep10.sh euler22 \
    # lr dec_depth tgt_depth mask_ratio epe dpe loss postfix

    # Target depth 
    bash sbatch_path1_maefeat_det_ep10.sh euler23 \
    0.001 3 12 0.9 intrinsic global infonce_patches _id3

# v12
    # bash sbatch_v12_det_ep10.sh nodelist lr\
    # pretrain_job_name
    bash sbatch_v12_det_ep10.sh euler10 0.0005 \
    v12_copy_vit_base_d3t12_f1_inim1_k3_m0.925in0.5out0.75_eperelative_tperelative_dperelative_c0.2_blr0.00015_0.0001_1e-06_infonce_patches_mixed_bs256x2x1_ep300_sgpeFalse_imagenet_t0.2_id0

    #debug
    tools/lazyconfig_train_net.py \
    --config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
    --resume --num-gpus 4 --dist-url tcp://127.0.0.1:11212 \
    pretrain_job_name=v12_copy_vit_base_d3t12_f1_inim1_k3_m0.925in0.5out0.75_eperelative_tperelative_dperelative_c0.2_blr0.00015_0.0001_1e-06_infonce_patches_mixed_bs256x2x1_ep300_sgpeFalse_imagenet_t0.2_id0\
    ckpt_dir=/srv/home/pmorgado/workspace/mae2cl/checkpoints \
    epochs=10 warmup_iters=1000 \
    dataloader.train.total_batch_size=24 optimizer.lr=5e-4

#mae
# partition nodelist ngpus accum_iter lr bs epochs
# ckpt_dir
# pretrain_job_name checkpoint_epoch
bash sbatch_v12_det_im480.sh \
morgadolab euler10 2 4 0.0005 7 10 \
/srv/home/wei96/checkpoints/mae2cl \
mae_pretrain_vit_base.pth 

bash sbatch_v12_det_im480.sh \
morgadolab euler28 4 0.0005 6 10 \
/srv/home/wei96/checkpoints/mae2cl \
v13_vit_base_d3t12_k2_m0.9in93out0.51_blr0.0005_0.0001_0.0001_infonce_patches_bs128x1x1_ep100_imagenet100_id0 \
latest




bash sbatch_v12_det_im480.sh \
morgadolab euler10 4 0.0005 6 10 \
/srv/home/wei96/checkpoints/mae2cl \
v12_vit_base_d3t12_k3_m0.925in0.5out0.75_blr8e-05_0.0003_1e-06_infonce_patches_bs256x8x1_ep800_imagenet_id0 \
0500


bash sbatch_v12_det_im480.sh \
morgadolab euler10 4 0.0005 6 10 \
/srv/home/wei96/checkpoints/mae2cl \
v13_vit_base_d4t12_k2_m0.925in134out0.5_blr0.00015_0.0003_0.0003_infonce_patches_bs256x4x4x4_ep800_imagenet_id0 \
0200


# debug on mmlab2 
#export DETECTRON2_DATASETS=/home/yibingwei/dataset
CUDA_VISIBLE_DEVICES=0 tools/lazyconfig_train_net.py \
--config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_im480.py \
--resume --num-gpus 1 --dist-url tcp://127.0.0.1:40011 --eval_only True\
pretrain_job_name=mae_pretrain_vit_base.pth modelzoo_dir=/home/yibingwei/model_zoo \
ckpt_dir=/home/yibingwei/checkpoints/mae2cl \
epochs=10 warmup_iters=1000 ckpt_epoch=latest \
bs=1 ngpus=1 optimizer.lr=0.0005 accum_iter=5 \
log.use_wandb=True log.wandb_entity=mae-vs-clr log.wandb_project=mae2cl_det

tools/lazyconfig_train_net.py \
--config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_im480.py \
--resume --num-gpus 2 --dist-url tcp://127.0.0.1:40001 \
pretrain_job_name=mae_pretrain_vit_base.pth modelzoo_dir=/home/yibingwei/model_zoo \
ckpt_dir=/home/yibingwei/checkpoints/mae2cl \
epochs=10 warmup_iters=1000 ckpt_epoch=latest \
bs=3 ngpus=2 optimizer.lr=0.0005 accum_iter=4 \
log.use_wandb=True log.wandb_entity=mae-vs-clr log.wandb_project=mae2cl_det

# ViTDet
# COCO | Mask RCNN | ViT
# MAE
    # original mae
    srun -N 1 -G 4 --job-name=det_mae --partition=morgadolab --time=24:00:00 --mem=128G --cpus-per-task=63 --nodelist=euler23 \  
    tools/lazyconfig_train_net.py \
    --config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
    --resume --num-gpus 4 \
    modelzoo_dir=/srv/home/pmorgado/yibing/modelzoo/ \
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
    nohup srun -N 1 -G 4 --job-name=detection_lr1 --partition=morgadolab --time=24:00:00 --mem=64G --cpus-per-task=64 --nodelist=euler23 \
    tools/lazyconfig_train_net.py \
    --config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
    --num-gpus 4 \
    pretrain_job_name="path1_maefeat_d3t12_compl\[0\,\ 0.25\]_m0.9_c0.2_epeintrinsic_dpeglobal_blr0.0005_infonce_patches_in100_vitb_bs128x1_ep100_id3" \
    ckpt_dir=/srv/home/pmorgado/workspace/mae2cl/checkpoints \
    epochs=10 warmup_iters=1000 \
    dataloader.train.total_batch_size=24 optimizer.lr=1e-3 &

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
#!/bin/bash
nodelist=$1

lr=$2

decoder_depth=$3
target_depth=$4
mask_ratio=$5
encoder_pe=$6
decoder_pe=$7
loss=$8
postfix=$9
pretrain_job_name="path1_maefeat_d${decoder_depth}t${target_depth}_compl0_0.25_m${mask_ratio}_c0.2_epe${encoder_pe}_dpe${decoder_pe}_blr0.0005_${loss}_in100_vitb_bs128x1_ep100${postfix}"
port=$RANDOM  #10000, 20000

mkdir -p /srv/home/pmorgado/workspace/mae2cl/checkpoints/det__${pretrain_job_name}__ep10_bs24_blr${lr}_im480
    
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --job-name=det_lr${lr}_${pretrain_job_name} # Job name
#SBATCH --partition=morgadolab
#SBATCH --nodelist=${nodelist}    # euler22 or euler23
#SBATCH --cpus-per-task=64 
#SBATCH --mem=128G	
#SBATCH --time=24:00:00	
#SBATCH --output=/srv/home/pmorgado/workspace/mae2cl/checkpoints/det__${pretrain_job_name}__ep10_bs24_blr${lr}_im480/log.out
#SBATCH --error=/srv/home/pmorgado/workspace/mae2cl/checkpoints/det__${pretrain_job_name}__ep10_bs24_blr${lr}_im480/log.err	

export DETECTRON2_DATASETS=/srv/home/pmorgado/datasets

tools/lazyconfig_train_net.py \
--config-file /srv/home/pmorgado/yibing/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_10ep.py \
--resume --num-gpus 4 --dist-url tcp://127.0.0.1:${port} \
pretrain_job_name=${pretrain_job_name} \
ckpt_dir=/srv/home/pmorgado/workspace/mae2cl/checkpoints \
epochs=10 warmup_iters=1000 \
dataloader.train.total_batch_size=24 optimizer.lr=${lr}
EOT
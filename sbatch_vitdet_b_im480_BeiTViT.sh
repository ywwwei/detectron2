#!/bin/bash
partition=$1
nodelist=$2
ngpus=$3
accum_iter=$4
lr=$5
bs=$6
epochs=$7
wd=$8
dp=$9
ckpt_dir=${10}
pretrain_job_name=${11}
checkpoint_epoch=${12}
modelzoo_dir=${13}
port=$((RANDOM%(50000-35565+1)+35565))  #35565, 50000

export DETECTRON2_DATASETS=/srv/home/groups/pmorgado/datasets
mkdir -p ${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im480
echo "Job: det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im480"
echo "mem=$((60*${ngpus}))G"
echo "cpus-per-task=$((12*${ngpus}))"
    
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=${ngpus}
#SBATCH --job-name=det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im480 # Job name
#SBATCH --partition=${partition}
#SBATCH --nodelist=${nodelist} 
#SBATCH --cpus-per-task=$((14*${ngpus})) 
#SBATCH --mem=$((30*${ngpus}))G	
#SBATCH --time=12:00:00	
#SBATCH --output=${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im480/log.out
#SBATCH --error=${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im480/log.err	

tools/lazyconfig_train_det2_mim.py \
--config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_im480_beitViT.py \
--resume --num-gpus ${ngpus} --dist-url tcp://127.0.0.1:${port} \
pretrain_job_name=${pretrain_job_name} modelzoo_dir=${modelzoo_dir} \
ckpt_dir=${ckpt_dir} \
epochs=${epochs} warmup_iters=1000 ckpt_epoch=${checkpoint_epoch} \
bs=${bs} ngpus=${ngpus} accum_iter=${accum_iter} optimizer.lr=${lr} \
optimizer.weight_decay=${wd} model.backbone.net.drop_path_rate=${dp} \
log.use_wandb=True log.wandb_entity=mae-vs-clr log.wandb_project=mae2cl_det
EOT
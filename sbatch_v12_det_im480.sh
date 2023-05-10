#!/bin/bash
partition=$1
nodelist=$2
ngpus=$3
accum_iter=$4
lr=$5
bs=$6
epochs=$7
ckpt_dir=$8
pretrain_job_name=$9
checkpoint_epoch=${10}
modelzoo_dir="/srv/home/wei96/modelzoo"
port=$((RANDOM%(50000-35565+1)+35565))  #35565, 50000

export DETECTRON2_DATASETS=/srv/home/groups/pmorgado/datasets
mkdir -p ${ckpt_dir}/det_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_im480
echo "Job: det_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_im480"
echo "mem=$((60*${ngpus}))G"
echo "cpus-per-task=$((12*${ngpus}))"
    
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=${ngpus}
#SBATCH --job-name=det_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_im480 # Job name
#SBATCH --partition=${partition}
#SBATCH --nodelist=${nodelist} 
#SBATCH --cpus-per-task=$((12*${ngpus})) 
#SBATCH --mem=$((60*${ngpus}))G	
#SBATCH --time=11:30:00	
#SBATCH --output=${ckpt_dir}/det_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_im480/log.out
#SBATCH --error=${ckpt_dir}/det_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_im480/log.err	

tools/lazyconfig_train_net.py \
--config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_im480.py \
--resume --num-gpus ${ngpus} --dist-url tcp://127.0.0.1:${port} \
pretrain_job_name=${pretrain_job_name} modelzoo_dir=${modelzoo_dir} \
ckpt_dir=${ckpt_dir} \
epochs=${epochs} warmup_iters=1000 ckpt_epoch=${checkpoint_epoch} \
bs=${bs} ngpus=${ngpus} optimizer.lr=${lr}\
log.use_wandb=True log.wandb_entity=mae-vs-clr log.wandb_project=mae2cl_det
EOT
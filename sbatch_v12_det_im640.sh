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
milestone1=${10}
milestone2=${11}
ckpt_dir=${12}
pretrain_job_name=${13}
checkpoint_epoch=${14}
modelzoo_dir=${15}
port=$((RANDOM%(50000-35565+1)+35565))  #35565, 50000

export DETECTRON2_DATASETS=/srv/home/groups/pmorgado/datasets
mkdir -p ${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_ms${milestone1}_${milestone2}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im640
echo "Job: det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_ms${milestone1}_${milestone2}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im640"
echo "mem=$((60*${ngpus}))G"
echo "cpus-per-task=$((12*${ngpus}))"
    
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=${ngpus}
#SBATCH --job-name=det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_ms${milestone1}_${milestone2}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im640 # Job name
#SBATCH --partition=${partition}
#SBATCH --nodelist=${nodelist} 
#SBATCH --cpus-per-task=$((15*${ngpus})) 
#SBATCH --mem=$((60*${ngpus}))G	
#SBATCH --time=11:30:00	
#SBATCH --output=${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_ms${milestone1}_${milestone2}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im640/log.out
#SBATCH --error=${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_ms${milestone1}_${milestone2}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im640/log.err	

tools/lazyconfig_train_net.py \
--config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_im640.py \
--resume --num-gpus ${ngpus} --dist-url tcp://127.0.0.1:${port} \
pretrain_job_name=${pretrain_job_name} modelzoo_dir=${modelzoo_dir} \
ckpt_dir=${ckpt_dir} \
epochs=${epochs} warmup_iters=1000 ckpt_epoch=${checkpoint_epoch} \
bs=${bs} ngpus=${ngpus} accum_iter=${accum_iter} optimizer.lr=${lr} \
optimizer.weight_decay=${wd} model.backbone.net.drop_path_rate=${dp} \
milestone1=${milestone1} milestone2=${milestone2} \
log.use_wandb=True log.wandb_entity=mae-vs-clr log.wandb_project=mae2cl_det
EOT
#!/bin/bash
nodelist=$1
ngpus=$2
lr=$3
bs=$4
epochs=$5
ckpt_dir=$6
pretrain_job_name=$7
modelzoo_dir=$8
port=$((RANDOM%(50000-35565+1)+35565))  #35565, 50000

export DETECTRON2_DATASETS=/srv/home/groups/pmorgado/datasets
mkdir -p ${ckpt_dir}/det_${pretrain_job_name}_ep${epochs}_bs${bs}x${ngpus}_blr${lr}_im640
echo "Job: det_${pretrain_job_name}_ep${epochs}_bs${bs}x${ngpus}_blr${lr}_im640"
echo "mem=$((60*${ngpus}))G"
echo "cpus-per-task=$((12*${ngpus}))"
    
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=${ngpus}
#SBATCH --job-name=det_${pretrain_job_name}_ep${epochs}_bs${bs}x${ngpus}_blr${lr}_im640 # Job name
#SBATCH --partition=morgadolab
#SBATCH --nodelist=${nodelist} 
#SBATCH --cpus-per-task=$((12*${ngpus})) 
#SBATCH --mem=$((60*${ngpus}))G	
#SBATCH --time=24:00:00	
#SBATCH --output=${ckpt_dir}/det_${pretrain_job_name}_ep${epochs}_bs${bs}x${ngpus}_blr${lr}_im640/log.out
#SBATCH --error=${ckpt_dir}/det_${pretrain_job_name}_ep${epochs}_bs${bs}x${ngpus}_blr${lr}_im640/log.err	

tools/lazyconfig_train_net.py \
--config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_im640.py \
--resume --num-gpus ${ngpus} --dist-url tcp://127.0.0.1:${port} \
pretrain_job_name=${pretrain_job_name} modelzoo_dir=${modelzoo_dir} \
ckpt_dir=${ckpt_dir} \
epochs=${epochs} warmup_iters=1000 \
bs=${bs} ngpus=${ngpus} optimizer.lr=${lr}
EOT
#!/bin/bash

# Default values
partition="morgadolab"
nodelist="euler10"
ngpus=2
accum_iter=8
image_size=1024
lr=1e-4
bs=1
epochs=100
wd=0.1
dp=0.1
ckpt_dir="/srv/home/wei96/checkpoints/mae2cl"
pretrain_job_name=""
checkpoint_epoch="latest"
modelzoo_dir="/srv/home/wei96/modelzoo"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --partition) partition="$2"; shift ;;
    --nodelist) nodelist="$2"; shift ;;
    --ngpus) ngpus="$2"; shift ;;
    --accum_iter) accum_iter="$2"; shift ;;
    --image_size) image_size="$2"; shift ;;
    --lr) lr="$2"; shift ;;
    --bs) bs="$2"; shift ;;
    --epochs) epochs="$2"; shift ;;
    --wd) wd="$2"; shift ;;
    --dp) dp="$2"; shift ;;
    --ckpt_dir) ckpt_dir="$2"; shift ;;
    --pretrain_job_name) pretrain_job_name="$2"; shift ;;
    --checkpoint_epoch) checkpoint_epoch="$2"; shift ;;
    --modelzoo_dir) modelzoo_dir="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

port=$((RANDOM%(50000-35565+1)+35565))  #35565, 50000

export DETECTRON2_DATASETS=/srv/home/groups/pmorgado/datasets
mkdir -p ${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im${image_size}
echo "Job: det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im${image_size}"
echo "mem=$((60*${ngpus}))G"
echo "cpus-per-task=$((12*${ngpus}))"

echo partition=${partition}
echo nodelist=${nodelist}
echo ngpus=${ngpus}
echo accum_iter=${accum_iter}
echo image_size=${image_size}
echo lr=${lr}
echo bs=${bs}
echo epochs=${epochs}
echo wd=${wd}
echo dp=${dp}
echo ckpt_dir=${ckpt_dir}
echo pretrain_job_name=${pretrain_job_name}
echo checkpoint_epoch=${checkpoint_epoch}
echo modelzoo_dir=${modelzoo_dir}

    
sbatch <<EOT
#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:${ngpus}
#SBATCH --job-name=det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im${image_size} # Job name
#SBATCH --partition=${partition}
# #SBATCH --nodelist=${nodelist}
#SBATCH --cpus-per-task=$((14*${ngpus})) 
#SBATCH --mem=$((30*${ngpus}))G    
#SBATCH --time=12:00:00    
#SBATCH --output=${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im${image_size}/log.out
#SBATCH --error=${ckpt_dir}/det_bs${bs}x${ngpus}x${accum_iter}_blr${lr}_wd${wd}_dp${dp}_${pretrain_job_name}_${checkpoint_epoch}_ep${epochs}_im${image_size}/log.err    

tools/lazyconfig_train_net.py \
--config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b.py \
--resume --num-gpus ${ngpus} --dist-url tcp://127.0.0.1:${port} \
pretrain_job_name=${pretrain_job_name} modelzoo_dir=${modelzoo_dir} \
ckpt_dir=${ckpt_dir} \
epochs=${epochs} warmup_iters=1000 ckpt_epoch=${checkpoint_epoch}_detectron2 \
bs=${bs} ngpus=${ngpus} accum_iter=${accum_iter} optimizer.lr=${lr} \
optimizer.weight_decay=${wd} model.backbone.net.drop_path_rate=${dp} \
log.use_wandb=True log.wandb_entity=mae-vs-clr log.wandb_project=mae2cl_det
EOT

#!/bin/bash
#SBATCH --job-name=bert_ddp_2n4g
#SBATCH --partition=short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=runs/logs/%x_%j.out

source $STORE/myvenv/bin/activate
export HF_HOME=$STORE/hf_cache
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

cd bert_squad_distributed

RUN=$STORE/HPCToolsLabIA/DISTRIBUTED/runs/ddp_2n4g_$SLURM_JOB_ID
mkdir -p $RUN

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5 > $RUN/nvidia-gpu-node0.log &
NVIDIA_SMI_PID=$!

srun bash launch_node.sh \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --bf16 True \
    --tf32 True \
    --optim adamw_torch \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size 64 \
    --learning_rate 6e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --pad_to_max_length False \
    --group_by_length True \
    --output_dir $LUSTRE_SCRATCH/bert_squad_output \
    --overwrite_output_dir True \
    --save_total_limit 2 \
    --report_to tensorboard \
    --disable_tqdm true \
    --logging_steps 100 \
    --logging_dir $LUSTRE_SCRATCH/bert_squad_logs

kill $NVIDIA_SMI_PID

cp -r $LUSTRE_SCRATCH/bert_squad_logs $RUN/logs
cp $LUSTRE_SCRATCH/bert_squad_output/*.json $RUN/
cp $SLURM_SUBMIT_DIR/runs/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $RUN/
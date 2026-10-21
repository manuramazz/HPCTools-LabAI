#!/bin/bash
#SBATCH --job-name=bert_profile_1gpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=runs/logs/%x_%j.out

source $STORE/myvenv/bin/activate
export HF_HOME=$STORE/hf_cache

cd bert_squad_distributed

RUN=$STORE/HPCToolsLabIA/DISTRIBUTED/runs/profile_1gpu_$SLURM_JOB_ID
mkdir -p $RUN

# Monitoreo de utilización de GPU en background
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5 > $LUSTRE_SCRATCH/gpu.log &
NVIDIA_SMI_PID=$!
export PROFILE=1

srun python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --bf16 True \
  --tf32 True \
  --optim adamw_torch \
  --dataloader_num_workers 8 \
  --per_device_train_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --pad_to_max_length False \
  --group_by_length True \
  --max_steps 30 \
  --output_dir $LUSTRE_SCRATCH/bert_squad_output \
  --overwrite_output_dir True \
  --save_total_limit 2 \
  --report_to tensorboard \
  --disable_tqdm true \
  --logging_steps 100 \
  --logging_dir $LUSTRE_SCRATCH/bert_squad_logs

kill $NVIDIA_SMI_PID

# Copiar a almacenamiento persistente (solo ficheros pequeños, sin pesos)
cp -r $LUSTRE_SCRATCH/bert_squad_logs $RUN/logs
cp $LUSTRE_SCRATCH/bert_squad_output/*.json $RUN/
cp $LUSTRE_SCRATCH/gpu.log $RUN/nvidia-gpu.log
cp $SLURM_SUBMIT_DIR/runs/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $RUN/
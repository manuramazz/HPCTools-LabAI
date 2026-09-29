#!/bin/bash
#SBATCH --job-name=solo-prueba
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

source $STORE/myvenv/bin/activate
export HF_HOME=$STORE/hf_cache

cd bert_squad_baseline

srun python ../pruebas.py

srun python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --auto_find_batch_size True \
  --per_device_train_batch_size 128 \
  --max_train_samples 500 \
  --bf16 True \
  --tf32 True \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $LUSTRE_SCRATCH/bert_squad_output \
  --report_to tensorboard \
  --disable_tqdm true \
  --logging_steps 100 \
  --overwrite_output_dir True \
  --logging_dir $LUSTRE_SCRATCH/bert_squad_logs

# Copiar a almacenamiento persistente los logs
cp -r $LUSTRE_SCRATCH/bert_squad_logs $STORE/HPCToolsLabIA/BASELINE/bert_squad_logs
cp -r $LUSTRE_SCRATCH/bert_squad_output $STORE/HPCToolsLabIA/BASELINE/bert_squad_output

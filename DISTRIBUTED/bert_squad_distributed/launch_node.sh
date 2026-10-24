
#!/bin/bash
torchrun --nnodes=$SLURM_NNODES --nproc_per_node=2 \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  run_qa.py "$@"
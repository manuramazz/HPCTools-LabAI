import torch
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}")
print(f"Memoria total: {props.total_memory / 1024**3:.1f} GB")
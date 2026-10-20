Una gpu nvidia A100 estándar tiene 2 GPUs y 40 GB de memoria, lo pone en la documentación del CESGA, no hace ni falta comporbarlo desde los nodos:
#### 64 a100 compute nodes with GPU accelerators: 4.096 cores : 
2x Intel Xeon Ice Lake 8352Y with 32 cores each
256GB of RAM memory (247GB for real use)
960GB SSD NVMe of local storage
2x NVIDIA A100 GPUs.


#### NVIDIA A100
CUDA Driver Version / Runtime Version: 11.5 / 11.2
CUDA Capability Major/Minor version number: 8.0
MEMORY: 40 GB of HBM2 BANDWIDTH 1555 GB/s
108 Multiprocessors, 64 CUDA Cores/MP: 6912 CUDA Cores
GPU Max Clock rate: 1410 MHz (1.41 GHz) 

#### Volumen y FLOPs del modelo:
[INFO|trainer.py:1793] ... Number of trainable parameters = 108,893,186
Memoria fija con adamw_torch y BF16 autocast: los pesos maestros quedan en FP32, así que cada parámetro ocupa 4 B (peso) + 4 B (gradiente) + 8 B (momentos m y v) = 16 B/param. Con 108.9M de parámetros son ≈ 1.74 GB, y el resto de la memoria son activaciones, que crece con el tamaño del batch. Esto justifica no tocar para nada ZeRo ni FDSP.
Sobre el tamaño del batch: el máximo posible sin que el tamaño del modelo con entre en la GPU:
21.3 GiB

Peak flops de NVIDA a100: 312 TFLOPs con mixed precision bf16.
FLOPs del entrenamiento siguiendo la fórmula 6 x N x D
Número de tokens: 4 épocas x 88524 samples x 384 tokens por entrada = 135.972.864 tokens
Tokens medidos sobre el programa: 
FLOPs = 135M x 108M x 6 = 8.88 x 10e16

**Versión base**
| FLOPs totales | TFLOP/s  | MFU |
|---|---|---|
| 8.88·10¹⁶ | 58.9 | 18.9 % |

Versión distribuida a completar obviamente (los FLOPs son los mismos)

Versión con --pad_to_max_length False:

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
"max_cuda_allocated_gib": 20.231595039367676,
"max_cuda_reserved_gib": 20.681640625,

Peak flops de NVIDA a100: 312 TFLOPs con mixed precision bf16.
FLOPs del entrenamiento siguiendo la fórmula 6 x N x D
Número de tokens: 4 épocas x 88524 samples x 384 tokens por entrada = 135.972.864 tokens


OPTIMIZACIÓN QUITANDO EL PADDING DE TOKENS
No se rellenan si el tamaño es menor a 384

Real tokens: 
Tokens con padding: 135M
Tokens reales medidos sobre el programa: 15115761 x 4 épocas = 60463044
FLOPS = 60463044 x 6 x 108893186 = 3.95 x 10e16

**SINGLE GPU**
| Version | Runtime | FLOPs totales | TFLOP/s  | MFU |
|---|---|---|---|---|
| BASELINE deliverable | 0:25:08.03 | 8.88·10¹⁶ | 58.9 | 18.9 % (contando tokens totales, por tanto una parte del cálculo es inútil) |
| No pad | 0:12:06 | 3.95 x 10e16 | 54.4 | 17.4 %  |
Se puede apreciar que el MFU es menor pero en realidad el tiempo es mucho menor por el no-padding. Los tokens pueden tener tamaños variables y esto puede hacer que no se aproveche tan bien como antes el cómputo en GPU, pero un gran porcentaje del cómputo era inútil. (tiempo de ejecución a la mitad).
{
    "epoch": 4.0,
    "eval_exact_match": 79.6972563859981,
    "eval_f1": 87.61559296607729,
    "eval_runtime": 15.9666,
    "eval_samples": 10784,
    "eval_samples_per_second": 675.409,
    "eval_steps_per_second": 84.426,
    "max_cuda_allocated_gib": 20.231595039367676,
    "max_cuda_reserved_gib": 20.798828125,
    "train_loss": 0.869132709640988,
    "train_real_tokens_per_epoch": 15115761,
    "train_runtime": 726.4731,
    "train_samples": 88524,
    "train_samples_per_second": 487.418,
    "train_steps_per_second": 7.62
}
Además:   "train_loss": 0.869132709640988 casi la misma que el baseline. Por tanto se confirma una mejora real sin impacto alguno en el resultado.


PROFILER de torch sobre la versión optimizada con el no-pad:
Medido: GPU ocupada el 78% del step, Memcpy y DataLoader ~0%, kernels ampere_bf16 (BF16 en Tensor Cores), ~33% del tiempo de GPU en Tensor Cores y ~la mitad en kernels elementwise.
Inferido: que los miles de kernels pequeños vienen del optimizador, el clipping y la acumulación de gradientes (el patrón de 4 llamadas por cada uno de los 199 tensores de parámetros encaja, pero no lo he comprobado directamente).

Versión distribuida a completar obviamente (los FLOPs son los mismos)
Entre nodos usa InfiniBand: Using network IB y canales via NET/IB/0 y NET/IB/1, con los dispositivos mlx5_2:1/IB y mlx5_bond_0:1/RoCE. No aparece NET/Socket, así que se descarta mi sospecha de que cayera a TCP.
Dentro de un nodo, las GPUs se hablan por memoria compartida del host: via SHM/direct/direct, sin NVLink ni P2P. Cuadra con que sean A100-PCIe.
Resultados:
{
    "epoch": 4.0,
    "eval_exact_match": 79.98107852412488,
    "eval_f1": 87.63487613812165,
    "eval_runtime": 4.4768,
    "eval_samples": 10784,
    "eval_samples_per_second": 2408.885,
    "eval_steps_per_second": 75.278,
    "max_cuda_allocated_gib": 20.647382259368896,
    "max_cuda_reserved_gib": 21.099609375,
    "train_loss": 1.0397457304717488,
    "train_real_tokens_per_epoch": 15115761,
    "train_runtime": 245.1938,
    "train_samples": 88524,
    "train_samples_per_second": 1444.147,
    "train_steps_per_second": 5.645
}

Profiling:

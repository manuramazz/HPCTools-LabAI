## Baseline implementation

Scripts `run_qa.py`, `trainer_qa.py`, `utils_qa.py` obtenidos de:
https://github.com/huggingface/transformers/tree/examples/pytorch/question-answering


Descargados el 2026-09-27. Tag 4.30.2 para coincidir con mi versión de transformers instalada localmente.

## Baseline Training execution — BERT-Base on SQuAD (Full Dataset)

### Results

| Metric | Value |
|---|---|
| Epochs | 2.0 |
| Train loss (final) | 0.9871 |
| Train runtime | 0:54:05.61 |
| Train samples | 88,524 |
| Train samples/second | 54.55 |
| Train steps/second | 4.546 |

### Configuration

| Parameter | Value |
|---|---|
| Model | bert-base-uncased |
| Dataset | SQuAD (full) |
| GPU | 1× NVIDIA A100 |
| Precision | FP32 / TF32 (default) |
| Batch size (per device) | 12 |
| Learning rate | 3e-5 |
| Max sequence length | 384 |
| Doc stride | 128 |

---
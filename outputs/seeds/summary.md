## Seed study (validation, mean ± SD)

| Model | Seeds | Macro F1 | Balanced acc. | Weighted F1 | Accuracy |
|---|---|---|---|---|---|
| multimodal | 3 | 0.7217 ± 0.0051 | 0.7362 ± 0.0051 | 0.8401 ± 0.0021 | 0.8370 ± 0.0031 |
| image_only | 3 | 0.7230 ± 0.0015 | 0.7160 ± 0.0127 | 0.8381 ± 0.0006 | 0.8383 ± 0.0029 |

## Per run

| Run | Macro F1 | Balanced acc. | Accuracy |
|---|---|---|---|
| multimodal_seed42 | 0.7265 | 0.7421 | 0.8381 |
| multimodal_seed43 | 0.7223 | 0.7328 | 0.8336 |
| multimodal_seed44 | 0.7163 | 0.7338 | 0.8394 |
| image_only_seed42 | 0.7247 | 0.7084 | 0.8414 |
| image_only_seed43 | 0.7228 | 0.7306 | 0.8355 |
| image_only_seed44 | 0.7216 | 0.7089 | 0.8381 |

## Data strategy (multimodal, seed 42)

| Setting | Macro F1 | Balanced acc. | Accuracy |
|---|---|---|---|
| multimodal_no_class_weights_seed42 | 0.7350 | 0.7293 | 0.8460 |
| multimodal_class_weights_seed42 | 0.7265 | 0.7421 | 0.8381 |

## Computational cost

| Model | Params (M) | Size (MB) | Inference (ms/img) | Train time (min) | Peak train mem (GB) |
|---|---|---|---|---|---|
| resnet50 | 23.52 | 89.7 | 17.24 | — | — |
| vit | 85.8 | 327.3 | 40.91 | — | — |
| multimodal | 24.06 | 91.8 | 20.65 | 38.7 | 0.35 |
| image_only | 24.03 | 91.7 | 18.55 | 38.2 | 0.34 |
| tabular_only | 0.03 | 0.1 | 1.98 | — | — |
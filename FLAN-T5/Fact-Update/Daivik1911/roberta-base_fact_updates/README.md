---
license: mit
base_model: roberta-base
tags:
- generated_from_trainer
model-index:
- name: roberta-base_fact_updates
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base_fact_updates

This model is a fine-tuned version of [roberta-base](https://huggingface.co/roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2890

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 15

### Training results

| Training Loss | Epoch | Step  | Validation Loss |
|:-------------:|:-----:|:-----:|:---------------:|
| 0.2718        | 1.0   | 734   | 0.2895          |
| 0.2817        | 2.0   | 1468  | 0.3295          |
| 0.3141        | 3.0   | 2202  | 0.2891          |
| 0.3464        | 4.0   | 2936  | 0.2919          |
| 0.2108        | 5.0   | 3670  | 0.2966          |
| 0.2811        | 6.0   | 4404  | 0.3113          |
| 0.4315        | 7.0   | 5138  | 0.2903          |
| 0.3629        | 8.0   | 5872  | 0.2890          |
| 0.2729        | 9.0   | 6606  | 0.2900          |
| 0.2273        | 10.0  | 7340  | 0.2891          |
| 0.2127        | 11.0  | 8074  | 0.2890          |
| 0.2929        | 12.0  | 8808  | 0.2895          |
| 0.3608        | 13.0  | 9542  | 0.2890          |
| 0.2816        | 14.0  | 10276 | 0.2890          |
| 0.3008        | 15.0  | 11010 | 0.2890          |


### Framework versions

- Transformers 4.38.2
- Pytorch 2.2.1
- Datasets 2.18.0
- Tokenizers 0.15.2

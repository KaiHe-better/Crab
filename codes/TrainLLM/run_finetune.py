from src.llmtuner.train.tuner import run_exp

import json

args = dict(
  stage="sft",                        # do supervised fine-tuning
  do_train=True,
  model_name_or_path="path/to/Meta-Llama-3.1-8B-Instruct",
  dataset="crab_train_data",             # use character dataset
  template="llama3",                     # use llama3 prompt template
  cutoff_len=2048,                      # the maximum length of input
  finetuning_type="lora",                   # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  output_dir="path/to/save/crab-llama3.1-lora",                  # the path to save LoRA adapters
  per_device_train_batch_size=2,               # the batch size
  gradient_accumulation_steps=8,               # the gradient accumulation steps
  lr_scheduler_type="cosine",                 # use cosine learning rate scheduler
  logging_steps=10,                      # log every 10 steps
  warmup_ratio=0.1,                      # use warmup scheduler
  save_steps=200,                      # save checkpoint every 1000 steps
  learning_rate=5e-5,                     # the learning rate
  num_train_epochs=3.0,                    # the epochs of training
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  loraplus_lr_ratio=16.0,                   # use LoRA+ algorithm with lambda=16.0
  use_unsloth=False,                      # use UnslothAI's LoRA optimization for 2x faster training
  bf16=True,                         # use float16 mixed precision training
  overwrite_output_dir=True,
)

run_exp(args)
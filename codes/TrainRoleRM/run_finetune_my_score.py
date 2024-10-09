import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.llmtuner.train.tuner import run_exp
import json


# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"

args = dict(
  stage="sft",                           # do supervised fine-tuning
  do_train=True,
  model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  dataset="data_score_train",            # use character dataset
  template="llama3",                     # use llama3 prompt template
  cutoff_len=2048,                       # the maximum length of input
  finetuning_type="lora",                # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  output_dir="output",                   # the path to save LoRA adapters
  per_device_train_batch_size=8,        # the batch size
  gradient_accumulation_steps=8,         # the gradient accumulation steps
  # lr_scheduler_type="cosine",          # use cosine learning rate scheduler
  warmup_steps=200,                      # use warmup scheduler
  logging_steps=10,                      # log every 10 steps
  save_steps=100,                        # save checkpoint every 1000 steps
  learning_rate=5e-5,                    # the learning rate
  num_train_epochs=99999,                # the epochs of training
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  # quantization_bit=4,                  # use 4-bit QLoRA
  loraplus_lr_ratio=16.0,                # use LoRA+ algorithm with lambda=16.0
  use_unsloth=False,                     # use UnslothAI's LoRA optimization for 2x faster training
  bf16=True,                             # use float16 mixed precision training
  overwrite_output_dir=True,
  flash_attn="off",                      # It is strongly recommended to train Gemma2 models with the `eager` attention implementation instead of `sdpa`.
  train_on_prompt=False,
)

run_exp(args)
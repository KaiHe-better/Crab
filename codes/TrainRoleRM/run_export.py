'''
Descripttion: 
Author: 
'''
'''
Descripttion: 
Author: 
'''
import json
from src.llmtuner.train.tuner import export_model

args = dict(
  model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct", # use official non-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="output_my",            # load the saved LoRA /demo_train_27b/checkpoint-1600
  template="llama3",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  export_dir="output_my/final_model",              # the path to save the merged model
  export_size=4,                       # the file shard size (in GB) of the merged model
  export_device="cpu",                    # the device used in export, can be chosen from `cpu` and `cuda`
  #export_hub_model_id="your_id/your_model",         # the Hugging Face hub ID to upload model
)

export_model(args)
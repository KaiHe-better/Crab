import json
from src.llmtuner.train.tuner import export_model

args = dict(
  model_name_or_path="path/to/Meta-Llama-3.1-8B-Instruct", # use official non-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="path/to/save/crab-llama3.1-lora",            # load the saved LoRA adapters
  template="llama3",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  export_dir="path/to/save/crab-llama3.1",              # the path to save the merged model
  export_size=4,                       # the file shard size (in GB) of the merged model
  export_device="cpu",                    # the device used in export, can be chosen from `cpu` and `cuda`
  #export_hub_model_id="your_id/your_model",         # the Hugging Face hub ID to upload model
)

print(args)
export_model(args)
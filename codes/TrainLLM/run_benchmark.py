import json
from src.llmtuner.chat import ChatModel
from src.llmtuner.extras.misc import torch_gc

import sys
model_id = sys.argv[1] # "crab-llama3.1"
template = "gemma" if "gemma" in model_id else "llama3"
args = dict(
  model_name_or_path="path/to/save/"+model_id,
  template=template,                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  use_unsloth=False,                     # use UnslothAI's LoRA optimization for 2x faster generation
  max_length=2048,
  max_new_tokens=128,
  temperature=0.8,
  top_p=1.0,
  do_sample=True,
)
chat_model = ChatModel(args)
with open('dataset/crab_benchmark.json') as f:
    data = json.load(f)
import random
random.seed(42)
random.shuffle(data)
print(len(data))

from tqdm import trange,tqdm
new_data = []
try:
  with open(f"results_to_score/{model_id}.jsonl","r") as f:
    start  = len(list(f))
except:
  start = 0
print("Start from", start)
import torch
with torch.no_grad():
  for d in tqdm(data[start:]):
      system = d['system']
      messages = []
      for i in range(0, len(d['conversations'])-1, 2):
          c = d['conversations'][i]
          if c['from'] == 'human':
              messages.append({"role": "user", "content": c['value']})
          c_1 = d['conversations'][i+1]
          response = chat_model.chat(messages, system=system)[0].response_text
          messages.append({"role": "assistant", "content": response})
          c_1['generated'] = response
      with open(f"results/{model_id}.jsonl","a") as f:
          f.write(json.dumps(d)+"\n")

torch_gc()

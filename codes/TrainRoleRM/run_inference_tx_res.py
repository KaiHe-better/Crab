
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ID', default="test", type=str, help='gpu device numbers')
parser.add_argument('--gpu', default="4", type=str, help='gpu device numbers')
parser.add_argument('--data_file', default='data/tx/test_gen_0802_llama_baseline/to_test/Meta-Llama-3-8B-Instruct.json', type=str, help='gpu device numbers')
parser.add_argument('--model_name_or_path', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str, help='model_name_or_path')
parser.add_argument('--adapter_name_or_path', default="checkpoint-900", type=str, help='adapter_name_or_path')
parser.add_argument('--test_num', default=5, type=int, help='test_num')
parser.add_argument('--temperature', default=0.8, type=float, help='temperature')
parser.add_argument('--top_k', default=50, type=int, help='temperature')
parser.add_argument('--top_p', default=0.95, type=float, help='temperature')
sys_args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = sys_args.gpu

import json
from src.llmtuner.chat import ChatModel
from src.llmtuner.extras.misc import torch_gc
import sys
import re

model_id = "llama3"
template = "gemma" if "gemma" in model_id else "llama3"

data_file = sys_args.data_file

# all_adapter_name_or_path = "output/"+sys_args.adapter_name_or_path
all_adapter_name_or_path = "output-llama3.1/"+sys_args.adapter_name_or_path

model_name_or_path = sys_args.model_name_or_path

args = dict(
  model_name_or_path=model_name_or_path,
  adapter_name_or_path=all_adapter_name_or_path,           

  template=template,                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  # quantization_bit=8,                    # load 4-bit quantized model
  use_unsloth=False,                     # use UnslothAI's LoRA optimization for 2x faster generation
  max_length=2048,
  max_new_tokens=128,
  # repetition_penalty=1,
  # fp16=True

  do_sample=True,
  top_k=sys_args.top_k,
  top_p=sys_args.top_p,
  temperature=sys_args.temperature,
  
)
chat_model = ChatModel(args)

with open(data_file) as f:
    data = json.load(f)


import random
random.seed(42)
random.shuffle(data)
from tqdm import trange,tqdm


patterns = ['Language Fluency', 'Language Relevance', 'Role Language', 'Role Knowledge', 'Emotional Expression', 'Interactive Engagement']
# patterns = ['LF', 'LR', 'RL', 'RK', 'EE', 'IE']


weights = [0.9, 1.5, 0.96, 0.96, 0.84, 0.84]
assert sum(weights)==6
weights_dic =  dict(zip(patterns, weights))


def extract_numbers(data):
    results = {}
    for index, pattern in enumerate(patterns):
        regex = f'{pattern}:\s*(\d+)'
        match = re.search(regex, data)
        if match:
            # 将找到的数字存入字典，转换为整数
            results[pattern] = int(match.group(1))
        else:
            new_regex = r"(\d+)\D*(\d+)\D*(\d+)\D*(\d+)\D*(\d+)\D*(\d+)"
            new_match = re.search(new_regex, data)
            if new_match:
                results[pattern] = int(new_match.group(index+1))
            else:
                results[pattern] = 2
    return results

def average_dict_values(dict_list):
    totals = {}
    counts = {}
    
    for d in dict_list:
        for key, value in d.items():
            if key in totals:
                totals[key] += value
                counts[key] += 1
            else:
                totals[key] = value
                counts[key] = 1
    
    averages = {key: round(totals[key] / counts[key],2) for key in totals}
    
    return averages

def calcuate_MAE_score(gold_dic, pred_dic):
    if gold_dic.keys() != pred_dic.keys():
        return "Error: Dictionaries do not have the same keys."

    total_difference = 0
    total_weight_difference = 0
    gold_list = []
    pred_list = []
    score_dic = {}
    for key in gold_dic:
        difference = abs(gold_dic[key] - pred_dic[key])
        total_difference += difference
        total_weight_difference += weights_dic[key]*difference

        score_dic[key] = difference
        gold_list.append(gold_dic[key])
        pred_list.append(pred_dic[key])

    return total_weight_difference, total_difference, gold_list, pred_list, score_dic



total_res = []
total_dic = []
total_w_dic = []
for d in tqdm(data[:sys_args.test_num]):

    system = d['system']
    messages = []
    
    gold_tmp = []
    pred_tmp = []

    each_score = []
    each_weight_score = []

    each_dic = []
    each_w_dic = []
    for i in range(0, len(d['conversations'])-1, 2):
        c = d['conversations'][i]
        if c['from'] == 'dialogue': 
           messages.append({"role": "user", "content": c['value']})
        c_1 = d['conversations'][i+1]

        response = chat_model.chat(messages, system=system)[0].response_text
        messages.append({"role": "assistant", "content": response})
        # c_1['generated'] = response
        
        pred_dic = extract_numbers(response)
        each_dic.append(pred_dic)
        c_1['score'] = pred_dic

        if c_1["from"] =="scorer" and "value" in c_1.keys():
            c_1.pop('value')

        pred_w_dic = {}
        for key in pred_dic:
            pred_w_dic[key] =  pred_dic[key]*weights_dic[key]
        each_w_dic.append(pred_w_dic)

    total_res.append(d)
    total_dic.append(average_dict_values(each_dic))
    total_w_dic.append(average_dict_values(each_w_dic))
    # break



total_res_dic = average_dict_values(total_dic)
total_w_res_dic = average_dict_values(total_w_dic)
print("total_res:", total_res_dic)
print("total_w_res:", total_w_res_dic)

total_res_dic_sum = round(sum(total_res_dic.values())/len(total_res_dic),2)
total_w_res_dic_sum = round(sum(total_w_res_dic.values())/len(total_res_dic),2)


if not os.path.exists('res/'):
    os.makedirs('res/')

dir = 'res/'+sys_args.adapter_name_or_path+"_"+str(sys_args.ID)+"_w_"+str(total_w_res_dic_sum)+"\\"+str(total_res_dic_sum)+'.json'
res_file = 'res/'+sys_args.adapter_name_or_path+"_"+str(sys_args.ID)+"_w_"+str(total_w_res_dic_sum)+"\\"+str(total_res_dic_sum)+'.json'

with open(res_file,"w") as f:
    f.write("no weight res :\n")
    f.write(json.dumps(total_res_dic)+"\n")
    f.write("weight res :\n")
    f.write(json.dumps(total_w_res_dic)+"\n\n")

    for res in zip(total_res):
        # f.write("score: "+str(s)+"\n")
        # f.write("weight score: "+str(ws)+"\n")
        
        f.write(json.dumps(res)+"\n")

torch_gc()

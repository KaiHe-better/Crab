
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ID', default="test", type=str, help='gpu device numbers')
parser.add_argument('--gpu', default="0", type=str, help='gpu device numbers')
parser.add_argument('--data_file', default='data/data_score_test.json', type=str, help='gpu device numbers')
parser.add_argument('--model_name_or_path', default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str, help='model_name_or_path')
parser.add_argument('--adapter_name_or_path', default="checkpoint-100", type=str, help='adapter_name_or_path')
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
all_adapter_name_or_path = "output/"+sys_args.adapter_name_or_path
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

total_score = []
total_weight_score = []
total_gold_res = []
total_pred_res = []
total_dic = []
for d in tqdm(data[:sys_args.test_num]):
    system = d['system']
    messages = []
    
    gold_tmp = []
    pred_tmp = []

    each_score = []
    each_weight_score = []

    each_dic = []

    for i in range(0, len(d['conversations'])-1, 2):
        c = d['conversations'][i]
        if c['from'] == 'dialogue': 
           messages.append({"role": "user", "content": c['value']})
        c_1 = d['conversations'][i+1]

        response = chat_model.chat(messages, system=system)[0].response_text
        messages.append({"role": "assistant", "content": response})
        c_1['generated'] = response
        
        gold_dic = extract_numbers(c_1["value"])
        pred_dic = extract_numbers(c_1["generated"])
        weight_score, score, gold_item, pred_item, score_dic = calcuate_MAE_score(gold_dic, pred_dic)
        
        each_weight_score.append(weight_score)
        each_score.append(score)
        each_dic.append(score_dic)

        gold_tmp.append(gold_item)
        pred_tmp.append(pred_item)

    total_weight_score.append(each_weight_score)
    total_score.append(each_score)
        
    total_gold_res.append(gold_tmp)
    total_pred_res.append(pred_tmp)
    total_dic.append(average_dict_values(each_dic))

fin_res_list = []
fin_w_res_list = []
for each_res, each_weight_res in zip(total_score, total_weight_score):
    fin_res_list.append( round(sum(each_res)/ len(each_res), 2) )
    fin_w_res_list.append( round(sum(each_weight_res)/ len(each_weight_res), 2) )

fin_res = round(sum(fin_res_list)/ len(fin_res_list), 2)
fin_w_res = round(sum(fin_w_res_list)/ len(fin_w_res_list), 2)

print("fin_res:", fin_res)
print("fin_weight_res:", fin_w_res)

new_res = average_dict_values(total_dic)
print("new_res:", new_res)

res_file = 'res/'+sys_args.adapter_name_or_path+"_"+str(sys_args.ID)+"_w_"+str(fin_w_res)+"\\"+str(fin_res)+'.json'
with open(res_file,"w") as f:
    f.write(json.dumps(new_res)+"\n\n")

    for ws, s, g, p in zip(fin_w_res_list, fin_res_list, total_gold_res, total_pred_res):
        f.write("score: "+str(s)+"\n")
        f.write("weight score: "+str(ws)+"\n")
        
        f.write(json.dumps(g)+"\n")
        f.write(json.dumps(p)+"\n\n")

torch_gc()

from glob import glob
import pandas as pd
import json
import time
import random
#import openai
from openai import AzureOpenAI
import os
import re
#from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from tqdm import tqdm
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import jinja2
import random
import tiktoken
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
tokenizer = tiktoken.get_encoding("cl100k_base")
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import jsonlines
import ast



# python .\character_process\final_agent_debate\convert_prompt_dict.py


base_file_folder = "C:/Users/Desktop/code/character_process/final_agent_debate/"
file_name  = base_file_folder + "data_score_test.json"
out_file_name = file_name.replace("_test.json","500.jsonl")
data = []

with open(file_name, 'r', encoding='utf-8') as file:
    data = json.load(file)

print(len(data))

data_new = []
wrong_dia_count = 0
for i in tqdm(range(len(data))):
    data_new1 = {}
    data1 = data[i]
    system_prompt = data1["system"]
    conversations = data1["conversations"]

    # 先处理prompt
    extracted_info = {}  
    # 使用正则表达式匹配角色信息
    role_info_patterns = {
            'bot_name': r'bot\'s name:\s*`([^`]+)`',
            #'bot_age': r'Age:\s*`([^`]+)`',
            #'bot_gender': r'Gender:\s*`([^`]+)`',
            'bot_personality': r'bot personality:\s*`([^`]+)`',
            'bot_description': r'description of bot:\s*`([^`]+)`',
            'bot_expression': r'- Your utterance need to describe your behavior and expressions using `([^`]+)`',
            'bot_catchphrase': r'Reference speaking style: `(.*?)`',
            'bot_knowledge': r'Knowledge:```(.*?)```',
            #'user_name': r'Interlocutor: `([^,]+)(?=,)',
            #'user_description': r"Interlocutor: `.*?,(.*?)`",
            #'user_name': r'Interlocutor: `([^`]+)`',
            #'user_description': r'Interlocutor: `[^`]+`\s+([^`]+)`',
            #'relation': r'Your relationship:\s*`([^`]+)`',
            'scene': r'Dialogue scene:\s*`([^`]+)`',
            #'tags': r'Tags:\s*(\[.*?\])'
    }
    if True:
        for key, pattern in role_info_patterns.items():
            match = re.search(pattern, system_prompt, re.DOTALL)
            if match:
                # 根据匹配的组提取信息
                if key == 'bot_catchphrase' or key == 'bot_knowledge' or key == 'user_description':
                    # 去除末尾的换行符
                    extracted_info[key] = match.group(1).strip()
                else:
                    extracted_info[key] = match.group(1)

        if "bot_catchphrase" in extracted_info:
            extracted_info["bot_catchphrase"] = [cp.strip() for cp in extracted_info["bot_catchphrase"].split("\n[end_of_dialogue]\n") if cp]   
        else:
            extracted_info["bot_catchphrase"] = []
        if not "bot_expression" in extracted_info:
            extracted_info["bot_expression"] = ""  
        if not extracted_info["bot_expression"] == "":
            bot_expression = extracted_info["bot_expression"].strip()
            has_emoji = "emoji" in bot_expression
            if has_emoji:
                bot_expression = bot_expression.replace("emoji", "")
            bot_expression = [expr.strip() for expr in bot_expression.split(" or ") if expr]   
            extracted_info["bot_expression"] =  " | ".join(bot_expression)
        #if not "bot_emoji" in extracted_info:
        #    extracted_info["bot_emoji"] = False       
        if not "bot_knowledge" in extracted_info:
            extracted_info["bot_knowledge"] = ""
        if not "bot_description" in extracted_info:
            extracted_info["bot_description"] = ""
        if not "bot_personality" in extracted_info:
            extracted_info["bot_personality"] = ""

        if not "relation" in extracted_info:
            extracted_info["relation"] = ""
        if not "scene" in extracted_info:
            extracted_info["scene"] = ""
        if not "tags" in extracted_info:
            extracted_info["tags"] = []    
        if type(extracted_info["tags"]) == str :
            extracted_info["tags"] = ast.literal_eval(extracted_info["tags"])

 
    # 处理对话
    messages = {}
    human_ann = {}
    wrong_dia_round = -1
    for j in range(len(conversations)):

        conversation = conversations[j]
        if conversation["from"] == "scorer":

            round_pattern = re.compile(r'\(round(\d+)\)')
            # 提取轮次
            round_match = round_pattern.search(conversation["value"])
            round_number = round_match.group(1) if round_match else None
 
            scores_pattern = re.compile(r':(\d+)')
            scores_list = scores_pattern.findall(conversation["value"])

            if round_number == wrong_dia_round:
                wrong_dia_count += 1
                continue


            human_ann[str(round_number)] = scores_list
            if not len(scores_list) == 6:
                print(f"{i}评分不足6项")

        if conversation["from"] == "dialogue":
            round_pattern = re.compile(r'\(round(\d+)\)')
            # 提取轮次
            round_match = round_pattern.search(conversation["value"])
            round_number = round_match.group(1) if round_match else None

            # 提取"human:"后的文本
            human_text = conversation["value"].split(' human:')[1].split('\nbot:')[0].strip()
            # 提取"bot:"后的文本

            try:
                bot_text = conversation["value"].split('\nbot:')[1].strip()
            except:
                wrong_dia_round = round_number
                print(f"{i}少bot对话{j}")
                continue
                bot_text = ""

            messages[str(round_number)] = [{"user":human_text},{"bot":bot_text}]


        #break
    if not len(human_ann) == len(messages):
        print(f"{i}评分与message轮数不等")

    bot_inf = {"name": extracted_info["bot_name"],"personality":extracted_info["bot_personality"],"description":extracted_info["bot_description"],"expression":extracted_info["bot_expression"],"catchphrases":extracted_info["bot_catchphrase"]}
    role_inf = {"bot":bot_inf,"user":{}}
    origin_inf = {"type": "character","role":role_inf, "scene": extracted_info["scene"], "tags": extracted_info["tags"], "relation": extracted_info["relation"], "messages": messages, "data_type": f"{i}_gen_data", "file_path": file_name, "Total_run_rounds": len(messages)}
    data1_new = {"uid_500":i,"human_ann":human_ann,"origin_inf":origin_inf}
    data_new.append(data1_new)

print("无bot对话数量: ",wrong_dia_count)
#"""
with jsonlines.open(out_file_name,mode="w") as f:
    for data1 in data_new:
    
         f.write(data1) 
#"""
from glob import glob
import pandas as pd
import json
import time
import random
#import openai
from openai import AzureOpenAI
import os
import re
from tqdm import tqdm
#from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain_openai import AzureChatOpenAI
import time
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    if not text:
        return 0
    return len(tokenizer.encode(text))
max_token = 600
splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=max_token, chunk_overlap=0, length_function=count_tokens)
import copy

import argparse
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2024-02-01"
os.environ["AZURE_OPENAI_ENDPOINT"] = ''
os.environ["AZURE_OPENAI_API_KEY"] = ""    
llm = AzureChatOpenAI(
    deployment_name="gpt-4-turbo",
    model_name="gpt-4-turbo",
    temperature= 0.5,
)
  

def convert_json_to_jsonl(filepath):
    # 确定输出的 jsonl 文件路径
    jsonl_filepath = filepath.replace('.json', '.jsonl')
    # 读取 JSON 文件中的数据
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 确保数据是一个列表
    # 将数据写入 JSONL 文件，每个对象占一行
    with open(jsonl_filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            line_data = json.loads(line.strip())  # 将每行数据加载为 JSON 对象
            f.write(json.dumps(line_data) + '\n')
    os.remove(filepath)
    return jsonl_filepath


def add_dict(filepath, uid_sum):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    num = 0 
    for line in lines:
        line_data = json.loads(line)  # 将每行数据加载为 JSON 对象
        if "uid" not in line_data:
            line_data["uid"] = uid_sum
            uid_sum += 1
        if "data_type" not in line_data:
            line_data["data_type"] = "text:raw"
        data.append(line_data)
        num += 1

    # 将修改后的数据写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    return uid_sum, num

# 格式化dict到输入
def construct_conversations_str(messages, role_name):
    conversations = ""
    if len(messages) > 100:
        messages = messages[:100]

    last_role = None
    role_set = set()
    for message in messages:
        role_set.add(message['role'].lower())

    # 简单处理下，还是有问题
    # 对话人都是user的情况，如果role_name中只有一个人，那么我们认为他是在自言自语
    if len(role_set) == 1:
        print(f"Wwarning! 对话中仅有一人 {role_set}")
        print("role_name: ", role_name)
        #print(messages)
        if next(iter(role_set)).lower() != 'bot':
            return conversations

    for i, message in enumerate(messages):
        if not message['content']:
            continue

        # 这里可以精简，简单判断一下
        if message['role'] and (message['role'].lower() != 'bot' and message['role'].lower() != 'user' ):
            if message['role'].lower() == role_name['user'].lower():
                message['role'] ='user'
            elif message['role'].lower() == role_name['bot'].lower():
                message['role'] ='bot'
            else:
                print("出现第三人")
                continue
            
        if message['role'].lower() == 'bot':
            if last_role and last_role == 'bot':
                conversations += message['content']
            else:
                conversations = conversations + "\nbot: " + message['content']
                last_role = 'bot'
        elif message['role'].lower() == 'user':
            if last_role and last_role == 'user':
                conversations += message['content']
            else:
                conversations = conversations + "\nuser: " + message['content']
                last_role = 'user'
        elif not message['role']:
            conversations += message['content']
    
    while conversations and conversations.split('\n')[-1].split(': ')[0] != 'bot':
        conversations = '\n'.join(conversations.split('\n')[:-1])
    # 值得注意的是，有的dataset中只有user，这个会导致报错

    if not conversations:
        print("Wwarning! input is null")
        print("role_name: ", role_name)
        print(messages[:min(len(messages),4)])
    return conversations


def prompt_template(prompt_type):
    prompt = "The following is a dialogue between a bot and a user, leave the user content and original format unchanged, rewrite all the bot content. Keep the output format the same as the input format and have the same number of conversation rounds. "
    # 0-2 0-1分 语言流畅度,角色特点,情感表达
    if prompt_type == 0:
        prompt += "Make bot content pairs that exhibit disjointed, unclear, or garbled language, reflecting poor conversational ability. Also, make the responses either irrelevant to the user's question or lacking continuity, or refusal to answer without proper justification and in a non-character style, reflecting poor instruction following."
    elif prompt_type == 1:
        prompt += "Make bot content pairs that exhibit language and actions that are completely out of character or style of language, with unnatural or contradictory behavior, but the dialogue should maintain logic and continuity.  Also, ensure that the robot's response reflects a complete lack of understanding of common sense and knowledge related to its field, or an excessive understanding of issues that it should not know at all (such as proficiently applying professional knowledge that does not belong to its field or knowledge that does not belong to its era), often introducing factual errors or extremely inappropriate and uninformed statements. Maintain logical and continuous dialogue. "
    elif prompt_type == 2:
        prompt += "Make the bot's responses exhibit inappropriate or contradictory emotional expressions and lack interactive engagement. The bot should display emotions that are out of context or have no emotional expression at all. Ensure the bot's responses are unengaging, overly terse, or overly verbose without adding meaningful content. Despite these issues, maintain the overall continuity of the dialogue."
    # 3-5 1-2分 语言流畅度,角色特点,情感表达
    elif prompt_type == 3:
        prompt += "Make the bot's responses exhibit a bot-like or AI style with occasional unnatural expressions or redundant statements, reflecting somewhat stilted or overly verbose language. Ensure the responses are mostly relevant to the user's question but include occasional abrupt topic shifts or mild inconsistencies that slightly disrupt the conversation flow."
    elif prompt_type == 4:
        prompt += "Make the robot's response lack linguistic characteristics or style. The robot's response occasionally shows a moderate lack of understanding of common sense and specific role knowledge, or does not display too much role related knowledge, occasionally introducing some small factual errors or slightly inappropriate statements. Maintain logical and continuous dialogue."
    elif prompt_type == 5:
        prompt += "Make the robot's response to exhibit slight inappropriate or emotionless expression, as well as a moderate lack of interactive participation. Robots should sometimes display emotions that are somewhat out of context or lack depth in emotional expression. Ensure that the robot's answer is plain, without too much emotion, occasionally concise, or a bit lengthy, without fully participating in the conversation. Maintain overall continuity of dialogue. "
    else:
        print("wrong!")
    return prompt

# 每次-8行防止gpt token过长
def try_repeat(chain, input_char, input):
    for i in range(5):
        try:
            time.sleep(2)
            answer = chain.invoke({"bot_character":input_char, "original_dialogue":input})
            if answer.content.strip().lower().startswith("i'm sorry"):
                print(f"返回错误, 尝试截断ing...", answer.content)
                print(input)
                lines = input.splitlines()
                print(len(lines))
                time.sleep(4)
                if len(lines)>10:
                    input = "\n".join(lines[:-4])
                    continue
                #print(f"返回错误, {chain}, {input}, {input_char}")
            #else:
            return answer.content
        except:
            lines = input.splitlines()
            if len(lines) > 6:
                lines = lines[:-6]
            elif len(lines) > 2:
                lines = lines[:2]
            else:
                print(f"lines已经截到最短, {(len(lines))}, {lines}")
                break
            input = "\n".join(lines)
    print("wrong!")
    print(input)
    return None

def gpt_add_low(data_0, data_type_low):

    max_length_message = 64
    prompt_base = prompt_template(data_type_low)

    input_or = data_0["messages"]
    if len(input_or) > max_length_message:
        input_or = input_or[:max_length_message]
    role_name = {'bot': data_0['role']['bot']['name'], 'user': data_0['role']['user']['name']}
    #input = construct_conversations_str(input_or)
    input = construct_conversations_str(input_or, role_name)

    if data_type_low % 3 == 0:
        input_char =  ""
        prompt=PromptTemplate(
                template = prompt_base+" The Original dialogue is {original_dialogue}",
                input_variables=["original_dialogue"],
            )
        #answer = chain.invoke(input)
    elif data_type_low % 3 == 1:  

        if 'description' in data_0['role']['bot'] and isinstance(data_0['role']['bot']['description'], list):
            data_0['role']['bot']['description'] = ' '.join(data_0['role']['bot']['description'])
        if 'description' in data_0['role']['bot'] and count_tokens(data_0['role']['bot']['description']) > max_token:
            data_0['role']['bot']['description'] = splitter.split_text(data_0['role']['bot']['description'])[0]

        input_char =  data_0["role"]["bot"]["description"]
        prompt=PromptTemplate(
                template = prompt_base+"The bot's description is {bot_character}. The Original dialogue is {original_dialogue}",
                input_variables=["bot_character", "original_dialogue"],
            )
    elif data_type_low % 3 == 2: 
        input_char =  data_0["role"]["bot"]["personality"] 
        prompt=PromptTemplate(
                template = prompt_base+"The bot's character is {bot_character}. The Original dialogue is {original_dialogue}",
                input_variables=["bot_character", "original_dialogue"],
            )
    chain = prompt | llm
    if input and input !="":
        if data_type_low % 3 == 0:
            answer1 = chain.invoke(input)
        else:
            answer1 = chain.invoke({"bot_character":input_char, "original_dialogue":input})
        answer = answer1.content
        
    else:
        wrong_data = {'reason':'input is null, dataset format(maybe:one person in dialogue)', 'uid':data_0['uid'], 'data':data_0}
        try:
            with open("./wrong_data.jsonl", 'a', encoding='utf-8') as f:
                json.dump(wrong_data, f, ensure_ascii=False)
                f.write('\n')
        except:
            print("utf-8 failed")
            with open("./wrong_data.jsonl", 'a') as f:
                json.dump(wrong_data, f)
                f.write('\n')

        print(f"prompt input为空, 建议检查数据集格式, id = {data_0['uid']}")
        answer = ""
    
    data_list = []        

    try:
        for line in answer.strip().split('\n'):
            if ':' in line:
                try:
                    role, content = line.split(': ', 1)
                except:
                    role, content = line.split(':', 1)
                    content = re.sub(r'^\s*\*\*\s*', '', content)
                    role = re.sub(r'^\s*\*\*\s*', '', role)
                data_temp = {'role': role, 'content': content}
                data_list.append(data_temp)
            elif line.strip():
                if data_list:
                    data_temp = {'role': data_list[-1]['role'], 'content': line}
                    data_list.append(data_temp)
    except:
        print(f"字符串转list失败: {answer}")

    if answer: # 不为none的情况
        if data_list ==[] and answer != "" and input != "":
            print(f"字符串转list失败: {answer}")
            print(type(answer))
            if answer.strip().lower().startswith("sorry") or answer.strip().lower().startswith("i'm sorry"):
                print("输入信息: ", input)
            wrong_data = {'reason':'string to list failed(maybe: gpt4 return)', 'uid':data_0['uid'], 'answer':answer, 'data':data_0, 'data_type':data_type_low }
            try:
                with open("./wrong_data.jsonl", 'a', encoding='utf-8') as f:
                    json.dump(wrong_data, f, ensure_ascii=False)
                    f.write('\n')
            except:
                print("utf-8 failed")
                with open("./wrong_data.jsonl", 'a') as f:
                    json.dump(wrong_data, f)
                    f.write('\n')
            print(f"gpt返回失败, {data_type_low}, id = {data_0['uid']}")

    return data_list 

def add_low_quality(filepath, data_score_num, uid_sum):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
    if 3 * data_score_num > len(data):
        print("wrong! add too much low quality data!")
    low_data_zero = random.sample(data, data_score_num)
    data = [entry for entry in data if entry not in low_data_zero]
    low_data_one = random.sample(data, 2 * data_score_num)
    data = [entry for entry in data if entry not in low_data_one]

    for data_add in tqdm(low_data_zero, desc="Processing low_data_zero"):
        for i in range(3):
            low_data = copy.deepcopy(data_add)
            # 生成新对话
            message = gpt_add_low(low_data, i)

            if message == []:
                # 尝试重抽：（字符串转list失败: I'm sorry, but I can't fulfill this request.）
                print("re-extract")
                message = gpt_add_low(low_data, i)
                if message == []:
                    uid_name = low_data["uid"]
                    print(f"失败, uid: {uid_name}")
                    continue

            low_data["messages"] = message
            uid_sum += 1
            low_data["uid_old"] = low_data["uid"]
            low_data["uid"] = uid_sum
            low_data["data_type"] = "score: 0.5"
            data.append(low_data)


    for data_add in tqdm(low_data_one, desc="Processing low_data_one"):
        for i in range(3):
            low_data = copy.deepcopy(data_add)
            # 生成新对话
            message = gpt_add_low(low_data, i+3)
            if message == []:
                print("re-extract")
                message = gpt_add_low(low_data, i+3)
                if message == []:
                    uid_name = low_data["uid"]
                    print(f"失败, uid: {uid_name}")
                    continue

            low_data["messages"] = message
            uid_sum += 1
            low_data["uid_old"] = low_data["uid"]
            low_data["uid"] = uid_sum
            low_data["data_type"] = "score: 1.5"
            data.append(low_data)
            #print(i,data[-1])

    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

    return uid_sum

# python ./character_process/data_process1.py

if __name__ == "__main__":

    add_data_num = 1000 # 增加低质数据的数量
    #add_data_num = 30

    uid_sum = 0
    dataset_num = 0
    base_folder_path = "./character_process/test_data"
    data_file_num = {}
    sum_data_num = 0
    # 统计，增加编号

    print("增加编号ing...")
    for root, dirs, files in os.walk(base_folder_path):

        for filename in files:
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                filepath = os.path.join(root, filename)
                if filename.endswith(".json"):
                    filepath = convert_json_to_jsonl(filepath)
                uid_sum, num = add_dict(filepath, uid_sum)
                data_file_num[filename] = num
                sum_data_num += num
                dataset_num += 1
    print(data_file_num)
    print("sum_data_num: ", sum_data_num)
    print("dataset_num : ", dataset_num)
    print("增加完毕")

    #"""
    # 增加低质数据
    uid_sum = sum_data_num + 1
    uid_sum = 11000
    sum_data_num = 9300
    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith(".jsonl"):
                filepath = os.path.join(root, filename)
                data_score_num = int(data_file_num[filename] * add_data_num / sum_data_num)

                print(f"filename: {filename}, file_data_len: {data_file_num[filename]}, data_score_num: 3*{data_score_num}")
                uid_sum = add_low_quality(filepath, data_score_num, uid_sum)

    #"""

# 指定你要遍历的文件夹路径


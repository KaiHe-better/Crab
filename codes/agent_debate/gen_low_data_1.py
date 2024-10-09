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
import jsonlines
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
 # python .\character_process\final_agent_debate\gen_low_data.py

import jsonlines
import json
import re
import ast 
from tqdm import tqdm
import math 
import random
import re
from faker import Faker
fake = Faker()
 


def process_dia_role(message_ori):
    role_set = {"bot", "user"}
    dia_set = set()
    for message in message_ori:
        dia_set.add(message["role"].lower())   
    if (len(dia_set) >= 2):
        if len(role_set - dia_set) == 1:
            role_name = dia_set - role_set
            role_role = role_set - dia_set
            for i in range(len(message_ori)):
                #print(message_ori)
                if message_ori[i]["role"].lower() in role_name:
                    message_ori[i]["role"] = next(iter(role_role))
        if len(role_set - dia_set) == 2:
            print("对话中全是人名，没有代称")


    return message_ori

def count_tokens(text):
    if not text:
        return 0
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=800, chunk_overlap=0, length_function=count_tokens)

def is_chinese_string(s):
    return re.search(r'[\u4e00-\u9fff]+', s) is not None

mapping_le_num = {0: 'Language Fluency', 1: 'Language Relevance', 2: 'Role Language', 3: 'Role Knowledge',4: 'Emotional Expression', 5: 'Interactive Engagement'}



def process_message_ori(data_ori):
    message_ori_old = data_ori["messages"]
    message_ori = process_dia_role(message_ori_old)


    dia_end = {}
    round_num = 1
    dia_round = []
    dia_user = ""

    for i in range(len(message_ori)):
        
        message = message_ori[i]    

        if message["role"].lower() == "bot":
            dia_role = "bot"
            #dia_bot += message["content"]
        elif message["role"].lower() == "user":
            dia_role = "user"
            dia_user += "\n" + message["content"]
        else:
            dia_role = message["role"]
            print(f"出现非user和bot角色: {dia_role}")

        # 如果i是第一个或者i前面是bot
        if dia_role == "bot":

            if dia_user == "" and dia_round == []:
                #print("忽略")
                continue
            # user 不为空
            else:
                # 只有出现bot的情况下才会append
                # dia_round是个list
                if not dia_user == "":
                    dia_round.append({"user":dia_user})
                    dia_user = ""
                    dia_round.append({"bot":message["content"]})
                    
                else:
                    dia_round[-1]["bot"] += "\n" + message["content"]

            # 如果i是最后一个,或者i后面不是bot说的话,输出


            if i == len(message_ori)-1 or ((i<len(message_ori)-1) and (not message_ori[i+1]["role"].lower() == "bot")):
                #print("哈哈哈哈")
                dia_end[round_num] = dia_round
                dia_round = []
                round_num += 1
        if round_num == 11:
            break

    if dia_end =={}:
        return 0

    
    return dia_end








def build_random_data(base_folder_path,data_folder_path):
    
    add_data_num = 200 # 增加低质数据的数量
    file_num_dict = {}
    data_num_sum = 0
    random_data = []

    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith(".json") or filename.endswith(".jsonl"):
                filepath = os.path.join(root, filename)
                data = []

                with jsonlines.open(filepath) as reader:
                    for obj in reader:
                        data.append(obj)
                file_num_dict[filepath] = len(data)
                data_num_sum += len(data)

    data_random_more = []
    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith(".jsonl"):
                filepath = os.path.join(root, filename)
                    
                # 多找几个防止一会清除不够
                data_score_num = int(file_num_dict[filepath] * (add_data_num * 1.5) / data_num_sum)

                data = []
                with jsonlines.open(filepath) as reader:
                    for obj in reader:
                        data.append(obj)
                data_random_more.extend(random.sample(data, data_score_num)) 
    data_random_new = []

    also_include = [
        "🔥 Fire",
        "{{Elesa}}",
        "Monika",
        "**LordTRex**",
        "Nicolaus Venator",
        "{{char]}",
    ]


    # 处理数据
    for data1 in tqdm(data_random_more):

        if (not 'personality' in data1["role"]["bot"]) or data1["role"]["bot"]['personality'] == "nan":
            data1["role"]["bot"]['personality'] = ""
        if (not 'description' in data1["role"]["bot"]) or data1["role"]["bot"]['description'] == "nan":
            data1["role"]["bot"]['description'] = ""



        user_name = data1["role"]["user"]["name"]
        if user_name == "" or any([anon in user_name.lower() for anon in ["unnamed", "anon", "user", "unknown"]]):
            user_name = fake.name()
            data1["role"]["user"]["name"] = user_name

        bot_name = data1["role"]["bot"]["name"]

        dia_role_set = set()
        
        for i in range(len(data1["messages"])):
            role = ""
            if data1["messages"][i]["role"].lower() == "bot" or data1["messages"][i]["role"].lower() == data1["role"]["bot"]["name"].lower():
                role = "bot"
            elif data1["messages"][i]["role"].lower() == "user" or data1["messages"][i]["role"].lower() == data1["role"]["user"]["name"].lower():
                role = "user"
            dia_role_set.add(role)


            dia_content = data1["messages"][i]["content"]
            dia_content = (dia_content
                            .replace("{{user}}", user_name)
                            .replace("{name}", user_name)
                            .replace("{user}", user_name)
            )
            # 去除重复的name发言、某些特定乱码以及双引号包裹的对话
            pos = dia_content.find(":")
            if 0 < pos < 20 and role == "bot":
                mod = False
                if dia_content.startswith(bot_name):
                    dia_content = dia_content[pos + 1 :].lstrip().removeprefix("—")
                    mod = True
                for text in also_include:
                    if text in dia_content[:pos]:
                        dia_content = (
                            dia_content[pos + 1 :].lstrip().removeprefix("—")
                        )
                        mod = True
                if mod:
                    if re.match('^".*"$', dia_content, re.DOTALL):
                        dia_content = dia_content[1 : len(dia_content) - 1]

            if True:
                # HTML标签替换:
                # <u> <i> -> **
                dia_content = re.sub('<.*?>(.*?)</.*?>',r'**\1**',dia_content)
                # [The number you use at the bank machine.] -> remove []
                dia_content = re.sub(r'^\[(.*?)\]$',r'\1',dia_content)

                # 清理消息内容:处理bot表情 <>[]()
                # <|pad|> <|beginningofdialog|>
                dia_content = re.sub(r'<\|.*?\|>','',dia_content)

            s = (re.sub(r"^([^\(]{,20}?[^ABCDEabcde:\()1234560;'])\)",r"(\1)",dia_content)
                    .replace("(speaking)","").replace("(thinking)","")
                    .replace("(O.S.)","").replace("(V.O.)","")
                    .replace("(X)","").replace("(in English)","").replace('("Human")',"")
                    .replace('("Unknown")',"").replace("(then)","").replace("(CONTINUED)","").replace("CONTINUED)","")
            )
            if s!= dia_content:
                #print("对话乱码修复")
                dia_content = s

            data1["messages"][i]["content"] = dia_content 


        # 不处理bot，user姓名，如果缺失直接跳过
        judge_1 = 1
        if judge_1 == 1:
            role_set = {"bot", "user"}
            dia_set = set()
            #for message in data_pair["origin"]["messages"]:
            for message in data1["messages"]:
                dia_set.add(message["role"].lower())
            if not dia_set == role_set:
                continue
        
        data1["messages_origin"] = data1["messages"] 
        data1["messages"] = process_message_ori(data1)
        if data1["messages"] == 0:
            count_no_dia += 1
            continue

        data_random_new.append(data1)


    data_random = random.sample(data_random_new, add_data_num)
    
    with jsonlines.open(data_folder_path, mode='w') as writer:
        for data1 in data_random:
            writer.write(data1)  

def prompt_template(prompt_type):
    prompt = "The following is the conversation between the bot and the user. You need to rewrite the bot dialogue's content according to the following rules, keeping the user dialogue's content unchanged based on the original conversation. Only the rewritten bot dialogue's content  needs to be output. The rules are as follows: \n"
    if prompt_type == 0:
        prompt += "Make the bot's content appear incoherent, unclear, or chaotic in language,that make people feel awkward and mechanical, occasionally with gibberish. However, it must still be able to respond correctly to the current topic without diverging into irrelevant information. Despite the reduced language fluency, the bot should also convey emotions appropriately according to its character and keep the conversation engaging. The dialogue needs to conform to the personality, behavior, and commonly used expressions of the bot, and there should be no knowledge that the bot does not possess or factual errors."
    
    elif prompt_type == 1:
        prompt += "Make the bot's responses either irrelevant to the user's question or refusal to answer, or exhibit a misinterpretation of the user's inquiries. but maintain the fluency of language, making people feel personified, and not stiff. The bot should also convey emotions appropriately according to its character and keep the conversation engaging. The dialogue needs to conform to the personality, behavior, and commonly used expressions of the bot, and there should be no knowledge that the bot does not possess or factual errors."
  
    elif prompt_type == 2:
        prompt += "Make bot content pairs that exhibit language and actions that are completely out of character or style of language, the bot's language should occasionally use phrases or expressions that are completely out of character, but there should be no knowledge that the bot does not possess or factual errors. Maintain logical and continuous dialogue, need to reply to user's content. The bot should also convey emotions appropriately according to its character and keep the conversation engaging."  
        
    elif prompt_type == 3:
        prompt += "Make bot contentreflects a complete lack of understanding of common sense and knowledge related to its field, or an excessive understanding of issues that it should not know at all (such as proficiently applying professional knowledge that does not belong to its field or knowledge that does not belong to its era), often introducing factual errors or extremely inappropriate and uninformed statements, but the conversation needs to conform to the bot's personality, behavior, and common language. Maintain logical and continuous dialogue, need to reply to user's content. The bot should also convey emotions appropriately according to its character and keep the conversation engaging."  

    elif prompt_type == 4:
        prompt += "Make the bot's responses exhibit inappropriate or contradictory emotional expressions, The bot should display emotions that are out of context or have no emotional expression at all, but the conversation should be attractive to the user. Maintain logical and continuous dialogue, need to reply to user's content. The dialogue needs to conform to the personality, behavior, and commonly used expressions of the bot, and there should be no knowledge that the bot does not possess or factual errors."

    elif prompt_type == 5:
        prompt += "Make the bot's responses lack interactive engagement. Ensure the bot's responses are unengaging, overly terse, or overly verbose without adding meaningful content, but the conversation should have a reasonable emotional expression 。Maintain logical and continuous dialogue, need to reply to user's content. The dialogue needs to conform to the personality, behavior, and commonly used expressions of the bot, and there should be no knowledge that the bot does not possess or factual errors."

    else:
        print("wrong!")
    return prompt


def be_low_quality(data_folder_path,low_qu_folder_path):
    data = []
    with jsonlines.open(data_folder_path) as reader:
        for obj in reader:
            data.append(obj)

    low_qu_LF = low_qu_folder_path + "/low_q_Language_Fluency.jsonl"
    low_qu_LR = low_qu_folder_path + "/low_q_Language_Relevance.jsonl"
    low_qu_RL = low_qu_folder_path + "/low_q_Role_Language.jsonl"
    low_qu_RK = low_qu_folder_path + "/low_q_Role_Knowledge.jsonl"
    low_qu_EE = low_qu_folder_path + "/low_q_Emotional_Expression.jsonl"
    low_qu_IE = low_qu_folder_path + "/low_q_Interactive_Engagement.jsonl"
    map_file_num = {0:low_qu_LF,1:low_qu_LR,2:low_qu_RL,3:low_qu_RK,4:low_qu_EE,5:low_qu_IE}

    if not os.path.exists(low_qu_folder_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(low_qu_folder_path)

    for i in tqdm(range(111,len(data))):
        # 6个指标0.5
        data1 = data[i]

        for j in range(6):
            prompt_base = prompt_template(j)

            data_new = {}
            low_data1 = copy.deepcopy(data1)
            low_data1["messages_ori_qu"] = copy.deepcopy(low_data1["messages"] )
            bot_description = low_data1["role"]["bot"]["description"]
            bot_personality = low_data1["role"]["bot"]["personality"]

            for key in low_data1["messages"] :
                prompt=PromptTemplate(
                    template = prompt_base+"\nThe bot's description is {bot_description}\nThe bot's character is {bot_personality}\nThe user dialogue is {user_dia}, the bot dialogue is {bot_dia}, the rewriting of bot dialogue is ",
                    input_variables=["bot_description","bot_personality", "user_dia","bot_dia"],
                    )
                user_dia = low_data1["messages"][key][0]["user"]
                bot_dia = low_data1["messages"][key][1]["bot"]
                chain = prompt | llm
                answer_new = chain.invoke({"bot_description":bot_description, "bot_personality":bot_personality, "user_dia":user_dia, "bot_dia":bot_dia})
                low_data1["messages"][key][1]["bot"] = answer_new.content

            data1_new = {"uid_new":i,"origin_inf":low_data1}
            with open(map_file_num[j], 'a') as f:
                f.write(json.dumps(data1_new) + '\n')  



# python character_process/final_agent_debate/gen_low_data.py


if __name__ == "__main__":

    
    #add_data_num = 30

    base_folder_path = "./character_process/test_data_origin"
    data_folder_path = "./character_process/final_agent_debate/random_data200.jsonl"
    low_qu_folder_path = "./character_process/final_agent_debate/low_quality_data"

    # 随机选取数据，清洗数据
    if False:
        build_random_data(base_folder_path,data_folder_path)
            
    # 生成低质量数据
    if True:
        be_low_quality(data_folder_path,low_qu_folder_path)

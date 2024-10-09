import jsonlines
import json
import re
import ast 
from tqdm import tqdm

def back_to_type(in_file_name, out_file_name, origin_file_name):
    data = []
    with jsonlines.open(in_file_name) as f:
        for obj in f:
            data.append(obj)
    print(len(data))

    data_ori = []
    with jsonlines.open(origin_file_name) as f:
        for obj in f:
            data_ori.append(obj)
    print(len(data_ori))



    # 处理数据
    data_new = []
    for data1 in tqdm(data):
        #print(json.dumps(data1,indent=4))
        
        # 处理对话信息
        dia_inf = data1["system"]
        #print(dia_inf)
        extracted_info = {}  
        # 使用正则表达式匹配角色信息
        role_info_patterns = {
            'bot_name': r'Name:\s*`([^`]+)`',
            'bot_age': r'Age:\s*`([^`]+)`',
            'bot_gender': r'Gender:\s*`([^`]+)`',
            'bot_personality': r'Personality:\s*`([^`]+)`',
            'bot_description': r'Description:\s*`([^`]+)`',
            'bot_expression': r'- Your utterance need to describe your behavior and expressions using `([^`]+)`',
            'bot_catchphrase': r'Reference speaking style: ```(.*?)```',
            'bot_knowledge': r'Knowledge:```(.*?)```',
            'user_name': r'Interlocutor: `([^,]+)(?=,)',
            'user_description': r"Interlocutor: `.*?,(.*?)`",
            #'user_name': r'Interlocutor: `([^`]+)`',
            #'user_description': r'Interlocutor: `[^`]+`\s+([^`]+)`',
            'relation': r'Your relationship:\s*`([^`]+)`',
            'scene': r'Scene:\s*`([^`]+)`',
            'tags': r'Tags:\s*(\[.*?\])'
        }
        # 遍历所有模式并尝试匹配
        for key, pattern in role_info_patterns.items():
            match = re.search(pattern, dia_inf, re.DOTALL)
            if match:
                # 根据匹配的组提取信息
                if key == 'bot_catchphrase' or key == 'bot_knowledge' or key == 'user_description':
                    # 去除末尾的换行符
                    extracted_info[key] = match.group(1).strip()
                else:
                    extracted_info[key] = match.group(1)

        if "bot_catchphrase" in extracted_info:
            #print(extracted_info["bot_catchphrase"])
            extracted_info["bot_catchphrase"] = [cp.strip() for cp in extracted_info["bot_catchphrase"].split("\n[end_of_dialogue]\n") if cp]   
            #print(json.dumps(extracted_info["bot_catchphrase"],indent=2))
        if type(extracted_info["tags"]) == str:
            extracted_info["tags"] = ast.literal_eval(extracted_info["tags"])
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
            extracted_info["tags"] = ""    


        bot_inf = {"name": extracted_info["bot_name"], "age": extracted_info["bot_age"], "gender": extracted_info["bot_gender"],"personality":extracted_info["bot_personality"],"description":extracted_info["bot_description"],"expression":extracted_info["bot_expression"],"catchphrase":extracted_info["bot_catchphrase"],"bot_knowledge":extracted_info["bot_knowledge"]}
        user_inf =  {"name": extracted_info["user_name"],"description":extracted_info["user_description"]}
        
        #print(json.dumps(extracted_info,indent=4))
        #print(type(extracted_info["tags"]))
        #print(dia_inf)
        
        # 处理对话

        
        messages = {}
        conversations = data1["conversations"]
        
        # 总数据
        dia_gold = {}
        dia_gen = {}
        # 轮数
        round_num = 1
        # 单轮数据
        dia_round_gold = []
        dia_round_gen = []

        for i in range(len(conversations)):
            conversation = conversations[i]
            dia1 = {}
            dia2 = {}
            if conversation["from"].lower() == "gpt":
                dia_role = "bot"
                dia2[dia_role] = conversation["generated"]
            elif conversation["from"].lower() == "human":
                dia_role = "user"
            dia1[dia_role] = conversation["value"]

            #print(dia1)
            if dia_role == "user":
                dia_round_gold.append(dia1)
                dia_round_gen.append(dia1)

            # 如果i是第一个或者i前面是bot
            if dia_role == "bot":
                if dia_round_gold == []:
                    continue
                else:
                    dia_round_gold.append(dia1)
                    dia_round_gen.append(dia2)
                # 如果i是最后一个,或者i后面不是bot说的话,输出
                if i == len(conversations)-1 or ((i<len(conversations)-1) and (conversations[i+1]["from"].lower() == "human")):
                    dia_gold[round_num] = dia_round_gold
                    dia_gen[round_num] = dia_round_gen
                    dia_round_gold = []
                    dia_round_gen = []
                    round_num += 1

        #print("conversations:\n",json.dumps(conversations,indent=4))
        #print("dia_gold:\n",json.dumps(dia_gold,indent=4))
        #print("dia_gen:\n",json.dumps(dia_gen,indent=4))
        role_inf = {"bot":bot_inf, "user":user_inf}
        file_parts = os.path.basename(in_file_name).split('_')
        file_desired_part = file_parts[1] + '_' + file_parts[2] if len(os.path.basename(in_file_name).split('_')) > 3 else in_file_name

        origin_inf = {"type": "character","role":role_inf, "scene": extracted_info["scene"], "tags": extracted_info["tags"], "relation": extracted_info["relation"], "messages": dia_gen,"message_origin":dia_gold, "data_type": f"{file_desired_part}_gen_data", "file_path": in_file_name, "Total_run_rounds": len(dia_gold)}

        if "uid"in data1:
            dia_uid = data1["uid"]
            uid_to_human_ann_list = [item for item in data_ori if item.get("uid") == dia_uid]
            if not uid_to_human_ann_list == []:
                uid_to_human_ann = uid_to_human_ann_list[0]["human_ann"]
            else:
                print(f"有值为空,检查uid = {dia_uid}")
                uid_to_human_ann = {}

        data1_new = {"human_ann":uid_to_human_ann,"origin_inf":origin_inf,"uid":dia_uid}
        data_new.append(data1_new)
        #print("data1_new:\n",json.dumps(data1_new,indent=4))

        # break



    print(f"Origin Samples: {len(data_new)}")
    sharegpt_dataset = []
    #for sample in tqdm(data_new):
    for sample in data_new:

        data1_samp = sample["origin_inf"]
        # 用于打分对话模型的prompt
        messages=data1_samp["messages"]
        
        sample["origin_inf"]["role"]["bot"]["catchphrases"] = sample["origin_inf"]["role"]["bot"]["catchphrase"] 
        #print(sample["origin_inf"]["role"]["bot"]["catchphrase"])    
        #print(len(sample["origin_inf"]["role"]["bot"]["catchphrase"]))
        del sample["origin_inf"]["role"]["bot"]["catchphrase"]

        sample["origin_inf"]["messages"] = messages
        sample["origin_inf"] = data1_samp
        sharegpt_dataset.append(sample)
        #break
    #"""
    print(f"Get Samples: {len(sharegpt_dataset)}")

    with jsonlines.open(out_file_name,mode="w") as f:
        for sample in sharegpt_dataset:
            f.write(sample)
    

import os
data_folder = "../tx/test_gen_0802_llama_baseline"


base_folder = "./"
in_folder = base_folder + data_folder
out_folder = base_folder + data_folder + "/to_ann"
origin_file_path = base_folder + "our_ann_3.3.jsonl"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

#print(out_folder)
for filename in os.listdir(in_folder):
    if filename.endswith('.jsonl'):
        in_file_path = os.path.join(in_folder, filename)
        out_file_path = os.path.join(out_folder, filename)
        
        #print(in_file_path)
        #print(out_file_path)
        back_to_type(in_file_path,out_file_path,origin_file_path)
        # break

    # python /home/py/characterLLM/scoreLLM2-main/data/tx/0_train_gen2ann_data.py
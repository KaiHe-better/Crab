import jinja2
import random
import tiktoken
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
tokenizer = tiktoken.get_encoding("cl100k_base")
from transformers import AutoTokenizer, AutoModelForCausalLM

def count_tokens(text):
    if not text:
        return 0
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=800, chunk_overlap=0, length_function=count_tokens)

mapping_le_num = {0: 'Language Fluency', 1: 'Language Relevance', 2: 'Role Language', 3: 'Role Knowledge',4: 'Emotional Expression', 5: 'Interactive Engagement'}

roleplaying_template = ''' # Please score the round of dialogue in the following 6 targets.

## Targets:
1.Language Fluency. This score evaluates the fluency and naturalness of the language, making the text feel organic, lifelike, and not rigid or stilted. The focus here is solely on the overall smoothness and flow of the language, without considering the specific content. The goal is to evaluate how natural and conversational the language sounds, irrespective of the grammatical correctness.
2.Language Relevance. This score evaluates how well the bot responds to the current topic, staying focused and relevant without introducing irrelevant information. The key consideration is whether the bot's response correctly addresses the specific instructions or questions posed, regardless of the content or quality of the response itself.
3.Role Language. This score evaluates how well the language used by each character in the dialogue matches their established personality and traits. The focus is on whether the characters speak in a style consistent with their individual personalities, creating a natural and authentic conversation. This rating considers only the overall language style, not the content or accuracy of the responses.
4.Role Knowledge. This score evaluates the level of understanding and using of common sense (basic knowledge) and role knowledge (as well as related background) by the character. If a character speaks against what they are supposed to know, they should be scored low。
5.Emotional Expression. This score evaluates how well the character's emotional responses, including expressions of empathy and emotional intelligence, align with their established personality and the context of the dialogue. 
6.Interactive Engagement. This score evaluates how engaging and motivating the character's dialogue is, encouraging the user to continue the conversation. The focus is on the overall conversational flow and interactivity, without considering the use of specialized vocabulary or any mismatches in communication styles. If the bot ends the dialogue with a question, it should receive a high score.

## Evaluate only the content of the bot's responses in the conversation. Provide six scores, separated by spaces, based on the following criteria:
0 - Negative, poor performance
1 - Dialogue does not reflect the indicator or does not quite meet the standards
2 - More in line with standards but still has some defects
3 - Perfectly meets the criteria

## The information of the bot is as follows:
bot's name: `{{bot.name}}`
bot personality: `{{bot.personality}}`
description of bot: `{{bot.description}}`
catchphrase:
{{cp}}
Dialogue scene: `{{scene}}`

# Please provide six scores (with a range of 0-3) for `{{bot.name}}` in the following dialogue based on the above information.
'''


def is_chinese_string(s):
    return re.search(r'[\u4e00-\u9fff]+', s) is not None

def construct_roleplaying_prompt(role):
    #print(role['role']['bot']['catchphrases'])
    #print(len(role['role']['bot']['catchphrases']))
    if 'knowledge' in role['role']['bot'] and isinstance(role['role']['bot']['knowledge'], list):
        role['role']['bot']['knowledge'] = ' '.join(role['role']['bot']['knowledge'])
    if 'knowledge' in role['role']['bot'] and count_tokens(role['role']['bot']['knowledge']) > 800:
        role['role']['bot']['knowledge'] = splitter.split_text(role['role']['bot']['knowledge'])[0]
    if 'description' in role['role']['bot'] and isinstance(role['role']['bot']['description'], list):
        role['role']['bot']['description'] = ' '.join(role['role']['bot']['description'])
    if 'description' in role['role']['bot'] and count_tokens(role['role']['bot']['description']) > 800:
        role['role']['bot']['description'] = splitter.split_text(role['role']['bot']['description'])[0]
        
    #print(role['role']['bot'])
    assert 'catchphrases' in role['role']['bot'] and isinstance(role['role']['bot']['catchphrases'], list)
    
    if 'catchphrases' in role['role']['bot'] and len(role['role']['bot']['catchphrases']) > 2:
        role['role']['bot']['catchphrases'] = random.sample(role['role']['bot']['catchphrases'], 2)
    
    

    # 简单去除包含中文字段
    if is_chinese_string(role['role']['bot']['description']):
        #print(bot_description)
        role['role']['bot']['description'] = ""
    if is_chinese_string(role['role']['bot']['personality']):
        #print(bot_personality)
        role['role']['bot']['personality']= ""
    if is_chinese_string(role['scene']):
        #print(dia_scene)
        role['scene'] = ""
        #chinese_field_num += 1

    template = jinja2.Template(roleplaying_template, trim_blocks=True, lstrip_blocks=True)
    has_emoji = role['role']['bot'].get('emoji', False)
    has_expression = bool(role['role']['bot'].get('expression', ""))
    #print(type(role['role']['bot']['catchphrases']))
    #print(role['role']['bot']['catchphrases'])

    template = jinja2.Template(roleplaying_template, trim_blocks=True, lstrip_blocks=True)
    has_emoji = role['role']['bot'].get('emoji', False)
    #print("type",type(has_emoji),has_emoji)
    if type(has_emoji) == str:
        print("?")
        if has_emoji.strip().lower() == "false":
            has_emoji = False
        if has_emoji.strip().lower() == "true":
            has_emoji = True
    bot_exp = role['role']['bot'].get('expression', "")
    #print("bot_exp: ",bot_exp)
    #print("has_emoji: ",has_emoji)
    bot_expressions = [f"`{e}`" for e in bot_exp.split(" | ") if e]
    if has_emoji:
        #print(has_emoji)
        bot_expressions.append("emoji")
    bot_expression = " or ".join(bot_expressions)



    text = template.render(
        bot=role['role']['bot'],
        user=role['role']['user'],
        cp="".join([t+"\n[end_of_dialogue]\n" for t in role['role']['bot']["catchphrases"]][:2]),
        relation=role.get('relation', ""),
        scene=role.get('scene', ""),
        tags=role.get('tags', []),
        bot_expression = bot_expression,
        )
    #print(text,"\n\n\n")
    return text

def construct_system_prompt(role):
    task_type = role['type']
    if task_type == 'character':
        return construct_roleplaying_prompt(role)
    # elif task_type == 'text_adventure':
    #     return construct_text_adventure_prompt(role)
    # elif task_type == 'character-numeric':
    #     return construct_roleplaying_with_numerical_prompt(role)
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    
def construct_dia_ann_pair(role_inf,human_ann,max_dia_token,uid):

    # 评分是否为空，如果为空，评分是测试数据
    if human_ann == {}:
        if_test = 1
    else:
        if_test = 0

    messages = role_inf.get('messages', [])

    if len(messages) > 100:
        messages = messages[:100]
    if len(human_ann) > 100:
        human_ann = human_ann[:100]
    #print(messages)
    sorted_messages = sorted([key for key in messages.keys()], key=lambda x: int(x))
    sorted_answer = sorted([key for key in human_ann.keys()], key=lambda x: int(x))
    if if_test==0 and(not sorted_messages == sorted_answer):
        if len(sorted_answer) > len(sorted_messages):
            sorted_answer = sorted_answer[:len(sorted_messages)]
            #print(f"len(messages){len(messages)}<len(answer){len(human_ann)},uid={uid},Truncated")
        else:
            print(f"warning, len(messages)>len(answer),uid={uid}")
    surplus_token_1 = max_dia_token

    dia_and_ann_list = []
    dia_ann_pair = []

    for k in sorted_messages:
        dia_value = f"(round{k}) "
        for dia1 in messages[k]:
            for key1 in dia1:
                if key1 =="user":
                    dia_value += f"human: {dia1[key1]} \n"
                else:
                    dia_value += f"bot: {dia1[key1]} \n"

        ans_value = f"(round{k}) "
        if if_test == 0:
            if not k in human_ann:
                # 缺少某轮数据，全部置为2
                print(f"uid {uid} lack {k} round dialogue") 
                human_ann[k] = ["2","2","2","2","2","2"]
            try:
                for i in range(len(human_ann[k])):
                    ans_value += mapping_le_num[i] + ":" + human_ann[k][i]
                    if not i == len(human_ann[k])-1:
                        ans_value += ", "
            except:
                print(json.dumps(human_ann,indent=2))
                print(f"human_ann[k],{uid}")
                print(json.dumps(human_ann[k],indent=2))

        input_ids_dia1 = tokenizer_llama.encode(dia_value, return_tensors='pt')
        token_length_dia1 = len(input_ids_dia1[0])

        if (surplus_token_1 - token_length_dia1) <= 0:
            dia_and_ann_list.append(dia_ann_pair)
            dia_ann_pair = []
            surplus_token_1 = max_dia_token

        dia_ann_pair.append({"from": "dialogue","value": dia_value})
        dia_ann_pair.append({"from": "scorer","value": ans_value})
        surplus_token_1 -= token_length_dia1
    dia_and_ann_list.append(dia_ann_pair)

    #print(messages)
    #print(human_ann)
    # if len(conversations) == 0:
    #     print(messages)
    # print(len(conversations))
    return dia_and_ann_list



sharegpt_dataset = []

def convert_data_format_to_sharegpt(sharegpt_dataset, role_inf,human_ann,uid):
    #tokenizer_llama = AutoTokenizer.from_pretrained(model_id)

    if True:
    #try:
        #print(role_inf)
        system = construct_system_prompt(role_inf)
        #print(system)
    #except:
    #    return None
    
    input_ids_prompt = tokenizer_llama.encode(system, return_tensors='pt')
    token_length_prompt = len(input_ids_prompt[0])
    #print(system)
    #print(token_length_prompt)
    max_dia_token = max_token - token_length_prompt
    dia_and_ann_list = construct_dia_ann_pair(role_inf,human_ann,max_dia_token,uid)
    #print(json.dumps(dia_and_ann_list,indent=4))
    for dia_and_ann in dia_and_ann_list:
        #print(json.dumps(dia_and_ann,indent=4))
        converted_data = {
            "uid": uid,
            "system": system,
            "conversations": dia_and_ann,
            }
        #print(json.dumps(converted_data,indent=4))
        if converted_data and len(converted_data["conversations"]) > 0:
            #print(converted_data)
            sharegpt_dataset.append(converted_data)


def use_tx_data_convert(in_file_name, out_file_name):
    data = []
    with jsonlines.open(in_file_name) as reader:
        for obj in reader:
            data.append(obj)

    print(f"Origin Samples: {len(data)}")
    sharegpt_dataset = []
    for data1 in tqdm(data):
        role_inf = data1["origin_inf"]
        human_ann = data1["human_ann"]
        uid = data1.get("uid", "")
        
        convert_data_format_to_sharegpt(sharegpt_dataset, role_inf, human_ann,uid)

        #print(json.dumps(sharegpt_dataset,indent=2))
        # break

    print(f"Get Samples: {len(sharegpt_dataset)}")

    #"""
    with open(out_file_name, "w", encoding="utf-8") as f:
        json.dump(sharegpt_dataset, f, indent=4, ensure_ascii=False)  
    #"""

#python /home/py/characterLLM/scoreLLM2-main/data/tx/1_convert_character_data_to_train.py
if __name__ == "__main__":
    import json
    import jsonlines
    from tqdm import tqdm


    max_token = 2030

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_id = "/home/py/LLM_models/llama3/llama3.1-8b-instruct"
    tokenizer_llama = AutoTokenizer.from_pretrained(model_id)
        
    data_type = 1
    # 是否是生成数据，生成的数据带有golden数据和生成数据两个值
    #if_gen_data = 1

    # 单文件转换做训练数据用
    if data_type == 0:
        file_num = 3.3
        base_file_name = "./"
        in_file_name = base_file_name + f"our_ann_{file_num}.jsonl"
        out_file_name = "data_score.json"
        use_tx_data_convert(in_file_name,out_file_name)

    # 多文件转换做测试数据用（使用生成的对话）
    elif data_type == 1:
        in_folder = "../tx/test_gen_0802/to_ann"
        out_folder = in_folder.replace("/to_ann", "/to_test")
        # out_folder = "../"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for filename in os.listdir(in_folder):
            if filename.endswith('.jsonl'):
                in_file_path = os.path.join(in_folder, filename)
                out_file_path = os.path.join(out_folder, filename.replace(".jsonl", ".json"))
                use_tx_data_convert(in_file_path,out_file_path)
            # break
    #"""

    # python /home/py/characterLLM/scoreLLM2-main/data/tx/1_convert_character_data_to_train.py


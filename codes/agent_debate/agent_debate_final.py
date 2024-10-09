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

if True:
    from langchain_openai import AzureChatOpenAI
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2024-02-01"
    os.environ["AZURE_OPENAI_ENDPOINT"] = ''
    os.environ["AZURE_OPENAI_API_KEY"] = ""    
    llm_agent1 = AzureChatOpenAI(
        deployment_name="gpt-4-turbo",
        model_name="gpt-4-turbo",
        #temperature= 0.5,
        temperature= 0.5,
    )

    llm_agent2 = AzureChatOpenAI(
        deployment_name="gpt-4-turbo",
        model_name="gpt-4-turbo",
        temperature= 0.5,
    )    
    llm_agent_end = AzureChatOpenAI(
        deployment_name="gpt-4-turbo",
        model_name="gpt-4-turbo",
        temperature= 0.5,
    )    
else:

    from langchain_openai import ChatOpenAI
    #from langchain.llms import OpenAI
    #from langchain_community.llms import OpenAI


    key = "."

    key_bytes = key.encode()
    os.environ["OPENAI_API_KEY"] = key_bytes.decode('utf-8')
    llm_agent1 = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        #model_name="gpt-4-turbo",
        temperature=0
    )
    llm_agent_end = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        #model_name="gpt-4-turbo",
        temperature=0
    )
print(llm_agent1)

import logging
from logging.handlers import RotatingFileHandler

# 配置日志记录
log_file = './character_process/agent_debate_zongtipingfen.log'
log_format = '%(asctime)s %(levelname)s: %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# 创建 RotatingFileHandler
file_handler = RotatingFileHandler(log_file, mode='w', backupCount=0)
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 创建 StreamHandler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)

parser = argparse.ArgumentParser()
# 必须更改  --name "Fifty Shades of Grey" 
parser.add_argument('--test_data_path', type=str, default='./character_process/out_test_data/harry_potter_test.jsonl', help='file name')

parser.add_argument('--base_folder_path', type=str, default='./character_process/out_test_data', help='file path')
parser.add_argument('--score_folder_path', type=str, default='./character_process/score_data', help='源文件评分后')
parser.add_argument('--wrong_folder_path', type=str, default='./character_process/wrong_GPT_score.jsonl', help='GPT评分失败')


parser.add_argument('--rounds', type=int, default='2', help='debate rounds')
parser.add_argument('--agents_num', type=int, default='2', help='agents num')
parser.add_argument('--if_reason', type=bool, default=False, help='是否分析原因')
parser.add_argument('--if_test', type=bool, default=True, help='是否是最终测试，如果是最终测试不输出中间结果')
parser.add_argument('--re_evaluate', type=bool, default=False, help='对于部分数据重新评测')

args = parser.parse_args()

prompt_base = """# Please score the round of dialogue in the following 6 areas.

## Score areas:
1.Language Fluency. This score evaluates the fluency and naturalness of the language, making the text feel organic, lifelike, and not rigid or stilted. The focus here is solely on the overall smoothness and flow of the language, without considering the specific content. The goal is to evaluate how natural and conversational the language sounds, irrespective of the grammatical correctness.
2.Language Relevance. This score evaluates how well the bot responds to the current topic, staying focused and relevant without introducing irrelevant information. The key consideration is whether the bot's response correctly addresses the specific instructions or questions posed, regardless of the content or quality of the response itself.
3.Role Language. This score evaluates how well the language used by each character in the dialogue matches their established personality and traits. The focus is on whether the characters speak in a style consistent with their individual personalities, creating a natural and authentic conversation. This rating considers only the overall language style, not the content or accuracy of the responses.
4.Role Knowledge. This score evaluates the level of understanding and using of common sense (basic knowledge) and role knowledge (as well as related background) by the character. If a character speaks against what they are supposed to know, they should be scored low.
5.Emotional Expression. This score evaluates how well the character's emotional responses, including expressions of empathy and emotional intelligence, align with their established personality and the context of the dialogue. 
6.Interactive Engagement. This score evaluates how engaging and motivating the character's dialogue is, encouraging the user to continue the conversation. The focus is on the overall conversational flow and interactivity, without considering the use of specialized vocabulary or any mismatches in communication styles. If the bot ends the dialogue with a question, it should receive a high score.

## Evaluate only the content of the bot's responses in the conversation. Provide six scores, separated by spaces, based on the following criteria:
0 - Negative, poor performance
1 - Dialogue does not reflect the indicator or does not quite meet the standards
2 - More in line with standards but still has some defects
3 - Perfectly meets the criteria
## The information of the bot is as follows:
"""

prompt_base_template = """
bot's name: `{{bot.name}}`
bot personality: `{{bot.personality}}`
description of bot: `{{bot.description}}`
Reference speaking style:
{{catchphrase}}
Dialogue scene: `{{scene}}`

# Please provide six scores (with a range of 0-3) for `{{bot.name}}` in the following dialogue based on the above information.
You should output score for each round, and only output score once per round (round n)
"""


def count_tokens(text):
    if not text:
        return 0
    return len(tokenizer.encode(text))

splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=800, chunk_overlap=0, length_function=count_tokens)

def is_chinese_string(s):
    return re.search(r'[\u4e00-\u9fff]+', s) is not None

mapping_le_num = {0: 'Language Fluency', 1: 'Language Relevance', 2: 'Role Language', 3: 'Role Knowledge',4: 'Emotional Expression', 5: 'Interactive Engagement'}

def construct_roleplaying_prompt(role):

    if 'knowledge' in role['role']['bot'] and isinstance(role['role']['bot']['knowledge'], list):
        role['role']['bot']['knowledge'] = ' '.join(role['role']['bot']['knowledge'])
    if 'knowledge' in role['role']['bot'] and count_tokens(role['role']['bot']['knowledge']) > 800:
        role['role']['bot']['knowledge'] = splitter.split_text(role['role']['bot']['knowledge'])[0]
    if 'description' in role['role']['bot'] and isinstance(role['role']['bot']['description'], list):
        role['role']['bot']['description'] = ' '.join(role['role']['bot']['description'])
    if 'description' in role['role']['bot'] and count_tokens(role['role']['bot']['description']) > 800:
        role['role']['bot']['description'] = splitter.split_text(role['role']['bot']['description'])[0]

    assert 'catchphrases' in role['role']['bot'] and isinstance(role['role']['bot']['catchphrases'], list)
    if 'catchphrases' in role['role']['bot'] and len(role['role']['bot']['catchphrases']) > 2:
        role['role']['bot']['catchphrases'] = random.sample(role['role']['bot']['catchphrases'], 2)

    # 简单去除包含中文字段
    if is_chinese_string(role['role']['bot']['description']):
        role['role']['bot']['description'] = ""
    if is_chinese_string(role['role']['bot']['personality']):
        role['role']['bot']['personality']= ""
    if (not "scene" in role) or (is_chinese_string(role['scene'])):
        role['scene'] = ""
        #chinese_field_num += 1

    template = jinja2.Template(prompt_base_template, trim_blocks=True, lstrip_blocks=True)
    has_emoji = role['role']['bot'].get('emoji', False)
    has_expression = bool(role['role']['bot'].get('expression', ""))


    template = jinja2.Template(prompt_base_template, trim_blocks=True, lstrip_blocks=True)
    has_emoji = role['role']['bot'].get('emoji', False)
    if type(has_emoji) == str:
        print("?")
        if has_emoji.strip().lower() == "false":
            has_emoji = False
        if has_emoji.strip().lower() == "true":
            has_emoji = True
    bot_exp = role['role']['bot'].get('expression', "")
    bot_expressions = [f"`{e}`" for e in bot_exp.split(" | ") if e]
    if has_emoji:
        bot_expressions.append("emoji")
    bot_expression = " or ".join(bot_expressions)



    text = template.render(
        bot=role['role']['bot'],
        user=role['role']['user'],
        cp="".join([t+"\n[end_of_dialogue]\n" for t in role['role']['bot']["catchphrases"]][:2]),
        relation=role.get('relation', ""),
        scene=role.get('scene', ""),
        tags=role.get('tags', []),
        bot_expression = bot_expression
        )
    return text



# python ./text_multiagent.py

prompt=PromptTemplate(
        template="请问，{country}的首都是哪里 ?",
        input_variables=["country"],
    )


# 格式化输入
# 只评价bot
def agent_input_pd(data):

    bot_name = data["role"]["bot"]["name"]
    user_name = data["role"]["user"]["name"]
    bot_personality = data["role"]["bot"]["personality"]
    bot_description = data["role"]["bot"]["description"]
    dia_scene = data["scene"]
    roles_relation = data["relation"]
    dia_tags = data["tags"]
    dialogue_message = data["messages"]

    dialogue_input = ""
    dialogue_input += "bot's name: {}\nbot personality: {}\ndescription of bot: {}\nDialogue scene: {}\nrelation between roles: {}\ndialogue:\n".format(bot_name, bot_personality, bot_description, dia_scene, roles_relation)
    dia_bot_num = 0
    dia_bot_word = 0
    for message_1 in dialogue_message:
        if message_1["role"] == "bot":
            dialogue_input += bot_name
            dia_bot_num += 1
            dia_bot_word += len(message_1["content"])
        else:
            dialogue_input += user_name
        dialogue_input += ": {}\n".format(message_1["content"])
    
    bot_avg_word = dia_bot_word / dia_bot_num                     
    dialogue_input += f"Please provide six ratings for \"{bot_name}\" dialogue(range float 0-3):"
    if args.if_reason:
        dialogue_input = dialogue_input + "Please provide specific reasons"

    # prompt 输入
    return bot_avg_word, dialogue_input, bot_name


# gpt4 评分
def agent_response(input):

    system_prompt = prompt_base 
    q_example = """###
    bot's name: `Hagrid`
    description of bot: `Hagrid: Hagrid is a half-giant wizard, who is surprisingly tall and full of beards. He works at Hogwarts as the Key Guardian and Hunting Ground Guard. He is a loyal friend and mentor to Harry.`

    # Please provide six scores (with a range of 0-3) for `Celestia` in the following dialogue based on the above information.
    (round1)
    Harry: "What? What?"
    Hagrid: "He wants payin' fer deliverin' the paper. Look in the pockets. Give him five Knuts,"
    (round2)
    Harry: "Knuts?"
    Hagrid: "The little bronze ones. Best be off, Harry, lots ter do today, gotta get up ter London an' buy all yer stuff fer school."
    (round3)
    Harry: "Wizards have banks?"
    Hagrid: "Just the one."
    """

    a_example = """
    (round1) Language Fluency:3, Language Relevance:3, Role Language:2, Role Knowledge:3, Emotional Expression:1, Interactive Engagement:2
    (round2) Language Fluency:3, Language Relevance:3, Role Language:1, Role Knowledge:3, Emotional Expression:2, Interactive Engagement:3
    (round3) Language Fluency:2, Language Relevance:3, Role Language:2, Role Knowledge:3, Emotional Expression:2, Interactive Engagement:2
    """


    if args.if_reason:
        system_prompt = system_prompt + "Please provide specific reasons."
    system_prompt = system_prompt + "The output format is as follows: round1: a: 3 b: 3 c: 2.5 d: 2.5 e: 3 f: 2.5 \nround2: a: 2 b: 1.2 c: 1.4 d: 2.8 e: 1.5 f: 1.0 (round represents the number of rounds of dialogue for the bot, followed by the rating for that round of conversation)"


    
    input_gpt = input[0]["content"]

    if len(input) > 2:
        previous_ans = input[-2]["content"]
        other_ans = input[-1]["content"]
        input_gpt += f"\nYour previous answer was {previous_ans}\n{other_ans}" 

        # 这是prompt的输入

    judge_a_a = 0   # 0 正常 1 无例子 2 kor
    # 是不是要例子？
    # 1 - 不要例子
    if judge_a_a == 1:
        prompt=PromptTemplate(
            template = system_prompt+"\n{original_dialogue}",
            #input_variables=["bot_character", "original_dialogue"],
            input_variables=["original_dialogue"],
        )
        summarize_response = llm_agent1.invoke(input_gpt).content

    elif judge_a_a == 0:
        messages = [SystemMessage( content = system_prompt),
                HumanMessage( content = q_example),
                AIMessage( content = a_example)]
        messages.append( HumanMessage(content = input_gpt) )
        if input[0]["role"] == "GPT_final":
            summarize_response = llm_agent_end.invoke( messages ).content
        else:
            summarize_response = llm_agent1.invoke( messages ).content
  

    # 这是每轮的输出
    if not args.if_test:
        logging.info(summarize_response)
    return summarize_response 


# 对分数进行一些修正
def score_correction(score_dict, bot_avg_word):
    judge_length = 0
    min_dia_length = 50
    # 这两个可以合起来写，但是为了之后拓展功能
    if bot_avg_word < min_dia_length:
        judge_length = 1
    if judge_length == 0:
        return score_dict
    if not args.if_test:
        logging.info("correcting...")
    if float(score_dict["c"]) >= 2.5:
        score_dict["c"] = str(float(score_dict["c"]) - 0.5)       
    if float(score_dict["d"]) >= 2.5:   
        score_dict["d"] = str(float(score_dict["d"]) - 1)
    if float(score_dict["e"]) >= 2:
        score_dict["e"] = str(float(score_dict["e"]) - 1)
    if float(score_dict["f"]) >= 2.5:
        score_dict["f"] = str(float(score_dict["f"]) - 1.5) 
    elif float(score_dict["f"]) >= 2 : 
        score_dict["f"] = str(float(score_dict["f"]) - 1)       
    
    return score_dict

def extract_summarize_score_multi(input):

    # 这里的input是prompt加上之前评分的所有信息
    if True:
        # 生成答案
        summarize_response = agent_response(input)

        #要不要匹配对话中bot的发言数量？
        pattern_seg = r'(\(\s*round\d+\s*\))' 
        segments = re.split(pattern_seg, summarize_response)

        paragraphs_dict = {}
        current_key = None
        current_paragraph = ""

        gpt_score_list = []
        for segment in segments:
            segment = segment.strip("\n").strip()
            if segment == "":
                continue
            if re.match(pattern_seg, segment):
                if current_key:
                    paragraphs_dict[current_key] = current_paragraph.strip()
                current_key = segment.strip()
                current_paragraph = ""
            else:
                current_paragraph += segment
        if current_key:
            paragraphs_dict[current_key] = current_paragraph.strip()
        #现在paragraphs_dict 是形如{(round1)：string}的键值对


        wrong_judge = 0 
        score_round = 0
        for key, value in paragraphs_dict.items():
            summarize_score = {}
            # 定义正则表达式模式，匹配评分标准和分数
            pattern = r"(\w+):\s*(\d+(?:\.\d+)?)"
            # 使用 re.findall() 查找所有匹配的评分标准和分数
            matches = re.findall(pattern, value)
            # 创建一个字典来存储评分结果
            summarize_score = {k: float(v) for k, v in matches}
            

            if len(summarize_score) == 6:
                pass    
            else:
                if len(summarize_score) < 4:
                    # 正则表达式模式，匹配一个或多个数字
                    pattern = r'\d+'
                    score_num = re.findall(pattern, value)
                    score_num = list(map(int, score_num))
                    summarize_score = {
                       "Language Fluency":score_num[0],
                       "Language Relevance":score_num[1],
                       "Role Language":score_num[2],
                       "Role Knowledge":score_num[3],
                       "Emotional Expression":score_num[4],
                       "Interactive Engagement":score_num[5],
                    }


                if not len(summarize_score) == 6:
                    print("summarize_response: ",summarize_response)
                    print("paragraphs_dict: ",paragraphs_dict)
                    print(f"key: {key}, value: {value}")
                    print("summarize_score: ",summarize_score)
                    paragraphs_dict +1
                    break
            paragraphs_dict[key] = summarize_score
        if True:
            score_round += 1
    if not args.if_test:
        logging.info("Summary of this round of ratings: ")
        logging.info(paragraphs_dict)
    return paragraphs_dict

def agent_construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        for key,value in agent_response.items():
            agent_answer = agent_answer + key + " "
            for k,v in value.items():
                agent_answer = agent_answer + k +":"+ str(v) + ", "
            agent_answer += "\n"
        response = "\n One agent solution: ```{}```".format(agent_answer)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n Using other agents and your previous answers as additional suggestions, can you update your answer? Examine your solution and that other agents step by step."""#.format(question)
    return {"role": "user", "content": prefix_string}



def extract_num(score_final):
    score_final_end = {}
    for round_key, scores in score_final.items():
        round_number = round_key.lstrip("(round").strip(")")
        score_final_end[round_number] = [int(min(3, round(score))) for score in scores.values()]

    return score_final_end


def multi_agent(role_inf_prompt,dia_value,ans_value,wait_time = 0) :

    judge_multi = 1 # 1：一次多轮评分 0：一次一个评分

    if not args.if_test:
        logging.info(f"\n\n{role_inf_prompt}")
        logging.info(dia_value)
        logging.info(f"{ans_value}")

    # 所有的agent输入信息的prompt都一致
    agent_contexts = [[{"role": f"agent{agent}", "content": role_inf_prompt + dia_value}] for agent in range(args.agents_num)] 
    for round in range(args.rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                if not args.if_test:
                    logging.info(f"round: {round}, agent: {i}, agent debate")
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]            # 把其他的信息加到一起
                message_other_agent = agent_construct_message(agent_contexts_other, role_inf_prompt, 2 * round - 1)
                agent_context.append(message_other_agent)
   
            # 这个逻辑有点怪，越变越多，回头看一看
            # 单行
            if judge_multi == 0:
                pass
            # 多行抽取
            elif judge_multi == 1:
                time.sleep(wait_time)
                score_list = extract_summarize_score_multi(agent_context)
                assistant_message = {"role": "assistant", "content": score_list}

            agent_context.append(assistant_message)
            
    # 这里是输出全部信息
    if not args.if_test:
        logging.info("\n\nlast ansewer")
        for i in range(len(agent_contexts)):
            logging.info(f"The rating of the {i} th agent is as follows: ")
            logging.info(agent_contexts[i][-1]["content"])


    input_final = role_inf_prompt + dia_value + "These are the solutions to the problem from other agents: \n"
    
    if not args.if_test:
        logging.info("\n\nAgent Debate Final Score:")

    for agent_context in agent_contexts:

        agent_answer1 = ""
        for key,value in agent_context[-1]["content"].items():
            agent_answer1 = agent_answer1 + key + " "
            for k,v in value.items():
                agent_answer1 = agent_answer1 + k +":"+ str(v) + ", "
            agent_answer1 += "\n"

        input_final += "agent {} solution: ```{}```\n".format(i, agent_answer1)
        if not args.if_test:
            logging.info(agent_context[-1]["content"])
    input_final += "Using other agents answers as additional suggestions, Can you rate the six aspects of the conversation separately?"


    input_final_dict = []
    input_final_dict.append({"role": "GPT_final", "content": input_final})
    if judge_multi == 0:
        pass
    if judge_multi == 1:
        time.sleep(wait_time)
        score_final = extract_summarize_score_multi(input_final_dict)

    if not args.if_test:
        logging.info(score_final)
    #"""
    score_final_end = extract_num(score_final)
    return score_final_end





def for_data_GPT(file_path, out_filename, begin_num = 0,wait_time = 0):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    count_run = 0

    print(len(data))
    begin_re_connect = 0
    # 先处理数据
    for i in tqdm(range(begin_num, len(data))):

        #time.sleep(wait_time)
        data1 = data[i]
        if args.re_evaluate:
            if data1["uid"] < begin_re_connect:
                continue


        role_inf = data1["origin_inf"]

        uid = data1.get("uid", "")
        role_inf_prompt = construct_roleplaying_prompt(role_inf)
        messages = role_inf.get('messages', [])
        bot_name = role_inf["role"]["bot"]["name"]


        if len(messages) > 10:
            messages = messages[:10]
       
        sorted_messages = sorted([key for key in messages.keys()], key=lambda x: int(x))
        
        if_ann = 0
        if "human_ann" in data1:
            if_ann = 1
            human_ann = data1["human_ann"]
            if len(human_ann) > 10:
                human_ann = human_ann[:10]
            sorted_answer = sorted([key for key in human_ann.keys()], key=lambda x: int(x))

            if (not sorted_messages == sorted_answer):
                if len(sorted_answer) > len(sorted_messages):
                    sorted_answer = sorted_answer[:len(sorted_messages)]
                    
                else:
                    print(f"warning, len(messages)>len(answer),uid={uid}")



        dia_value = ""
        ans_value = ""
        for k in sorted_messages:
            dia_value += f"(round{k})\n"
            for dia1 in messages[k]:
                for key1 in dia1:
                    if key1 =="user":
                        dia_value += f"human: {dia1[key1]}\n"
                    else:
                        dia_value += f"bot: {dia1[key1]}\n"

            if if_ann:
                ans_value += f"(round{k}) "
                if not k in human_ann:
                    # 缺少某轮数据，全部置为2
                    print(f"uid {uid} lack {k} round dialogue") 
                    human_ann[k] = ["2","2","2","2","2","2"]
                
                for i in range(len(human_ann[k])):
                    ans_value += mapping_le_num[i] + ":" + human_ann[k][i]
                    if not i == len(human_ann[k])-1:
                        ans_value += ", "
                    else:
                        ans_value += "\n"



        if role_inf["messages"] == []:
            logging.info(f"Wrong! messages - {i} is null!")
        else:    
            
            data1["GPT_score"] =  multi_agent(role_inf_prompt,dia_value,ans_value,wait_time) 
            with open(out_filename, 'a') as f:
                f.write(json.dumps(data1) + '\n')  




# python .\character_process\final_agent_debate\agent_debate_final_copy.py


if __name__ == "__main__":

    file_path = args.test_data_path
    # 多轮评价, 信息以及所有对话
    base_file_path = "./character_process/final_agent_debate/"
    file_path = base_file_path + "our_ann_3.3blend_test.jsonl"
    file_path = base_file_path + "data_score500.jsonl"

    #file_name = "low_q_Emotional_Expression.jsonl"
    #file_path = base_file_path + "low_quality_data/" + file_name
    # 单轮分别评价，信息以以及单论对话
    #file_path = "./character_process/generate_annoatation_files_0619_random_multi.jsonl"

    out_filename = file_path.replace(".jsonl","_gpt_score.jsonl")
    #out_filename = base_file_path + "low_quality_data/low_q_GPT_score/" +file_name.replace(".jsonl", "_gpt_score.jsonl")

    # 从第几项开始
    begin_num = 0
    # 每轮执行前需要等待多长时间，例如30s，2轮3agent，就是30*(2*3+1)=3.5min
    wait_time = 0
    
    for_data_GPT(file_path, out_filename, begin_num,wait_time)







    



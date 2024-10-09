import os
import shutil
import json
from tqdm import tqdm
import re
import pandas as pd
import csv
import numpy as np
import copy
import chardet
import tiktoken
import argparse
import jsonlines
import zipfile
from collections import Counter
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import string

#from langchain import PromptTemplate, LLMChain
#from langchain.chat_models import  AzureChatOpenAI
#from langchain_community.chat_models import AzureChatOpenAI

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain_openai import AzureChatOpenAI


#from langchain.chat_models import ChatOpenAI
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
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number


parser = argparse.ArgumentParser()
# 必须更改  --name "Fifty Shades of Grey" 
parser.add_argument('--name', type=str, default='神探狄仁杰1', help='novel name')
# 可能更改  --is_series True
parser.add_argument("--is_series", type=bool, default=False, help="小说的路径是直接以txt存在的False, 还是一个文件夹里一系列相同的小说True")
# 可能更改  --novel_text_type "epub"
parser.add_argument("--novel_text_type", type=str, default="txt",  choices=["txt", "epub"], help="txt直接处理, epub先转txt, mobi建议格式工厂转epub")
parser.add_argument('--folder_path', default="./StoryGPT" , type=str, help='base save folder path')
# 可能更改  --language 1
parser.add_argument('--language', default=0 , type=int, help='0 - English novel, 1 - Chinese novel')
# 看情况更改
parser.add_argument('--chapter_type', default=1 , type=int, help='目前只控制英文小说 0 - 章节是两种形式都可能, 1 - 章节是Chapter形式, 2 - 章节是罗马数字形式')
parser.add_argument('--max_token_len', default=2000 , type=int, help='对话抽取时最大token数')
parser.add_argument('--key_type', default="azure", choices=["azure", "openai"], type=str, help='gpt的key类型')
# 看情况更改
parser.add_argument('--extract_method', default=0 , type=int, help='抽取小说方法， 0 - kor, 1 - 非kor')
# 可能更改，重连  --extract_begin 0
parser.add_argument('--extract_begin', default=0 , type=int, help='抽取断点重连，从这个值开始抽取直到最后')
# 看情况更改
parser.add_argument('--extract_end', default=-1 , type=int, help='最少抽取多少章节，-1为抽全篇')
# 看情况更改
parser.add_argument('--repair_raw_text', default=True , type=bool, help='true：如果对话txt文本有些莫名的换行符进行删除，但是可能会造成新的问题')
# 看情况更改
parser.add_argument('--extract_type', default=0 , type=int, choices=[0, 1], help='0是小说，1是剧本')

parser.add_argument('--simple_prompt', default=1 , type=int, help='精简prompt以减少花费, 0 - 原prompt, 1 - 新写的精简prompt')
parser.add_argument('--max_find_lines', default=10 , type=int, help='支持跨越多少行寻找目标角色，也即控制段内行间距不超过该值')
parser.add_argument('--max_token_num_dia', default=1000 , type=int, help='对话最多多少token')
parser.add_argument('--max_roles', default=30 , type=int, help='最多多少个角色可以被抽取')
parser.add_argument('--min_roles', default=3 , type=int, help='最少多少个角色可以被抽取，对话不足强制补满')
parser.add_argument('--min_roles_dia', default=30 , type=int, help='最少一个角色多少句对话才可以被抽取')
parser.add_argument('--fix_roles_dia', default=-1 , type=int, help='固定抽取多少角色, -1为抽完全根据上两个指标定义, 大于min_roles_dia的不超过max_roles的所有角, 按对话数倒叙')
parser.add_argument('--use_gpt_rewrite', default=False , type=bool, help='是否用gpt重写场景和标签, 不用gpt就是简单的把场景提取出来, tag=[]')
parser.add_argument('--dialogue_ratio', default=0.7 , type=float, help='主角配角对话超过多少算双人对话')
parser.add_argument('--adopted_for_others', default=0 , type=int, help='对于双人对话的第三人采取什么策略: 0 - 丢弃, 1 - 算配角说的, 2 - 算主角说的')
parser.add_argument('--find_role_range', default=1000 , type=int, help='找主角上下文的范围+-1000字符')
args = parser.parse_args()

# --language 1 --chapter_type 0

novel_name = args.name


def epub_to_txt(epub_path, txt_path):
    book = epub.read_epub(epub_path)
    text = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text() + '\n\n'
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

if args.novel_text_type == "epub":
    if args.is_series:
        series_folder = f'{args.folder_path}/novel/epub/{args.name}'
        output_folder = f'{args.folder_path}/novel/{args.name}_epub2'
        os.makedirs(output_folder, exist_ok=True)
        with tqdm(total=len(os.listdir(series_folder)), desc="Processing .epub files") as pbar:  
            for filename in os.listdir(series_folder):
                if filename.endswith(".epub"):
                    epub_path = os.path.join(series_folder, filename)
                    txt_name = os.path.splitext(filename)[0] + "_epub2.txt"
                    txt_path = os.path.join(output_folder, txt_name)
                    epub_to_txt(epub_path, txt_path)
                if filename.endswith(".txt"):
                    shutil.copy(os.path.join(series_folder, filename),os.path.join(output_folder, filename))
                pbar.update(1)  
    else:
        print("epub 格式转换中...")
        epub_path = f'{args.folder_path}/novel/epub/{args.name}.epub'
        txt_path = f'{args.folder_path}/novel/{args.name}_epub2.txt'
        epub_to_txt(epub_path, txt_path)
        print(f"{args.name}文件 epub 格式转换完成...")

    novel_name = f"{args.name}_epub2"




# 1 - 新小说抽取

base_folder = args.folder_path
novel_folder = f"{base_folder}/novel_data_history/{novel_name}"       # 处理后的文件夹
save_folder = f"{novel_folder}/{novel_name}_extract"                    # 抽取的文件
save_folder_path =  f"{novel_folder}/reorganized_story_{novel_name}"    # 第二部分用的
save_variables = f'{novel_folder}/middle_variable'                      # 存储一些中间变量

Process_variables = {}

if not os.path.exists(novel_folder):
    os.makedirs(novel_folder)
    print(f"已创建_{novel_name}_文件夹")

#"""
# 创建相应存储文件夹

if not args.is_series:        # 单本小说
    novel_file_path = f'{novel_folder}/{novel_name}.txt'  
    if not os.path.exists(f"{base_folder}/novel/{novel_name}.txt"):
        print(f"注意，{base_folder}/novel/{novel_name}.txt文件不存在!")
    if not os.path.exists(novel_file_path):
        print("复制小说ing...")
        shutil.copy(f"{base_folder}/novel/{novel_name}.txt",novel_file_path)
else:                       # 多部系列小说在一个文件夹下
    novel_file_path = f'{novel_folder}/{novel_name}'  
    if not os.path.exists(f"{base_folder}/novel/{novel_name}"):
        print(f"注意，{base_folder}/novel/{novel_name}文件夹不存在!")
    else:
        if not os.listdir(f"{base_folder}/novel/{novel_name}"):
            print(f"注意，文件夹 {base_folder}/novel/{novel_name} 为空!")
    if not os.path.exists(novel_file_path):
        print("复制文件夹ing...")
        shutil.copytree(f"{base_folder}/novel/{novel_name}",novel_file_path)

if not os.path.exists(save_folder):
  os.makedirs(save_folder)
  print(f"已创建{novel_name}_extract文件夹")
else:
    print('文件夹',save_folder,'已经存在')


def detect_encoding(filename):
    with open(filename, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']
    
print("is novel series? ",args.is_series)

def contains_roman_numerals(text):
    pattern = r"\b([IVX]+)\b"  #IVXLCDM
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return True
    return False

def has_digit(text):
    for char in text:
        if char.isdigit():
            return True
    return False

def contains_all_english_digits(text):
    english_digits = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                      "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                      "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                      "eighty", "ninety", "hundred", "thousand", "million", "billion"]
    text = text.lower()
    if contains_roman_numerals(text): # 罗马数字
      return True
    text = text.replace(" ", "")
    for digit in english_digits:

      if text.find(digit) != -1:
        return True
    if has_digit(text): # 阿拉伯数字
      return True
    return False

def is_chapter_line(line,chapter_type):

    if chapter_type == 0:
        line_judge = line.replace(" ", "").replace(",", "").replace(".", "").replace("'", "").replace("“", "").replace("’", "").replace("”", "").replace(";", "").replace(":", "").replace("?", "").replace("‘", "").replace("—", "").replace("-", "").replace("(", "").replace(")", "").replace("\"", "").replace("*", "").replace("!", "")
        return line.strip().startswith('CHAPTER') or line.strip().startswith('Chapter') or contains_roman_numerals(line_judge) or line.strip().startswith('VOLUME')
    elif chapter_type == 1:
        return line.strip().startswith('CHAPTER') or line.strip().startswith('Chapter') or line.strip().startswith('VOLUME')
    elif chapter_type == 2:
        line_judge = line.replace(" ", "").replace(",", "").replace(".", "").replace("'", "").replace("“", "").replace("’", "").replace("”", "").replace(";", "").replace(":", "").replace("?", "").replace("‘", "").replace("—", "").replace("-", "").replace("(", "").replace(")", "").replace("\"", "").replace("*", "").replace("!", "")
        return contains_roman_numerals(line_judge)
    elif chapter_type == 3:         # 针对的是1./n这种类型
        pattern = r"\d+\."
        return re.fullmatch(pattern, line.strip())
    else:
        # 如果 chapter_type 不是 0、1、2 中的一个值，可能需要根据实际情况进行处理
        raise ValueError("Invalid chapter_type value")


# 切片 - 切章节和chunk


if not args.is_series:   
    # 检测文件编码
    file_encoding = detect_encoding(novel_file_path)
    print("file_encoding: ",file_encoding)

    # 处理 raw_text
    # 根据检测到的编码来打开文件
    raw_text = open(novel_file_path, encoding=file_encoding).read()
else:
    raw_text = ""
    files_series = os.listdir(novel_file_path)
    files_series.sort()  # 按文件名排序
    num = 0
    with tqdm(total=len(files_series)) as pbar:
        for file_name in files_series:
            file_series_path = os.path.join(novel_file_path, file_name)
            if os.path.isfile(file_series_path):
                num += 1 
                raw_text = raw_text+ f"\n VOLUME {num} - {os.path.splitext(file_name)[0]} \n"

                encoding = detect_encoding(file_series_path)
                with open(file_series_path, 'r', encoding=encoding) as f:
                    raw_text += f.read()
            pbar.update(1)  # 更新进度条


# 处理一下txt中出现的换行问题
if args.repair_raw_text  and args.language == 0:
    # 根据换行符分割成行的列表
    lines = raw_text.splitlines()
    raw_text = ""
    # 逐行处理
    judge_space= 0

    for i in range(len(lines)):
        if is_chapter_line(lines[i], args.chapter_type):
            raw_text += "\n" + lines[i] + "\n"
        else:
            if lines[i].strip() == "":
                if i<len(lines)-1 and  lines[i+1].strip() == "":
                    continue
                elif judge_space == 1:
                    judge_space = 0
                    continue
                else:
                    raw_text += lines[i] + "\n"
            else:
                if lines[i].strip()[-1].isdigit() or (lines[i].strip()[-1] in string.punctuation):
                    raw_text += lines[i] + "\n"
                else:
                    raw_text += lines[i]
                    if lines[i][-1] !=" ":
                        raw_text += " "
                    if i<len(lines)-2 and lines[i+1].strip() == "":
                        judge_space = 1



# 切章节
#chapters = [raw_text]


# 用正则简单看下对话数
def count_dialogues(raw_text):
    # 使用正则表达式匹配任意跨行的对话
    raw_text1 = raw_text.replace('\n','')
    dialogue_pattern = re.compile(r'["\'“‘](.+?)["\'”’]', re.DOTALL)
    # 找到所有匹配的对话
    dialogues = re.findall(dialogue_pattern, raw_text)

    # 返回对话的数量
    return len(dialogues)
# 统计对话数量

dialogue_count = count_dialogues(raw_text)
raw_text_len = len(raw_text.replace(r'\n',''))
Process_variables["Total char in novel"] = raw_text_len
Process_variables["Total number of dialogues"] = dialogue_count
print(Process_variables)

# 一些切章节的函数（英文）
# contains_roman_numerals(text)： 判断是否是罗马数字
# has_digit(text)： 是否还有数字
# is_chapter_line(line,line_judge,chapter_type): # 判断不同章节类型


if args.language == 0:          #英语抽章节
    chapters = []
    chapters_name = []
    chapter_contents = []
    i = 0
    Flag_content = 2
    for line in raw_text.split('\n'):
        Flag = False
        line_judge = line.replace(" ", "").replace(",", "").replace(".", "").replace("'", "").replace("“", "").replace("’", "").replace("”", "").replace(";", "").replace(":", "").replace("?", "").replace("‘", "").replace("—", "").replace("-", "").replace("(", "").replace(")", "").replace("\"", "").replace("*", "").replace("!", "")

        if is_chapter_line(line, args.chapter_type):
            i = i+1
            # 遇到章节标题,将之前章节内容添加到结果列表
            head = line.strip()
            if args.chapter_type ==  3:
                head_num = head[0:-1]
            else:
                head_num = head[7:min(20,len(head))]
            if (contains_all_english_digits(head_num)) or contains_roman_numerals(line_judge): #or has_digit(head[:11])): #and (head.find("'") == -1):
                Flag = True
                if Flag and Flag_content<=1:
                    Flag = False
                Flag_content = 0 # 记录上一行是不是标题，0代表是

        if line:
            Flag_content += 1

        if Flag:
            if chapter_contents:
                chapters.append('\n'.join(chapter_contents))
                chapter_contents = []
            # 记录当前章节标题
            chapters_name.append(line)
        else:
            # 累积章节内容
            chapter_contents.append(line)

    # 添加最后一个章节内容
    if chapter_contents:
        chapters.append('\n'.join(chapter_contents))

    Process_variables["chapters_name"] = chapters_name
    Process_variables["len(chapters)"] = len(chapters)
    print(f"len(chapters):{len(chapters)}")

elif args.language == 1:
    # 中文切章节
    chapters = []
    chapter_contents = []

    for line in raw_text.split('\n'):
        Flag = False
        if line.strip().startswith('第'):
            # 遇到章节标题,将之前章节内容添加到结果列表

            head = line.strip()
            head = head[:min(30,len(head))]
            if head.find('章',1)>0 or head.find('部',1)>0:
                print(head)
                Flag = True

        if Flag:
            if chapter_contents:
                chapters.append('\n'.join(chapter_contents))
                chapter_contents = []
            # 记录当前章节标题
            # chapters.append(line)
        else:
            # 累积章节内容
            chapter_contents.append(line)

    # 添加最后一个章节内容
    if chapter_contents:
        chapters.append('\n'.join(chapter_contents))

    print(len(chapters))

else:
    print("warning! Currently not supported for other languages")


# 切chunk


enc = tiktoken.get_encoding("cl100k_base")

# 定义divide函数，用来切分超长文本
def divide_str(s, sep=['\n', '.', '。']):
    mid_len = len(s) // 2  # 中心点位置
    best_sep_pos = len(s) + 1  # 最接近中心点的分隔符位置
    best_sep = None  # 最接近中心点的分隔符
    for curr_sep in sep:
        sep_pos = s.rfind(curr_sep, 0, mid_len)  # 从中心点往左找分隔符
        if sep_pos > 0 and abs(sep_pos - mid_len) < abs(best_sep_pos - mid_len):
            best_sep_pos = sep_pos
            best_sep = curr_sep
    if not best_sep:  # 没有找到分隔符
        return s, ''
    return s[:best_sep_pos + 1], s[best_sep_pos + 1:]
def strong_divide(s):
    left, right = divide_str(s)

    if right != '':
        return left, right

    whole_sep = ['\n', '.', '，', '、', ';', ',', '；',\
                 '：', '！', '？', '(', ')', '”', '“', \
                 '’', '‘', '[', ']', '{', '}', '<', '>', \
                 '/', '''\''', '|', '-', '=', '+', '*', '%', \
               '$', '''#''', '@', '&', '^', '_', '`', '~',\
                 '·', '…']
    left, right = divide_str(s, sep=whole_sep)

    if right != '':
        return left, right

    mid_len = len(s) // 2
    return s[:mid_len], s[mid_len:]

# 以1500 token为限，切分chunk，输出总chunk数量
# max_token_len = 24000 #基本是一章，太大了
max_token_len = args.max_token_len   # 对话抽取5000效果可以,但是有点犯病
chunk_text = []
for chapter in chapters:

    split_text = chapter.split('\n')
    curr_len = 0
    curr_chunk = ''
    tmp = []

    for line in split_text:
        line_len = len(enc.encode( line ))

        if line_len <= max_token_len - 5:
            tmp.append(line)
        else:
            path = [line]
            tmp_res = []

            while path:
                my_str = path.pop()
                left, right = strong_divide(my_str)

                len_left = len(enc.encode( left ))
                len_right = len(enc.encode( right ))

                if len_left > max_token_len - 15:
                    path.append(left)
                else:
                    tmp_res.append(left)

                if len_right > max_token_len - 15:
                    path.append(right)
                else:
                    tmp_res.append(right)

            for line in tmp_res:
                tmp.append(line)

    split_text = tmp

    for line in split_text:
        line_len = len(enc.encode( line ))

        if line_len > max_token_len:
            print('warning line_len = ', line_len)

        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = line
            curr_len = line_len

    if curr_chunk:
        chunk_text.append(curr_chunk)
    # break
print("章节数：",len(chapters))
Process_variables["max_token_len"] = max_token_len
Process_variables["分块后块数"] = len(chunk_text)
print(f"分块后块数: {len(chunk_text)}")


# key配置（OpenAI, azure）

if args.key_type == "openai":
    from langchain_openai import ChatOpenAI
    from langchain.llms import OpenAI

    key = ''
    key_bytes = key.encode()
    os.environ["OPENAI_API_KEY"] = key_bytes.decode('utf-8')
    llm = OpenAI(
        model_name="gpt-3.5-turbo",
        #model_name="gpt-4-turbo",
        temperature=0
    )
elif  args.key_type == "azure":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2024-02-01"
    os.environ["AZURE_OPENAI_ENDPOINT"] = ''
    os.environ["AZURE_OPENAI_API_KEY"] = ""    
    llm = AzureChatOpenAI(
        #deployment_name="gpt-4-32k",
        deployment_name="gpt-4-turbo",
        model_name="gpt-4-turbo",
        temperature= 0.,
    )
    print(llm)
if args.language == 0:  # 英文
    # 定义kor中一个称为抽取schema的数据结构
    if args.simple_prompt == 0:
        schema = Object(
            id="script",
            description="Extract Dialogue in order From Novel and identify the role involved in the dialogue, ignore the non-dialogue parts",
            attributes=[
                Text(
                    id="role",
                    description="The character who is speaking, use context to predict the name of the role.",
                ),
                Text(
                    id="dialogue",
                    description="The dialogue spoken by the characters in the sentence",
                )
            ],
            examples=[
                (
                    '''``"IT WASN'T A NIGHTMARE!" Ron yelled. "PROFESSOR, SIRIUS BLACK WAS STANDING OVER ME!" 
                    Professor McGonagall stared at him.
                    "Don't be ridiculous, Weasley, how could he possibly have gotten through the portrait hole?"
                    "Ask him! if he saw --" Glaring suspiciously at Ron, Professor McGonagall pushed the Portrait back open and went outside. "Sir Cadogan, did you just let a man enter Gryffindor Tower?" "Certainly, good lady!" cried Sir Cadogan.
                    There was a stunned silence. "You -- you did? But the password!" 
                    "He had 'em! Had the whole week's, my lady!"``''',
                    [
                        {"role": "Ron","dialogue": "IT WASN'T A NIGHTMARE! PROFESSOR, SIRIUS BLACK WAS STANDING OVER ME!"},
                        {"role": "Professor McGonagall", "dialogue": "Don't be ridiculous, Weasley, how could he possibly have gotten through the portrait hole?"},
                        {"role": "Ron","dialogue": "Ask him! if he saw --"},
                        {"role": "Professor McGonagall", "dialogue": "Sir Cadogan, did you just let a man enter Gryffindor Tower?"},
                        {"role": "Sir Cadogan","dialogue": "Certainly, good lady!"},
                        {"role": "Professor McGonagall","dialogue": "You -- you did? But but the password!"},
                        {"role": "Sir Cadogan","dialogue": "He had 'em! Had the whole week's, my lady!"}
                    ],
                )
            ],
            many=True,
        )
    elif args.simple_prompt == 1:
        schema = Object(
            id="script",
            description="Extract Dialogue in order From Novel and identify the role involved in the dialogue, ignore the non-dialogue parts",
            attributes=[
                Text(
                    id="role",
                    description="The character who is speaking, use context to predict the name of the role.",
                ),
                Text(
                    id="dialogue",
                    description="The dialogue spoken by the characters in the sentence",
                )
            ],
            examples=[
                (
                    '''``"Ask him! if he saw --" Ron yelled. Glaring suspiciously at Ron, Professor McGonagall pushed the Portrait back open and went outside. "Sir Cadogan, did you just let a man enter Gryffindor Tower?" "Certainly, good lady!" cried Sir Cadogan.
                    There was a stunned silence. "You -- you did? But the password!" 
                    "He had 'em! Had the whole week's, my lady!"``''',
                    [
                        {"role": "Ron","dialogue": "Ask him! if he saw --"},
                        {"role": "Professor McGonagall", "dialogue": "Sir Cadogan, did you just let a man enter Gryffindor Tower?"},
                        {"role": "Sir Cadogan","dialogue": "Certainly, good lady!"},
                        {"role": "Professor McGonagall","dialogue": "You -- you did? But but the password!"},
                        {"role": "Sir Cadogan","dialogue": "He had 'em! Had the whole week's, my lady!"}
                    ],
                )
            ],
            many=True,
        )        
elif args.language == 1: # 中文
    schema = Object(
        id="script",
        description="Extract Dialogue in order From Novel, ignore the non-dialogue parts",
        attributes=[
            Text(
                id="role",
                description="The character who is speaking, use context to predict the name of the role.",
            ),
            Text(
                id="dialogue",
                description="The dialogue spoken by the characters in the sentence",
            ),
        ],
        many=True,
    )
else:
    print("warning! Currently not supported for other languages")
chain = create_extraction_chain(llm, schema)
# 非kor对话抽取
ex_ins = []
ex_outs = []
task_prompt_dia = """Your task is to extract dialog text from each novel paragraph.
For each paragraph, if there is dialog information in the sentence (dialog text is generally between quotation marks),
infer the speaker and spoken text based on context, and output in the format |.
If there is no dialog in sentence, do not output anything."""
def csv_to_json(text):
    lines = []
    for line in text.splitlines():
        if '|' in line:
            lines.append(line)

    reader = csv.reader(lines, delimiter='|')

    result = []
    for row in reader:
        if row[0] == 'role' and row[1] == 'dialogue':
            continue
        result.append({
            "role": row[0],
            "dialogue": row[1]
        })

    return result
def extract_dialogue_wo_kor(input_text):
    messages_dia = [SystemMessage( content = task_prompt_dia),
                HumanMessage( content = example_in_dia),
                AIMessage( content = example_out_dia)]
    input_text = '\n###\n' + input_text
    messages_dia.append( HumanMessage(content = input_text) )
    response = llm( messages_dia ).content

    json_result = csv_to_json(response)
    return json_result

system_prompt = """
Summarize the key points of the following text in a concise way, using bullet points.
"""
q_example = """###
Text:
洪七公、周伯通、郭靖、黄蓉四人乘了小船，向西驶往陆地。黄蓉不住向周伯通详问骑鲨游海之事，周伯通兴起，当场就要设法捕捉鲨鱼，与黄蓉大玩一场。
郭靖见师父脸色不对，问道：“你老人家觉得怎样”洪七公不答，气喘连连。他被欧阳锋点中之后，穴道虽已解开，内伤却又加深了一层。黄蓉喂他服了几颗九花玉露丸，痛楚稍减，气喘仍是甚急。
老顽童不顾别人死活，仍是嚷着要下海捉鱼，黄蓉向他连使眼色，要他安安静静的，别吵得洪七公心烦。周伯通并不理会，只闹个不休。黄蓉皱眉道：“你要捉鲨鱼，又没饵引得鱼来，吵些甚么”

Summarize in BULLET POINTS form:
"""
a_example = """
- 洪七公等四人乘船西行,洪七公因受内伤加重而气喘不止
- 周伯通要捉鲨鱼玩,被黄蓉阻止
"""
messages = [SystemMessage( content = system_prompt),
            HumanMessage( content = q_example),
            AIMessage( content = a_example)]

#"""

print("save_folder: ",save_folder)
print("len(chunk_text): ",len(chunk_text))
print("chunk_text[0][:200]: ",chunk_text[0][:200])

# 全篇
if args.extract_end <= 0:
    extract_part = len(chunk_text)  # 全篇
else :
    extract_part = args.extract_end

for i in tqdm(range(args.extract_begin, min(len(chunk_text),extract_part))):  #断点重连
    save_name = os.path.join(save_folder, f"{i}_dialogue.txt")

    if not os.path.exists(save_name) or os.path.getsize(save_name) < 5:
        if os.path.exists(save_name):
            print('re-generate dialogue id = ', i)
        query_text = f"{chunk_text[i]}"

        # 小说抽取
        if args.extract_type == 0:
            if args.extract_method == 0:
                try:
                    dialogue_response = chain.run( query_text )["data"]
                    with open(save_name, 'w', encoding='utf-8') as f:
                        if 'script' not in dialogue_response:
                            print('Error: response does not contain key "script"')
                        else:
                            for chat in dialogue_response['script']:
                                json_str = json.dumps(chat, ensure_ascii=False)
                                f.write(json_str+"\n")
                except:
                    print(f"第{i}段对话kor抽取失败")
                    dialogue_response_wo_kor = extract_dialogue_wo_kor( query_text )
                    with open(save_name, 'w', encoding='utf-8') as f:
                            for chat in dialogue_response_wo_kor:
                                json_str = json.dumps(chat, ensure_ascii=False)
                                f.write(json_str + '\n')
            elif args.extract_method == 1:
                dialogue_response_wo_kor = extract_dialogue_wo_kor( query_text )
                with open(save_name, 'w', encoding='utf-8') as f:
                    for chat in dialogue_response_wo_kor:
                        json_str = json.dumps(chat, ensure_ascii=False)
                        f.write(json_str + '\n')
            else:
                print("extract_method is wrong!")

    
        # 剧本对话抽取   
        elif args.extract_type == 1:
            pass
        else:
            print("no extract type")

    save_name_sum = os.path.join(save_folder, f"{i}_sum.txt")

    if not os.path.exists(save_name_sum) or os.path.getsize(save_name_sum) < 5:
        if os.path.exists(save_name_sum):
            print('re-summarize id = ',i )
        #dealing with summarize
        messages = [SystemMessage( content = system_prompt),
                HumanMessage( content = q_example),
                AIMessage( content = a_example)]

        new_input = f"###\nText:\n{chunk_text[ i ]}\nSummarize in BULLET POINTS form:"

        messages.append( HumanMessage(content = new_input) )

        summarize_response = llm( messages ).content

        with open(save_name_sum, 'w', encoding='utf-8') as f:
            f.write( summarize_response )

    raw_text_save_name = os.path.join(save_folder, f"{i}_raw.txt")
    if not os.path.exists(raw_text_save_name) or os.path.getsize(raw_text_save_name) < 5:
        with open(raw_text_save_name, 'w', encoding='utf-8') as f:
            f.write( chunk_text[i] )

#"""
# 由对话和摘要重组小说



if not os.path.exists(save_folder_path):
  os.makedirs(save_folder_path)

#folder_path = save_folder 
story_name_en = os.path.basename(save_folder).split("_")[0]
print("story_name_en",story_name_en)
# 测试ID

# 默认的保存路径
save_jsonl_path = f"{save_folder_path}/reorganized_{story_name_en}.jsonl"
save_txt_path = f"{save_folder_path}/reorganized_{story_name_en}.txt"

# 默认抽取出的dialogue和summary文件位置/如果有不同请在此处和底部自动程序中修改



## 通过切分好的chunk
### 非顺序，避免有些章节没出现，会更慢
file_list = []
sort_num = []
# 遍历文件夹中的文件
for file_name in os.listdir(save_folder):
    if file_name.endswith("_raw.txt"):
        file_path = os.path.join(save_folder, file_name)

        # 将文件路径添加到列表中
        file_list.append(file_path)
        sort_num.append(int(file_name.split('_')[0]))
# 按照文件名中的数字进行排序
sorted_files = [file for _, file in sorted(zip(sort_num, file_list))]

chunk_text = []

# 读取文件内容并存储到列表中
for file_path in sorted_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        chunk_text.append(text)


### 顺序，和上面运行一个就行

print("len(chunk_text): ",len(chunk_text))
# 给定summarzie_file = os.path.join(save_folder, f"{id}_sum.txt")
# 先检查这个文件是否存在
# 然后使用utf-8编码打开，检查每一行，如果strip后，行首是'-'，则把后面的字符串append到一个list chunk_sum中
# 提取raw文本对应的对话以及总结


# 给定长文本raw_text(切分原文)。
# 使用换行符\n或者。 来对这个字符串进行切割，忽略掉strip之后是空的子字符串
# 将每一段话的起点位置存储在一个list of int , starts中
# 将每一段话的结束位置存储在一个list of int , ends中
# 并且将每一个子字符串的存储在一个list of str, lines中

def divide_raw2lines(raw_text_line):
    previous_str = ''
    starts = []
    ends = []
    lines = []
    for i in range(len(raw_text_line)):
        previous_str += raw_text_line[i]
        # 这个是有问题的， 如果一个txt本身有问题而产生的错误划分
        # 还有，如果一个人说了一句话，“啊啊啊。”a如是说道，在句子中会被错误截断
        if raw_text_line[i] in ('\n','。','.','"','”','\'','’'):
            if i < len(raw_text_line)-1 and raw_text_line[i] in ('。','.'):
                if raw_text_line[i+1] in ('"','”','\'','’','.'):
                    continue
            if i > 0 and raw_text_line[i] in ('\n','"','”','\'','’') :    
                if  raw_text_line[i-1] in (',','，') :
                    continue
            strip_str = previous_str.strip(' "“”\r\n')
            if len(strip_str)>0:
                lines.append(strip_str)
                starts.append(i - len(strip_str))
                ends.append(i)
            previous_str = ''
        else:
            pass
    # 处理最后一个句子
    if previous_str.strip():  # 如果不为空
        lines.append(previous_str.strip())
        starts.append(ends[-1] + 1 - len(previous_str))
        ends.append(ends[-1] + 1)
    return lines , starts , ends

# 英文单词
# m是总结，n是源文本
def compute_word_recall(query, datas):
    M = len(query)
    N = len(datas)
    freqs = np.zeros((M, N), dtype=int)
    for m in range(M):
        q_words = set(re.findall(r'\b\w+(?:[-\']\w+)*\b', query[m].lower()))

        for n in range(N):
            for word in q_words:
                a = re.findall(r'\b\w+(?:[-\']\w+)*\b', datas[n].lower())
                if word in a:
                    freqs[m][n] += 1
    EPSILON = 1e-8 
    query_words_count = [len(sent.split()) for sent in query]
    #recalls = freqs / np.array(query_words_count)[:, None]
    recalls = freqs / (np.array(query_words_count)[:, None] + EPSILON)
    return recalls

# 中文汉字

# 已知if '\u4e00' <= char <= '\u9fa5': 可以判断一个char是否是中文字
# 我希望实现一个函数，这个函数的输入是两个list of string, 长度为M的query 和 长度为N的datas
# 输出是一个M*N的numpy float数组 recalls
# 先计算freqs[m][n] 表示query的第m句中的每一个中文字，是否在datas[n]中是否出现，如果出现，则freqs[m][n]加一
# 然后计算recalls[m][n]是freqs[m][n]除掉 query[m]中所有中文字的个数

def compute_char_recall(query, datas):
    M = len(query)
    N = len(datas)
    freqs = np.zeros((M, N), dtype=int)

    for m in range(M):
        q_chars = set()
        for char in query[m]:
            if '\u4e00' <= char <= '\u9fa5':
                q_chars.add(char)

        for n in range(N):
            for char in q_chars:
                if char in datas[n]:
                    freqs[m][n] += 1

    query_chars_count = [len(set(char for char in sent if '\u4e00'<= char <= '\u9fa5'))
                         for sent in query]
    recalls = freqs / np.array(query_chars_count)[:, None]
    return recalls

# import copy
def summary2line(chunk_sum, lines):
    if args.language == 0:  # 英文
        s = compute_word_recall(chunk_sum, lines)
    elif args.language == 1: # 中文
        s = compute_char_recall(chunk_sum, lines)
    else:
        print("warning! Currently not supported for other languages")
    color_map = {}  # color_map用于记录已经访问过的子问题
    ans_Q = {}    # ans_Q用于记录每个子问题的最优解
    ans_div = {}   # ans_div用于记录每个子问题的分割位置
    flags = {}    # flags用于记录每个子问题是否选择了左侧路径。

    M = len( chunk_sum )
    N = len( lines )
    # m*n，m行总结，n列原文本，理论上m<n
    # 这两行计算总结文本和原始文本的行数分别为M和N
    for n in range(0, N):
        if n==0:
            # 这个位置如果报错一般是chunk_sum=[]
            ans_Q[ (0,0) ] = s[0,0] # 0,0是他的相似度（召回率）
            ans_div[ (0,0 ) ] = []
        else:
            ans_Q[ (0,n) ] = ans_Q[ (0,n-1) ] + s[0,n]  # 0,n是当前相似度+之前相似度总和
            ans_div[ (0,n) ] = []
    for m in range(1,M):
        ans_Q[(m,m)] = ans_Q[(m-1,m-1)] + s[m,m]  # 斜着计算
        ans_div[ (m,m) ] =  ans_div[ (m-1,m-1) ].copy() # 因为m<n，这是m=n的情况，总结匹配只可能大于等于m*m对角线
        ans_div[ (m,m) ].append(m)          # 初始值：m个总结比m-1多一个m
    def find_Q( m , n ):
        if m < 0 or n < 0:
            print('error out bound', m , ' ' , n )
            return 0, []
        if (m,n) in ans_Q.keys(): #如果计算过，直接返回
            return ans_Q[(m,n)], ans_div[(m,n)]
        if (m,n) in color_map.keys():
            print('error repeated quest ', m , ' ', n )
            return 0, []
        else:
            color_map[(m,n)] = 1
        current_div = []

        # 递归地计算左侧和上方路径的召回率和分割位置。
        left, left_div = find_Q( m, n-1 )     # 计算对于上一个句子n-1，这个总结m与上个总结m-1
        right, right_div = find_Q( m-1, n-1 )
        if left > right:          # 如果上个句子n-1对于当前的总结m匹配度要更高，这里一直往上找，找到匹配度最高的，认为是m与n匹配
            ans = left + s[m][n]    # 分数是累加的
            flags[(m,n)] = False    # 不划分
            current_div = left_div  #划分等同于上个句子
        else:
            ans = right + s[m][n]   # 如果上个句子n-1对于上个总结m-1匹配度要更高
            flags[(m,n)] = True     # 划分，因为m至少要匹配到一个句子  如果看m,n+1,回来找 m，n以及m-1，n，看看是不是还能往上归类
            current_div = right_div.copy()
            current_div.append(n-1)   #把n-1放进去
        # ans = max(  , ) + s[m][n]

        ans_Q[(m,n)] = ans
        ans_div[(m,n)] = current_div.copy()

        return ans, current_div

    score, divs = find_Q(M-1,N-1)
    divs.append(N-1)

    return score, divs 
# div 划分的句子位置(只对n来讲)


# 感觉有点问题，因为句子未必是按顺序，但是问题不大，因为一模一样的句子很少见
def dialogue2line(dia_texts, lines):
    if args.language == 0:  # 英文
        s_dialogue = compute_word_recall(dia_texts, lines)
    elif args.language == 1: # 中文
        s_dialogue = compute_char_recall(dia_texts, lines)
    else:
        print("warning! Currently not supported for other languages")

    # m是对话，n是文本
    M, N = s_dialogue.shape
    if M==0 or N==0:
        return []
    dp = np.zeros((M, N))
    dp[0] = s_dialogue[0]
    prev_indices = np.zeros((M, N), dtype=int)
    for i in range(1, M):
        for j in range(N):
            max_prev_index = np.argmax(dp[i-1])
            dp[i][j] = dp[i-1][max_prev_index] + s_dialogue[i][j]
            prev_indices[i][j] = max_prev_index
  
    max_end_index = np.argmax(dp[-1])
    sequence = []
    for i in range(M-1, -1, -1):
        sequence.append(max_end_index)
        max_end_index = prev_indices[i][max_end_index]
    sequence.reverse()

    return sequence


def jsonl_sorted(dialogues, chunk_sum, divs, dia_texts, seq):

    combined_data = []
    combined_text = ""
    for index in sorted(seq + divs):
        if index in seq:
            combined_data.append({
                "role" : dialogues[seq.index(index)]["role"],
                'text': dialogues[seq.index(index)]["dialogue"],
                'if_scene': False
            })
            combined_text = combined_text + dialogues[seq.index(index)]["role"] + ":" + dialogues[seq.index(index)]["dialogue"] +"\n"
            seq[seq.index(index)] = -1
        if index in divs:
            combined_data.append({
                "role" : "scene" ,
                'text': chunk_sum[divs.index(index)],
                'if_scene': True
            })
            combined_text = combined_text + "scene" + ":" + chunk_sum[divs.index(index)] +"\n"
            divs[divs.index(index)]=-1
    return combined_data, combined_text

Process_variables["novel_folder"] = novel_folder
Process_variables["len(chunk_text)"] = len(chunk_text)
print(f"novel_folder: {novel_folder}")
print("len(chunk_text): ", len(chunk_text))

final_jsonl = []
final_txt = ""

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

sort_num = []
for file_name in os.listdir(save_folder):
  if not int(file_name.split('_')[0]) in sort_num:
    sort_num.append(int(file_name.split('_')[0]))
sorted_file_list = sorted(sort_num)

for i in tqdm(range(1,len(chunk_text)), desc="Processing", total=len(chunk_text)-1, unit="item"):
    try:
        # story_name_en = 'shediaoyingxiongzhuan'
        raw_text = chunk_text[ i ]
        dialoge_file = os.path.join(save_folder, f"{sorted_file_list[i]}_dialogue.txt")
        summarzie_file = os.path.join(save_folder, f"{sorted_file_list[i]}_sum.txt")

        chunk_sum = []
        if os.path.exists(summarzie_file):
            with open(summarzie_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('-'):
                        chunk_sum.append(line.strip()[1:].strip())
        # 这里需要对"I'm sorry, but I can't fulfill this request."这种回复重新抽取
        if chunk_sum ==[]:
            print(f"第{i}段chunk_sum = [], 尝试重新抽取中...")
            system_prompt = """
            Summarize the key points of the following text in a concise way, using bullet points, use " - " before the every summary
            """

            q_example = """###
            Text:
            Professor McGonagall's voice trembled as she went on. "That's not all. They're saying he tried to kill the Potter's son, Harry. But -- he couldn't. He couldn't kill that little boy. No one knows why, or how, but they're saying that when he couldn't kill Harry Potter, Voldemort's power somehow broke -- and that's why he's gone.　　Dumbledore nodded glumly.
            "It's -- it's true?" faltered Professor McGonagall. "After all he's done... all the people he's killed... he couldn't kill a little boy? It's just astounding... of all the things to stop him... but how in the name of heaven did Harry survive?"
            "We can only guess," said Dumbledore. "We may never know."

            Summarize in BULLET POINTS form:
            """

            a_example = """
            - Voldemort's attempt to kill Harry Potter failed, leading to his downfall.
            - Professor McGonagall expresses astonishment at Voldemort's failure to kill Harry. Dumbledore think that the reason for Harry's survival may never be fully understood.
            """
            messages = [SystemMessage( content = system_prompt),
                HumanMessage( content = q_example),
                AIMessage( content = a_example)]

            new_input = f"""###
            Text:
            {raw_text}

            Summarize in BULLET POINTS form:"""
            messages.append( HumanMessage(content = new_input) )
            summarize_response = llm( messages ).content
            # 重写
            with open(summarzie_file, 'w', encoding='utf-8') as f:
                f.write( summarize_response )
            with open(summarzie_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('-'):
                        chunk_sum.append(line.strip()[1:].strip())
            if chunk_sum != []:
                print(f"第{i}段-总结-重新抽取失败")
            #else:
            #    print("重新抽取失败")


        if os.path.exists(dialoge_file):
            with open(dialoge_file, encoding='utf-8') as f:
                dialogues = []
                for line in f:
                    dialogue = json.loads(line)
                    dialogues.append(dialogue)
        # 这里需要对对话为0的情况下重新抽取
        judge_dia = False
        if dialogues != []:
            try:
                if not 'dialogue' in dialogues[0]:
                    pass
            except:
                print(dialogue)
            if not 'dialogue' in dialogues[0]:
                judge_dia = True

        if dialogues ==[] or judge_dia:
            print(f"第{i}段dialogues = [], 尝试重新抽取中...")
            messages_dia = [SystemMessage( content = task_prompt_dia),
                            HumanMessage( content = example_in_dia),
                            AIMessage( content = example_out_dia)]
            re_input_text = '\n###\n' + raw_text
            messages_dia.append( HumanMessage(content = re_input_text) )
            response = llm( messages_dia ).content
            json_result = csv_to_json(response)

            with open(dialoge_file, 'w', encoding='utf-8') as f:
                for chat in json_result:
                    json_str = json.dumps(chat, ensure_ascii=False)
                    f.write(json_str + '\n')
            with open(dialoge_file, encoding='utf-8') as f:
                dialogues = []
                for line in f:
                    dialogue = json.loads(line)
                    dialogues.append(dialogue)
                if dialogues == []:
                    print(f"第{i}段-对话-重新抽取失败")

        unique_dialogue = []
        for item in dialogues:
            if item not in unique_dialogue:
                unique_dialogue.append(item)
        # 当抽取犯病的时候这个会炸，问题不大
        # {'role': 'role', 'Mr. Sedley': 'Amelia', "I shall leave the fellow half my property, he said":'She did not speak.'}
        
        dia_texts = [data['dialogue'] for data in unique_dialogue]   
        unique_chunk_sum = []
        for item in chunk_sum:
            if item not in unique_chunk_sum:
                unique_chunk_sum.append(item)
        chunk_sum = unique_chunk_sum
        dialogues = unique_dialogue
        lines, starts, ends = divide_raw2lines(raw_text)

        if len(chunk_sum)> len(lines):
            # 这个一般是出现一些别的问题，直接忽略可以，比如lines是标题，总结根据一句标题分析了一大堆
            print(f"第{i}段出错，原因：原文小于总结数，建议查看源文本")
            continue
        if chunk_sum == []:
            # 一般是抽取失败了
            print(f"第{i}段出错，原因：chunk_sum = []")

        score, divs = summary2line(chunk_sum, lines)  #summary匹配
        seq = dialogue2line(dia_texts, lines) #对话匹配

        combined_data, combined_text = jsonl_sorted(dialogues, chunk_sum, divs.copy(), dia_texts, seq.copy())
        # 如果需要保存每个chunk的，在此处保存
        final_jsonl.append(combined_data)
        final_txt = final_txt + combined_text + "\n"
    except:
        print("第" + str(i) + "个chunk出错")
        pass
print(save_jsonl_path)
with open(save_jsonl_path, "w", encoding="utf-8") as file:
    # 遍历数据列表中的每个字典
    for record in final_jsonl:
        # 将字典转换为JSON格式的字符串
        json_record = json.dumps(record, ensure_ascii=False)
        # 将转换后的JSON字符串写入文件，并添加换行符
        file.write(json_record + "\n")
with open(save_txt_path, "w", encoding="utf-8") as file:
    file.write(final_txt)



# 对特定小说抽取chatbot

# 读取总结之后输出的jsonl
# 总结所有角色出现的频率，可视化
# 想个办法截取特定角色
# 如果太长则切一下

# 参数设置

# 支持跨越多少行寻找目标角色，也即控制段内行间距不超过该值
max_find_lines = 10  # 要改后面对应函数

max_token_num = 500 # 要改后面对应函数，最大token数

# target_role支持 空字符串(默认前三个)或者List of string 如果出错默认保存第一个
target_role = ['郭靖', "欧阳锋"]
target_role = ['']

# 输入文件路径

#input_name = save_jsonl_path 
if os.path.exists(save_jsonl_path):
    print("true")
else:
  print("未找到文件")
# 保存路径
savepath = f"{novel_folder}/reorganized_story_{story_name_en}/texts"
if not os.path.exists(savepath):
  print("need_build")
  os.makedirs(savepath)

print("save_jsonl_path: ",save_jsonl_path)
with open(save_jsonl_path, encoding='utf8') as f:
    first_line = f.readline().strip()  # 读取第一行并去除首尾空白字符
    try:
        json.loads(first_line)  # 尝试解码第一行
        print("第一行是有效的 JSON 数据。")
    except json.JSONDecodeError as e:
        print("第一行不是有效的 JSON 数据。")
        print("错误消息:", str(e))


# input_name = '/content/reorganized_xiaoaojianghu.jsonl'
# 用utf8编码为我读取jsonl文件，并且把每一行解析成一个list of json, data

# # 输入文件路径
print(f"novel_folder:{novel_folder}")

enc = tiktoken.get_encoding("cl100k_base")

data_in_chunk = []
with open(save_jsonl_path, encoding='utf8') as f:
    for line in f:
      data_in_chunk.append(json.loads(line))

data = []
for chunk in data_in_chunk:
    for d in chunk:
        data.append(d)
        
Process_variables["len of json"] = len(data)
print("第二部分json行数, len of json: ",len(data))
for i,d in enumerate(data):
    if d['role'] == 'scene':
        first_scene_id = i
        break
print("first_scene_id: ", first_scene_id)
Process_variables["first_scene_id"] = first_scene_id

role_counts = Counter()
for line in data:
    role = line['role']
    role_counts[role] += 1
    #if role == "-":
    #  print(line)

common_roles = role_counts.most_common(args.max_roles)
status = 0

role_name = []
role_name_count = []

for role, count in common_roles:
    status = status + 1
    if role != 'scene':
        role_name.append(role)
    if status % 5 == 0:
        print(role, count)
    else:
        print(role, count, end=' ')
    role_name_count.append([role,count])
sorted_roles = sorted(common_roles, key=lambda x: x[1], reverse=True)
sorted_roles_clear = [role[0] for role in sorted_roles if role[0] != "scene"]
Process_variables["role_name_count"] = role_name_count

# 名为role_best的list of str，其按升序记录了所有的角色，
# 请编写一个Python程序，其判断target_role的类型，如果是空字符串或者None，就将role_best前三行提取到一个名为role_extract的list of string；
# 如果target_role为一个list of string，则逐个判断其中的每一个字符串是否在role_best中出现，如果出现，则提取到一个名为role_extract的list of string，否则打印一行错误信息，
# 在遍历完target_role后，如果role_extract不足三个，就用role_best从前往后补充至三个

def extract_roles(role_best, target_role):
    role_extract = []

    if target_role is None or target_role == ""or target_role == [""]:
        role_extract = role_best[:3]
    elif isinstance(target_role, list):
        for role in target_role:
            if role in role_best:
                role_extract.append(role)
            else:
                print(f"Error: Role '{role}' not found in role_best!")
    else:
        print("Error: Invalid target_role type!")
        return

    if len(role_extract) < 1:
        additional_roles = role_best[:1 - len(role_extract)]
        role_extract.extend(additional_roles)

    return role_extract
role_extract = extract_roles(sorted_roles_clear, target_role)


def output_scene_chat_id(data, target_role_single):

  chat_ids = []

  # 先寻找所有出现角色的节点
  for i,d in enumerate(data):
      if d['role'] == target_role_single:
          chat_ids.append(i)

  previous_scene_ids = []

  # 对于每一个chat_ids，向前寻找scene的节点
  for chat_id in chat_ids:
      ans = first_scene_id
      for j in range(chat_id, first_scene_id,-1):
          if data[j]['role'] == 'scene':
              ans = j
              break
      previous_scene_ids.append(ans)
  return chat_ids, previous_scene_ids


## 分块，决定texts组织内容
def divide_chats2chunks(chat_ids, previous_scene_ids):
    max_find_lines = 12

    chat_ids_in_chunk = []
    current_chunk = []

    for chat_id in chat_ids:
        if not current_chunk:
            current_chunk.append(chat_id)
            continue

        if chat_id - current_chunk[-1] <= max_find_lines:
            current_chunk.append(chat_id)
        else:
          chat_ids_in_chunk.append(current_chunk)
          current_chunk = [chat_id]

    if current_chunk:
        chat_ids_in_chunk.append(current_chunk)

    chat_id2previous_scene_id = {}

    for previous, chat_id in zip(previous_scene_ids, chat_ids):
        chat_id2previous_scene_id[chat_id] = previous
        if previous > 0:
            if data[previous-1]['role'] != target_role:
                chat_id2previous_scene_id[chat_id] -= 1
    # chat_ids的分块， chat_id对应的旁白id
    return chat_ids_in_chunk, chat_id2previous_scene_id

# 组织texts
# 计算一下每一句所花的token数量
# 
scene_may_set = ['旁白', '', 'scene','Scene','narrator' , 'Narrator']

def count_token( my_str ):
    return len(enc.encode(my_str))
def data2str( data ):
    role = data['role']
    if role in scene_may_set:
        return 'scene:' + data['text']
    else:
        return role + ':「' + data['text'] + '」'
    
# 我们现在需要把这东西变成最终的texts文本

def id2texts(data, chat_ids_in_chunk, chat_id2previous_scene_id):
    # chat_ids_in_chunk：一个列表，其中每个元素是一个包含对话 ID 的子列表，表示一个文本块中的对话。
    # 分好块后的，例如 [1,2,4],[100,110,120],[555]
    # 每个对话节点的token数量

    #max_token_num = 2000
    max_token_num = args.max_token_num_dia
    line_token = [count_token(data2str(d)) for d in data]
    from ast import Break

    # final_chunks，用于存储最终的文本块。
    # print_count，用于跟踪打印次数。
    # appended_key，用于记录已添加的关键索引。
    final_chunks = []
    print_count = 0
    appended_key = []
    appended_start = []

    role_chunks = []
    a = 0
    max_token = 0
    judge = 0 # 判断是否分段
    for chunk in chat_ids_in_chunk:
        N = len(chunk)
        current_i = 0
        judge = 0
        while current_i < N-1:
            # 找到当前id 以及对应的 场景id
            consider_chat_id = chunk[current_i]
            previous_scene_id = chat_id2previous_scene_id[consider_chat_id]
            # 理论上是希望max_token按分段,就是上面注释掉的原代码，但是由于所有的一组对话的背景都是一个
            # 导致start都是一样的，最终导致分段失败
            #保底
            withdraw_start = previous_scene_id
            withdraw_end = consider_chat_id
            current_count = sum(line_token[previous_scene_id:consider_chat_id+1])

            if judge == 1:
                withdraw_start = consider_chat_id
                current_count = sum(line_token[withdraw_start:consider_chat_id+1])

            while current_count < max_token_num and current_i < N-1:
                consider_end = chunk[current_i + 1]
                consider_count = sum(line_token[previous_scene_id:consider_end+1])

                if judge == 1:
                    consider_count = sum(line_token[withdraw_start:consider_end+1])
                if consider_count < max_token_num:
                    current_count = consider_count
                    if not judge == 1:
                        withdraw_start = previous_scene_id
                    withdraw_end = consider_end
                    current_i += 1
                else:
                    break

            if withdraw_end+1 not in appended_key:

                chunk_str = ''
                role_chunks_s = []
                max_token = max(max_token,sum(line_token[withdraw_start:withdraw_end+1]))

                if appended_start and (appended_start[-1] == withdraw_start):
                    for i in range(withdraw_start, withdraw_end+1):
                        chunk_str += data2str(data[i]) + '\n'
                        role_chunks_s.append(data[i])

                    role_chunks[-1] = role_chunks_s
                    final_chunks[-1] = chunk_str
                    appended_key[-1] = withdraw_end+1
                else:
                    appended_key.append(withdraw_end+1)
                    appended_start.append(withdraw_start)

                    if judge == 1:
                        chunk_str += "(接上段)" + '\n'
                        while data[previous_scene_id]['role'] in scene_may_set:
                            chunk_str += data2str(data[previous_scene_id]) + '\n'
                            previous_scene_id +=1
                    for i in range(withdraw_start, withdraw_end+1):
                        chunk_str += data2str(data[i]) + '\n'
                        role_chunks_s.append(data[i])
                    role_chunks.append(role_chunks_s)
                    final_chunks.append(chunk_str)

            if current_i < N-1:       # 这个操控judge 1还是0
                judge = 1       # 按token分段
                judge = 0       # 不分段，token小于10k感觉可以不用分段

            current_i += 1
    print("max_token:",max_token)
    #return appended_key, final_chunks
    return appended_key, final_chunks, role_chunks

# 找到主要人物和次要人物对话的所有段落
# 主角和配角，对于主角为role_cur_name，找到他与特定配角role_supporting_name，所有对话id（在主角的的块上的id）

def lead_support_dia_id(role_cur_name, role_supporting_name, role_chunks):
    n = 0
    dia_in_role_id = [] # 主角和次要人物对话的所有段落的id

    for chunk_a in role_chunks:
        count = 0
        for i in range(len(chunk_a)):
            if chunk_a[i]['role'] == role_supporting_name:
                count += 1
            if count>2:        # 出场次数超过两次认为算出场
                dia_in_role_id.append(n)
                break
        n += 1
    return dia_in_role_id


#重写场景以及tag prompt - new, 简单处理下，token更少点（大概吧)

if args.language == 0:  # 英文
    system_prompt_scene_sum = """
    Combine 'scene=' and dialogue, use a sentence to rewrite a new scene description, and output scene or dialogue style tags, Tag is two words, separate words with spaces' '. The rewritten scene and tag output start with '- scene=' and '- tag=' respectively
    """
    q_example_scene_sum = """###
    Text:
    scene: Di Gong was thinking alone in the inn when he was suddenly visited by a young man wearing a soap robe, who was Li Yuanfang.
    dialogue: [{'role':'Di Gong','text':'Who are you?'}, {'role': 'Youth', 'text': 'It is said that Di Gong is a god of reasoning, often able to cut off one's identity through temperament and clothing. Xiao Ke is just about to see and see.'}, {'role ':' Di Gong ',' text ':' I think when you visit late at night, you don't want to play hide and seek with me, do you? '}, {'role': 'Youth', 'text': 'I just want to prove that Di Gong is really as divine as the legend, or is he just a notorious figure?' {'role ':' Di Gong ',' text ':' I am over the age of ancient times and have long passed the age of competitiveness. My reputation is even more of an outsider to me. Moreover, I, Di Huaiying, have a false reputation and real talent, which may not be something that a young person like you can say in just one sentence. '} {'role': 'Youth', 'text': 'This should be considered eloquent words.'} {'role ':' Di Gong ',' text ':' Whatever you think. '. However, I have a premonition that there may be some gains today, and in order not to waste time, I have decided to give it a try, {'role ':' Youth ',' text ':' Please go ahead. '}, {'role': 'Di Gong', 'text': 'With a straight waist, legs slightly apart, hands recorded, the typical sitting posture of a subordinate officer in the Guard. His face was haggard, his complexion pale, but his cheeks were flushed, which was caused by excessive blood loss. This can be proven by the bloodstains oozing from your shoulders. In such a late night, sneaking into the room from the window to see me, he must not want to be discovered. So, who else could be an officer who was seriously injured and his whereabouts mysterious? Li Yuan? Fang, the captain of the security guard escorting the Turkic mission, the first wanted criminal in the court
    Summarize scene and tag in Text:
    """
    a_example_scene_sum = """
    - scene = In the inn, Di Gong was thinking alone when a young man wearing a soap robe suddenly visited. This young man was Li Yuanfang, and the two engaged in a series of conversations and reasoning.
    - tag = Inference Suspense
    """
elif args.language == 1: # 中文
    #  Do not involve the content of the conversation,
    system_prompt_scene_sum = """
    Combine 'scene=' and dialogue, use a sentence to rewrite a new scene description, and output scene or dialogue style tags, Tag is two words, separate words with spaces' '. The rewritten scene and tag output start with '- scene=' and '- tag=' respectively
    """
    q_example_scene_sum  = """###
    Text:
    scene: 狄公在驿馆中独自思考，被一名身穿皂袍的年轻人突然访问，这个年轻人就是李元芳。
    dialogue: [{'role': '狄公', 'text': '你是谁？'}, {'role': '青年', 'text': '都说狄公推理如神，常能以气质衣着断人身份，小可正想见识见识。'}, {'role': '狄公', 'text': '我想，你深夜来访，总不是想和我玩捉迷藏吧？'}, {'role': '青年', 'text': '我只想证明一下，狄公真像传说中那么神，还是浪得虚名。'}, {'role': '狄公', 'text': '我已年逾古稀，早就过了争强好胜的年纪，名声对我来说更是身外之物。而且，我狄怀英是浪得虚名，还是有真才实学，恐怕也不是你一个年轻人一句话就能评说的。'}, {'role': '青年', 'text': '这应该算是巧言令色吧。'}, {'role': '狄公', 'text': '随你怎么想。不过，我已经预感到，今天可能会有些收获，为了不浪费时间，我还是决定试一试。'}, {'role': '青年', 'text': '请吧。'}, {'role': '狄公', 'text': '腰杆挺直，腿微分，双手据案，典型的卫军下级军官的坐姿。面容憔悴，脸色苍白，而双颊却有红晕，此乃失血过多所致，这一点，从你双肩渗出的血迹可以得到证明。如此深夜，从窗户潜进房中见我，定是不欲被人发现行迹。那么，一个军官，身负重伤，行踪诡秘，还会是谁呢？李元芳，护送突厥使团的卫队长，朝廷第一号通缉犯！'}, {'role': '李元芳', 'text': '如不是亲眼所见，我真是不敢相信！不错，我正是李元芳。'}]
    Summarize scene and tag in Text:
    """
    a_example_scene_sum  = """
    - scene = 驿馆中，狄公在独自思考，此时一名身穿皂袍的年轻人突然造访，这个年轻人就是李元芳，二人展开了一系列的对话和推理。
    - tag = 推理 悬疑
    """
else:
    print("warning! Currently not supported for other languages")


 #利用gpt4重写场景，顺便把tag也搞了
 #不用gpt就是简单的把场景提取出来，tag=\[ \]
 #通过use_gpt_rewrite参数控制，默认开启
def re_scene_tag(n, role_chunks, dia_in_role_id, use_gpt=True):  # 主角与配角的第n段对话
    scene_sum = ''
    dia_data = []
    for chunks in role_chunks[dia_in_role_id[n]]:
        if chunks['role'] in scene_may_set:
            scene_sum += chunks['text']
        else:
            chunk_no_scene = {}
            chunk_no_scene['role'] = chunks['role']
            chunk_no_scene['text'] = chunks['text']
            dia_data.append(chunk_no_scene) # 去除场景的所有对话
    tag_new = []

    if not args.use_gpt_rewrite:
        return  dia_data, scene_sum, tag_new
    messages = [SystemMessage( content = system_prompt_scene_sum),
                HumanMessage( content = q_example_scene_sum),
                AIMessage( content = a_example_scene_sum)]
    # 新prompt
    new_input = f"""###
    Text:
    scene: {scene_sum}
    dialogue: {dia_data}
    Summarize scene and tag in Text:"""

    messages.append( HumanMessage(content = new_input) )
    response = llm( messages )
    content = response.content

    try:
        content = content.replace("tag=", "tag =")
        scene, tag = content.split('- tag =')
    except:
        print("warning:",content)
        scene, tag = content.split('\n')
    scene_new = scene.replace("- scene = ", "").replace("- scene=", "").strip()
    tag = tag.strip()
    tag_new = tag.split()

    return  dia_data, scene_new, tag_new

person_Template= {
    "name": "",
    "data": {
        "age": "",
        "gender": "",
        "personality": "",
        "catchphrases": [],
        "description": "",
        "emoji": False,
        "expression": True,
        "forbidden": [],
        "knowledge": ""
    }
}
data_Template = {
            "type": "character",
            "role": {
                "bot": {
                    "name": "",
                    "age": "",
                    "gender": "",
                    "personality": "",
                    "catchphrases": [],
                    "description": "",
                    "emoji": False,
                    "expression": True,
                    "forbidden": [],
                    "knowledge": ""
                },
                "user": {
                    "name": "",
                    "description": ""
                }
            },
            "scene": "",
            "tags": [],
            "relation": "",
            "messages": []
        }

# 筛选双人对话
two_person_list = {}
def filter_dialogue(role_chunks, leading_role, two_person_list, if_deduplication=True):
    dialogue_two_person = [] # 存储筛选后的对话, 对话id-配角名称

    #for chunk_a in role_chunks:
    for i in range(len(role_chunks)):   # role_chunks 多个对话块组成的list
        role_list = {}            # 对话中，其他角色列表及出现次数
        leading_count = 0         # 主角对话计数
        scene_count = 0           # 场景计数，去除
        for chunk in role_chunks[i]:   # role_a 一个对话块，多轮对话
            role = chunk['role']
            if role == leading_role:
                leading_count += 1
            elif role in scene_may_set:
                scene_count += 1
            elif role in role_list:
                role_list[role] += 1
            else:
                role_list[role] = 1
        # 其他角色最大对话计数('role', max_other_count)
        support_role_maybe=['',0]
        if role_list:
            support_role_maybe = max(role_list.items(), key=lambda x: x[1])
        else:
            print(f"warning! no others! {leading_role}的第{i}段对话，对话总长度:{len(role_chunks[i])}，主角对话长度：{leading_count}，场景长度：{scene_count}")
        if (len(role_chunks[i])-scene_count) <= 0:
            continue
        # 主角对话占比超过80%
        if (support_role_maybe[1]+leading_count) / (len(role_chunks[i])-scene_count) >= args.dialogue_ratio:
            # 将[对话id, 配角名称]添加到主角筛选对话的list中
            if not if_deduplication:
                dialogue_two_person.append([i, support_role_maybe[0]])
            # 接下来去重
            else:
                if not leading_role in two_person_list or support_role_maybe[0] != two_person_list.get(leading_role):
                    dialogue_two_person.append([i, support_role_maybe[0]])
                    two_person_list[support_role_maybe[0]] = leading_role
                else:
                    continue
    return dialogue_two_person


# 首先找到提取的人

if args.fix_roles_dia >=1:
    role_num = args.fix_roles_dia       # 这个是强制提取多少人，设置这个不用跑后面
else:
    role_num = len(sorted_roles_clear)
    # 提取对话数大于30的人
    for i in range(len(sorted_roles_clear)):
        for name, number in common_roles:
            if name == sorted_roles_clear[i]:
                break
        if number <= args.min_roles_dia:
            role_num = i-1
            break
    if role_num < args.min_roles:
        role_num = args.min_roles
  



# 最终

print(f"共{role_num}个符合条件的角色, 角色名如下：")
Process_variables["role_num"] = role_num
print(' '.join(sorted_roles_clear[:role_num]))


print("")
uni_scene_id = []
data_all = []
# 对于每个人提取对话
two_person_list= {}
for i in range(0,role_num):
    leading_role = sorted_roles_clear[i]
    print(f"第{i+1}个角色：{leading_role}")

    # 对话分块，联系场景
    chat_ids, previous_scene_ids = output_scene_chat_id(data, leading_role)

    # chat_ids_in_chunk：分块后的对话id, chat_id2previous_scene_id：对话id-先前的场景id键值对
    chat_ids_in_chunk, chat_id2previous_scene_id = divide_chats2chunks(chat_ids, previous_scene_ids)

    # appended_key：结束id, final_chunks：对话块, role_chunks：键值对形式的对话（以leading_role为主，包含场景）
    appended_key, final_chunks, role_chunks = id2texts(data, chat_ids_in_chunk, chat_id2previous_scene_id)

    dialogue_two_person = filter_dialogue(role_chunks, leading_role, two_person_list, True) # 这样抽会重复，感觉有点问题
    dia_in_two_role_id = [item[0] for item in dialogue_two_person if isinstance(item[0], int)]

    for j in tqdm(range(len(dialogue_two_person))):
        data_dig2 = copy.deepcopy(data_Template)
        #try:
        # dia_data，去除场景的对话
        dia_data, scene_new, tag_new = re_scene_tag(j,role_chunks,dia_in_two_role_id)
        #dia_data, scene_new, tag_new = re_scene_tag(j,role_chunks,dia_in_two_role_id,False)
        data_dig2['role']['bot']['name'] = leading_role           # 主角角色名
        data_dig2['role']['user']['name'] = dialogue_two_person[j][1]    # 配角角色名
        data_dig2['scene'] = scene_new
        data_dig2['tags'] = tag_new
        for data_part in dia_data:
            message = {}
            if args.adopted_for_others == 0:
                if data_part['role'] == leading_role or data_part['role'] == dialogue_two_person[j][1]: # 只保留两人对话，其他人的话丢弃
                #if data['role']:      # 将其他人的话安在配角身上
                    message = {
                        "content": data_part['text'],
                        "role": "bot" if data_part['role'] == leading_role else "user"
                    }
                    data_dig2["messages"].append(message)
            elif args.adopted_for_others == 1:
                if data['role']:      # 将其他人的话安在配角身上
                    message = {
                        "content": data_part['text'],
                        "role": "bot" if data_part['role'] == leading_role else "user"
                    }
                    data_dig2["messages"].append(message)
            elif args.adopted_for_others == 2:
                if data['role']:      # 将其他人的话安在主角身上
                    message = {
                        "content": data_part['text'],
                        "role": "user" if data_part['role'] ==  dialogue_two_person[j][1] else "bot"
                    }
                    data_dig2["messages"].append(message)  
            else:
                print("There is no strategy adopted by a third party for a two person conversation like this! ")             
        data_all.append(data_dig2)

# 存之前等一下：这里要把人称代词，空值和 unknown重跑

print("抽取修复ing...")
# 参数


max_distance = 50000    # 判断对话的最大字母距离
max_token = 100000      # 判断对话+加上下文的最大token
context_word1 = 10000   # 上下文只找前后文的word数
context_word2 = 2000    # 上下文前后文word数+原文


# 找到原文索引
def find_all_occurrences(text, pattern):
    occurrences = []
    start_index = 0

    while True:
        index = text.find(pattern, start_index)
        if index == -1:
            break
        occurrences.append(index)
        start_index = index + 1
    return occurrences

# 暴力
def find_approximate_numbers(position_set):
    position_set1 = []
    max_find_num_in_lst = 3
    judge_pos_force = []

    for lst in position_set:
        if lst == [] or len(lst) >= max_find_num_in_lst:
            continue
        position_set1.append(lst)
        if len(lst) == 1:
            # 收集所有单list的文件
            judge_pos_force.append(lst)
    position_set = position_set1

    judge_pos_force.sort()
    judge_pos_new =[]
    mid = len(judge_pos_force) // 2
    for lst in judge_pos_force:
        if abs(lst[0] - judge_pos_force[mid][0])<10000:
            judge_pos_new.append(lst)
    if len(judge_pos_new) >= len(judge_pos_force)//2 and len(judge_pos_new) >2:
        # 不用暴力
        result = [judge_pos_new[0][0], judge_pos_new[-1][0]]
        return result
    
    
    print("暴力求解ing...等待...")
    print("求解数组: ",position_set)
    # 暴力，很慢，数太大了
    def generate_combinations(position_set, current_combination, index, result):
        if index == len(position_set):
            if len(current_combination) > len(position_set) / 2:
                result.append(current_combination)
            return

        generate_combinations(position_set, current_combination + [], index + 1, result)
        for num in position_set[index]:
            generate_combinations(position_set, current_combination + [num], index + 1, result)
    
    Force_result = []
    generate_combinations(position_set, [], 0, Force_result)
    print("暴力求解end")
    
    up_result = []
    for lst in Force_result:
        judge = 0
        for i in range(len(lst) - 1):
            if lst[i] >= lst[i+1]:
                judge +=1
                break
        if judge == 0:
            up_result.append(lst)
    distance = up_result[0][-1]-up_result[0][0]
    result = [up_result[0][0], up_result[0][-1]]
    for up_re in  up_result:
        if up_re[-1]-up_re[0] < distance:
            distance = up_re[-1]-up_re[0]
            result = [up_re[0], up_re[-1]]
    return result





def find_nearest_number(position_set):
    nearest_pair = [0,-1]
    distance = -1
    for pos_start in position_set[0]:
        start = pos_start
        mid = pos_start
        
        for i in range(1, len(position_set)):
            for pos_mid in position_set[i]:
                if pos_mid > mid:
                    mid = pos_mid
                    break
        if (mid-start>0 and mid-start<distance) or distance<=0:
            distance = mid-start
            nearest_pair[0] = start
            nearest_pair[1] = mid
    if nearest_pair[1] - nearest_pair[0] > max_distance:

        nearest_pair = find_approximate_numbers(position_set)
    return nearest_pair

def find_role(data,raw_text,role_who):
    dialogue = ""
    dialogue = dialogue + "scene: " + data["scene"] + " /n "
    position_set = []
    for message in data["messages"]:
        content = message["content"]
        # 添加 prompt
        if message["role"] == "bot":
            dialogue += data["role"]["bot"]["name"]
        else:
            dialogue += data["role"]["user"]["name"]
        dialogue = dialogue +":\"" + content  + "\" "

        # 寻找源文本
        small_content_num = 100
        if len(content)>200:
            occurrences = []
            now_con = 0
            while occurrences == []:
                content1 = content[now_con:now_con+small_content_num]
                occurrences = find_all_occurrences(raw_text, content1)
                if occurrences != []:
                    break
                now_con += small_content_num
                
                if now_con+small_content_num>len(content)-1:
                    occurrences = find_all_occurrences(raw_text, content[now_con:])
                    break
        else:
            occurrences = find_all_occurrences(raw_text, content)
        if occurrences != []: 
            position_set.append(occurrences)
    if position_set == []:
        return ""
    enc = tiktoken.get_encoding("cl100k_base")
    start_end = find_nearest_number(position_set)

    system_prompt_find = f"""
    Find who the "{role_who}" in the dialogue is based on the context, only output a name or a role, Don't have the same name as another character in the dialogue, Use specific person names instead of pronouns and null, don't use "he","speaker"," "
    """
    q_example_find = f"""###
    dialogue:
        scene: A man hiding in the closet emerges and extinguishes the candle, startling both the widow and Grip, who then fully recites "Polly put the kettle on."The man confronts the sleeping Barnaby, threatening the widow with his newfound knowledge of Barnaby's existence and implying he has power over her. /n He:"Stay, You teach your son well." She:"I have taught him nothing that you heard to-night. Depart instantly, or I will rouse him." He:"You are free to do so. Shall I rouse him?" She:"You dare not do that." He:"I dare do anything, I have told you. " She:"Would you kill him in his sleep?"
    context:
        But even this failed to awaken the sleeper. He turned over towards the fire, and his head drooped heavily upon it. The widow and her unwelcome visitor gazed at him and at each other for a moment, and then she motioned him towards the door.
        'Stay,' he whispered. 'You teach your son well.''I have taught him nothing that you heard to-night. Depart instantly, or I will rouse him.'
        'You are free to do so. Shall I rouse him?'
        'You dare not do that.'
        'I dare do anything, I have told you.'
        'Would you kill him in his sleep?' cried the widow, throwing herself between them.
    who is "she":
    """
    a_example_find = """
        widow
    """


    if len(enc.encode( raw_text[start_end[0]:start_end[1]]+dialogue ))>  max_token:
        print("token太多辣")
        print("context: ", len(enc.encode(raw_text[start_end[0]:start_end[1]])))
        print("start_end",start_end)
        print("dialogue: ", len(enc.encode(dialogue )))
        messages_find = [SystemMessage( content = system_prompt_find),
            HumanMessage( content = q_example_find),
            AIMessage( content = a_example_find)]
        new_input_find = f"###\ndialogue:\n{dialogue}\ncontext:\n{raw_text[start_end[0]-context_word1:start_end[0]+context_word1]}\nwho is \"{role_who}\":"
        messages_find.append( HumanMessage(content = new_input_find) )
        role = llm( messages_find ).content

        if role.strip() in ["narrator", "", "i", "you", "he", "she", "it", "they", "we", "unknown", "man", "woman", "-", "user", "speaker", "speaker 1", "speaker 2"]  or len(role)>30:
            messages_find = [SystemMessage( content = system_prompt_find),
                HumanMessage( content = q_example_find),
                AIMessage( content = a_example_find)]
            new_input_find = f"###\ndialogue:\n{dialogue}\ncontext:\n{raw_text[start_end[1]-context_word1:start_end[1]+context_word1]}\nwho is \"{role_who}\":"
            messages_find.append( HumanMessage(content = new_input_find) )
            role = llm( messages_find ).content
    else:
        messages_find = [SystemMessage( content = system_prompt_find),
            HumanMessage( content = q_example_find),
            AIMessage( content = a_example_find)]
        new_input_find = f"###\ndialogue:\n{dialogue}\ncontext:\n{raw_text[start_end[0]-context_word2:(start_end[0]+start_end[1])//2]}\nwho is \"{role_who}\":"
        messages_find.append( HumanMessage(content = new_input_find) )
        role = llm( messages_find ).content

        if role.strip() in ["narrator", "", "i", "you", "he", "she", "it", "they", "we", "unknown", "man", "woman", "-", "user", "speaker", "speaker 1", "speaker 2"]  or len(role)>30:
            messages_find = [SystemMessage( content = system_prompt_find),
                HumanMessage( content = q_example_find),
                AIMessage( content = a_example_find)]
            new_input_find = f"###\ndialogue:\n{dialogue}\ncontext:\n{raw_text[(start_end[0]+start_end[1])//2+1:start_end[1]+context_word2]}\nwho is \"{role_who}\":"
            messages_find.append( HumanMessage(content = new_input_find) )
            role = llm( messages_find ).content
    return role


role_bot_num = 0
role_user_num = 0
data_wrong = []
wrong_num = 0

progress_bar = tqdm(total=len(data_all))  # 这个进度条不准，凑合看看
count = 0
for line in data_all:
    count += 1
    # 解析 JSON 数据
    data = line
    modify = 0
    # 如果第一人称主角可以为"i"
    role_lead = data["role"]["bot"]["name"]
    role_sup = data["role"]["user"]["name"]
    if data["role"]["bot"]["name"].lower().strip() in ["narrator", "", " ", "you", "he", "she", "it", "they", "we", "unknown", "man", "woman","-", "user", "speaker", "speaker 1", "speaker 2", "girl", "boy"]:
        modify += 1
        role_bot_num += 1
        role_who = data["role"]["bot"]["name"]
        if data["role"]["bot"]["name"].lower().strip() in ["", " "]:
            judge_bot = 0
            for message_a in data["messages"]:
                if message_a["role"] == "bot":
                    judge_bot = 1
                    break
            if judge_bot == 0:
                continue
        try:
            role = find_role(data,raw_text,role_who)
            if role.strip() in ["narrator", ""," ", "i", "you", "he", "she", "it", "they", "we", "unknown", "man", "woman","-", "user", "speaker", "speaker 1", "speaker 2", "girl", "boy"] or role == data["role"]["user"]["name"] or len(role)>30:
                data_wrong.append(data)
                print(f"fail - lead - {role} - {count-1}行，（原主角{role_lead},原配角{role_sup}）")
                wrong_num += 1
                continue
            else:
                data["role"]["bot"]["name"] = role
        except:
            data_wrong.append(data)
            print(f"wrong:{count-1}数据")
            wrong_num += 1
               #aaa
    if data["role"]["user"]["name"].lower().strip() in ["narrator", "", " ", "i", "you", "he", "she", "it", "they", "we", "unknown", "-", "user", "speaker", "speaker 1", "speaker 2"]:  

        # 如果只有主角自言自语，没有配角""，跳过
        if data["role"]["user"]["name"].lower().strip() in ["", " "]:
            judge_bot = 0
            for message_a in data["messages"]:
                if message_a["role"] == "user":
                    judge_bot = 1
                    break
            if judge_bot == 0:
                continue
        modify += 1
        role_user_num += 1
        data_user = data
        role_who = data["role"]["user"]["name"]
        try:
            role = find_role(data,raw_text,role_who)
            if role.strip() in ["narrator", "", "i", "you", "he", "she", "it", "they", "we", "unknown","-", "user", "speaker", "speaker 1", "speaker 2"] or role == data["role"]["bot"]["name"] or len(role)>30:
                print(f"fail - sup - {role} - {count-1}行，（原主角{role_lead},原配角{role_sup}）")
                wrong_num += 1
                data_wrong.append(data)
                continue
            else:
                data["role"]["user"]["name"] = role
        except:
            data_wrong.append(data)
            wrong_num += 1
            print(f"wrong:{count-1}数据")
    progress_bar.update(1)
progress_bar.close()

print(f"wrong—role_bot_num: {role_bot_num}")            
print(f"wrong—role_user_num: {role_user_num}")   
print(f"wrong_num: {wrong_num}")



save_jsonl_path = f"{novel_folder}/{story_name_en}_dialogue.jsonl"
with open(save_jsonl_path, "w", encoding="utf-8") as file:
    # 遍历数据列表中的每个字典
    for record in data_all:
        # 将字典转换为JSON格式的字符串
        json_record = json.dumps(record, ensure_ascii=False)
        # 将转换后的JSON字符串写入文件，并添加换行符
        file.write(json_record + "\n")
shutil.copy(save_jsonl_path, f"{base_folder}/dataset_json/{story_name_en}_dialogue.jsonl")

done_file_path = f'{base_folder}/novel/done' 
if not args.is_series:        # 单本小说
    try:
        shutil.copy(f"{base_folder}/novel/{novel_name}.txt", f"{done_file_path}/{novel_name}.txt")
    except:
        pass
else:                       # 多部系列小说在一个文件夹下
    try:
        novel_file_path = f'{novel_folder}/{novel_name}'  
        shutil.copytree(f"{base_folder}/novel/{novel_name}",f"{done_file_path}/{novel_name}")
    except:
        pass

with open(save_variables+".jsonl", 'w') as file:
    json.dump(Process_variables, file, ensure_ascii=False)

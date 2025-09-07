# -*- coding: utf-8 -*-

import os
import random
import json
import time
import argparse
import openai  # 只锟斤拷锟斤拷 openai 模锟斤拷
from tqdm import tqdm
from src.tools_for_LLM4EA import *
from zhipuai import ZhipuAI

# engine = "deepseek-chat"
no_random = False
api_key = "your-api-key"  # 填写您的API密钥


def get_glm_response(prompt: str, api_key: str, model: str = "glm-4-flash") -> str:
    """
    调用GLM模型获取回答
    :param prompt: 用户输入的提示
    :param api_key: API密钥
    :param model: 使用的模型名称，默认为 "glm-4-flash"
    :return: GLM模型的回答
    """
    # 初始化客户端
    client = ZhipuAI(api_key=api_key)

    # 调用模型生成回答
    try:
        # 调用模型生成回答
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": " You are an expert in entity classification tasks."},
                {"role": "user", "content": prompt}
            ],
        )
    except Exception as e:
        print(f"Error encountered:{e}. Continuing with next task.")
        response = None
        return response

    # 返回生成的文本
    return response.choices[0].message


# def try_get_response(prompt, messages=[], max_tokens=100, max_try_num=3):
# 	try_num = 0
# 	flag = True
# 	response = None
# 	while flag:
# 		try:
# 			# request LLM
# 			response = openai.ChatCompletion.create(
# 				model=engine,
# 				messages=messages + [{"role": "user", "content": prompt}],
# 				max_tokens=max_tokens,
# 				temperature=0.2
# 			)
# 			flag = False
# 		except openai.OpenAIError as e:
# 			try_num += 1
# 			if try_num >= max_try_num:
# 				break
# 	return response, (not flag)
import requests

# def try_get_response(prompt, messages=[], max_tokens=100, max_try_num=3):
#     try_num = 0
#     response = None
#     while try_num < max_try_num:
#         try:
#             # 锟斤拷锟襟本碉拷 LLM
#             response = requests.post(f"http://localhost:8000/v1", json={
#                 "messages": messages + [{"role": "user", "content": prompt}],
#                 "max_tokens": max_tokens,
#                 "temperature": 0.2
#             })
#             response.raise_for_status()  # 锟斤拷锟斤拷锟斤拷锟斤拷欠锟缴癸拷
#             response_json = response.json()
#             # print('Response JSON:', response_json)  # 锟斤拷锟斤拷锟斤拷锟?
#
#             if "choices" in response_json and response_json["choices"]:
#                 full_text = response_json["choices"][0]["text"]
#                 if "[Output]:" in full_text:
#                     output_sentence = full_text.split("[Output]:")[-1].strip()
#                     return output_sentence, True
#                 else:
#                     raise ValueError("Invalid response format: '[Output]:' not found")
#             else:
#                 raise ValueError("Invalid response format")
#         except (requests.RequestException, ValueError) as e:
#             print(f"Attempt {try_num + 1} failed: {str(e)}")
#             try_num += 1
#
#     print('Final response:', response)
#     return response, False

import requests


def try_get_response(prompt, messages, max_try_num=3):
    try_num = 0
    flag = True
    response = None
    response_json = None
    url = "http://localhost:8001/v1"  # 本地大模型服务的URL
    log_print = True

    while flag and try_num < max_try_num:
        try:
            # 构建请求数据
            request_data = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2  # 设置 temperature 参数
            }

            # 发送请求到本地部署的大模型服务
            response = requests.post(url, json=request_data)

            # 检查请求是否成功
            if response.status_code == 200:
                response_json = response.json()
                if log_print:
                    print('### RAW RESPONSE:')
                # print(response_json)

                # 检查响应内容是否包含预期的字段
                if "choices" in response_json and len(response_json["choices"]) > 0 and "text" in \
                        response_json["choices"][0]:
                    flag = False
                else:
                    raise Exception("Unexpected response format")
            else:
                raise Exception(f"HTTP error: {response.status_code}")
        except Exception as e:
            if log_print:
                print(f'### ERROR: {e}')
            try_num += 1

    if log_print:
        if not flag:
            print('### RESPONSE:')
            print(response_json["choices"][0]["text"])
        else:
            print('### RESPONSE: [ERROR]')

    if response_json and "choices" in response_json and len(response_json["choices"]) > 0 and "text" in \
            response_json["choices"][0]:
        return response_json, True
    else:
        return {"choices": [{"text": ""}]}, False


def generate_prompt(entity_id, ent_info):
    # neighbors = [f"({', '.join(list(neigh))})" for neigh in entity["neighbors"]]
    #
    # prompt = f"Your task is to give a one-sentence brief introduction for given [Entity], based on 1.YOUR OWN KNOWLEDGE; 2.[Knowledge Tuples]. NOTICE, introduction is just one sentence and less than 50 tokens."
    # prompt += "Here is a example:\n[KNOWLEDGE]: Given [Entity] Gun Hellsivik and its related [Knowledge Tuples]: [(Gun Hellsvik, member of, Moderate Party)].\n[Input] What is Gun Hellsivik? Please give a one-sentence brief introduction based on YOUR OWN KNOWLEDGE and [Knowledge Tuples]\n[Output]: Gun Hellsvik was a Swedish politician and member of the Moderate Party, knownfor serving as the Minister of Justice in Sweden.\n"
    # prompt += "Now please answer:\n"
    # prompt += f"[KNOWLEDGE]: Given [Entity] {entity['name']} and its related [Knowledge Tuples]: [{', '.join(neighbors)}].\n"
    # prompt += f"[Input]: What is {entity['name']}? Please give a one-sentence brief introduction based on YOUR OWN KNOWLEDGE and [Knowledge Tuples].\n"
    # prompt += "[Output]: "
    desc = ent_info['desc']
    print(desc)
    name = ent_info['name']

    prompt = f'''Given the entity's name and description:
            Name: {name}
            Description: {desc}

            Please classify the entity into one of the following categories:
            1. **People**: Individuals or notable personalities.
            2. **Organizations**: Companies or groups.
            3. **Locations**: Countries, cities, or landmarks.
            4. **Events**: Historical or significant occurrences.
            5. **Creative Works**: Books, movies, or artworks.
            6. **Products**: Items or services for sale.
            7. **Scientific Concepts**: Terms or ideas in the scientific field.
            8. **Geographical Features**: Natural landmarks like mountains or lakes.

            Return only the category name in the following format:
            **Category**: [Your category here]

            - If the entity does not clearly fit one of the categories, return "unknown" without any additional text or explanation.
            - Do not include any examples or context in the output.
            - Ensure the output is strictly in the format: **Category**: [Your category here]

'''

    return prompt


def extract_output(res):
    output_start = res.find("[Output]:")

    if output_start != -1:

        output_text = res[output_start + len("[Output]:"):].strip()
        return output_text
    else:
        return "[ERROR] Output not found."


def process_res(res: str):
    ###############
    ### extract entity description from repsonse
    ###############
    text_list = res.strip().split(":")
    text = ":".join(text_list[1:]) if len(text_list) > 1 else text_list[0]
    text = text.strip().split("\n")[0]
    return text.strip()


def extract_category(model_output):
    import re

    print('model_output:', model_output)

    # 尝试匹配固定格式
    categories = re.findall(r'\*\*Category\*\*: ([\w\s]+)', model_output)

    # 如果没有匹配到内容，尝试匹配数字格式
    if not categories:
        number_categories = re.findall(r'\*\*Category\*\*: (\d)', model_output)
        if number_categories:
            # 将数字转换为文字类别
            category_map = {
                '1': 'People',
                '2': 'Organizations',
                '3': 'Locations',
                '4': 'Events',
                '5': 'Creative Works',
                '6': 'Products',
                '7': 'Scientific Concepts',
                '8': 'Geographical Features'
            }
            categories = [category_map.get(num, 'unknown') for num in number_categories]

    print('categories:', categories)

    return categories[0] if categories else 'unknown'


def get_description(idx, neigh=0, max_tokens=50):
    description = {}
    i = 1
    import json

    # 定义文件路径
    file_path = r'D:\jinquanxin\LLM4EA\data\FBYG15K\candidates\description1'
    output_file_path = r'D:\jinquanxin\LLM4EA\data\FBYG15K\candidates\description_with_category1_new.json'

    # 读取 JSON 文件
    def read_json_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def write_json_file(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    # 处理 JSON 数据
    def process_data(data):
        ent = {}
        for entity_id, entity_info in tqdm(data.items(), desc=f"Processing {idx}"):
            messages = ''
            # response, get_res = try_get_response(generate_prompt(entity_id,entity_info),messages)
            response = get_glm_response(generate_prompt(entity_id, entity_info), api_key)
            # response_content = response["choices"][0]['text']
            response_content = response.content if response is not None else ''
            print(response_content)
            category_info = extract_category(response_content)

            entity_info['category'] = category_info
            print(category_info)
            ent[entity_id] = entity_info

        write_json_file(ent, output_file_path)
        print(f"Data has been written to {output_file_path}")

    data = read_json_file(file_path)
    process_data(data)

    return description


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data")
parser.add_argument("--data", type=str, default="FBYG15K")
parser.add_argument("--desc_file", type=str, default="description")
parser.add_argument("--cand_file", type=str, default="cand")
parser.add_argument("--neigh", type=int, default=25)
parser.add_argument("--ent", type=int, default=0)
parser.add_argument("--max_tokens", type=int, default=80)
parser.add_argument("--no_random", action="store_true")
parser.add_argument("--random_seed", type=int, default=20231201)
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

### random setting
no_random = args.no_random
os.environ['PYTHONHASHSEED'] = str(args.random_seed)
random.seed(args.random_seed)

description = get_description(0)









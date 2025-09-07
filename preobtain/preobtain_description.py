import os
import random
import json
import time
import argparse
import openai
from tqdm import tqdm
from tools_for_ChatEA import *
import requests

engine = "deepseek-chat"
no_random = False
from zhipuai import ZhipuAI
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

    try:
        # 调用模型生成回答
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": " You are an expert in the task of generating entity descriptions."},
                {"role": "user", "content": prompt}
            ],
        )
    except Exception as e:
        print(f"Error encountered:{e}. Continuing with next task.")
        response = None
        return response

    # 返回生成的文本
    return response.choices[0].message
#
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
def try_get_response(prompt, messages=[], max_tokens=80, max_try_num=3):
    try_num = 0
    response = None
    while try_num < max_try_num:
        try:
            # 锟斤拷锟襟本碉拷 LLM
            response = requests.post(f"http://localhost:8001/v1", json={
                "messages": messages + [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2
            })
            response.raise_for_status()  # 锟斤拷锟斤拷锟斤拷锟斤拷欠锟缴癸拷
            response_json = response.json()
            # print('Response JSON:', response_json)  # 锟斤拷锟斤拷锟斤拷锟?

            if "choices" in response_json and response_json["choices"]:
                full_text = response_json["choices"][0]["text"]
                if "[Output]:" in full_text:
                    output_sentence = full_text.split("[Output]:")[-1].strip()
                    return output_sentence, True
                else:
                    raise ValueError("Invalid response format: '[Output]:' not found")
            else:
                raise ValueError("Invalid response format")
        except (requests.RequestException, ValueError) as e:
            print(f"Attempt {try_num + 1} failed: {str(e)}")
            try_num += 1

    print('Final response:', response)
    return response, False

def generate_prompt(entity):
    neighbors = [f"({', '.join(list(neigh))})" for neigh in entity["neighbors"]]

    import numpy as np
    import faiss

    from transformers import AutoModel, AutoTokenizer
    import torch
    local_model_path = '../model/all-MiniLM-L6-v2'

    # 从本地目录加载模型和分词器
    model = AutoModel.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # 读取FAISS索引和模板映射
    index = faiss.read_index("./npy_data/vector_database.index")
    template_map = np.load("./npy_data/template_map.npy", allow_pickle=True).item()

    def get_prompt(input_text):
        # 使用分词器将文本转化为模型输入
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        # 将输入传递给模型
        with torch.no_grad():
            output = model(**inputs)

        # 获取模型输出的最后一个隐藏状态（默认是[0]）
        input_vector = output.last_hidden_state.mean(dim=1).cpu().numpy()  # 平均池化获得句子的向量表示

        # 使用FAISS进行相似度检索
        D, I = index.search(input_vector, 1)  # 获取与输入文本最相似的模板索引
        best_match_index = I[0][0]  # 取最相似的索引
        prompt_template = template_map[best_match_index]  # 根据索引获取模板

        return prompt_template

    # 示例输入
    template = f"[KNOWLEDGE]: Given [Entity] {entity['name']} and its related [Knowledge Tuples]: [{', '.join(neighbors)}]."

    template_new = get_prompt(template)

    prompt = f"Your task is to give a one-sentence brief introduction for given [Entity], based on 1.YOUR OWN KNOWLEDGE; 2.[Knowledge Tuples]. NOTICE, introduction is just one sentence and less than 50 tokens."
    # prompt += "Here is a example:\n[KNOWLEDGE]: Given [Entity] Gun Hellsivik and its related [Knowledge Tuples]: [(Gun Hellsvik, member of, Moderate Party)].\n[Input] What is Gun Hellsivik? Please give a one-sentence brief introduction based on YOUR OWN KNOWLEDGE and [Knowledge Tuples]\n[Output]: Gun Hellsvik was a Swedish politician and member of the Moderate Party, knownfor serving as the Minister of Justice in Sweden.\n"
    prompt += f"Here is a example:{template_new}\n"
    prompt += "Now please answer:\n"
    prompt += f"[KNOWLEDGE]: Given [Entity] {entity['name']} and its related [Knowledge Tuples]: [{', '.join(neighbors)}].\n"
    prompt += f"[Input]: What is {entity['name']}? Please give a one-sentence brief introduction based on YOUR OWN KNOWLEDGE and [Knowledge Tuples].\n"
    prompt += "[Output]: "

    return prompt


def process_res(res:str):
    ###############
    ### extract entity description from repsonse
    ###############
    text_list = res.strip().split(":")
    text = ":".join(text_list[1:]) if len(text_list) > 1 else text_list[0]
    text = text.strip().split("\n")[0]
    return text.strip()

# def get_description(idx, ng:NeighborGenerator, entities, neigh=0, max_tokens=50):
#     description = {}
#     for eid in tqdm(entities, desc=f"{idx:2d}", position=idx):
#         ent = ng.get_neighbors(eid, neigh)
#         description[eid] = {"name": ent["name"], "desc": ""}
#         res, get_res = try_get_response(generate_prompt(ent), max_tokens=max_tokens)
#         desc_text = res["choices"][0]["message"]["content"] if get_res else "[ERROR]"
#         description[eid]["desc"] = process_res(desc_text)
#     return description
def write_file_new(data, file_path):
    # 追加模式写入文件
    with open(file_path, 'a',encoding='utf-8') as f:

        # 将字典转换为 JSON 字符串
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        json_data = json_data.lstrip('{').rstrip('}')
        # 写入文件，每个 JSON 对象后添加换行符
        f.write(json_data + ','+'\n')
def get_description(idx, ng: NeighborGenerator, entities, neigh=0, max_tokens=50):
    description = {}
    output_file_path = r'D:\jinquanxin\LLM4EA\data\FBDB15K\candidates\description1.json'

    i = 1
    for eid in tqdm(entities, desc=f"{idx:2d}", position=idx):
        desc={}
        ent = ng.get_neighbors(eid, neigh)
        description[eid] = {"name": ent["name"], "desc": ""}
        desc[eid] = {"name": ent["name"], "desc": ""}
        #res, get_res = try_get_response(generate_prompt(ent), max_tokens=max_tokens)
        res=get_glm_response(generate_prompt(ent),api_key)
        res_content=res.content if res is not None else ""
        print(res_content)
        # desc_text = res["choices"][0]["message"]["content"] if get_res else "[ERROR]"

        # if res is not None:
        #     res = res.split('\n')
        #     res = res[0]
        # else:
        #
        #     res = []

        #print('process_res:', res)
        desc_text = res_content if res else "[ERROR]"
        description[eid]["desc"] = str(desc_text)
        desc[eid]["desc"] = str(desc_text)
        write_file_new(desc, output_file_path)

        # description[eid]["desc"] = extract_output(desc_text)
        # 锟斤拷印每锟斤拷实锟斤拷锟斤拷锟斤拷锟斤拷锟较?

    return description


def generate_entity_description(data, data_dir, cand_file="cand", desc_file="description", ent=0, neigh=0, max_tokens=100, use_time=True):
    ng = NeighborGenerator(data=data, data_dir=data_dir, use_time=use_time, use_desc=False, cand_file=cand_file, desc_file=desc_file)
    entities = ng.get_all_entities()
    print('entities len:', len(entities))
    if ent > 0:
        if not no_random:
            random.shuffle(entities)
        entities = entities[:ent]
    st = time.time()
    
    description = get_description(0, ng, entities, neigh, max_tokens)

    time_cost = time.time() - st
    h, m, s = transform_time(int(time_cost))
    print(f'Time Cost : {h}hour, {m:02d}min, {s:02d}sec')

    return description


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data")
parser.add_argument("--data", type=str, default="FBDB15K")
parser.add_argument("--desc_file", type=str, default="description1_cot")
parser.add_argument("--cand_file", type=str, default="cand")
parser.add_argument("--neigh", type=int, default=25)
parser.add_argument("--ent", type=int, default=0)
parser.add_argument("--max_tokens", type=int, default=80)
parser.add_argument("--no_random", action="store_true")
parser.add_argument("--random_seed", type=int, default=20231201)
args = parser.parse_args()

### random setting
no_random = args.no_random
os.environ['PYTHONHASHSEED'] = str(args.random_seed)
random.seed(args.random_seed)

### change these settings, according to your LLM
openai.api_base = f"http://localhost:8000/v1"
# openai.api_base = f"https://api.deepseek.com"
# openai.api_key = "sk-a9e23d16bb2c46d2a7c20e7a81d6d6a2"
# engine = openai.Model.list()["data"][0]["id"]

#engine='deepseek-chat'
### generate entity description previously
use_time = True if args.data in ["icews_wiki", "icews_yago"] else False

description = generate_entity_description(data=args.data, data_dir=args.data_dir, cand_file=args.cand_file, desc_file=args.desc_file, ent=args.ent, neigh=args.neigh, max_tokens=args.max_tokens, use_time=use_time)

### save entity description
with open(os.path.join(args.data_dir, args.data, "candidates", args.desc_file), "w", encoding="utf-8") as fw:
    json.dump(description, fw, ensure_ascii=False, indent=4)


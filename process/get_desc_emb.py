import torch
from transformers import BertTokenizer, BertModel
import pickle
import json

# 初始化BERT模型和tokenizer
model_name = r'E:\code\实体对齐代码\model\bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 输入文件路径和输出文件路径
input_file = r'E:\code\实体对齐代码\实体对齐\my-LLM-EA\data\0_1\candidates\description'  # 替换为你的文本文件路径
output_file = 'entity_description_embeddings.pkl'

# 准备一个字典来保存实体id和描述嵌入向量
entity_embeddings = {}

# 读取实体描述文件，并将其转换为JSON格式
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 对每个实体描述进行编码
for entity_id, entity_info in data.items():
    description = entity_info.get('desc', '')

    if description:
        # 编码描述信息
        inputs = tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)

        # 将输入移动到GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # 获取[CLS]标记的嵌入向量
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # 保存到字典中
        entity_embeddings[int(entity_id)] = embedding

# 将结果保存为.pkl文件
with open(output_file, 'wb') as f:
    pickle.dump(entity_embeddings, f)

print(f"Entity embeddings saved to {output_file}")
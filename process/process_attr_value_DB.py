import torch
from transformers import BertTokenizer, BertModel
import pickle
import re

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_path = r'E:\code\实体对齐代码\model\bert-base-uncased'

# 加载预训练的 BERT 模型和 tokenizer，并转移到设备
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(tokenizer_path).to(device)

# 设置模型为评估模式 (不更新梯度)
model.eval()

# 格式化实体的属性和值
def format_attributes(entity_attributes):
    formatted_text = ""
    for attr, value in entity_attributes.items():
        formatted_text += f"[{attr}] {value} "
    return formatted_text.strip()

# 对实体的属性进行编码
def encode_attributes_with_structure(entity_attributes, max_length=512):
    formatted_text = format_attributes(entity_attributes)

    # 使用 tokenizer 编码输入文本，并转移到设备
    inputs = tokenizer(formatted_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)

    # 使用 BERT 模型进行前向传播，获取编码结果
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取 [CLS] token 的向量作为实体的嵌入表示
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # [batch_size, hidden_size]
    return cls_embedding

# 处理包含多个实体的文件，只保留属性和值部分
def process_entities(entity_file, id2name_file, output_file):
    # 从 id2name 文件中读取实体 ID 映射
    with open(id2name_file, 'r',encoding='utf-8') as f:
        id2entity = {line.split()[0]: line.split()[1] for line in f.readlines()}

    entity_embeddings = {}
    with open(entity_file, 'r',encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 3:
                continue  # 如果格式不正确，跳过该行

            entity_key = parts[0]  # 实体的键，例如 /m/06rf7
            attribute = parts[1].split('/')[-1]  # 属性，例如 location.geocode.longitude
            value = parts[2]  # 属性值，例如 9.70404945

            # 去除属性值中的不必要的后缀（例如 ^^^<...>）
            value = re.sub(r'\^\^<.*?>', '', value).strip('"')

            # 使用 id2entity 映射来获取实体的 ID
            if entity_key in id2entity.values():  # 确保 entity_key 在 id2entity 中
                # 获取实体 ID
                entity_id = [k for k, v in id2entity.items() if v == entity_key][0]
            else:
                print(f"警告: 未找到实体键 {entity_key} 对应的 ID")
                continue  # 跳过没有对应 ID 的实体

            # 将属性添加到对应实体 ID 的字典中
            if entity_id not in entity_embeddings:
                entity_embeddings[entity_id] = {}
            entity_embeddings[entity_id][attribute] = value

    # 为每个实体 ID 生成嵌入
    final_embeddings = {}
    for entity_id, attributes in entity_embeddings.items():
        embedding = encode_attributes_with_structure(attributes)
        final_embeddings[entity_id] = embedding.cpu().numpy()  # 转换为 NumPy 数组以便存储

    # 保存嵌入到 .pkl 文件
    with open(output_file, 'wb') as f:
        pickle.dump(final_embeddings, f)

    print(f"实体嵌入已保存至 {output_file}")

# 示例文件路径
entity_file = r'E:\学习资料\mmkb-master\DB15K\DB15K_NumericalTriples.txt'  # 实体文件
id2entity_file = './norm/ent_ids_2'  # id和实体映射文件
output_file = './norm/DB_attr_entity_embeddings.pkl'  # 输出的嵌入文件

# 处理并生成嵌入
process_entities(entity_file, id2entity_file, output_file)

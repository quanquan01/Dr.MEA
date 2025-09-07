import json


def read_file(file_path, encodings=['utf-8', 'gbk']):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return [line.strip().split('\t') for line in file]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode file {file_path} with any of the provided encodings")


def read_id2name(file_path):
    id2name = {}
    lines = read_file(file_path)
    for line in lines:
        id, name = line
        id2name[int(id)] = name
    return id2name


def build_entity_neighbors(id2name, triplet_files):
    entity_neighbors = {id: [] for id in id2name.keys()}

    for triplet_file in triplet_files:
        triplets = read_file(triplet_file)
        for triplet in triplets:
            if len(triplet) != 3:
                continue
            #head, relation, tail, time1, time2 = map(int, triplet)\
            head, relation, tail= map(int, triplet)

            if head in entity_neighbors:
                #entity_neighbors[head].append([head, relation, tail, time1, time2])
                entity_neighbors[head].append([head, relation, tail])
            if tail in entity_neighbors:
                #entity_neighbors[tail].append([head, relation, tail, time1, time2])
                entity_neighbors[tail].append([head, relation, tail])

    return entity_neighbors


def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# 指定文件路径
id2name1 = './norm/ent_ids_1_new_1'
id2name2 = './norm/ent_ids_2_new'
triplet_file1 = './norm/triples_1'
triplet_file2 ='./norm/triples_2'

# 读取实体ID
id2name1 = read_id2name(id2name1)
id2name2 = read_id2name(id2name2)
id2name = {**id2name1, **id2name2}

# 读取五元组文件
triplet_files = [triplet_file1, triplet_file2]

# 构建实体邻居字典
entity_neighbors = build_entity_neighbors(id2name, triplet_files)

# 保存到文件
output_file = './norm/candidates/neighbors_FB_YG.json'
save_to_file(entity_neighbors, output_file)

# 打印输出
print(f"Entity neighbors saved to '{output_file}'")
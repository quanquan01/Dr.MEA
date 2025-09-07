import json
import codecs

# 定义土耳其语字符到英文字符的映射
turkish_to_english = {
    'Å':'A',
    'Ã³':'A',
    'Ã':'A',
    'Ä':'A',
    'ı': 'i',
    'İ': 'I',
    'ğ': 'g',
    'ş': 's',
    'ç': 'c',
    'ö': 'o',
    'ü': 'u',
    'İ': 'I',
    '©':'',
    '¶':'',
    'º':'',
    '¼':'',
    'È':''

}

def convert_turkish_to_english(text):
    return ''.join(turkish_to_english.get(c, c) for c in text)

def read_file(file_path, encodings=['utf-8', 'gbk']):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.readlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode file {file_path} with any of the provided encodings")

def read_id2name(file_path):
    id2name = {}
    lines = read_file(file_path)
    for line in lines:
        id, name = line.strip().split('\t')
        # 使用 codecs.unicode_escape 解码转义的 Unicode 序列
        name = codecs.decode(name, 'unicode_escape')
        # 转换土耳其语字符为英文字符
        name = convert_turkish_to_english(name)
        id2name[id] = name
    return id2name

def read_id2rel(file_path):
    id2rel = {}
    lines = read_file(file_path)
    for line in lines:
        id, rel = line.strip().split('\t')
        # 使用 codecs.unicode_escape 解码转义的 Unicode 序列
        rel = codecs.decode(rel, 'unicode_escape')
        # 转换土耳其语字符为英文字符
        rel = convert_turkish_to_english(rel)
        id2rel[id] = rel
    return id2rel

def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict

def generate_output(id2name1, id2rel1, id2name2, id2rel2):
    ent1 = read_id2name(id2name1)
    rel1 = read_id2rel(id2rel1)
    ent2 = read_id2name(id2name2)
    rel2 = read_id2rel(id2rel2)
    #time=read_id2name(time_id)

    merged_ent = merge_dictionaries(ent1, ent2)
    merged_rel = merge_dictionaries(rel1, rel2)
    #merged_time = merge_dictionaries(time, time)

    output = {
        "ent": merged_ent,
        "rel": merged_rel,

    }
    return output

def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
# 读取文件
id_name1='./norm/ent_ids_1_new_1'
id_rel1='./norm/rel_ids1_new'
id_name2='./norm/ent_ids_2_new'
id_rel2='./norm/rel_ids2_new'



# 生成输出
output = generate_output(id_name1, id_rel1, id_name2, id_rel2)
save_to_file(output, './norm/candidates/name_dict_FB_DB.json')
# 打印输出
import json
print(json.dumps(output, indent=4))
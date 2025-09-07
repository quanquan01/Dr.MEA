# -*- coding: utf-8 -*-
import os
import json



def merge_dict(dic1, dic2):
    return {**dic1, **dic2}


def transform_idx_to_int(dic:dict):
    return {int(idx):data for idx, data in dic.items()}


def transform_time(seconds:int):
    h, m, s = 0, 0, 0
    h = seconds // 3600
    seconds -= h * 3600
    m = seconds // 60
    seconds -= m * 60
    s = seconds
    return h, m, s


def count_ranks(ranks):
    ###############
    ### Count the distribution of numbers of 0-1, 1-10, 10-20, >20 ranks
    ###############
    count = [0, 0, 0, 0]
    span = ['[ 0,  1)', '[ 1, 10)', '[10, 20)', '[20, --)']
    for r in ranks:
        if r == 0:
            count[0] += 1
        else:
            if r < 10:
                count[1] += 1
            else:
                if r < 20:
                    count[2] += 1
                else:
                    count[3] += 1
    total = len(ranks)
    print('Count of Ranks: ')
    for i in range(len(count)):
        print(f'  {span[i]} : {count[i]} , {count[i] / total:.2%}')


# generate entity neighbors information
class NeighborGenerator(object):
    def __init__(self, data, data_dir='data', cand_file='cand', desc_file='description', use_time=True, use_desc=True, use_name=True):
        self.use_time = use_time
        self.use_desc = use_desc
        self.use_name = use_name
        print(data_dir)
        self.use_attr=True
        self.path = os.path.join(data_dir, data, 'candidates')
        self.ref, self.rank, self.cand, self.cand_score = self.load_candidates(cand_file=cand_file)
        self.neighbors = self.load_neighbors()
        self.attr=self.load_attr()
        self.categories=self.load_category()
        if use_time:
            self.ent_name, self.rel_name, self.time_dict = self.load_name_dict()
        else:
            self.ent_name, self.rel_name = self.load_name_dict()
        if use_desc:
            self.description = self.load_description(desc_file=desc_file)

        self.entities = sorted([int(e) for e in self.cand.keys()])

    # initialize, load data
    def load_candidates(self, cand_file='cand'):
        file_path=os.path.join(self.path, cand_file)
        file_path=file_path.replace('\\', '/')
        with open(file_path, 'r', encoding='utf-8') as fr:
            origin_cand = json.load(fr)
        ref, rank, cand, cand_score = {}, {}, {}, {}
        for eid, data in origin_cand.items():
            eid = int(eid)
            ref[eid] = data['ref'] if 'ref' in data else ''
            rank[eid] = data['ground_rank'] if 'ground_rank' in data else ''     # ranks from method based on embeddings
            cand[eid] = data['candidates']   if 'candidates' in data else ''      # {ent_id: [cand_id_1, cand_id_2, ..., cand_id_20], ...}
            cand_score[eid] = data['cand_sims']   if 'cand_sims' in data else ''  # similarity scores from method based on embeddings
        return ref, rank, cand, cand_score
    def load_neighbors(self):
        with open(os.path.join(self.path, 'neighbors'), 'r', encoding='utf-8') as fr:
            neighbors = json.load(fr)           # {ent_id: [ [h1, r1, t1, ts1, te1], [h2, r2, t2, ts2, te2], ... ], ...}
        return transform_idx_to_int(neighbors)


    def load_attr(self):
        with open(os.path.join(self.path,"attribute"),'r',encoding='utf-8') as fr:
            attr=json.load(fr)
        return transform_idx_to_int(attr)

    def load_category(self):
        with open(os.path.join(self.path,"category"),'r',encoding='utf-8') as fr:
            cat=json.load(fr)
        return transform_idx_to_int(cat)

    def load_name_dict(self):
        with open(os.path.join(self.path, 'name_dict'), 'r', encoding='utf-8') as fr:
            name_dict = json.load(fr)
    
        ent_name = transform_idx_to_int(name_dict['ent'])
        rel_name = transform_idx_to_int(name_dict['rel'])
    

        if self.use_time:
            if 'time' in name_dict:
                time_dict = transform_idx_to_int(name_dict['time'])
            else:
                time_dict = {}
                print("Warning: 'time' key not found in name_dict.")
        else:
            time_dict = {}
    
        return ent_name, rel_name
        
    def load_description(self, desc_file='description'):
        with open(os.path.join(self.path, desc_file), 'r', encoding='utf-8') as fr:
            origin_desc = json.load(fr)
        desc = {int(eid):d["desc"] for eid, d in origin_desc.items()}   # {ent_id: desc, ...}
        return desc

    # API
    def get_all_entities(self):
        all_ent = set()
        for eid, cand in self.cand.items():
            all_ent.update([eid] + cand)
        return sorted(list(all_ent))
    def get_entities(self):
        return self.entities
    def get_ref_ent(self, ent_id:int):
        return self.ref[ent_id]
    def get_base_rank(self, ent_id:int):
        return self.rank[ent_id]
    def get_neighbors(self, ent_id:int, neigh_num=0):
        ###############
        ### Output: {'ent_id': ent_id, 'name': 'XXXX', 'neighbors': {'(h1, r1, XXXX, temp1_s, temp1_e)', ..., '(XXXX, r2, t2, temp2_s, temp2_e)'}
        ###############
        if ent_id in self.neighbors:
            neigh = self.neighbors[ent_id]
            if len(neigh) > 0:
                neigh_num = len(neigh) if neigh_num == 0 or neigh_num > len(neigh) else neigh_num
                neigh = neigh[:neigh_num]
                new_neigh = []
                if len(neigh[0]) == 3:
                    for h, r, t in neigh:
                        if self.use_name:
                            new_neigh.append((self.ent_name[h], self.rel_name[r], self.ent_name[t]))
                        else:
                            # replace entity name with entity id, when forbidding using name
                            new_neigh.append((self.ent_name[h], self.rel_name[r], self.ent_name[t]))
                            #new_neigh.append((f"'{h}'", self.rel_name[r], f"'{t}'"))
                else:
                    for h, r, t, ts, te in neigh:
                        if self.use_name:
                            new_neigh.append((self.ent_name[h], self.rel_name[r], self.ent_name[t]))
                        else:
                            new_neigh.append((self.ent_name[h], self.rel_name[r], self.ent_name[t]))
                            #new_neigh.append((f"'{h}'", self.rel_name[r], f"'{t}'", self.time_dict[ts], self.time_dict[te]))
                neigh = new_neigh
        else:
            neigh = []

        return_dict = {"ent_id": ent_id, "name": self.ent_name[ent_id], "neighbors": neigh,"category": self.categories[ent_id]['category']}
        if self.use_desc:
            return_dict["desc"] = self.description[ent_id]
        if self.use_attr:

            if ent_id in self.attr:
                #print(self.attr[ent_id]['properties'])
                return_dict['attr']=self.attr[ent_id]['properties']
            else:
                return_dict['attr']={}
        return return_dict

    def get_candidates(self, ent_id:int, neigh_num=0):
        ###############
        ### Output: [{'ent_id':cand_ent_1_id, 'name': 'XXXX', 'score': XXXX, 'neighbors': {...}}}, {'ent_id': cand_ent_2_id, ...}, ..., {'ent_id': cand_ent_20_id, ...}]
        ###############
        cand = []
        for score, cand_id in zip(self.cand_score[ent_id], self.cand[ent_id]):
            cand_ent = self.get_neighbors(cand_id, neigh_num)
            cand_ent['score'] = round(score, 3)
            cand.append(cand_ent)
        return cand


# evaluate
def evaluate_alignment(ranks, hit_k=[1, 5, 10]):
    ###############
    ### Input: ranks of all entities
    ### Output: Hits@K, MRR
    ###############
    hits = [0] * len(hit_k)
    mrr = 0
    for r in ranks:
        mrr += 1 / (r + 1)
        for j in range(len(hit_k)):
            if r < hit_k[j]:
                hits[j] += 1
    total_num = len(ranks)
    mrr /= total_num
    hits = [round(hits[i] / total_num, 4) for i in range(len(hit_k))]
    
    return hits, mrr


# save and load result
def save_result(result:dict, save_dir:str):
	##########
	## save experiment results
	## result = { ent_id: {"base_rank": XXXX, "llm_rank": XXXX, "iteration": X, tokens_count: {...}}, ...}
	##########
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		idx = 0
	else:
		for _, _, files in os.walk(save_dir):
			file_idxs = [int(name.split(".")[0].split('_')[-1]) for name in files]
		idx = max(file_idxs) + 1 if len(file_idxs) > 0 else 0
	save_path = os.path.join(save_dir, f"result_{idx}.json")
	with open(save_path, "w", encoding="utf-8") as fw:
		json.dump(result, fw, ensure_ascii=False, indent=4)


def load_result(save_dir:str):
    result = {}
    for _, _, files in os.walk(save_dir):
        for name in files:
            with open(os.path.join(save_dir, name), "r", encoding="utf-8") as fr:
                r = json.load(fr)
            result = merge_dict(result, r)
    return result


def get_new_result_idx(save_dir:str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        idx = 0
    else:
        for _, _, files in os.walk(save_dir):
            file_idxs = [int(name.split(".")[0].split('_')[-1]) for name in files]
        idx = max(file_idxs) + 1 if len(file_idxs) > 0 else 0
    return idx
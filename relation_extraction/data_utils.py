#处理数据

import re
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm


'''
{"h": {"name": "罗秀春", "pos": [17, 20]}, "t": {"name": "张默", "pos": [9, 11]}, "relation": "父母", "text": "-家庭张国立和父亲张默是张默和前妻罗秀春的儿子，据了解，张国立与罗女士相识于少年时代，长"}
'''
#也就是说从这里开始就默认为输入的实体1就是头实体，而输入的实体2就是尾实体 不存在主客体识别的部分
'''
d = {"h": {"e-actor-person": "韩学义", "pos": [0, 3]}, "t": {"e-co-script": "《民警家的贼》", "pos": [80, 87]}, "relation": "er-music design for script", "text": "韩学义（1949一）河南巩县人。中国音协陕西省分会会员，中国剧协陕西分会会员。二级作曲家，先后任西安市豫剧团党支部副书记、团长。如《樊梨花归唐》、《逼上梁山》、《民警家的贼》、《红珠女》、《胭脂》等。他编著的《常用豫剧打击乐谱》和《豫剧唱腔选》被《中国戏曲音乐集成•河南卷》等选用。1985年被聘为《中国戏曲音乐集成•河南卷》编委和撰稿人。所撰写《弦中之弹》获河南省戏曲音乐征文优秀奖"}
'''
## e1_type = list(d["h"].keys())[0]
## e2_type = list(d["t"].keys())[0]
## print(e1_type)
## print(e2_type)

class MyTokenizer(object):
    def __init__(self, pretrained_model_path=None, mask_entity=False):
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.mask_entity = mask_entity


    def tokenize(self, item):
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        #新加一个entity_type e1和e2的实体类型
        e_head_type = list(item['h'].keys())[0]
        e_tail_type = list(item['t'].keys())[0]



        #头实体==主体 主体在后面 客体在前：
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        # 主体在前 符合正常逻辑
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False

        sent0 = self.bert_tokenizer.tokenize(sentence[:pos_min[0]])#划分第一个实体出现之前的句子
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])#第一个出现的实体
        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])#第一个实体之后到第二个实体出现之前的句子
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])#第二个出现的实体
        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])#第二个实体之后的句子

        #pos_head 和 pos_tail 始终都是以给定的head和tail为主
        if rev: # head在tail后面
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)] # tail先出现
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ] # head后出现
        else: # head在tail前面
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2

        #这一块相当于生成一个输入bert模型中的向量格式[CLS]...[SEP]
        #并且返回实体所在位置 pos1开始位置，pos2结束位置
        re_tokens = ['[CLS]']
        cur_pos = 0
        # pos1和pos2 分别代表头实体和尾实体
        pos1 = [0, 0]
        pos2 = [0, 0]
        #pos_head 到达head_entity前的长度和包含实体长度的列表
        for token in tokens:
            token = token.lower() #变成小写字母
            if cur_pos == pos_head[0]:
                pos1[0] = len(re_tokens)
                re_tokens.append('[unused1]')
            if cur_pos == pos_tail[0]:
                pos2[0] = len(re_tokens)
                re_tokens.append('[unused2]')
            re_tokens.append(token) # 不管执不执行if语句 我都是要添加token的 而我if语句中出现的特殊标记符 那是为了方便我知道哪里是实体1、2的开始位置和结束位置 下面的步骤同理
            #上面的就截至到了不管哪个实体先出现的，截止获取到实体的第一个位置，下面是获取实体的第二个位置：
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused3]')
                pos1[1] = len(re_tokens)
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused4]')
                pos2[1] = len(re_tokens)
            cur_pos += 1 #往前推进一个字符
        #最后所有的都读取完了 那么会有一个终止符[SEP] 最后的re_tokens里面包含 tokens之外 还有实体的起始特殊字符以及前后出现的[SEP]和[CLS]
        re_tokens.append('[SEP]')
        return re_tokens[1:-1], pos1, pos2
    #re_tokens: [CLS] E(My) E(dog) E(name) E(is) [unused1] E(BOb) E(unused3) E(and) E(she) E(is) [unused2] E(adorable) [unused4] E(...)... [SEP]

#掩码mask有时候也相当于是id号是吧？
#下面的函数是将出现实体的地方赋值为1

#实体对应类型
'''def convert_pos_to_entity_type_id(e_pos, e_type, max_len=512):
    e2type = ['e-unk'] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e2type[i] = e_type
    return e2type'''

#实体掩码
def convert_pos_to_mask(e_pos, max_len=512):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def read_data(input_file, tagset_file, tokenizer=None, max_len=512):#原有部分是128
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    tags = []

    with open(tagset_file, 'r', encoding='utf-8') as f_tag:
        tagset = [i.strip() for i in f_tag.readlines()]
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            item = json.loads(line) # loads是将json解析成了一个字典格式 所以读取都是整个文件一起读的
            tag = item['relation']

            if tag in tagset:
                tags.append(tag)

                if tokenizer is None:
                    tokenizer = MyTokenizer()
                tokens, pos_e1, pos_e2= tokenizer.tokenize(item) #这里的tokens就是上面的re_tokens

                if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len and \
                        pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                    tokens_list.append(tokens)
                    e1_mask = convert_pos_to_mask(pos_e1, max_len)
                    e2_mask = convert_pos_to_mask(pos_e2, max_len)
                    e1_mask_list.append(e1_mask)
                    e2_mask_list.append(e2_mask)


                    '''tag = item['relation']
                    with open(tagset_file, 'r', encoding='utf-8') as f_tag:
                        l = f_tag.readlines()
                        tagset = [i.strip() for i in l]
                        if tag in tagset:
                            tags.append(tag)'''

    return tokens_list, e1_mask_list, e2_mask_list, tags


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))

#这两个函数其实是对实体的操作 标签：B-PER、id(1)这种类型的
def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        l = f_in.readlines()
        tagset = [i.strip() for i in l]
        #tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        l = f_in.readlines()
        tagset = [i.strip() for i in l]
        #tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


# data process for bert
class SentenceREDataset(Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=512):#原先是128
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path # 这个是关系标签的路径
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.tokenizer = MyTokenizer(pretrained_model_path=self.pretrained_model_path)#这里就相当于返回一个bert可读的输入向量
        self.max_len = max_len

        self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.tags = read_data(data_file_path, tagset_file= self.tagset_path, tokenizer=self.tokenizer, max_len=self.max_len)
        self.tag2idx = get_tag2idx(self.tagset_path) # 找到关系tag对应的id值

    #所以设置这个长度是需要给DataLoader的
    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_e1_mask = self.e1_mask_list[idx]
        sample_e2_mask = self.e2_mask_list[idx]
        sample_tag = self.tags[idx]

        encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len, padding='max_length')#pad_to_max_length=True 这个参数好像被弃用了
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']

        sample_tag_id = self.tag2idx[sample_tag]

        sample = {
            # 这个就是要将数值转换为tensor类型
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'e1_mask': torch.tensor(sample_e1_mask),
            'e2_mask': torch.tensor(sample_e2_mask),
            'tag_id': torch.tensor(sample_tag_id), # 标签值的id号
        }
        return sample
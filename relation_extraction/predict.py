import json
from idlelib.iomenu import encoding

import jsonlines
import time
import re
import torch


from data_utils import MyTokenizer, get_idx2tag, convert_pos_to_mask
from model import SentenceRE
import argparse




def predict(hparams, text, entity1, entity2):
    #device = hparams.device
    #seed = hparams.seed
    #torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    device = torch.device('cpu')

    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = SentenceRE(hparams)
    model.load_state_dict(torch.load(model_file,weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    tokenizer = MyTokenizer(pretrained_model_path)

    '''text = input("输入中文句子：")
    entity1 = input("句子中的实体1：")
    entity2 = input("句子中的实体2：")'''

    match_obj1 = re.search(entity1, text)
    match_obj2 = re.search(entity2, text)
    if not match_obj1 or not match_obj2:
        return 'UNKNOWN'
    #if match_obj1 and match_obj2:  # 姑且使用第一个匹配的实体的位置
    e1_pos = match_obj1.span()
    e2_pos = match_obj2.span()

    item = {
        'h': {
            'name': entity1,
            'pos': e1_pos
        },
        't': {
            'name': entity2,
            'pos': e2_pos
        },
        'text': text
    }
    tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
    encoded = tokenizer.bert_tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    e1_mask = torch.tensor([convert_pos_to_mask(pos_e1, max_len=attention_mask.shape[1])]).to(device)
    e2_mask = torch.tensor([convert_pos_to_mask(pos_e2, max_len=attention_mask.shape[1])]).to(device)

    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)[0]
        logits = logits.to(device)
    pred_rel = idx2tag[logits.argmax(0).item()]

    return pred_rel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_path", type=str,
                        default='pretrained_models/bert-base-chinese')
    parser.add_argument("--tagset_file", type=str,
                        default='YuOpera Dataset/new relations.txt')
    parser.add_argument("--model_file", type=str, default="E:\毕业论文\save_model_based_on_loss.pt")
    parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
    parser.add_argument('--dropout', type=float, default=0.5, required=False, help='dropout')

    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Device to use for computation ('cpu' or 'cuda')")


    hparams = parser.parse_args()

    file = 'new_rules_entity_pairs.json'
    #file = 'rest_text.jsonl'
    spo_file = "spo_file_new_relations.jsonl"
    # count = 0
    # accumulated_time = 0  # 初始化累积时间
    with jsonlines.open(spo_file, mode='w') as writer:
        with open(file, encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                #count += 1
                spo = []
                text = line['text']
                entities_pairs = line['entity_pairs']
                if not entities_pairs: #是空的
                    print('NULL')
                    continue
                else:
                    for entity1, type1, entity2, type2 in entities_pairs:
                        #start_time = time.time()
                        relation_pred = predict(hparams, text, entity1, entity2)
                        spo.append([entity1, type1, entity2, type2, relation_pred])
                d = {'text': text, 'spo': spo}
                writer.write(d)
                print(f'Writing...')

    print(f"文件写入完毕！:)")

if __name__ == '__main__':
    main()
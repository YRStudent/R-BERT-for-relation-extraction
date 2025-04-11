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
                        default='YuOpera Dataset/relation.txt')
    parser.add_argument("--model_file", type=str, default=r"G:\较优的pt文件模型\re-pt较优文件-部分关系\save_model_based_on_loss.pt")
    parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
    parser.add_argument('--dropout', type=float, default=0.5, required=False, help='dropout')

    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Device to use for computation ('cpu' or 'cuda')")


    hparams = parser.parse_args()

    #file = "output.jsonl"
    '''file = '../rest_text.jsonl'
    spo_file = "../spo_file.jsonl"'''
    # 总共1454条数据
    test_labelled_file = 'YuOpera Dataset/test_data.jsonl'
    pred_labelled_file = 'pred_labelled_data_spo.jsonl'
    '''
    {"h": {"e-actor-person": "史大成", "pos": [0, 3]}, "t": {"e-co-music": "三弦", "pos": [44, 46]}, "relation": "er-perform music", "text": "史大成（1888-1987年）豫剧琴师。河南省扶沟县人。青年时，从师白义然、娄中良，学习三弦、啖呐等乐器，长时期从事豫剧伴奏工作，先后在扶沟、太康、通许、开封、溪河、密县等地,与许多豫剧演员合作。新中国成立后在郑州参加人民剧团（后改为河南豫剧院二团）。1956年，河南省首届戏曲观摩演出大会中，曾荣获音乐伴奏一等奖。1960年调省戏曲学校任教。"}

    '''
    count = 0
    with jsonlines.open(pred_labelled_file, mode='w') as writer:
        with open(test_labelled_file, encoding='utf-8') as f:
            for line in jsonlines.Reader(f):
                count += 1
                pred_spo = []
                text = line['text']

                pos_h = line['h']['pos']
                pos_t = line['t']['pos']
                entity1 = text[pos_h[0]:pos_h[1]]
                entity2 = text[pos_t[0]:pos_t[1]]

                '''if not spo: #是空的
                    print('NULL')
                    continue
                else:'''
                start_time = time.time()

                relation_pred = predict(hparams, text, entity1, entity2)
                pred_spo.append([entity1, entity2, relation_pred])

                end_time = time.time()
                interval_time = end_time - start_time
                if interval_time > 10:
                    print('第{}条,TIMEOUT'.format(count))
                    d = {'text': text, 'spo': pred_spo}
                    writer.write(d)
                    continue #跳过当前数据 继续下一条数据
                else:
                    d = {'text': text, 'spo': pred_spo}
                    writer.write(d)
                    print(f'Writing...')

    print(f"文件写入完毕！:)")

if __name__ == '__main__':
    main()
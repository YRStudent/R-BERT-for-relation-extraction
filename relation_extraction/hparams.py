import os
import argparse
import torch


'''here = os.path.dirname(os.path.abspath(__file__))#'../pretrained_models/bert-base-chinese'
default_pretrained_model_path = os.path.join(here, 'root/autodl-tmp/relation-extraction-master/pretrained_models/bert-base-chinese')
default_train_file = os.path.join(here,'root/autodl-tmp/relation-extraction-master/YuOpera Dataset/train_data.json' )#'../datasets/train_small.jsonl'
default_validation_file = os.path.join(here,'root/autodl-tmp/relation-extraction-master/relation-extraction-master/YuOpera Dataset/valid_data.json' )#'../datasets/val_small.jsonl'
default_test_file = os.path.join(here,'root/autodl-tmp/relation-extraction-master/relation-extraction-master/YuOpera Dataset/test_data.jsonl')
default_output_dir = os.path.join(here, 'root/autodl-tmp/relation-extraction-master/saved_models')#'../saved_models'
default_log_dir = os.path.join(default_output_dir, 'runs')
default_tagset_file = os.path.join(here,'root/autodl-tmp/relation-extraction-master/豫剧资源/relation.txt' )#'../datasets/relation.txt'

default_model_file = os.path.join(default_output_dir, 'model.pth')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')'''

parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_path", type=str, default='root/autodl-tmp/relation-extraction-master/pretrained_models/bert-base-chinese')
parser.add_argument("--train_file", type=str, default='root/autodl-tmp/relation-extraction-master/YuOpera Dataset/train_data.json')
parser.add_argument("--validation_file", type=str, default='root/autodl-tmp/relation-extraction-master/relation-extraction-master/YuOpera Dataset/valid_data.json')
parser.add_argument("--test_file", type=str, default='root/autodl-tmp/relation-extraction-master/relation-extraction-master/YuOpera Dataset/test_data.jsonl')
parser.add_argument("--output_dir", type=str, default='root/autodl-tmp/relation-extraction-master/saved_models')
parser.add_argument("--log_dir", type=str, default='runs')
parser.add_argument("--tagset_file", type=str, default='root/autodl-tmp/relation-extraction-master/豫剧资源/relation.txt')

parser.add_argument("--model_file", type=str, default='model.pth')
parser.add_argument("--checkpoint_file", type=str, default='checkpoint.json')

# model
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
parser.add_argument('--dropout', type=float, default=0.5, required=False, help='dropout')

parser.add_argument('--device', type=str, default='cpu',choices=['cpu', 'cuda'], help="Device to use for computation ('cpu' or 'cuda')")#'cuda'
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--max_len", type=int, default=128)   #到时候这个这里改一下 512
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--validation_batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=40) #原先的参数是训练20轮 干脆到时候训练是20好了 现在是训到快20就差不多稳定了
parser.add_argument("--learning_rate", type=float, default=1e-5)#0.00001
parser.add_argument("--weight_decay", type=float, default=0)

hparams = parser.parse_args()

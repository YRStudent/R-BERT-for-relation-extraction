import os
import time
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter, writer
from sklearn import metrics
import argparse
from data_utils import SentenceREDataset, get_idx2tag, load_checkpoint, save_checkpoint
from model import SentenceRE
from early_stopping import EarlyStopping

def validation(val_loader, model, criterion, device):

    validation_loss = 0.0
    step = 0
    model.eval()
    with torch.no_grad():  # 不用梯度下降，意思是validation不参与反向传播
        tags_true = []
        tags_pred = []
        for val_i_batch, val_sample_batched in enumerate(val_loader):
            step += 1
            token_ids = val_sample_batched['token_ids'].to(device)
            token_type_ids = val_sample_batched['token_type_ids'].to(device)
            attention_mask = val_sample_batched['attention_mask'].to(device)
            e1_mask = val_sample_batched['e1_mask'].to(device)
            e2_mask = val_sample_batched['e2_mask'].to(device)
            tag_ids = val_sample_batched['tag_id']
            logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            pred_tag_ids = logits.argmax(1)
            # 因为logits是二维张量，所以1也等同于-1
            tags_true.extend(tag_ids.tolist())
            # tolist()是用于将张量格式转换位python的list类型
            tags_pred.extend(pred_tag_ids.tolist())  # 这里的extend和append还不太一样，extend相当于直接往列表序列的末尾一次多量的加入其他序列中的元素

            # 修改loss部分忽略padding部分带来的损失 新加的
            '''loss = criterion(logits.view(-1, logits.size(-1)), tag_ids.view(-1))  # 这个view(-1)就相当于是扁平化张量，让他们变成一维的
            masked_loss = loss * attention_mask.view(-1)  # 按照anttention_mask将padding部分的损失设置为0
            mean_loss = masked_loss.sum() / attention_mask.sum()
            validation_loss += mean_loss.item()'''
            # 计算validation的损失
            loss = criterion(logits, tag_ids.to(device)) #实际的值 tag_ids 和预测的值 logits
            validation_loss += loss.item()

    #新加的 来忽略评估中的padding带来的影响
    '''val_mask = attention_mask.view(-1) == 1
    filtered_tags_true = [tag for tag, mask in zip(tag_ids.view(-1).tolist(), val_mask.tolist()) if mask]
    filtered_tags_pred = [pred for pred, mask in zip(pred_tag_ids.view(-1).tolist(), val_mask.tolist()) if mask]'''

    Val_Loss = validation_loss / step
    val_f1 = metrics.f1_score(tags_true, tags_pred, average='weighted',zero_division=0)
    val_precision = metrics.precision_score(tags_true, tags_pred, average='weighted',zero_division=0)
    val_recall = metrics.recall_score(tags_true, tags_pred, average='weighted',zero_division=0)
    val_accuracy = metrics.accuracy_score(tags_true, tags_pred)

    '''val_f1 = metrics.f1_score(filtered_tags_true, filtered_tags_pred, average='weighted', zero_division=0)
    val_precision = metrics.precision_score(filtered_tags_true, filtered_tags_pred, average='weighted', zero_division=0)
    val_recall = metrics.recall_score(filtered_tags_true, filtered_tags_pred, average='weighted', zero_division=0)
    val_accuracy = metrics.accuracy_score(filtered_tags_true, filtered_tags_pred)'''


    return Val_Loss, val_f1, val_precision, val_recall, val_accuracy, tags_true, tags_pred

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_path", type=str,
                        default= r'E:\python projects\another computer\relation-extraction-master\pretrained_models\bert-base-chinese')
parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
parser.add_argument('--dropout', type=float, default=0.5, required=False, help='dropout')

parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                    help="Device to use for computation ('cpu' or 'cuda')")  # 'cuda'
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--max_len", type=int, default=512)  # 到时候这个这里改一下 512
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--validation_batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=100)  # 原先的参数是训练20轮 干脆到时候训练是20好了 现在是训到快20就差不多稳定了
parser.add_argument("--learning_rate", type=float, default=1e-6)  # 0.00001——>0.0001——>0.000001
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--tagset_file", type=str,
                        default=r'E:\python projects\another computer\relation-extraction-master\YuOpera Dataset\relation.txt')
hparams = parser.parse_args()


'''model_file = hparams.model_file
checkpoint_file = hparams.checkpoint_file'''

max_len = hparams.max_len  # 512
train_batch_size = hparams.train_batch_size  # 这个默认是8
validation_batch_size = hparams.validation_batch_size  # 这个也是8
epochs = hparams.epochs  # 设定为40
best_val_loss = 1e-18
learning_rate = hparams.learning_rate
weight_decay = hparams.weight_decay
tagset_file = hparams.tagset_file

device = torch.device(hparams.device if torch.cuda.is_available() else "cpu")
# 不知道这个seed起着什么作用
'''seed = hparams.seed
torch.manual_seed(seed)'''

idx2tag = get_idx2tag(tagset_file)

hparams.tagset_size = len(idx2tag)

model = model = SentenceRE(hparams).to(device)

test_dataset = SentenceREDataset(r'E:\python projects\another computer\relation-extraction-master\YuOpera Dataset\test_data.json', tagset_path= tagset_file, pretrained_model_path=r'E:\python projects\another computer\relation-extraction-master\pretrained_models\bert-base-chinese', max_len = 512)
criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)  # 交叉熵分类 损失函数
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
model.load_state_dict(torch.load(r"G:\relation-extraction-master\saved_models\model.bin"))

_, _, _, _, _, tags_true, tags_pred = validation(test_loader, model, criterion, device)

#print(classification_report(y_test, y_pred, labels=labels, digits=3))
print(metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys()),
                                    target_names=list(idx2tag.values())))
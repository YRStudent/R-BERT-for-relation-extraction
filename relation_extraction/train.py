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




def train(model, train_loader, optimizer, criterion, device):


    '''# load checkpoint if one exists
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_file))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0'''

    running_loss = 0.0
    step = 0
    '''writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))'''

    model.train()
    #[(id,内容),()...] id是从0开始的
    for i_batch, sample_batched in enumerate(train_loader):
        #print(len(train_loader))
        step += 1
        token_ids = sample_batched['token_ids'].to(device) #这个token_ids是给每个字的id号进行索引用的
        token_type_ids = sample_batched['token_type_ids'].to(device) #这个是依据段落所给文本按照逻辑来划分句子的id号
        attention_mask = sample_batched['attention_mask'].to(device) #总之这个玩意儿就是让模型去读取重要的那些token值，去分配一个什么权重。。。内部的计算我就不管了
        e1_mask = sample_batched['e1_mask'].to(device) #在被mask的地方赋值为1 其余为0
        e2_mask = sample_batched['e2_mask'].to(device)
        tag_ids = sample_batched['tag_id'].to(device)

        # optimizer zero grad
        # 在反向传播之前，要将之前的参数梯度清零
        model.zero_grad()
        logits = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
        # caculate loss ; tag_id == 'relation'
        loss = criterion(logits, tag_ids) #计算输入的对数和目标之间的损失

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        #修改loss部分忽略padding部分带来的损失 新加的
        '''loss = criterion(logits.view(-1,logits.size(-1)), tag_ids.view(-1))#这个view(-1)就相当于是扁平化张量，让他们变成一维的
        masked_loss = loss * attention_mask.view(-1) #按照attention_mask将padding部分的损失设置为0
        mean_loss = masked_loss.sum() / attention_mask.sum()
        running_loss += mean_loss.item()'''


        '''# back propagation on loss
        loss.backward()
        # gradient descent 更新参数
        optimizer.step()
        optimizer.zero_grad()
        #model.zero_grad()'''

        '''if i_batch % 10 == 9:#因为i_batch是从0开始的，所以满10是9 因此取余之后应当是9
            writer.add_scalar('Training/training loss', running_loss / 10, epoch * len(train_loader) + i_batch)
            # 此前所有epoch训练的条数+当前的i_batch 表示当前一共训练到第多少条
            running_loss = 0.0'''
    Train_Loss = running_loss / step
    #print('Training Loss: {}'.format(Train_Loss))
    return Train_Loss

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

'''def test(model, test_loader, device):
    model.eval()
    step = 0
    with torch.no_grad():  # 不用梯度下降，意思是validation不参与反向传播
        tags_true = []
        tags_pred = []
        for val_i_batch, val_sample_batched in enumerate(test_loader):
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
            tags_pred.extend(pred_tag_ids.tolist())
    return  tags_true, tags_pred'''

if __name__=="__main__":
    writer = SummaryWriter()
    # haprams 初始化的一些东西
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_path", type=str,
                        default='/root/autodl-tmp/relation-extraction++/pretrained_models/bert-base-chinese')
    parser.add_argument("--train_file", type=str,
                        default='/root/autodl-tmp/relation-extraction++/YuOpera Dataset/train_data.json')
    parser.add_argument("--validation_file", type=str,
                        default='/root/autodl-tmp/relation-extraction++/YuOpera Dataset/valid_data.json')
    parser.add_argument("--test_file", type=str,
                        default='/root/autodl-tmp/relation-extraction++/YuOpera Dataset/test_data.jsonl')

    '''parser.add_argument("--output_dir", type=str, default='/root/autodl-tmp/relation-extraction++/saved_models.pt')
    parser.add_argument("--log_dir", type=str, default='runs')'''
    parser.add_argument("--tagset_file", type=str,
                        default='/root/autodl-tmp/relation-extraction++/YuOpera Dataset/new relations.txt')
    parser.add_argument("--model_file", type=str, default='/root/autodl-tmp/relation-extraction++/model.pth')
    '''parser.add_argument("--checkpoint_file", type=str, default='checkpoint.json')'''

    # model
    parser.add_argument('--embedding_dim', type=int, default=768, required=False, help='embedding_dim')
    parser.add_argument('--dropout', type=float, default=0.5, required=False, help='dropout')

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="Device to use for computation ('cpu' or 'cuda')")  # 'cuda'
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_len", type=int, default=512)  # 到时候这个这里改一下 512
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--validation_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)  # 原先的参数是训练20轮 干脆到时候训练是20好了 现在是训到快20就差不多稳定了
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # 0.00001——>0.0001——>0.000001
    parser.add_argument("--weight_decay", type=float, default=0)

    hparams = parser.parse_args()

    pretrained_model_path = hparams.pretrained_model_path
    train_file = hparams.train_file
    validation_file = hparams.validation_file
    test_file = hparams.test_file
    tagset_file = hparams.tagset_file

    model_file = hparams.model_file
    '''checkpoint_file = hparams.checkpoint_file'''

    max_len = hparams.max_len  # 512
    train_batch_size = hparams.train_batch_size  # 这个默认是8
    validation_batch_size = hparams.validation_batch_size  # 这个也是8
    epochs = hparams.epochs  # 设定为40
    best_val_loss = 1e-18
    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay

    device = torch.device(hparams.device if torch.cuda.is_available() else "cpu")
    # 不知道这个seed起着什么作用
    '''seed = hparams.seed
    torch.manual_seed(seed)'''

    idx2tag = get_idx2tag(tagset_file)

    hparams.tagset_size = len(idx2tag)



    model = SentenceRE(hparams).to(device)

    # 加载数据集
    validation_dataset = SentenceREDataset(validation_file, tagset_path=tagset_file,
                                           pretrained_model_path=pretrained_model_path,
                                           max_len=max_len)
    train_dataset = SentenceREDataset(train_file, tagset_path=tagset_file,
                                      pretrained_model_path=pretrained_model_path,
                                      max_len=max_len)
    test_dataset = SentenceREDataset(test_file, tagset_path=tagset_file,
                                     pretrained_model_path=pretrained_model_path,max_len = max_len)

    val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)  # 所以它的长度应该是总的条数/batch_size
    test_loader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False)

    print('Load Data Done.')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Loss
    criterion = torch.nn.CrossEntropyLoss().to(device)  # 交叉熵分类 损失函数

    es = EarlyStopping(patience=7, verbose=True, path='/root/autodl-tmp/relation-extraction++/save_model_based_on_loss.pt')
    val_loss_l, train_loss_l, val_f1_l, train_f1_l = [], [], [], []
    # Training data loop

    print('Start Training...')
    for epoch in range(1,epochs+1):
        print("Epoch: {}".format(epoch))
        Train_Loss = train(model, train_loader, optimizer, criterion, device)
        print("Train Loss: {}".format(Train_Loss))
        validation_loss, val_f1, _, _, _,_,_ = validation(val_loader, model, criterion, device)
        print("Validation Loss: {}".format(validation_loss))
        print("Validation F1: {}".format(val_f1))
        _,train_f1,_,_,_,_,_ = validation(train_loader, model, criterion, device)
        print("Train F1: {}".format(train_f1))


        val_loss_l.append(validation_loss)
        train_loss_l.append(Train_Loss)
        val_f1_l.append(val_f1*100)
        train_f1_l.append(train_f1*100)

        if val_f1 > best_val_loss:
            best_val_loss = val_f1
            torch.save(model.state_dict(), model_file)

        es(validation_loss,model)
        if es.early_stop:
            print('Early Stopping')
            break

    plt.plot(range(1, len(train_f1_l) + 1), train_f1_l, label="Train_F1", color="#0000FF")
    plt.plot(range(1, len(val_f1_l) + 1), val_f1_l, label="Val_F1", color="#CE6090")
    plt.plot(range(1, len(train_loss_l) + 1), train_loss_l, label="Train_Loss", color="#F9A31A")
    plt.plot(range(1, len(val_loss_l) + 1), val_loss_l, label="Val_Loss", color="#79C5CE")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # 计算test的matrix
    # 加载最后的模型进行后续测试或评估
    # 通过F1判断保存的模型文件
    #model.load_state_dict(torch.load(model_file))
    # 通过Loss指标是否降低保存模型文件
    model.load_state_dict(torch.load('/root/autodl-tmp/relation-extraction++/save_model_based_on_loss.pt'))

    _, _, _, _, _, tags_true, tags_pred = validation(test_loader, model, criterion, device)

    #print(classification_report(y_test, y_pred, labels=labels, digits=3))
    print(metrics.classification_report(tags_true, tags_pred, labels=list(idx2tag.keys()),
                                        target_names=list(idx2tag.values())))
    writer.close()
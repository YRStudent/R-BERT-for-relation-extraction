import os
import logging
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

here = os.path.dirname(os.path.abspath(__file__))

# 它的输入是一个句子和两个实体的位置掩码，输出的位实体间关系类别的logits
class SentenceRE(nn.Module):

    def __init__(self, hparams):
        super(SentenceRE, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
        self.embedding_dim = hparams.embedding_dim
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size
        #新加的
        #self.entity_type_size = hparams.entity_type_size

        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path,attn_implementation="eager")

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim) #全连接层
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 3) #对拼接后的向量进行LayerNorm操作
        self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.tagset_size)
        # 最终的全连接层 用于输出预测关系的logits
        # 这个是嵌入多特征的层

        '''#新加的
        self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.entity_type_size)'''

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output, cross_attentions = self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True, return_dict=False)


        #BertModel()最后输出较多，具体参考HuggingFace
        #这里输出：sequence_output 是last_hidden_state 形状是[batch_size, sequence_length, hidden_size]
        #hidden_size: 隐藏层的大小 比如768
        #pooled_output: 相当于pooler_output 形状是[batch_size, hidden_size]
        #通常对于第一个token -> [CLS] token 的隐藏状态进行额外的线性变换和激活（通常是Tanh）得到的，这个在分类任务中会用到 就是池化层的概念，就相当于降维然后提取最为主要的特征 也通常被认为是提取语义
        #cross attentions (batch_size, num_head, sequence_length, sequence_length)


        # 下面这一部分就是具体的forward运算过程 和entity_type其实并没有太大关系
        # 每个实体的所有token向量的平均值
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e1_h = self.activation(self.dense(e1_h))
        e2_h = self.activation(self.dense(e2_h))

        # [cls] + 实体1 + 实体2
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.drop(concat_h))

        return logits

    @staticmethod
    def entity_average(hidden_output, e_mask):
        # 平均隐藏状态向量？
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """

        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # unsqueeze() 起到升维的作用，参数：在哪个地方增加一个维度，0：在张量最外层加一个维度； 1：在张量第一个位置加维度 -1：在最后一个位置加维度

        #torch.bmm(input, mat2) 对输入的张量进行矩阵的乘法运算 注意，输入输出的维度必须是3维
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        # squeeze :删除长度为1的轴
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
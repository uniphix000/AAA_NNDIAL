#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import logging
import copy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

use_cuda = torch.cuda.is_available()

class Network(nn.Module):
    def __init__(self, args, datareader):
        super(Network, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        self.encoder_type = args.encoder_type  # 默认为lstm #Todo CNN可以尝试加一下
        self.datareader = datareader
        self.voc_size = len(self.datareader.vocab)

        # intent encoder
        self.encoder = Encoder(self.args.embed_size, self.hidden_size, self.voc_size)

        # parameters for belief tracker
        self.reqseg = self.datareader.reqseg
        self.infoseg = self.datareader.infoseg
        self.chgseg = [0, 1]

        # three trackers
        self.info_tracker = Info_Tracker(self.infoseg, self.voc_size, self.hidden_size)
        self.req_tracker = Req_Tracker(self.reqseg, self.voc_size, self.hidden_size)
        self.chg_tracker = Req_Tracker(self.chgseg, self.voc_size, self.hidden_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        for i in range(self.args.max_epoch):
            logging.info('--------------------Round {0}---------------------'.format(i))
            while True:
                data = self.datareader.read(mode='train')
                if data == None:
                    break

                '''
                source_0, source_len_1, masked_source_2, masked_source_len_3,\
                target_4, target_len_5, masked_target_6, masked_target_len_7,\
                snapshot_8, change_9, goal_10, inf_trk_label_11, req_trk_label_12,\
                db_degree_13, srcfeat_14, tarfeat_15, finished_16, utt_group_17 = data  # 除了goal和finished以外长度都为对话轮次长
                '''
                self.recurr(data)


    def _data_iterator(self, data, i):
        '''
        每次弹出一个轮次的数据，不包括goal和finished，并将部分变成variable
        :param data:
        :param i:
        :return:
        '''
        data_piece = []
        turn_number = len(data[0])
        for j in range(len(data)):
            if type(data[j]) is not bool:
                if len(data[j]) == turn_number:
                    if j in range(8):
                        data_piece.append(variable_tensor(data[j][i], 'Long'))
                    else:
                        data_piece.append(data[j][i])
        assert len(data_piece) == len(data) - 2
        return data_piece

    def recurr(self, data):
        assert data[-2] == True  # 确认对话已经结束
        turn_number = len(data[0])

        # initial belief_0
        belief_0 = torch.zeros((1, self.infoseg[-1]))
        belief_0 = [belief_0[0][i-1] + 1  if i in self.infoseg[1:] else belief_0[0][i-1] for i in range(1, self.infoseg[-1]+1)]
        belief_0 = torch.FloatTensor(belief_0).unsqueeze(1)
        pre_belief = variable_tensor(belief_0)

        # initial pre_target
        masked_target_tm1 = torch.ones((1,len(data[2][0])))
        masked_target_len_tm1 = torch.ones((1,data[3][0]))

        # initial pre_target position features
        pre_target_position = -torch.ones((1,data[17][0]))

        # initial posterior # fixme
        '''
        belief_tm1, masked_target_tm1, masked_target_len_tm1,
                    target_feat_tm1, posterior_tm1
        '''

        for i in range(turn_number):
            '''
            (source_t, target_t, source_len_t, target_len_t,
                    masked_source_t, masked_target_t,
                    masked_source_len_t, masked_target_len_t,
                    utt_group_t, snapshot_t, success_reward_t, sample_t,
                    change_label_t, db_degree_t,
                    inf_label_t, req_label_t, source_feat_t, target_feat_t,
                    belief_tm1, masked_target_tm1, masked_target_len_tm1,
                    target_feat_tm1, posterior_tm1)
            '''
            # loss
            loss = 0

            # 抽取一轮数据
            data_piece = self._data_iterator(data, i)

            source, source_len, masked_source, masked_source_len,\
            target, target_len, masked_target, masked_target_len,\
            snapshot, change, inf_trk_label, req_trk_label,\
            db_degree, srcfeat, tarfeat, utt_group = data_piece

            # encode
            self.encoder.forward(masked_source)

            # belief tracking
            # informable slots
            belief_t = []
            for j in range(len(self.info_tracker.slots_box)):

                # 当前slot的上一轮次的value的分布
                slot_belief_value = pre_belief[self.infoseg[j]:self.infoseg[j+1]]

                # 对ngram的position进行解码
                slot_source_position = srcfeat[0][self.infoseg[j]:self.infoseg[j+1]]
                value_source_position = srcfeat[1][self.infoseg[j]:self.infoseg[j+1]]
                slot_target_position = tarfeat[0][self.infoseg[j]:self.infoseg[j+1]]
                value_target_position = tarfeat[1][self.infoseg[j]:self.infoseg[j+1]]

                # tracking
                new_slot_belief_value = self.info_tracker.slots_box[j].forward(slot_belief_value, masked_source, masked_target, \
                                                       masked_source_len, masked_target_len,\
                                                       slot_source_position, value_source_position,\
                                                       slot_target_position, value_target_position)

                slot_belief_label = variable_tensor(inf_trk_label[self.infoseg[j]:self.infoseg[j+1]], 'Long')

                loss += self.loss(new_slot_belief_value.view(1, -1), torch.max(slot_belief_label, 0)[1])  # 交叉熵不接受one-hot向量
                                                                                                         # 而是接收正确的indices
                belief_t.append(slot_belief_label)

            inf_belief_t = pre_belief

            # requestable slots
            for k in range(len(self.reqseg)-1):

                # current feature idx
                bn = self.infoseg[-1] + 2*k  # 排列在infor之后，每个占两位

                # 解码位置信息
                slot_source_position = srcfeat[0][bn]
                value_source_position = srcfeat[1][bn]
                slot_target_position = tarfeat[0][bn]
                value_target_position = tarfeat[1][bn]

                # tracking
                new_slot_belief_value = self.req_tracker.slots_box[k].forward(masked_source, masked_target, masked_source_len,\
                                                                              masked_target_len_tm1, slot_source_position,\
                                                                              value_source_position, slot_target_position,\
                                                                              value_target_position)

                slot_belief_label = variable_tensor(req_trk_label[2*k:2*(k+1)], 'Long')

                loss += self.loss(new_slot_belief_value.view(1, -1), torch.max(slot_belief_label, 0)[1])  # 交叉熵不接受one-hot向量

                belief_t.append(slot_belief_label)

            # offer-change tracker 是否变更了候选
            minus_1 = [-1]
            new_slot_belief_value = self.chg_tracker.slots_box[0].forward(masked_source, masked_target, masked_source_len,\
                                                                              masked_target_len_tm1, minus_1, minus_1,\
                                                                              minus_1, minus_1)

            slot_belief_label = variable_tensor(change, 'Long')

            loss += self.loss(new_slot_belief_value.view(1, -1), torch.max(slot_belief_label, 0)[1])  # 交叉熵不接受one-hot向量

            belief_t.append(change)

            print loss




class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.V = vocab_size
        self.embedding = nn.Embedding(self.V, self.embed_size)
        #self.cnn = nn.Conv1d()
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input):
        last_hidden = self.sort_batch(input)

    def sort_batch(self, input):  # Todo:使用padpackage
        """
        一句user input
        :return:
        """
        #input = Variable(torch.LongTensor(input)).cuda() if use_cuda else Variable(torch.LongTensor(input))
        self.batch_size = len(input)
        embed = self.embedding(input)
        #sentence_lens = self.data[3]
        #lst = list(range(self.batch_size))
        #lst = sorted(lst, key=lambda d: -sentence_lens[d])
        outputs, (h, c) = self.lstm(embed.unsqueeze(0))
        return h


class Info_Tracker(nn.Module):
    def __init__(self, info_seg, voc_size, hidden_size):
        super(Info_Tracker, self).__init__()
        self.info_seg = info_seg  # 该slot下value的个数
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.slots_box = []
        for i in range(len(self.info_seg)-1):
            belief_size = self.info_seg[i+1] - self.info_seg[i]
            self.slots_box.append(CNNInfoTracker(belief_size, self.voc_size, self.voc_size, self.hidden_size))


class CNNInfoTracker(nn.Module):
    def __init__(self, belief_size, input_vocsize, output_vocsize, hidden_size):
        '''
        对某个slot的value进行tracking,依照的是Jordan RNN部分的公式
        :return:
        '''
        super(CNNInfoTracker, self).__init__()

        # parameters
        self.belief_size = int(belief_size/1.5)  # fixme ???
        self.input_vocsize = input_vocsize
        self.output_vocsize = output_vocsize
        self.hidden_size = hidden_size

        # Linear
        self.Ws = nn.Linear(5*self.hidden_size, self.belief_size)
        self.Wt = nn.Linear(5*self.hidden_size, self.belief_size)
        self.Wv = nn.Linear(1, self.belief_size)
        self.Wn = nn.Linear(1, self.belief_size)
        self.W = nn.Linear(self.belief_size, 1)
        self.B = nn.Linear(1, 1)

        # softmax
        self.softmax = nn.Softmax()

        # 初始化CNN
        self.source_CNN = CNNEncoder(self.input_vocsize, self.hidden_size)
        self.target_CNN = CNNEncoder(self.output_vocsize, self.hidden_size)

    def forward(self, slot_belief_value, source_input, pre_target_input, source_lens, pre_target_lens,
                slot_source_position, value_source_position, slot_target_position, value_target_position):

        # CNN encoder
        source_ngram_feature, source_utterance_feature = self.source_CNN.forward(source_input, source_lens)
        pre_target_ngram_feature, pre_target_utterance_feature = self.target_CNN.forward(pre_target_input, pre_target_lens)

        # padding
        source_ngram_feature = torch.cat([source_ngram_feature, torch.zeros(1, 1, source_ngram_feature.size()[2])], 1)
        pre_target_ngram_feature = torch.cat([pre_target_ngram_feature, torch.zeros(1, 1, \
                                                                                    pre_target_ngram_feature.size()[2])], 1)

        # new belief 对每个value的概率进行跟更新
        assert len(slot_source_position)==len(value_source_position)==len(slot_target_position)==len(value_target_position)
        value_num = len(slot_source_position)

        g_j = []
        for value in range(value_num-1):                                            # fixme 这里的-1到底该怎么替换
            # source features
            slot_source_ngram_feature_value = torch.sum(\
                source_ngram_feature[:,self._normalize_slice(slot_source_position, \
                                                             source_ngram_feature.size()[1], value),:], 1)
            value_source_ngram_feature_value = torch.sum(\
                source_ngram_feature[:,self._normalize_slice(value_source_position, \
                                                             source_ngram_feature.size()[1], value),:], 1)
            source_feature = torch.cat([slot_source_ngram_feature_value, \
                                        value_source_ngram_feature_value, \
                                        source_utterance_feature], 1)  # (1, 5*h_s)

            # target features
            slot_target_ngram_feature_value = torch.sum(\
                pre_target_ngram_feature[:,self._normalize_slice(slot_target_position, \
                                                                 pre_target_ngram_feature.size()[1], value),:], 1)
            value_target_ngram_feature_value = torch.sum(\
                pre_target_ngram_feature[:,self._normalize_slice(value_target_position, \
                                                                 pre_target_ngram_feature.size()[1], value),:], 1)
            target_feature = torch.cat([slot_target_ngram_feature_value, \
                                        value_target_ngram_feature_value, \
                                        pre_target_utterance_feature], 1)  # (1, 5*h_s)
            # fixme 怎么构造元素乘
            tmp_0 = variable_tensor([1], 'Float')
            tmp_1 = variable_tensor([1], 'Float')
            tmp_2 = variable_tensor([1], 'Float')
            g_jv = self.W (F.sigmoid( self.Ws(source_feature) + self.Wt(target_feature) +\
                                       slot_belief_value[value] * self.Wv(tmp_0) +\
                slot_belief_value[-1] * self.Wn(tmp_1)))
            g_j.append(g_jv)
        g_j = torch.cat(g_j)
        g_j = torch.cat([g_j, self.B(tmp_2)])
        b_j = self.softmax(g_j.view(1, -1))

        return b_j

    def _normalize_slice(self, position_info, replace, value):
        new_list = []
        for n in position_info[value]:
            if n != -1:
                new_list.append(n)
            else:
                new_list.append(replace-1)
        return new_list

class Req_Tracker(nn.Module):
    def __init__(self, req_seg, voc_size, hidden_size):
        super(Req_Tracker, self).__init__()
        self.req_seg = req_seg
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.slots_box = []
        for i in range(len(self.req_seg) - 1):
            self.slots_box.append(CNNReqTracker(self.voc_size, self.voc_size, self.hidden_size))


class CNNReqTracker(nn.Module):
    def __init__(self, input_vocsize, output_vocsize, hidden_size):
        super(CNNReqTracker, self).__init__()
                # parameters
        self.belief_size = 4  # fixme ???
        self.input_vocsize = input_vocsize
        self.output_vocsize = output_vocsize
        self.hidden_size = hidden_size

        # Linear
        self.Ws = nn.Linear(5*self.hidden_size, self.belief_size)
        self.Wt = nn.Linear(5*self.hidden_size, self.belief_size)
        self.Wv = nn.Linear(1, self.belief_size)
        self.Wn = nn.Linear(1, self.belief_size)
        self.W = nn.Linear(self.belief_size, 1)
        self.B = nn.Linear(1, 1)

        # softmax
        self.softmax = nn.Softmax()

        # 初始化CNN
        self.source_CNN = CNNEncoder(self.input_vocsize, self.hidden_size)
        self.target_CNN = CNNEncoder(self.output_vocsize, self.hidden_size)

    def forward(self, source_input, pre_target_input, source_lens, pre_target_lens, slot_source_position,\
                value_source_position, slot_target_position,value_target_position):

        # CNN encoder
        source_ngram_feature, source_utterance_feature = self.source_CNN.forward(source_input, source_lens)
        pre_target_ngram_feature, pre_target_utterance_feature = self.target_CNN.forward(pre_target_input, pre_target_lens)

        # padding
        source_ngram_feature = torch.cat([source_ngram_feature, torch.zeros(1, 1, source_ngram_feature.size()[2])], 1)
        pre_target_ngram_feature = torch.cat([pre_target_ngram_feature, torch.zeros(1, 1, \
                                                                                    pre_target_ngram_feature.size()[2])], 1)

        # new belief 对每个value的概率进行跟更新
        assert len(slot_source_position)==len(value_source_position)==len(slot_target_position)==len(value_target_position)
        value_num = len(slot_source_position)

        # source features
        slot_source_ngram_feature_value = torch.sum(\
                source_ngram_feature[:,self._normalize_slice(slot_source_position, \
                                                             source_ngram_feature.size()[1]),:], 1)
        value_source_ngram_feature_value = torch.sum(\
            source_ngram_feature[:,self._normalize_slice(value_source_position, \
                                                         source_ngram_feature.size()[1]),:], 1)
        source_feature = torch.cat([slot_source_ngram_feature_value, \
                                    value_source_ngram_feature_value, \
                                    source_utterance_feature], 1)  # (1, 5*h_s)

        # target features
        slot_target_ngram_feature_value = torch.sum(\
            pre_target_ngram_feature[:,self._normalize_slice(slot_target_position, \
                                                             pre_target_ngram_feature.size()[1]),:], 1)
        value_target_ngram_feature_value = torch.sum(\
            pre_target_ngram_feature[:,self._normalize_slice(value_target_position, \
                                                             pre_target_ngram_feature.size()[1]),:], 1)
        target_feature = torch.cat([slot_target_ngram_feature_value, \
                                    value_target_ngram_feature_value, \
                                    pre_target_utterance_feature], 1)  # (1, 5*h_s)

        # fixme 怎么构造元素乘
        tmp_0 = variable_tensor([1], 'Float')
        tmp_1 = variable_tensor([1], 'Float')
        tmp_2 = variable_tensor([1], 'Float')

        g_j = self.W (F.sigmoid( self.Ws(source_feature) + self.Wt(target_feature)))
        g_j = torch.cat(g_j)
        g_j = torch.cat([g_j, self.B(tmp_2)])
        b_j = self.softmax(g_j.view(1, -1))

        return b_j


    def _normalize_slice(self, position_info, replace, value=None):
        new_list = []
        if value == None:
            for n in position_info:
                if n != -1:
                    new_list.append(n)
                else:
                    new_list.append(replace-1)
        else:
            for n in position_info[value]:
                if n != -1:
                    new_list.append(n)
                else:
                    new_list.append(replace-1)
        return new_list

class CNNEncoder(nn.Module):
    def __init__(self, voc_size, hidden_size):
        super(CNNEncoder, self).__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.kernel_num = 1
        self.kernel_size = (3, 1)

        # embedding
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)  # fixme

        # 1st conv
        self.conv2d_1 = nn.Conv2d(1, self.kernel_num, self.kernel_size, padding=1)
        # 2nd conv
        self.conv2d_2 = nn.Conv2d(1, self.kernel_num, self.kernel_size, padding=1)
        # 3rd conv
        self.conv2d_3 = nn.Conv2d(1, self.kernel_num, self.kernel_size, padding=1)


    def forward(self, input, sentence_lens):
        # embedding
        embed = self.embedding(input)  # (l, h_s)

        # 1st conv
        conv1 = self._conv_and_pool(embed.unsqueeze(0), False)

        # 2nd conv
        conv2 = self._conv_and_pool(conv1, False)

        # 3rd conv
        conv3 = self._conv_and_pool(conv2)
        return torch.cat([conv1, conv2], 2), conv3

    def _conv_and_pool(self, input, pool=True):
        input = F.relu(self.conv2d_1(input.unsqueeze(0)))
        input = input.squeeze(0)[:,:,1:-1]  # (1, l, h_s)
        input = F.tanh(torch.sum(F.avg_pool2d(input, (3,1)), 1)) if pool else input  # (l, h_s)
        return input


def variable_tensor(list, type=None):
    if type == 'Long':
        return Variable(torch.LongTensor(list)).cuda() if use_cuda else Variable(torch.LongTensor(list))
    elif type == 'Float':
        return Variable(torch.FloatTensor(list)).cuda() if use_cuda else Variable(torch.FloatTensor(list))
    else:
        return Variable(list).cuda() if use_cuda else Variable(list)
























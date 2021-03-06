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
import itertools
from Queue import PriorityQueue
from ConfigParser import SafeConfigParser
import numpy as np
from copy import deepcopy
import operator


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

use_cuda = torch.cuda.is_available()

class Network(nn.Module):
    def __init__(self, args, config, datareader):
        super(Network, self).__init__()
        self.args = args
        self.config = config
        self.hidden_size = self.args.hidden_size
        self.encoder_type = args.encoder_type  # 默认为lstm #Todo CNN可以尝试加一下
        self.datareader = datareader
        self.voc_size = len(self.datareader.vocab)

        # init intent encoder
        self.encoder = Encoder(self.args.embed_size, self.hidden_size, self.voc_size)

        # parameters for belief tracker
        self.reqseg = self.datareader.reqseg
        self.infoseg = self.datareader.infoseg
        self.chgseg = [0, 1]

        # init three trackers
        self.info_tracker = Info_Tracker(self.infoseg, self.voc_size, self.hidden_size)
        self.req_tracker = Req_Tracker(self.reqseg, self.voc_size, self.hidden_size)
        self.chg_tracker = Req_Tracker(self.chgseg, self.voc_size, self.hidden_size)

        # loss function
        self.loss = nn.NLLLoss()

        # belief size
        belief_size = 2*len(self.reqseg) + 3*(len(self.infoseg) - 1)  # 23

        # init policy
        # self.args.policy_flag
        self.policy = Policy(belief_size, 6, self.hidden_size, self.hidden_size)

        # init decoder
        self.decoder = Decoder(self.policy, self.datareader.vocab, self.voc_size, self.hidden_size, self.config)

        # optimizer
        self.optimizer = Opitmzier(self.encoder, self.info_tracker, self.req_tracker, \
                                  self.chg_tracker, self.policy, self.decoder, self.args.lr)


    def train(self):
        self._set_model_satate('train')
        self._cuda_model()
        for i in range(self.args.max_epoch):
            logging.info('--------------------Round {0}---------------------'.format(i))
            total_loss = 0
            count = 0
            while True:
                data = self.datareader.read()
                if data == None:
                    logging.info('Round Completed!')
                    break

                '''
                source_0, source_len_1, masked_source_2, masked_source_len_3,\
                target_4, target_len_5, masked_target_6, masked_target_len_7,\
                snapshot_8, change_9, goal_10, inf_trk_label_11, req_trk_label_12,\
                db_degree_13, srcfeat_14, tarfeat_15, finished_16, utt_group_17 = data  # 除了goal和finished以外长度都为对话轮次长
                '''
                ###############tansform#################
                # print self.datareader.vocab
                # for idx,len in enumerate(data[3]):
                #     idx_list = data[2][idx][:len]
                #     sentence = [self.datareader.vocab[id] for id in idx_list]
                #     print sentence
                # for idx,len in enumerate(data[7]):
                #     idx_list = data[6][idx][:len]
                #     sentence = [self.datareader.vocab[id] for id in idx_list]
                #     print sentence
                # print data[2]
                # print data[3]
                # print data[6]
                # print data[7]

                # dialogue_recur
                dialogue_loss, target, predict = self.recurr(data)

                if type(dialogue_loss) == int:
                    continue

                total_loss += dialogue_loss.data[0]
                count += 1
                logging.info('count: {0}'.format(count))
            logging.info('avg_loss: {0}'.format(total_loss/(count+0.000000000001)))


    def _set_model_satate(self, state):
        if state == 'train':
            self.encoder.train()
            self.info_tracker.train()
            self.req_tracker.train()
            self.chg_tracker.train()
            self.policy.train()
            self.decoder.train()
        elif state == 'eval':
            self.encoder.eval()
            self.info_tracker.eval()
            self.req_tracker.eval()
            self.chg_tracker.eval()
            self.policy.eval()
            self.decoder.eval()


    def _cuda_model(self):
        self.encoder = self._cuda_model_one(self.encoder)
        self.decoder = self._cuda_model_one(self.decoder)
        for model in self.info_tracker.slots_box:
            model = self._cuda_model_one(model)
        for model in self.req_tracker.slots_box:
            model = self._cuda_model_one(model)
        for model in self.chg_tracker.slots_box:
            model = self._cuda_model_one(model)
        self.policy = self._cuda_model_one(self.policy)


    def _cuda_model_one(self, model):
        return model.cuda() if use_cuda else model


    def _zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        for model in self.info_tracker.slots_box:
            model.zero_grad()
        for model in self.req_tracker.slots_box:
            model.zero_grad()
        for model in self.chg_tracker.slots_box:
            model.zero_grad()
        self.policy.zero_grad()
        self.decoder.zero_grad()


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
                        var = variable_tensor(data[j][i], 'Long') if type(data[j][i]) == list else variable_tensor([data[j][i]], 'Long')
                        data_piece.append(var)
                    else:
                        data_piece.append(data[j][i])
        if len(data_piece) == len(data) - 2 :
            return data_piece
        else:
            logging.info('data error')
            return None


    def recurr(self, data):

        #assert data[-2] == True  # 确认对话已经结束
        turn_number = len(data[0])

        # initial belief_0
        belief_0 = torch.zeros((1, self.infoseg[-1]))
        belief_0 = [belief_0[0][i-1] + 1  if i in self.infoseg[1:] else belief_0[0][i-1] for i in range(1, self.infoseg[-1]+1)]
        belief_0 = variable_tensor(belief_0, 'Float').unsqueeze(1)
        pre_belief = belief_0

        # initial pre_target
        masked_target_tm1 = variable_tensor([1] * len(data[6][0]), 'Long')
        masked_target_len_tm1 = variable_tensor([data[7][0]], 'Long')

        # initial pre_target position features
        pre_target_position = -torch.ones((1,data[17][0]))

        # initial posterior # fixme
        '''
        belief_tm1, masked_target_tm1, masked_target_len_tm1,
                    target_feat_tm1, posterior_tm1
        '''
        # dialogue_loss
        dialogue_loss = 0
        predict = []

        ##
        for i in range(turn_number):
            self._set_model_satate('train')
            self._zero_grad()
            '''
            data的格式在上面，这是recur的参数
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
            if data_piece == None:
                print ('data_piece error')
                return 0, [], []
            else:
                source, source_len, masked_source, masked_source_len,\
                target, target_len, masked_target, masked_target_len,\
                snapshot, change, inf_trk_label, req_trk_label,\
                db_degree, srcfeat, tarfeat, utt_group = data_piece

            # encode
            intent_t = self.encoder.forward(masked_source, masked_source_len)

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
                new_slot_belief_value = self.info_tracker.slots_box[j].forward(slot_belief_value, masked_source, masked_target_tm1, \
                                                       masked_source_len, masked_target_len_tm1,\
                                                       slot_source_position, value_source_position,\
                                                       slot_target_position, value_target_position)

                slot_belief_label = variable_tensor(inf_trk_label[self.infoseg[j]:self.infoseg[j+1]], 'Long')

                loss += self.loss(new_slot_belief_value.view(1, -1), torch.max(slot_belief_label, 0)[1])  # 交叉熵不接受one-hot向量
                                                                                                          # 而是接收正确的indices
                # summary
                tmp = [torch.sum(slot_belief_label[:-2]), slot_belief_label[-2], slot_belief_label[-1]]
                belief_t.append(torch.cat(tmp, 0))

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
                new_slot_belief_value = self.req_tracker.slots_box[k].forward(masked_source, masked_target_tm1, masked_source_len,\
                                                                              masked_target_len_tm1, slot_source_position,\
                                                                              value_source_position, slot_target_position,\
                                                                              value_target_position)

                slot_belief_label = variable_tensor(req_trk_label[2*k:2*(k+1)], 'Long')

                loss += self.loss(new_slot_belief_value.view(1, -1), torch.max(slot_belief_label, 0)[1])  # 交叉熵不接受one-hot向量

                belief_t.append(slot_belief_label)

            # offer-change tracker 是否变更了候选
            minus_1 = [-1]
            new_slot_belief_value = self.chg_tracker.slots_box[0].forward(masked_source, masked_target_tm1, masked_source_len,\
                                                                              masked_target_len_tm1, minus_1, minus_1,\
                                                                              minus_1, minus_1)

            slot_belief_label = variable_tensor(change, 'Long')

            loss += self.loss(new_slot_belief_value.view(1, -1), torch.max(slot_belief_label, 0)[1])  # 交叉熵不接受one-hot向量
            belief_t.append(variable_tensor(change, 'Long'))
            belief_t = torch.cat(belief_t, 0).float()
            db_degree_t = variable_tensor(db_degree[-6:], 'Float')

            # policy and decode
            new_loss, predict_box = self.decoder.decode(masked_source, masked_source_len, masked_target, masked_target_len, intent_t, belief_t,\
                                 db_degree_t, utt_group, snapshot)

            # debug
            # target_sent = ' '.join([self.datareader.vocab[idx] for idx in data[6][i]])
            # print 'target:',target_sent
            # predict_sent = ' '.join([self.datareader.vocab[idx] for idx in predict_box])
            # print 'predict',predict_sent
            # predict.append(predict_box)

            loss += new_loss

            loss.backward()

            # step
            self.optimizer.step()

            dialogue_loss += loss

            # next iter
            #print masked_target_tm1
            #print masked_target
            #print '-------------------------turn over-------------------'
            masked_target_tm1 = masked_target
            masked_target_len_tm1 = masked_target_len_tm1

        return dialogue_loss, data[6], predict


    def read(self, masked_source_t):
        return self.encoder.forward(masked_source_t, None)


    def track(self, pre_belief, masked_source, masked_target_tm1, srcfeat_t, tarfeat_tm1):

        # belief tracking
        # informable slots
        belief_t = []
        full_belief_t = []
        for j in range(len(self.info_tracker.slots_box)):

            # 当前slot的上一轮次的value的分布
            slot_belief_value = pre_belief[self.infoseg[j]:self.infoseg[j+1]]

            # 对ngram的position进行解码
            slot_source_position = srcfeat_t[0][self.infoseg[j]:self.infoseg[j+1]]
            value_source_position = srcfeat_t[1][self.infoseg[j]:self.infoseg[j+1]]
            slot_target_position = tarfeat_tm1[0][self.infoseg[j]:self.infoseg[j+1]]
            value_target_position = tarfeat_tm1[1][self.infoseg[j]:self.infoseg[j+1]]

            # tracking
            new_slot_belief_value = self.info_tracker.slots_box[j].forward(slot_belief_value, masked_source, masked_target_tm1, \
                                                   None, None,\
                                                   slot_source_position, value_source_position,\
                                                   slot_target_position, value_target_position)

            new_slot_belief_value = new_slot_belief_value.squeeze(0)
            # full
            full_belief_t.append(new_slot_belief_value)

            # summary
            tmp = [torch.sum(new_slot_belief_value[:-2]), new_slot_belief_value[-2], new_slot_belief_value[-1]]
            belief_t.append(torch.cat(tmp, 0))

        inf_belief_t = pre_belief

        # requestable slots
        for k in range(len(self.reqseg)-1):
            # current feature idx
            bn = self.infoseg[-1] + 2*k  # 排列在infor之后，每个占两位

            # 解码位置信息
            slot_source_position = srcfeat_t[0][bn]
            value_source_position = srcfeat_t[1][bn]
            slot_target_position = tarfeat_tm1[0][bn]
            value_target_position = tarfeat_tm1[1][bn]

            # tracking
            new_slot_belief_value = self.req_tracker.slots_box[k].forward(masked_source, masked_target_tm1, None,\
                                                                          None, slot_source_position,\
                                                                          value_source_position, slot_target_position,\
                                                                          value_target_position)

            new_slot_belief_value = new_slot_belief_value.squeeze(0)

            full_belief_t.append(new_slot_belief_value)

            belief_t.append(new_slot_belief_value)

        # offer-change tracker 是否变更了候选
        minus_1 = [-1]
        new_slot_belief_value = self.chg_tracker.slots_box[0].forward(masked_source, masked_target_tm1, None,\
                                                                          None, minus_1, minus_1,\
                                                                          minus_1, minus_1)

        new_slot_belief_value = new_slot_belief_value.squeeze(0)

        full_belief_t.append(new_slot_belief_value)
        belief_t.append(new_slot_belief_value)
        belief_t = torch.cat(belief_t, 0).float()
        flatten_belief_t_tensor = torch.cat(full_belief_t, 0)
        full_belief_t_numpy = [lst.data.numpy() for lst in full_belief_t]
        #full_belief_t = torch.cat(full_belief_t, 0).data.numpy()

        return full_belief_t_numpy, flatten_belief_t_tensor, belief_t


    def talk(self, masked_intent_t, belief_t, degree_t, masked_source_t=None, masked_target_t=None, \
             scoreTable=None, forced_sample=None):
        responses, sample, prob = self.decoder.talk( masked_intent_t, belief_t, degree_t[-6:], masked_source_t, masked_target_t,
                        scoreTable, forced_sample)
        return responses, sample, prob




class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.V = vocab_size
        self.embedding = nn.Embedding(self.V, self.embed_size)
        #self.cnn = nn.Conv1d()
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True)#, bidirectional=True)

    def forward(self, input, lens):
        outputs, (h_last, c_last) = self.sort_batch(input[:lens.data[0]]) if type(lens) != type(None) else self.sort_batch(input)
        #h_last = torch.sum(h_last, 0) * 0.5  # (1, h_s)
        #c_last = torch.sum(c_last, 0) * 0.5
        return h_last.squeeze(0), c_last.squeeze(0)

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
        outputs, (h, c) = self.lstm(embed.unsqueeze(0))  # outputs:(1,m_l, 2*h_s) h:(2, 1, h_s)
        return outputs, (h, c)


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
        self.softmax = nn.LogSoftmax()

        # 初始化CNN
        self.source_CNN = CNNEncoder(self.input_vocsize, self.hidden_size)
        self.target_CNN = CNNEncoder(self.output_vocsize, self.hidden_size)

    def forward(self, slot_belief_value, source_input, pre_target_input, source_lens, pre_target_lens,
                slot_source_position, value_source_position, slot_target_position, value_target_position):

        # CNN encoder
        source_ngram_feature, source_utterance_feature = self.source_CNN.forward(source_input, source_lens)
        pre_target_ngram_feature, pre_target_utterance_feature = self.target_CNN.forward(pre_target_input, pre_target_lens)

        # padding
        zero_tensor = variable_tensor(torch.zeros(1, 1, source_ngram_feature.size()[2]), 'Float')
        source_ngram_feature = torch.cat([source_ngram_feature, zero_tensor], 1)
        pre_target_ngram_feature = torch.cat([pre_target_ngram_feature, zero_tensor], 1)

        # new belief 对每个value的概率进行跟更新
        assert len(slot_source_position)==len(value_source_position)==len(slot_target_position)==len(value_target_position)
        value_num = len(slot_source_position)

        g_j = []
        for value in range(value_num-1):                                            # fixme 这里的-1到底该怎么替换
            #print value
            # source features
            # print slot_source_position
            # print source_ngram_feature
            # print self._normalize_slice(slot_source_position, \
            #                                                  source_ngram_feature.size()[1], value)
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
            # print slot_target_position
            # print self._normalize_slice(slot_target_position, \
            #                                                      pre_target_ngram_feature.size()[1], value)
            # print pre_target_ngram_feature
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
        if len(position_info[value]) == 0:
            new_list.append(replace-1)
            return new_list
        for n in position_info[value]:
            if (n != -1) & (n<=(replace-1)):
                new_list.append(n)
            elif (n > (replace-1)) | (n == -1):
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
        self.softmax = nn.LogSoftmax()

        # 初始化CNN
        self.source_CNN = CNNEncoder(self.input_vocsize, self.hidden_size)
        self.target_CNN = CNNEncoder(self.output_vocsize, self.hidden_size)

    def forward(self, source_input, pre_target_input, source_lens, pre_target_lens, slot_source_position,\
                value_source_position, slot_target_position,value_target_position):

        # CNN encoder
        source_ngram_feature, source_utterance_feature = self.source_CNN.forward(source_input, source_lens)
        pre_target_ngram_feature, pre_target_utterance_feature = self.target_CNN.forward(pre_target_input, pre_target_lens)

        # padding
        zero_tensor = variable_tensor(torch.zeros(1, 1, source_ngram_feature.size()[2]), 'Float')
        source_ngram_feature = torch.cat([source_ngram_feature, zero_tensor], 1)
        pre_target_ngram_feature = torch.cat([pre_target_ngram_feature, zero_tensor], 1)

        # new belief 对每个value的概率进行跟更新
        #assert len(slot_source_position)==len(value_source_position)==len(slot_target_position)==len(value_target_position)
        #value_num = len(slot_source_position)

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
        # print slot_target_position
        # print self._normalize_slice(slot_target_position, \
        #                                                      pre_target_ngram_feature.size()[1])
        # print pre_target_ngram_feature
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
        if len(position_info) == 0:
            new_list.append(replace-1)
            return new_list
        if value == None:
            for n in position_info:
                if (n != -1) & (n<=(replace-1)):
                    new_list.append(n)
                elif (n > (replace-1)) | (n == -1):
                    new_list.append(replace-1)
        else:
            for n in position_info[value]:
                if (n != -1) & (n<=(replace-1)):
                    new_list.append(n)
                elif (n > (replace-1)) | (n == -1):
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
        embed = self.embedding(input[:sentence_lens.data[0]])  if type(sentence_lens) != type(None) else self.embedding(input)# (l, h_s)
        #embed = self.embedding(input)
        # 1st conv
        conv1 = self._conv_and_pool(embed.unsqueeze(0), False)

        # 2nd conv
        conv2 = self._conv_and_pool(conv1, False)

        # 3rd conv
        conv3 = self._conv_and_pool(conv2)
        return torch.cat([conv1, conv2], 2) , conv3

    def _conv_and_pool(self, input, pool=True):
        input = F.relu(self.conv2d_1(input.unsqueeze(0)))
        input = input.squeeze(0)[:,:,1:-1]  # (1, l, h_s)
        input = F.tanh(torch.sum(F.avg_pool2d(input, (3,1)), 1)) if pool else input  # (l, h_s)
        return input


class Policy(nn.Module):
    def __init__(self, belief_size, degree_size, ihidden_size, ohidden_size):
        super(Policy, self).__init__()

        # parameters
        self.Wba = nn.Linear(belief_size, ohidden_size)
        self.Wda = nn.Linear(degree_size, ohidden_size)
        self.Wia = nn.Linear(ihidden_size, ohidden_size)


    def encode(self, belief_t, degree_t, intent_t):  # 公式(8)
        return F.tanh( self.Wba(belief_t) + self.Wda(degree_t) + self.Wia(intent_t))
        #return F.tanh( self.Wia(intent_t))


class Decoder(nn.Module):
    def __init__(self, policy, vocab, voc_size, hidden_size, config):
        super(Decoder, self).__init__()

        # policy
        self.policy = policy

        # vocab
        self.vocab = vocab

        # setting
        self.topk           = config.topk
        self.beamwidth      = config.beamwidth
        self.repeat_penalty = config.repeat_penalty
        self.token_reward   = config.token_reward
        self.alpha          = config.alpha
        self.q_limit        = 10000

        # special token accumulation table
        self.recordTable = {'s':{},'v':{}}
        for idx in range(len(self.vocab)):
            w = self.vocab[idx]
            if w.startswith('[VALUE_'):
                self.recordTable['v'][idx] = 0.0
            elif w.startswith('[SLOT_'):
                self.recordTable['s'][idx] = 0.0

        # parameters
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(voc_size, hidden_size)
        self.lstmcell = nn.LSTMCell(2 * hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, voc_size)
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        self.dropout = nn.Dropout(0.7)


    def decode(self, masked_source_t, masked_source_len_t, masked_target_t, masked_target_len_t, intent_t, \
               belief_t, degree_t, utt_group_t, snapshot_t):  # sample?
        # loss
        loss = 0

        # action
        action_t = self.policy.encode(belief_t, degree_t, intent_t[0])

        # init input
        input_word = self.embed(variable_tensor([1], 'Long'))  # fixme 选什么做初始化
        #input_word = F.sigmoid(self.embed(masked_source_t[0]))

        # init hideen
        (h, c) = intent_t

        right_count, false_count = 0, 0
        predict_word = []
        # recur
        for i in range(1, masked_target_len_t.data[0] - 1):
            h, c = self.lstmcell(torch.cat([input_word, action_t], 1), (h, c))
            #h, c = self.lstmcell(input_word, (h, c))
            output = self.softmax(self.linear(h))
            _, predict = torch.max(output, 1)
            predict_word.append(predict.data[0])
            if predict.data[0] == masked_target_t[i].data[0]:
                right_count += 1
            else:
                false_count += 1
            loss += self.loss(output, masked_target_t[i])
            input_word = self.embed(masked_target_t[i])
        print right_count * 1.0 /(right_count+false_count )

        return loss, predict_word


    def talk(self, masked_intent_t, belief_t, degree_t, masked_source_t=None, masked_target_t=None,\
             scoreTable=None, forced_sample=None):

        sample_t = prob_t = None

        action = self.policy.encode(belief_t, degree_t, masked_intent_t[0]).data.numpy()

        # store end node
        endnodes = []

        # iterate through action
        for a in range(action.shape[0]):
            pre_endnode = len(endnodes)
            number_required = min(  (self.topk+1)/action.shape[0],
                                    self.topk-len(endnodes))

            # hidden layers to be stored
            h0 = np.zeros(self.hidden_size)
            c0 = np.zeros(self.hidden_size)
            # starting node
            node = BeamSearchNode(masked_intent_t[0],masked_intent_t[1],None,1,0,1,\
                    record=deepcopy(self.recordTable))
            nodes= PriorityQueue()
            # put it in the queue
            nodes.put(( -node.eval(self.repeat_penalty,self.token_reward,\
                    scoreTable,self.alpha), node))
            qsize = 1
            # start beam search

            # intent h,c
            (h, c) = masked_intent_t
            while True:

                # give up when decoding takes too long
                if qsize>self.q_limit: break

                # fetch the best node
                score, n = nodes.get()

                # if end of sentence token
                if n.wordid==1 and n.prevNode!=None:
                    endnodes.append((score,n))
                    # if reach maximum # of sentences required
                    if len(endnodes)-prev_endnode>=number_required:break
                    else:   continue

                # decode for one step using decoder
                nextnodes = self.talk_forward(n,
                        masked_intent_t, belief_t, degree_t, action[a,:], (h, c),
                        scoreTable)
                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put( (score,nn) )
                # increase qsize
                qsize += len(nextnodes)-1

        # choose nbest paths, back trace them
        if len(endnodes)==0:
            endnodes = [nodes.get() for n in range(self.topk)]
        utts = []
        for score,n in sorted(endnodes,key=operator.itemgetter(0)):
            utt,att,gates = [],[],[]
            utt.append(n.wordid)
            # back trace
            while n.prevNode!=None:
                n = n.prevNode
                utt.append(n.wordid)

            utt,att,gates = utt[::-1],att[::-1],gates[::-1]
            utts.append([utt,att,gates])
        return utts, sample_t, prob_t


    def talk_forward(self, n, masked_intent_t, belief_t, degree_t, action_t, h_t, scoreTable):

        input_word = self.embed(variable_tensor([n.wordid], 'Long'))
        action_t = variable_tensor(action_t, 'Float').view(1, -1)
        h, c = self.lstmcell(torch.cat([input_word, action_t], 1), (n.h, n.c))
        output = self.softmax(self.linear(h))

        _, predict = torch.topk(output, self.beamwidth)

        nextnodes = []
        for pr in predict.data[0]:
            if pr == 0:
                continue

            # loglikelihood of current word
            logp = output[0][pr].data[0]

            # update record for new node
            new_record = deepcopy(n.record)
            if new_record['s'].has_key(pr):
                new_record['s'][pr] += 1
            if new_record['v'].has_key(pr):
                new_record['v'][pr] += 1

            # create new node and score it
            node = BeamSearchNode(h,c,n,pr,\
                    n.logp+logp,n.leng+1,new_record)

            # store nodes
            nextnodes.append( \
                    (-node.eval(self.repeat_penalty,self.token_reward,\
                    scoreTable,self.alpha), node))

        return nextnodes



class Opitmzier(nn.Module):
    def __init__(self, encoder, info_tracker, req_tracker, chg_tracker, policy, deocder, lr):
        super(Opitmzier, self).__init__()
        self.lr = lr
        self.encoder_optimier = self._init_opitmizer(encoder)
        self.info_tracker_optimizer = []
        self.req_tracker_optimizer = []
        self.chg_tracker_optimizer = []
        for slot in info_tracker.slots_box:
            self.info_tracker_optimizer.append(self._init_opitmizer(slot))
        for slot in req_tracker.slots_box:
            self.req_tracker_optimizer.append(self._init_opitmizer(slot))
        for slot in chg_tracker.slots_box:
            self.chg_tracker_optimizer.append(self._init_opitmizer(slot))
        self.policy_optimier = self._init_opitmizer(policy)
        self.deocder_optimier = self._init_opitmizer(deocder)

    def _init_opitmizer(self, model):
        optimizer = optim.Adam(
          model.parameters(),
          lr = self.lr
        )
        return optimizer

    def step(self):
        self.encoder_optimier.step()
        for optimizer in self.info_tracker_optimizer:
            optimizer.step()
        for optimizer in self.req_tracker_optimizer:
            optimizer.step()
        for optimizer in self.chg_tracker_optimizer:
            optimizer.step()
        self.policy_optimier.step()
        self.deocder_optimier.step()


class BeamSearchNode(object):

    def __init__(self,h,c,prevNode,wordid,logp,leng,record):
        self.h = h
        self.c = c
        self.prevNode = prevNode
        self.wordid = wordid
        self.logp   = logp
        self.leng   = leng
        self.record = record

    def eval(self, repeatPenalty, tokenReward, scoreTable, alpha=1.0):
        reward = 0
        # repeat penalty
        if repeatPenalty=='inf':
            # value repeat is not allowed
            for k,v in self.record['v'].iteritems():
                if v>1: reward -= 1000
            # slot repeat is slightly allowed
            for k,v in self.record['s'].iteritems():
                if v>1: reward -= pow(v-1,2)*0.5
        # special token reward
        if tokenReward and scoreTable!=None:
            for k,v in self.record['v'].iteritems():
                if v>0 and scoreTable.has_key(k):
                    reward += scoreTable[k]

        return self.logp/float(self.leng-1+1e-6)+alpha*reward


def variable_tensor(list, type=None):
    if type == 'Long':
        return Variable(torch.LongTensor(list)).cuda() if use_cuda else Variable(torch.LongTensor(list))
    elif type == 'Float':
        return Variable(torch.FloatTensor(list)).cuda() if use_cuda else Variable(torch.FloatTensor(list))
    else:
        return Variable(list).cuda() if use_cuda else Variable(list)


def flatten(lst):
    return list(itertools.chain.from_iterable(lst))






















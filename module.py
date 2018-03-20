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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

use_cuda = torch.cuda.is_available()

class Network(nn.Module):
    def __init__(self, args, datareader):
        super(Network, self).__init__()
        self.args = args
        self.encoder_type = args.encoder_type
        self.datareader = datareader
        self.V = len(self.datareader.Vocab)
        self.encoder = Encoder(self.args.embed_size, self.args.hidden_size, self.V)

    def train(self):
        for i in range(self.args.max_epoch):
            logging.info('--------------------Round {0}---------------------'.format(i))
            while True:
                data = self.datareader.read()
                if data == None:
                    break
                source, source_len, masked_source, masked_source_len,\
                target, target_len, masked_target, masked_target_len,\
                snapshot, change, goal, inf_trk_label, req_trk_label,\
                db_degree, srcfeat, tarfeat, finished, utt_group = data

                self.encoder.forward(data)








class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.V = vocab_size
        self.embedding = nn.Embedding(self.V, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        #self.cnn = nn.Conv1d()

    def forward(self, data):
        self.data = data
        _, _ = self.sort_batch()




    def sort_batch(self,):  # Todo:使用padpackage
        input = Variable(torch.LongTensor(self.data[2])).cuda() if use_cuda else Variable(torch.LongTensor(self.data[2]))
        self.batch_size = len(input)
        embed = self.embedding(input)
        #sentence_lens = self.data[3]
        #lst = list(range(self.batch_size))
        #lst = sorted(lst, key=lambda d: -sentence_lens[d])
        outputs, (h, c) = self.lstm(embed)




































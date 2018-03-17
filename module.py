#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optom
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.encoder = Encoder()


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, batch_first=True, bidirectional=True)
        #self.cnn = nn.Conv1d()


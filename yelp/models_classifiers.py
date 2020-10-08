import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import to_gpu
import json
import os
import numpy as np


class LSTMClassifier(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noutput=1, noise_r=0.2, hidden_init=False, dropout=0, gpu=False):
        super(LSTMClassifier, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        # self.linear = nn.Linear(nhidden, ntokens)
        self.linear = nn.Linear(nhidden, noutput)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)
        
    def forward(self, whichdecoder, indices, lengths, noise=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        # if hidden.requires_grad:
        #     hidden.register_hook(self.store_grad_norm)
        x = self.linear(hidden)
        output = F.sigmoid(x)

        return output

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)
        hidden, cell = state

        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        return hidden
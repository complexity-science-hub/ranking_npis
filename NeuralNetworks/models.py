import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
import os




######## TRANSFORMER ############################################################################
class TransformerModel(nn.Module):

    #def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        
        self.ninp = config['network']['emsize']
        self.nhead = config['network']['nhead']
        self.nhid = config['network']['nhid']
        self.nlayers = config['network']['nlayers']
        self.ntoken = config['network']['input_size']
        self.nout = config['network']['output_size']
        self.predec_size = config['network'].get('predecoder_size')
        if self.predec_size is None:
            self.predec_size = 0
        
        self.dropout = config['train']['dropout']
        
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.ninp, self.dropout)
        self.encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, self.nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = nn.Linear(self.ntoken, self.ninp)
        #self.ninp = ninp
        #self.decoder = nn.Linear(ninp, ntoken)
        
        if self.predec_size > 0:
            self.predecoder = nn.Linear(self.ninp, self.predec_size)
            self.decoder = nn.Linear(self.predec_size, self.nout)
        else:
            self.decoder = nn.Linear(self.ninp, self.nout)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        if self.predec_size > 0:
            output = self.predecoder(output)
            tanh = nn.Tanh()
            output = tanh(output)
        
        output = self.decoder(output)
        
        return output



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

###############################################################################################################



######## LSTM #################################################################################################
class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.input_size = config['network']['input_size']
        self.output_size = config['network']['output_size']
        self.hidden_size = config['network']['hidden_size']
        self.n_layers = config['network']['n_layers']
        self.learning_rate = config['train']['learning_rate']
        self.dropout = config['train']['dropout']
        
        # RNN Architecture
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.encoder = nn.LSTM(self.hidden_size, self.hidden_size,
                               self.n_layers, dropout=self.dropout)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, inp, hidden):
        embedded_input = self.input_layer(inp)
        lstm_output, hidden = self.encoder(embedded_input, hidden)
        output = self.decoder(lstm_output)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(self.n_layers, 1, self.hidden_size),
                torch.zeros(self.n_layers, 1, self.hidden_size))

#################################################################################################################
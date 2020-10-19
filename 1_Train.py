import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import random
import copy
import os

from utils import HandlingNets
from models import PositionalEncoding
from models import TransformerModel
from models import LSTMModel
from utils import get_data_tensor
from utils import ma
from utils import get_net_output


# General setting

model_type = 'LSTM' #'Transformer', 'LSTM' # 'LSTM', 'Transformer'

measure_level = 'L2'

measure_mode = 'theta' # 'theta' ONLY!

output_mode = 'r' # 'r', 'dr', 'drrel'

output_tanh = 0 # 0: 'linear', >0: 'tanh', only used wiyth Transformer

continent_knockout = '' # '': all, 'NoEurope', 'NoAsia', 'NoAmerica'



# Loading data

filename = './Data/data.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)


# Sorting country names

countries = data['countries']
countries = sorted(countries)


# Continent knockout countries selection

if continent_knockout == '':
    countries_to_remove = []
    test_countries = ['United Kingdom',
                     'Kazakhstan',
                     'Iceland',
                     'Denmark',
                     'Mexico',
                     'Bosnia and Herzegovina',
                     'Netherlands',
                     'Spain',
                     'US - Illinois',
                     'US - Indiana']

elif continent_knockout == 'NoEurope':
    countries_to_remove = ['Albania',
                         'Austria',
                         'Belgium',
                         'Bosnia and Herzegovina',
                         'Croatia',
                         'Czechia',
                         'Denmark',
                         'Estonia',
                         'Finland',
                         'France',
                         'Germany',
                         'Ghana',
                         'Greece',
                         'Hungary',
                         'Iceland',
                         'Ireland',
                         'Italy',
                         'Kosovo',
                         'Liechtenstein',
                         'Lithuania',
                         'Mauritius',
                         'Montenegro',
                         'Netherlands',
                         'North Macedonia',
                         'Norway',
                         'Poland',
                         'Portugal',
                         'Romania',
                         'Senegal',
                         'Serbia',
                         'Slovakia',
                         'Slovenia',
                         'Spain',
                         'Sweden',
                         'Switzerland',
                         'United Kingdom']
    test_countries = ['US - Louisiana', 'Thailand', 'US - Maine', 'US - Michigan',
                     'India', 'US - California']

elif continent_knockout == 'NoAsia':
    countries_to_remove = ['China',
                         'India',
                         'Indonesia',
                         'Japan',
                         'Kazakhstan',
                         'Korea, South',
                         'Kuwait',
                         'Malaysia',
                         'New Zealand',
                         'Singapore',
                         'Syria',
                         'Taiwan*',
                         'Thailand']
    test_countries = ['United Kingdom',
                     'Slovenia',
                     'Iceland',
                     'Denmark',
                     'US - Colorado',
                     'Netherlands',
                     'Spain',
                     'Mexico',
                     'US - Alabama',
                     'US - California']

elif continent_knockout == 'NoAmerica':
    countries_to_remove = ['Brazil',
                         'Canada',
                         'Ecuador',
                         'El Salvador',
                         'Honduras',
                         'Mexico',
                         'US - Alabama',
                         'US - Alaska',
                         'US - Arizona',
                         'US - California',
                         'US - Colorado',
                         'US - Connecticut',
                         'US - Delaware',
                         'US - Florida',
                         'US - Georgia',
                         'US - Hawaii',
                         'US - Idaho',
                         'US - Illinois',
                         'US - Indiana',
                         'US - Iowa',
                         'US - Kansas',
                         'US - Kentucky',
                         'US - Louisiana',
                         'US - Maine',
                         'US - Maryland',
                         'US - Massachusetts',
                         'US - Michigan',
                         'US - New York',
                         'US - Wisconsin',
                         'US - Wyoming']
    test_countries = ['United Kingdom',
                     'Kazakhstan',
                     'Iceland',
                     'Denmark',
                     'Indonesia',
                     'Netherlands',
                     'Spain',
                     'New Zealand']

countries = sorted(list(set(countries) - set(countries_to_remove)))


# Countries for training

train_countries = list(set(countries) - set(test_countries))



# Neural network configuration

config = dict()

input_size = len(data['Italy']['l2_theta'][0]) + 1
output_size = 1

if model_type == 'Transformer':
    
    config['network'] = dict()
    config['network']['input_size'] = input_size
    config['network']['output_size'] = output_size

    config['network']['emsize'] = 128 # embedding dimension
    config['network']['nhead'] = 8
    config['network']['nhid'] = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
    config['network']['nlayers'] = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    config['network']['predecoder_size'] = output_tanh

    config['train'] = dict()
    # normal training
    config['train']['learning_rate'] = 0.000005
    config['train']['batch_size'] = 20
    config['train']['weight_decay'] = 0.

    # general
    config['train']['dropout'] = 0.2
    config['train']['cuda'] = True
    
elif model_type == 'LSTM':

    config['network'] = dict()
    config['network']['input_size'] = input_size
    config['network']['output_size'] = output_size
    config['network']['hidden_size'] = 100
    config['network']['n_layers'] = 2

    config['train'] = dict()
    # normal training
    config['train']['learning_rate'] = 0.0001
    config['train']['batch_size'] = 20
    config['train']['weight_decay'] = 0.
    config['train']['validation_batch_size'] = 1

    # general
    config['train']['dropout'] = 0.4
    config['train']['cuda'] = True
    
else:
    raise ValueError('Wrong model name')



# Setting Cuda

cuda = config['train']['cuda']



# Setting output filname

file_net = model_type + '_' + measure_level + '_' + measure_mode + '_' + output_mode
if model_type == 'Transformer':
    file_net += '_' + str(config['network']['emsize'])
    if config['network']['predecoder_size'] > 0:
        file_net += '_tanh' + str(config['network']['predecoder_size'])
elif model_type == 'LSTM':
    file_net += '_' + str(config['network']['n_layers']) + 'x' + str(config['network']['hidden_size'])
if continent_knockout != '':
    file_net += '_' + continent_knockout
file_net += '.pt'


# Setting max networks and iterations

max_networks = 20
n_iter = 300000


# HandlingNets object

hn = HandlingNets(config, model_type, measure_level, measure_mode, output_mode, countries, train_countries,
                  test_countries, best_loss_ratio=1.5)

if hn.load_saved_nets(file_net):
    nnet = len(hn.nets_min_test_loss)
else:
    nnet = 0

rnn_input_size = config['network']['input_size']
batch_size = config['train']['batch_size']
cuda = config['train']['cuda']



#Main Loop

while True:
    
    min_test_loss = 1.e6
    
    loss = 0.0
    train_loss_seq = []
    test_loss_seq = []

    if model_type == 'Transformer':
        model = TransformerModel(config)
    elif model_type == 'LSTM':
        model = LSTMModel(config)
    if cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                             lr=config['train']['learning_rate'],
                             weight_decay=config['train']['weight_decay'])
    criterion = torch.nn.MSELoss()
    
    optimizer.zero_grad()
        
    for it in range(n_iter):
        model.train()
        country = random.choice(train_countries)

        inp, target = get_data_tensor(data, country, measure_mode, output_mode=output_mode, cuda=cuda)
        
        out_nn, _ = get_net_output(inp, model_type, model, cuda)

        temp_loss = criterion(out_nn, target)
        loss += temp_loss
            
        if (it + 1) % batch_size == 0:
            loss.backward()
            optimizer.step()

            train_loss_seq.append(loss.item()/batch_size)
                
            # Test
            model.eval()
            test_loss = 0.0
            for c in test_countries:
                inp, target = get_data_tensor(data, c, measure_mode, output_mode=output_mode, cuda=cuda)

                out_nn, _ = get_net_output(inp, model_type, model, cuda)

                test_loss += criterion(out_nn, target)
            test_loss_seq.append(test_loss.item()/len(test_countries))
                
            if test_loss_seq[-1] < min_test_loss:
                min_test_loss = test_loss_seq[-1]
                best_state_dict = copy.deepcopy(model.state_dict())
                
            if nnet == 0:
                nets_min_test_loss = -1.
            else:
                nets_min_test_loss = min(hn.nets_min_test_loss)
            print('Best Model - best test loss:{:.4f} (present loss:{:.4f} - it:{}) - Nets found:{} ({:.4f})'.
                      format(min_test_loss, test_loss_seq[-1], it, nnet, nets_min_test_loss))

            loss = 0.0
            optimizer.zero_grad()
    
    elig = hn.set_net(min_test_loss, best_state_dict, train_loss_seq, test_loss_seq)
    if elig:
        hn.save_nets(save_file=file_net)
        nnet = len(hn.nets_min_test_loss)
    

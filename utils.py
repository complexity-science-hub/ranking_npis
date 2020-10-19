import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
import os



def get_data_tensor(data, country, measure_mode, output_mode, cuda):
    if measure_mode == 'delta':
        xx = data[country]['l2_delta'][:-1]
        xx[0] = data[country]['l2_theta'][0]
        raise ValueError('NO DELTA')
    elif measure_mode == 'theta':
        xx = data[country]['l2_theta'][:-1]
    else:
        raise ValueError('measure_mode: \'delta\' or \'theta\'')
    
    inp = np.zeros((xx.shape[0], xx.shape[1]+1)).astype(float)
    inp[:,:xx.shape[1]] = xx
    inp[:,-1] = data[country]['r'][:-1]

    if output_mode == 'r':
        target = data[country]['r'][1:]
    elif output_mode == 'dr':
        target = data[country]['r'][1:] - data[country]['r'][:-1]
    elif output_mode == 'drrel':
        target = (data[country]['r'][1:] - data[country]['r'][:-1]) / data[country]['r'][:-1]
    
    inp = torch.FloatTensor(inp)
    target = torch.FloatTensor(target)
    
    inp = inp.view(inp.shape[0],1,inp.shape[1])
    target = target.view(target.shape[0],1,-1)
    
    if cuda:
        inp = inp.cuda()
        target = target.cuda()
    
    return inp, target




class HandlingNets(object):
    def __init__(self, config=None, model_type=None, measure_level=None, measure_mode=None, output_mode=None,
                 countries=None, train_countries=None, test_countries=None, best_loss_ratio=1.5,
                 max_nets=20):
        self.config = config
        self.nets_min_test_loss = []
        self.nets_best_state_dict = []
        self.nets_train_loss = []
        self.nets_test_loss = []
        self.countries = countries
        self.test_countries = test_countries
        self.train_countries = train_countries
        self.output_mode = output_mode
        self.measure_mode = measure_mode
        self.measure_level = measure_level
        self.model_type = model_type
        self.file_net = None
        self.best_loss_ratio = best_loss_ratio
        self.max_nets = max_nets
        
    def load_saved_nets(self, file_net=None):
        if file_net is None:
            if self.file_net is None:
                print('Input file name is missing. Nets not loaded')
                raise ValueError()
        else:
            self.file_net = file_net
        if os.path.isfile(self.file_net):
            self.file_net = file_net
            in_dict = torch.load(file_net)
            self.config = in_dict['config']
            self.countries = in_dict['countries']
            self.train_countries = in_dict['train_countries']
            self.test_countries = in_dict['test_countries']
            self.measure_level = in_dict['measure_level']
            self.measure_mode = in_dict['measure_mode']
            self.model_type = in_dict['model_type']
            self.output_mode = in_dict['output_mode']
            self.best_loss_ratio = in_dict['best_loss_ratio']
            self.nets_min_test_loss = in_dict['nets_min_test_loss']
            self.nets_best_state_dict = in_dict['nets_best_state_dict']
            self.nets_train_loss = in_dict['nets_train_loss']
            self.nets_test_loss = in_dict['nets_test_loss']
            return True
        else:
            print('File does not exist')
            return False
            
    def set_net(self, min_test_loss, best_state_dict, train_loss, test_loss):
        if len(self.nets_min_test_loss) > 0:
            eligible = False
            if min_test_loss < min(self.nets_min_test_loss) * self.best_loss_ratio:
                eligible = True
        else:
            eligible = True
        insertable = False
        if eligible:
            if len(self.nets_min_test_loss) == self.max_nets:
                if min_test_loss < max(self.nets_min_test_loss):
                    ind = np.where(min_test_loss < np.array(self.nets_min_test_loss))[0]
                    ind2del = ind[np.argmax(np.array(self.nets_min_test_loss)[ind])]
                    del self.nets_min_test_loss[ind2del]
                    insertable = True
            else:
                insertable = True
            if insertable:
                self.nets_min_test_loss.append(min_test_loss)
                self.nets_best_state_dict.append(best_state_dict)
                self.nets_train_loss.append(train_loss)
                self.nets_test_loss.append(test_loss)
        # verify new min_test_loss list
        ind = []
        for k, mtl in enumerate(self.nets_min_test_loss):
            if mtl > min(self.nets_min_test_loss) * self.best_loss_ratio:
                ind.append(k)
        for i in sorted(ind, reverse=True):
            del self.nets_min_test_loss[i]
            del self.nets_best_state_dict[i]
            del self.nets_train_loss[i]
            del self.nets_test_loss[i]
        return eligible
            
    def get_dict(self):
        out_dict = {}
        out_dict['config'] = self.config
        out_dict['countries'] = self.countries
        out_dict['train_countries'] = self.train_countries
        out_dict['test_countries'] = self.test_countries
        out_dict['model_type'] = self.model_type
        out_dict['measure_level'] = self.measure_level
        out_dict['measure_mode'] = self.measure_mode
        out_dict['output_mode'] = self.output_mode
        out_dict['best_loss_ratio'] = self.best_loss_ratio
        out_dict['nets_min_test_loss'] = self.nets_min_test_loss
        out_dict['nets_best_state_dict'] = self.nets_best_state_dict
        out_dict['nets_train_loss'] = self.nets_train_loss
        out_dict['nets_test_loss'] = self.nets_test_loss
        return out_dict
    
    def save_nets(self, save_file=None):
        if save_file is None:
            if self.file_net is None:
                print('Output file name is missing. Nets not saved')
                raise ValueError()
        else:
            self.file_net = save_file
        torch.save(self.get_dict(), self.file_net)



def out2r(out, r, output_mode):
    if output_mode == 'r':
        pass
    elif output_mode == 'dr':
        out = out + r
    elif output_mode == 'drrel':
        out = r * (out + 1.)
    return out


def ma(x, wlen, mode='center'):
    x = np.array(x).astype(float)
    out = np.zeros_like(x)
    if mode == 'center':
        fr0 = -int(np.floor(wlen/2))
        to0 = int(np.floor(wlen/2) + 1)
    elif mode == 'left':
        fr0 = -wlen + 1
        to0 = 1
    elif mode == 'right':
        fr0 = 0
        to0 = wlen
    else:
        raise ValueError
    
    for k in range(len(x)):
        fr = k + fr0
        to = k + to0
        if fr < 0:
            fr = 0
        if to > len(x):
            to = len(x)
        out[k] = x[fr:to].mean()
        
    return out



def get_net_output(inp, model_type, model, cuda, hidden=None):
    if model_type == 'Transformer':
        return model(inp), None
    elif model_type == 'LSTM':
        if hidden is None:
            hidden = model.init_hidden()
        if cuda:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        return model(inp, hidden)
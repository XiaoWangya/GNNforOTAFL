import torch, random,io, os
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
from copy import deepcopy

class Net(nn.Module):
    def __init__(self,input_shape,output_shape, hidden = 64):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(input_shape, hidden).double().cuda()
        self.BN = nn.BatchNorm1d(hidden).double().cuda()
        self.layer2 = nn.Linear(hidden, output_shape).double().cuda()
        self.BN2 = nn.BatchNorm1d(output_shape).double().cuda()
    def forward(self,x):
        x = self.layer1(x)
        x = self.BN(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = self.BN2(x)
        x = nn.Sigmoid()(x)
        return x

class IRS_block(nn.Module):
    def __init__(self,number_nodes,input_shape, output_shape,hidden = 128, operation = 'mean'):
        super(IRS_block,self).__init__()
        self.input_shape = input_shape                                                                                                  
        self.output_shape = output_shape
        self.number_nodes = number_nodes
        self.nets_ue = Net(input_shape,output_shape, hidden = hidden)
        self.nets_bs = Net(input_shape,output_shape, hidden = hidden)
        self.combine_net = Net(output_shape*2 + self.input_shape, output_shape, hidden = hidden)
        self.operation = operation

    def forward(self,x):
        """
            x: batch \times number_nodes \times input_feature_size
        """
        batch, number_channel, input_feature_size = x.shape
        output = torch.zeros(batch, number_channel-2, self.output_shape).double().cuda()
        for ii in range(number_channel-2):
            output[:,ii,:] = self.nets_ue(x[:,ii,:])
        # aggregation operation
        output_aggregate = output.mean(dim = 1)
        # combination operation
        result = self.combine_net(torch.cat((output_aggregate, self.nets_bs(x[:,-1,:]), x[:,-2,:]), 1))
        return result

class BS_block(nn.Module):
    def __init__(self,number_nodes,input_shape, output_shape,hidden = 128, operation = 'mean'):
        super(BS_block,self).__init__()
        self.input_shape = input_shape                                                                                                  
        self.output_shape = output_shape
        self.number_nodes = number_nodes
        self.nets_ue = Net(input_shape,output_shape, hidden = hidden)
        self.nets_irs = Net(input_shape,output_shape, hidden = hidden)
        self.combine_net = Net(output_shape*2 + self.input_shape, output_shape, hidden = hidden)
        self.operation = operation

    def forward(self,x):
        """
            x: batch \times number_nodes \times input_feature_size
        """
        batch, number_channel, input_feature_size = x.shape
        output = torch.zeros(batch, number_channel-2, self.output_shape).double().cuda()
        for ii in range(number_channel-2):
            output[:,ii,:] = self.nets_ue(x[:,ii,:])
        # aggregation operation
        output_aggregate = output.mean(dim = 1)
        # combination operation
        result = self.combine_net(torch.cat((output_aggregate, self.nets_irs(x[:,-2,:]), x[:,-1,:]), 1))
        return result

class User_block(nn.Module):
    def __init__(self, number_nodes, input_shape, output_shape, order, hidden = 128, operation = 'mean'):
        super(User_block,self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.number_nodes = number_nodes
        self.order = order
        self.user_nets = Net(input_shape,output_shape, hidden = hidden)
        self.irs_net = Net(input_shape,output_shape, hidden = hidden)
        self.bs_net = Net(input_shape,output_shape, hidden = hidden)
        self.combine_net = Net(output_shape*3+input_shape, output_shape, hidden = hidden)
        self.operation = operation

    def forward(self,x):
        """
            x: batch \times number_nodes \times input_feature_size
        """
        batch, number_channel, input_feature_size = x.shape # number_channel = number_nodes + 2
        output_user = torch.zeros(batch, self.number_nodes-2, self.output_shape).double().cuda()
        for ii in range(self.number_nodes-2):
            if ii < self.order:
                output_user[:,ii,:] = self.user_nets(x[:,ii,:])
            else:
                output_user[:,ii,:] = self.user_nets(x[:,ii+1,:])
        # aggregation operation
        if self.operation == 'mean':
            output_aggregate = output_user.mean(dim = 1)
        if self.operation == 'max':
            output_aggregate = output_user.max(dim = 1)[0]
        # combination operation
        result = torch.cat((output_aggregate, self.irs_net(x[:,-2,:]), self.bs_net(x[:,-1,:]), x[:,self.order,:]), 1)
        result = self.combine_net(result)
        return result

class Initial_layer(nn.Module):
    def __init__(self, number_nodes, input_shape, output_shape, hidden = 128, operation = 'mean'):
        super(Initial_layer, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.number_nodes = number_nodes
        self.nets = Net(input_shape, output_shape, hidden = hidden)
        self.combine_net_bs = Net(output_shape,output_shape, hidden = hidden)
        self.combine_net_irs = Net(output_shape,output_shape, hidden = hidden)
        self.operation = operation

    def forward(self,x):
        """
            x: batch \times number_nodes \times input_feature_size
        """
        batch, _, input_feature_size = x.shape
        output = torch.zeros(batch, self.number_nodes+2, self.output_shape).double().cuda()
        for ii in range(self.number_nodes):
            output[:,ii,:] = self.nets(x[:,:,ii])
        result = output[:,:self.number_nodes,:].mean(dim = 1)
        output[:,-2,:] = self.combine_net_irs(result)
        output[:,-1,:] = self.combine_net_bs(result)
        return output

        
        
class Graph_layer(nn.Module):
    def __init__(self,size_input, size_output, number_nodes):
        super(Graph_layer, self).__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.number_nodes = number_nodes
        self.user_nodes = User_block(number_nodes, size_input, size_output, order = 0, operation='max')
        self.irs_node = IRS_block(number_nodes, size_input, size_output, operation='mean')
        self.bs_node = BS_block(number_nodes, size_input, size_output, operation='mean')
    
    def forward(self, x):
        batch, number_nodes_with_irs, size_input_feature = x.shape# number_nodes_with_irs_bs = number_nodes+2
        output = torch.zeros(batch, number_nodes_with_irs, self.size_output).double().cuda()
        for ii in range(number_nodes_with_irs-2):
            self.user_nodes.order = ii
            output[:,ii,:] = self.user_nodes(x)
        output[:,-2,:] = self.irs_node(x)
        output[:,-1,:] = self.bs_node(x)
        return output

class Graph_net(nn.Module):
    def __init__(self, size_input, number_node, number_irs_elements, number_layer = 2, size_hidden = [64,256,128]):
        super(Graph_net, self).__init__()
        #check available input
        if len(size_hidden)!= number_layer+1:
            print('Enter right number of layers as well as the size of hidden layers')
        self.initial_layer = Initial_layer(number_node, size_input, size_hidden[0])
        self.hidden_layers = [Graph_layer(size_hidden[i], size_hidden[i+1], number_node) for i in range(number_layer)]
        self.output_layers_node = nn.Linear(size_hidden[number_layer],1).double().cuda()
        self.output_layers_irs = nn.Linear(size_hidden[number_layer], number_irs_elements).double().cuda()
        self.output_layers_eta = nn.Linear(size_hidden[number_layer], 1).double().cuda()
        self.number_layer = number_layer
        self.number_node = number_node
        self.number_irs_elements = number_irs_elements

    def forward(self, x):
        batch, _,_ = x.shape
        x = self.initial_layer(x)
        for _ in range(self.number_layer):
            # y = torch.ones(x.shape[0], x.shape[1], self.hidden_layers[_].user_nodes.output_shape).cuda().double()
            x = self.hidden_layers[_](x)
            # x = y.clone()
        # xx = x.clone().flatten(start_dim = 1)
        result = torch.zeros(batch,self.number_node + self.number_irs_elements + 1).cuda().double()
        for _ in range(self.number_node):
            result[:,_] = self.output_layers_node(x[:,_,:]).squeeze()
        result[:,-self.number_irs_elements-1:-1] = self.output_layers_irs(x[:,-2,:])
        result[:,-1] = self.output_layers_eta(x[:,-1,:]).squeeze()
        return torch.sigmoid(result)
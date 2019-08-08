#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

"""
CS224N 2018-19: Homework 5
"""


### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, hidden_size):
        super(Highway, self).__init__()
        self.hidden_size = hidden_size
        self.h_projection = nn.Linear(hidden_size, hidden_size, bias=True)
        self.h_gate = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, conv_out: torch.Tensor):
        x_proj = F.relu(self.h_projection(conv_out))
        x_gate = torch.sigmoid(self.h_gate(conv_out))
        x_highway = x_gate*x_proj +(1-x_gate)*conv_out
        return x_highway
### END YOUR CODE 


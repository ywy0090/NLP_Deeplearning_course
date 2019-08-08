#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self,sent_len, embed_size, k, f):
        super(CNN, self).__init__()
        # window size
        self.k = k
        self.f = f
        self.max_word_len = 21
        self.conv = nn.Conv1d(in_channels=embed_size, out_channels=f, kernel_size=k, stride=1)
        self.maxpool = nn.MaxPool1d(self.max_word_len-k+1)

    def forward(self, x_reshape):
        x_conv = self.conv(x_reshape)
        x_maxpool = self.maxpool(F.relu(x_conv))
        return x_maxpool

### END YOUR CODE


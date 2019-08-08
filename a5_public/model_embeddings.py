#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.embed_layer = nn.Embedding(len(vocab.char2id), embed_size)

        ### END YOUR CODE

    def forward(self, input: torch.Tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j

        sent_len, batch_size, max_word_len = input.size()
        # define layers
        cnn_layer = CNN(sent_len, embed_size=self.embed_size, k=5, f=self.embed_size)
        highway_layer = Highway(self.embed_size)
        dropout_layer = nn.Dropout(0.3)
        # forward tensor
        # input = input.view(batch_size, sent_len*max_word_len)
        embeded = self.embed_layer(input)
        embeded = embeded.view(sent_len*batch_size, self.embed_size, max_word_len)
        x_conv = cnn_layer(embeded)
        x_conv = torch.squeeze(x_conv, 2)
        x_highway = highway_layer(x_conv)
        x_emd_word = dropout_layer(x_highway)
        x_emd_word = x_emd_word.view(sent_len, batch_size, self.embed_size)
        return x_emd_word
        ### END YOUR CODE


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size, bias=True)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id), bias=True)
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        ### END YOUR CODE
    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        scores = []
        embed_input = self.decoderCharEmb(input)
        output, dec_hidden = self.charDecoder(embed_input, dec_hidden)
        for s in torch.split(output, 1, dim=0):
            s = torch.squeeze(s, 0)
            s = self.char_output_projection(s)
            scores.append(s)
        scores = torch.stack(scores)
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, dec_hidden = self.forward(char_sequence, dec_hidden)
        loss = nn.CrossEntropyLoss()
        loss_word_dec = 0
        for i, s in enumerate(torch.split(scores,1, dim=0)):
            p = F.softmax(s)
            p = torch.squeeze(p, 0)
            char_tensor = char_sequence[i]
            loss_char_dec = loss(p, char_tensor)
            loss_word_dec += loss_char_dec
        return loss_word_dec
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        w_id, batch_size, hidden_size = initialStates[0].size()
        decodedWords = []
        tmp_words = []
        current_batch_chars = [self.target_vocab.char2id['{'] for i in range(batch_size)]
        for i in range(max_length):
            embed_input = self.decoderCharEmb(torch.tensor(current_batch_chars, device=device))
            embed_input = torch.unsqueeze(embed_input, 0)
            output, initialStates = self.charDecoder(embed_input, initialStates)
            output = torch.squeeze(output, 0)
            output = self.char_output_projection(output)
            p = F.softmax(output)
            idx_list = torch.argmax(p, dim=1).tolist()
            current_batch_chars = [self.target_vocab.id2char[idx] for idx in idx_list]
            tmp_words.append(current_batch_chars)
            current_batch_chars = [self.target_vocab.char2id[wd] for wd in current_batch_chars]
        # truncate string
        for i in range(batch_size):
            wd = [w[0] for w in tmp_words]
            try:
                end = wd.index('}')
            except ValueError:
                end = len(wd)
            if end > 0:
                wd = wd[0:end]
            else:
                wd = ""
            decodedWords.append(''.join(wd))
        return decodedWords
        ### END YOUR CODE


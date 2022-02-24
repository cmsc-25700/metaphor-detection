"""
Class containing LSTM Model 
"""
import torch.nn as nn
import torch
import torch.nn.functional as torch_function
from allennlp.nn.util import sort_batch_by_length, masked_softmax
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class LSTM_Classifier(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_layers,
                dropout_rate, num_classes = 2, bidirectional = False):

        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers,
                            dropout_rate, batch_first = True)
        # If bidirectional, expand hidden size
        
        if bidirectional:
            direction = 2
        else:
            direction = 1
        
        self.attention_layer = nn.Linear(hidden_size * direction, 1)
        self.output_layer = nn.Linear(hidden_size * direction, num_classes)
        # for now, using a single dropout rate for all layers.
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = torch_function.log_softmax()
    
    def forward(self, inputs, lengths):
        '''
        Forward function for LSTM Model.
        Inputs:
            - Input: batch of inputs. (batch_size, sequence_length)
            -
        '''
        #dropout to embedding
        embedding = self.dropout(inputs)
        #sort the input by decreasing order of length.
        input_sorted, lengths_sorted, unsorted_indices_input, _ = sort_batch_by_length(embedding, lengths)
        #pack input
        packed_input = pack_padded_sequence(input_sorted, lengths_sorted.data.tolist(), batch_first = True)
        packed_output, _ = self.LSTM(packed_input)
        #unpack output
        unpacked_output = pad_packed_sequence(packed_output, batch_first = True)
        #return the initial ordering 
        output = unpacked_output[unsorted_indices_input]

        ##attention
        attentions = self.attention_layer(unpacked_output).squeeze(dim = -1)
        mask = (attentions != 0)
        #if using GPU
        if inputs.is_cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)
        #softmax for attention
        softmax_attention = masked_softmax(attentions, mask).unsqueeze(dim = 1)
        #matrix product of attention and the LSTM layer output
        input_encoded = torch.bmm(softmax_attention, output).squeeze(dim = 1)

        ##linear layer
        input_encoded = self.dropout(input_encoded)
        output_final = self.output(input_encoded)
        output_final = torch_function.log_softmax(output_final, dim = -1)

        return output_final

        
        

import random

import torch as torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.models as models
from .baseRNN import BaseRNN

from dataset.collate_fn import calc_im_seq_len

class EncoderCRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).
    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, vocab_size, hidden_size=128,
                 backend='resnet18',
                 input_dropout_p=0, dropout_p=0,
                 n_layers=2, bidirectional=True, rnn_cell='gru', variable_lengths=True):
        super(EncoderCRNN, self).__init__(vocab_size, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.fully_conv = True
        self.variable_lengths = variable_lengths
        self.feature_extractor = getattr(models, backend)(pretrained=True)
        self.cnn = nn.Sequential(
            self.feature_extractor.conv1,
            self.feature_extractor.bn1,
            self.feature_extractor.relu,
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            self.feature_extractor.layer1,
            self.feature_extractor.layer2,
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            self.feature_extractor.layer3,
            # self.feature_extractor.layer4,
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        )
        self.rnn_num_layers = n_layers
        self.rnn_hidden_size = hidden_size
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1

        self.rnn = self.rnn_cell(self.get_block_size(self.cnn), hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        # check cnn output lenght compatible to calculations
        for i in range(20):
            length = random.randint(50,300)
            height1, width1 = self._get_output_ratio(length)

            width2 = calc_im_seq_len(length)
            if (width2 != width1):
                raise Exception("error, orig width is: {} ; width through network is: {} ; calculated width is: {} .".format(
                    length, width1, width2
                ))
            if height1 != 1:
                raise Exception("hight after network should be one, but is: {}".format(height1))

    def squeeze_hidden(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = h.view(self.rnn_num_layers, self.n_directions, -1, self.rnn_hidden_size)

        h = h[-1].squeeze(0)
        if self.n_directions == 2:
            h = h[0] + h[1]
        h = h.unsqueeze(0)
        return h

    def get_block_size(self, layer):
        #print('block size is:')
        #print(layer[-2][-1].bn2.weight.size()[0])
        return layer[-2][-1].bn2.weight.size()[0]


    def forward(self, x, seq_lens, mask):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        features = self.cnn(x)
        features = self.features_to_sequence(features)

        features = self.input_dropout(features)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(features, seq_lens, batch_first=True)
        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)

        output, hidden = self.rnn(embedded, hidden)

        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        max_length = output.size(1)

        return output, hidden, max_length

    def features_to_sequence(self, features):
        b, c, h, w = features.size()
        assert h == 1, "the height of out must be 1"
        #if not self.fully_conv:
        #    features = features.permute(0, 3, 2, 1)#b,w,h,c
        #    features = self.proj(features)
        #else:
        #    features = features.permute(3, 0, 2, 1) #w,b,h,c
        features = features.permute(0, 3, 2, 1)  # w,b,h,c
        features = features.squeeze(2)#b,w,c
        #print('features_to_sequence - features size:')
        #print(features.shape)
        return features

    def init_hidden(self, batch_size, gpu=False):
        h0 = Variable(torch.zeros( self.rnn_num_layers * 2,
                                   batch_size,
                                   self.rnn_hidden_size))
        if gpu:
            h0 = h0.cuda()
        return h0

    def _get_output_ratio(self, length):
        bs = 1
        input = Variable(torch.rand((bs, 3, 64, length)))
        output_feat = self.cnn(input)
        seq_input = self.features_to_sequence(output_feat)  # w,b,c
        return output_feat.data.size(2), output_feat.data.size(3)
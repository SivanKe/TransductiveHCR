import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.models as models
import string
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset.collate_fn import calc_im_seq_len
import random

class CrnnAttention(nn.Module):
    def __init__(self,
                 lexicon=None,
                 backend='resnet18',
                 rnn_hidden_size=128,
                 trg_emb_dim=128,
                 trg_hidden_dim=128,
                 dropout=0.5,
                 rnn_num_layers=2,
                 rnn_dropout=False,
                 seq_proj=[0, 0]):
        super().__init__()
        self.lexicon = lexicon

        self.num_classes = len(self.lexicon)

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
            #self.feature_extractor.layer4,
            nn.MaxPool2d(kernel_size=(3,1), stride=(2,1), padding=(1,0))
        )

        self.fully_conv = True#seq_proj[0] == 0
        if not self.fully_conv:
            self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn = nn.GRU(self.get_block_size(self.cnn),
                          rnn_hidden_size, rnn_num_layers,
                          batch_first=False,
                          dropout=rnn_dropout, bidirectional=True)
        self.decoder = LSTMAttentionDot(
            trg_emb_dim,
            trg_hidden_dim,
            batch_first=False
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )

        DecoderRNN




        trg_vocab_size = self.num_classes
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

        self.init_weights()

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


    def _get_output_ratio(self, length):
        bs = 1
        input = Variable(torch.rand((bs, 3, 64, length)))
        output_feat = self.cnn(input)
        seq_input = self.features_to_sequence(output_feat)  # w,b,c
        return output_feat.data.size(2), output_feat.data.size(3)

    def forward(self, , decode=False):
        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)
        features = self.cnn(x)
        #print('in network forward, features size is:')
        #print(features.shape)
        features = self.features_to_sequence(features) #b,w,c
        #print('in network forward, features after features_to_sequence size is:')
        #print(features.shape)
        # total_length = features.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(features, seq_lens,
                                            batch_first=False)
        packed_output, _ = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=False)
        # NOTICE: SHOULD I DO THIS OR NOT????
        # output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs

                                        #total_length=total_length)
        #print('in network forward, rnn output size is:')
        #print(output.shape)
        seq = self.linear(output)
        if not self.training:
            seq = self.softmax(seq)
            if decode:
                orig_seq = seq

                seq = self.decode(seq, self.lexicon)
                return seq, orig_seq
        return seq

    def init_hidden(self, batch_size, gpu=False):
        h0 = Variable(torch.zeros( self.rnn_num_layers * 2,
                                   batch_size,
                                   self.rnn_hidden_size))
        if gpu:
            h0 = h0.cuda()
        return h0

    def features_to_sequence(self, features):
        b, c, h, w = features.size()
        assert h == 1, "the height of out must be 1"
        if not self.fully_conv:
            features = features.permute(0, 3, 2, 1)#b,w,h,c
            features = self.proj(features)
        else:
            features = features.permute(3, 0, 2, 1) #w,b,h,c
        features = features.squeeze(2)#w,b,c
        #print('features_to_sequence - features size:')
        #print(features.shape)
        return features

    def get_block_size(self, layer):
        #print('block size is:')
        #print(layer[-2][-1].bn2.weight.size()[0])
        return layer[-2][-1].bn2.weight.size()[0]

    @staticmethod
    def pred_to_string(pred, lexicon):
        seq = []
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label)
        return CRNN.label_to_string(seq, lexicon)

    @staticmethod
    def label_to_string(label, lexicon, allow_repitions=False):
        out = []
        for i in range(len(label)):
            if len(out) == 0:
                if label[i] != -1:
                    out.append(label[i])
            else:
                if label[i] != -1 and ((allow_repitions) or (label[i] != label[i - 1])):
                    out.append(label[i])
        out = ''.join(lexicon[i] for i in out if i in lexicon)
        return out

    @staticmethod
    def decode(pred, lexicon, num_decode=None):
        pred = pred.cpu().data.numpy().transpose((1,0,2))
        num_preds = pred.shape[0]
        seq = []
        if num_decode is None:
            num_decode = pred.shape[0]
        for i in range(min(num_decode,num_preds)):
            seq.append(CRNN.pred_to_string(pred[i], lexicon))
        return seq

    @staticmethod
    def decode_flatten(labels, label_lens, lexicon, num_decode=None):
        labels = labels.cpu().data.numpy()
        label_lens = label_lens.cpu().data.numpy()
        if labels.shape[0] != np.sum(label_lens):
            raise Exception("error - size of all labels flatten must equal sum of all labels lens")
        seq = []
        if num_decode is None:
            num_decode = len(label_lens)
        cur_position = 0
        for i in range(min(num_decode, len(label_lens))):
            #print(CRNN.label_to_string(labels[cur_position:cur_position+label_lens[i]], lexicon))
            seq.append(CRNN.label_to_string(labels[cur_position:cur_position+label_lens[i]], lexicon, allow_repitions=True))
            cur_position = cur_position+label_lens[i]
        return seq

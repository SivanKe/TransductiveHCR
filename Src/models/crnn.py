import torch
import torch.nn as nn
from torch.autograd import Variable
import operator
import torchvision.models as models
import models.load_models as my_models
import string
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset.collate_fn import calc_im_seq_len
import random
import ctcdecode
import copy
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class CRNN(nn.Module):
    def __init__(self,
                 lexicon=None,
                 backend='resnet18',
                 base_model_dir=None,
                 rnn_hidden_size=128,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 seq_proj=[0, 0],
                 do_beam_search=False,
                 dropout_conv=False,
                 dropout_rnn=False,
                 dropout_output=False,
                 cuda=True,
                 do_ema=False,
                 ada_after_rnn=False,
                 ada_before_rnn=False):
        super().__init__()
        self.lexicon = lexicon
        print(lexicon)
        self.do_beam_search = do_beam_search
        self.num_classes = len(self.lexicon)
        self.ada_after_rnn = ada_after_rnn
        self.ada_before_rnn = ada_before_rnn

        self.feature_extractor = getattr(my_models, backend)(pretrained=True, model_dir=base_model_dir)
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

        self.dropout_conv = dropout_conv
        self.dropout_rnn = dropout_rnn
        self.dropout_output = dropout_output
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.dropout1d = nn.Dropout(p=0.5)

        self.fully_conv = True#seq_proj[0] == 0
        if not self.fully_conv:
            self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        if self.dropout_rnn:
            self.rnn = nn.GRU(self.get_block_size(self.cnn),
                              rnn_hidden_size, rnn_num_layers,
                              batch_first=False, bidirectional=True, dropout=0.5)
        else:
            self.rnn = nn.GRU(self.get_block_size(self.cnn),
                              rnn_hidden_size, rnn_num_layers,
                              batch_first=False, bidirectional=True, dropout=0.5)

        self.linear = nn.Linear(rnn_hidden_size * 2, self.num_classes + 1)
        self.softmax = nn.Softmax(dim=2)

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

        if self.do_beam_search:
            sorted_letters = [item[1] for item in sorted(lexicon.items(), key=operator.itemgetter(0))]
            sorted_keys = [item[0] for item in sorted(lexicon.items(), key=operator.itemgetter(0))]
            #print(sorted_keys)
            #print(sorted_letters)
            self.label_str = ['_'] + sorted_letters

            #print(label_str)

            print('vocab size is: {}'.format(len(self.label_str)))
            self.beam_decode = ctcdecode.CTCBeamDecoder(self.label_str, blank_id=0, beam_width=20)

        if cuda:
            self.cuda()
        if do_ema:
            self.avg_param = self.copy_model_params()  # initialize
            if cuda:
                for i in range(len(self.avg_param)):
                    self.avg_param[i].cuda()

        if ada_after_rnn:
            self.domain_classifier_rnn = nn.Sequential()
            self.domain_classifier_rnn.add_module('d_fc1', nn.Linear(rnn_hidden_size*2, 100))
            self.domain_classifier_rnn.add_module('d_bn1', nn.BatchNorm1d(100))
            self.domain_classifier_rnn.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier_rnn.add_module('d_fc2', nn.Linear(100, 2))
            self.domain_classifier_rnn.add_module('d_softmax', nn.LogSoftmax())

        if ada_before_rnn:
            self.domain_classifier_cnn = nn.Sequential()
            self.domain_classifier_cnn.add_module('d_fc1', nn.Linear(self.get_block_size(self.cnn), 100))
            self.domain_classifier_cnn.add_module('d_bn1', nn.BatchNorm1d(100))
            self.domain_classifier_cnn.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier_cnn.add_module('d_fc2', nn.Linear(100, 2))
            self.domain_classifier_cnn.add_module('d_softmax', nn.LogSoftmax())



    def _get_output_ratio(self, length):
        bs = 1
        input = Variable(torch.rand((bs, 3, 64, length)))
        output_feat = self.cnn(input)
        seq_input = self.features_to_sequence(output_feat)  # w,b,c
        return output_feat.data.size(2), output_feat.data.size(3)

    def vat_forward(self, x, seq_lens):
        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)
        features = self.cnn(x)
        if self.dropout_conv:
            features = self.dropout2d(features)
        # print('in network forward, features size is:')
        # print(features.shape)
        features = self.features_to_sequence(features)  # b,w,c
        # print('in network forward, features after features_to_sequence size is:')
        # print(features.shape)
        # total_length = features.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(features, seq_lens,
                                            batch_first=False)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=False)
        if self.dropout_output:
            output = self.dropout1d(output)
            # total_length=total_length)
        # print('in network forward, rnn output size is:')
        # print(output.shape)
        seq = self.linear(output)
        return seq

    def vat_forward_cnn(self, x):
        features = self.cnn(x)
        if self.dropout_conv:
            features = self.dropout2d(features)
        # print('in network forward, features size is:')
        # print(features.shape)
        features = self.features_to_sequence(features)  # b,w,c
        return features

    def vat_forward_rnn(self, x_size, features, seq_lens):
        hidden = self.init_hidden(x_size, next(self.parameters()).is_cuda)
        packed_input = pack_padded_sequence(features, seq_lens,
                                            batch_first=False)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=False)
        if self.dropout_output:
            output = self.dropout1d(output)
            # total_length=total_length)
        # print('in network forward, rnn output size is:')
        # print(output.shape)
        seq = self.linear(output)
        return seq

    def forward(self, x, seq_lens, decode=False, do_beam_search=False, ada_alpha=None, mask=None):

        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)
        features = self.cnn(x)
        if self.dropout_conv:
            features = self.dropout2d(features)
        #print('in network forward, features size is:')
        #print(features.shape)
        if self.ada_before_rnn and (not decode):
            ada_features_cnn = self.choose_features_to_ada(features, mask)
            reverse_ada_features_cnn = ReverseLayerF.apply(ada_features_cnn, ada_alpha)
            ada_output_cnn = self.domain_classifier_cnn(reverse_ada_features_cnn)
        features = self.features_to_sequence(features) #b,w,c
        #print('in network forward, features after features_to_sequence size is:')
        #print(features.shape)
        # total_length = features.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(features, seq_lens,
                                            batch_first=False)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=False)

        if self.ada_after_rnn and (not decode):
            w,b,c = output.size()

            output_ada = torch.masked_select(output.cpu(), mask.byte().cpu()).view(-1, c).cuda()
            reverse_output_ada = ReverseLayerF.apply(output_ada, ada_alpha)
            ada_output_rnn = self.domain_classifier_rnn(reverse_output_ada)

        if self.dropout_output:
            output = self.dropout1d(output)
                                        #total_length=total_length)
        #print('in network forward, rnn output size is:')
        #print(output.shape)
        seq = self.linear(output)
        if not self.training:
            seq = self.softmax(seq)
            if decode:
                orig_seq = seq

                seq = self.decode(seq, self.lexicon, do_beam_search=do_beam_search)
                return seq, orig_seq
        if (self.ada_after_rnn or self.ada_before_rnn) and (not decode):
            if self.ada_after_rnn and self.ada_before_rnn:
                return seq, ada_output_cnn, ada_output_rnn
            elif self.ada_after_rnn:
                return seq, None, ada_output_rnn
            else:
                return seq, ada_output_cnn, None
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
    def choose_features_to_ada(self,features, mask):

        b, c, h, w = features.size()
        assert h == 1, "the height of out must be 1"
        ada_features_cnn = features.squeeze(2)
        ada_features_cnn = ada_features_cnn.permute(0, 2, 1)  # b,w,c
        # select only valid features
        # transpose mask to do (max_len,bs,ch)=>(bs,max_len,ch)
        ada_features_cnn = torch.masked_select(ada_features_cnn.cpu(), np.transpose(mask, (1, 0, 2)).byte().cpu()).view(-1, c).cuda()
        return ada_features_cnn

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


    def decode(self, pred, lexicon, num_decode=None, do_beam_search=False):
        pred = pred.transpose(1, 0)
        num_preds = pred.size(0)
        if do_beam_search:
            beam_results, beam_scores, timesteps, out_seq_len = self.beam_decode.decode(pred)
            seq = []


            for i in range(num_preds):
                seq_len = out_seq_len[i][0]
                res = beam_results[i][0].data.numpy()
                cur_out = ''.join([lexicon[x] for x in res[0:seq_len] if x>0])

                seq.append(cur_out)

            return seq

        else:
            pred = pred.cpu().data.numpy()

            seq = []
            if num_decode is None:
                num_decode = pred.shape[0]
            for i in range(min(num_decode,num_preds)):
                seq.append(CRNN.pred_to_string(pred[i], lexicon))
        return seq


    def decode_flatten(self, labels, label_lens, lexicon, num_decode=None):
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

    def copy_model_params(self):

        flatten = copy.deepcopy(list(p.data for p in self.parameters()))

        return flatten

    def load_params(self, flattened):
        for p, avg_p in zip(self.parameters(), flattened):
            p.data.copy_(avg_p)

    def udate_ema(self):
        param_copy = self.copy_model_params()
        for i in range(len(self.avg_param)):
            self.avg_param[i] = 0.9 * self.avg_param[i] + 0.1 * param_copy[i]

    def start_test(self):
        print('load avg params')
        self.original_param = self.copy_model_params()  # save current params
        self.load_params(self.avg_param)  # load the average

    def end_test(self):
        print('load train params')
        self.load_params(self.original_param)  # restore parameters



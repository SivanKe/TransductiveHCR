import torch
from torch import nn
from models.crnn import CRNN
from models.Attention.Seq2Seq import CrnnAttentionModel
from collections import OrderedDict

def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            print("notice that {} is not loaded".format(str(k)))
            new_dict[k] = v
    target.load_state_dict(new_dict)

def load_model(lexicon, seq_proj=[0, 0], backend='resnet18', base_model_dir=None, snapshot=None, cuda=True,
               do_beam_search=False,
               dropout_conv=False,
               dropout_rnn=False,
               dropout_output=False,
               do_ema=False,
               ada_after_rnn=False,
               ada_before_rnn=False,
               rnn_hidden_size=128
               ):
    net = CRNN(lexicon=lexicon, seq_proj=seq_proj, backend=backend, base_model_dir=base_model_dir, do_beam_search=do_beam_search,
               dropout_conv=dropout_conv,
               dropout_rnn=dropout_rnn,
               dropout_output=dropout_output,
               do_ema=do_ema,
               ada_after_rnn=ada_after_rnn,
               ada_before_rnn=ada_before_rnn,
               rnn_hidden_size=rnn_hidden_size
               )
    #net = nn.DataParallel(net)
    if snapshot is not None:
        print('snapshot is: {}'.format(snapshot))
        #print(torch.load(snapshot))
        load_weights(net, torch.load(snapshot))
    if cuda:
        print('setting network on gpu')
        net = net.cuda()
        print('set network on gpu')
    return net

def load_attn_model(lexicon, seq_proj=[0, 0], backend='resnet18', snapshot=None, cuda=True, dropout=0):
    net = CrnnAttentionModel(lexicon=lexicon, seq_proj=seq_proj, backend=backend, dropout=dropout)
    #net = nn.DataParallel(net)
    if snapshot is not None:
        print('snapshot is: {}'.format(snapshot))
        load_weights(net, torch.load(snapshot))
    if cuda:
        print('setting network on gpu')
        net = net.cuda()
        print('set network on gpu')
    return net

import torch.nn as nn
import torch.nn.functional as F

from .EncoderCrnn import EncoderCRNN
from .DecoderRNN import DecoderRNN
import numpy as np
from dataset.text_data_attn import Consts as AttnConsts

class CrnnAttentionModel(nn.Module):
    def __init__(self,
                 lexicon=None,
                 backend='resnet18',
                 rnn_hidden_size=128,
                 trg_emb_dim=128,
                 trg_hidden_dim=128,
                 dropout=0.5,
                 rnn_num_layers=2,
                 rnn_dropout=False,
                 seq_proj=[0, 0],
                 decode_function = F.log_softmax):
        super(CrnnAttentionModel, self).__init__()
        self.decode_function = decode_function
        self.vocab_size = len(lexicon)
        print('lexicon in model:')
        print(lexicon)
        self.lexicon = lexicon
        self.encoder = EncoderCRNN(self.vocab_size, dropout_p=dropout)
        self.decoder = DecoderRNN(self.vocab_size, hidden_size=trg_hidden_dim,
            sos_id=1, eos_id=2, bidirectional=True)

    def predict(self, ret_dict):
        """ Make prediction given `src_seq` as input.
        Args:
            src_seq (list): list of tokens in source language
        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        lengths = ret_dict['length']

        outputs = np.stack([step.view(-1).data for step in ret_dict['sequence']]).transpose()

        out_predictions = []
        for output, length in zip(outputs, lengths):

            cur_out = ''.join([self.lexicon[output[i]] for i in range(length) if
                               (self.lexicon[output[i]] not in [AttnConsts.BLANK_CHAR, AttnConsts.EOS_CHAR])])
            out_predictions.append(cur_out)
        return out_predictions

    def padded_seq_to_txt(self, padded_seq, lengths):
        out_txts = []

        for output, length in zip(padded_seq, lengths):
            cur_out = ''.join([self.lexicon[output[i]] for i in range(1,length-1)])
            out_txts.append(cur_out)
        return out_txts


    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, x, seq_lens, mask, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden, max_length = self.encoder(x, seq_lens, mask)
        encoder_hidden = self.encoder.squeeze_hidden(encoder_hidden)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              max_encoder_length =max_length)
        return result
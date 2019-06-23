import torch
import numpy as np
from torch.autograd import Variable

def calc_im_seq_len(img_width):
    w1 = np.floor((img_width-1)/2)+1
    w2 = np.floor((w1 - 1) / 2) + 1
    w3 = np.floor((w2 - 1) / 2) + 1
    #w4 = np.floor((w3 - 1) / 2) + 1
    if (w3 <=1).sum() > 0:
        raise Exception("{} images have too small w".format((w5 <=1).sum()))
    return w3

def pad_sequence(sequences, batch_first=False, padding_value=0):
    r"""Pad a list of variable length Variables with zero

    ``pad_sequence`` stacks a list of Variables along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    `B` is batch size. It's equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        from torch.nn.utils.rnn import pad_sequence
        a = Variable(torch.ones(25, 300))
        b = Variable(torch.ones(22, 300))
        c = Variable(torch.ones(15, 300))
        pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Variable of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Variables
            in sequences are same.

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements.

    Returns:
        Variable of size ``T x B x *`` if batch_first is False
        Variable of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    starting_dims, max_len = max_size[:-1], max_size[-1]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences),) + starting_dims + (max_len,)

    out_variable = sequences[0].new(*out_dims).fill_(padding_value)
    for i, variable in enumerate(sequences):
        length = variable.size(2)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
            raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        out_variable[i, :,:,:length] = variable

    return out_variable

def pad_no_order(max_len, sequences, padding_value=0):
    out_dims = (len(sequences), max_len)
    out_variable = sequences[0].new(*out_dims).fill_(padding_value)
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # use index notation to prevent duplicate references to the variable
        out_variable[i, :length] = variable
    return out_variable

def text_collate(batch, do_mask):
    img = list()
    seq = list()
    seq_len = list()
    img_width = list()
    for sample in batch:
        if len(sample["img"].shape) == 2:
            sample["img"] = np.expand_dims(sample["img"], 2)
        img.append(torch.from_numpy(sample["img"].transpose((2, 0, 1))).float())
        seq.append(sample["seq"])
        seq_len.append(sample["seq_len"])
        img_width.append(sample["img_width"])
    img, seq, seq_len, img_width = zip(*sorted(zip(img, seq, seq_len, img_width), key=lambda tpl: tpl[3], reverse=True))
    img = pad_sequence(img, batch_first=True, padding_value=0)
    #print('in collate_fn - image size is:')
    #print(img.shape)
    seq = list(seq)
    bs = len(seq)

    padded_seq = pad_no_order(max(seq_len), [torch.from_numpy(np.array(sublist)) for sublist in seq])
    seq = torch.Tensor(np.array([item for sublist in seq for item in sublist])).int()

    seq_len = torch.Tensor(list(seq_len)).int()
    im_seq_len = list(calc_im_seq_len(np.array(img_width)).astype(int))
    max_im_seq_len = max(im_seq_len)

    #print('in collate_fn - im_seq_len')
    #print(im_seq_len[:10])
    batch = {"img": img, "seq": seq, "seq_len": seq_len,
             "im_seq_len": im_seq_len, "seq_text": sample["seq_text"],
             "im_path": sample["im_path"], 'padded_seq':padded_seq}
    if do_mask:
        mask = np.zeros((bs, max_im_seq_len))
        for i in range(bs):
            mask[i, :im_seq_len[i]] = 1
        mask = np.expand_dims(np.transpose(mask), axis=2)
        batch = {**batch, **{'mask': torch.from_numpy(mask).type(torch.FloatTensor)}}
    return batch

def collate_comp(batch):
    img1 = list()
    img_width1 = list()
    img2 = list()
    img_width2 = list()
    for sample in batch:
        img1.append(torch.from_numpy(sample["img1"].transpose((2, 0, 1))).float())
        img2.append(torch.from_numpy(sample["img2"].transpose((2, 0, 1))).float())
        img_width1.append(sample["img_width1"])
        img_width2.append(sample["img_width2"])
    bs = len(img1)
    img1, img_width1 = zip(*sorted(zip(img1, img_width1), key=lambda tpl: tpl[1], reverse=True))
    img2, img_width2 = zip(*sorted(zip(img2, img_width2), key=lambda tpl: tpl[1], reverse=True))
    img1 = pad_sequence(img1, batch_first=True, padding_value=0)
    img2 = pad_sequence(img2, batch_first=True, padding_value=0)
    # print('in collate_fn - image size is:')
    # print(img.shape)

    im_seq_len1 = list(calc_im_seq_len(np.array(img_width1)).astype(int))
    im_seq_len2 = list(calc_im_seq_len(np.array(img_width2)).astype(int))
    max_im_seq_len1 = max(im_seq_len1)
    max_im_seq_len2 = max(im_seq_len2)

    # print('in collate_fn - im_seq_len')
    # print(im_seq_len[:10])
    batch = {"img1": img1,
             "im_seq_len1": im_seq_len1,
             "im_path1": sample["im_path1"],
             "img2": img2,
             "im_seq_len2": im_seq_len2,
             "im_path2": sample["im_path2"]
             }
    mask1 = np.zeros((bs, max_im_seq_len1))
    for i in range(bs):
        mask1[i, :im_seq_len1[i]] = 1
    mask1 = np.expand_dims(np.transpose(mask1), axis=2)
    mask2 = np.zeros((bs, max_im_seq_len2))
    for i in range(bs):
        mask2[i, :im_seq_len2[i]] = 1
    mask2 = np.expand_dims(np.transpose(mask2), axis=2)
    batch = {**batch, **{'mask1': torch.from_numpy(mask1).type(torch.FloatTensor),
                         'mask2': torch.from_numpy(mask2).type(torch.FloatTensor)}}
    return batch

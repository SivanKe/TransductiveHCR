from torch.utils.data import Dataset
import json
import os
import cv2
import pickle as pkl
from random import shuffle
import numpy as np

class Consts(object):
    SOS_CHAR = 'sos'
    EOS_CHAR = 'eos'
    BLANK_CHAR = 'blank'

def lexicon_to_attn(lexicon):
    for key,val in lexicon.items():
        lexicon[key] = val+3
    lexicon[Consts.BLANK_CHAR] = 0
    lexicon[Consts.SOS_CHAR] = 1
    lexicon[Consts.EOS_CHAR] = 2
    return lexicon

class TextDataset(Dataset):
    def __init__(self, data_path, lexicon, base_path=None, transform=None, fonts=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.lexicon = lexicon

        with open(data_path, 'r') as f:
            lines = f.readlines()
        shuffle(lines)
        all_records = [line.split("   *   ") for line in lines if len(line.split("   *   ")) > 1]
        if base_path:
            all_records = [(os.path.join(base_path, rec[0]),rec[1]) for rec in all_records]

        if fonts is not None:
            self.all_records = []
            for font in fonts:
                self.all_records = self.all_records + [(rec[0] + '_' + font + '.png', rec[1]) for rec in all_records
                                                       if os.path.exists(os.path.join(base_path, rec[0] + '_' + font + '.png'))]
        else:
            self.all_records = all_records


    def abc_len(self):
        return len(self.lexicon)

    def get_lexicon(self):
        return {v: k for k, v in self.lexicon.items()}


    def __len__(self):
        return len(self.all_records)

    def __getitem__(self, idx):
        im_path = self.all_records[idx][0]
        text = self.all_records[idx][1]

        # img = cv2.imread(os.path.join(self.data_path, "data", name))
        img = cv2.imread(im_path)
        if img is None:
            raise Exception("image: {}. Could not read image.".format(im_path))
        try:
            seq = self.text_to_seq(text)
        except Exception as e:
            print('exception proccessing line of image: {}'.format(im_path))
            print(e)
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "seq_text":text, 'im_path':im_path}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):
        # 'a':97, '0':48
        #self.lexicon['།'] = self.lexicon['་']
        #self.lexicon['༎'] = self.lexicon['་']
        try:

            word = [self.lexicon[Consts.SOS_CHAR]] + [self.lexicon[c] for c in text if c not in ['\n', ' ']] + [
                self.lexicon[Consts.EOS_CHAR]]
        except Exception as e:
            print('text is: "{}"'.format(text))
            for c in text:
                if c not in self.lexicon:
                    print('letter "{}" appears, but is not in lexicon'.format(c))
            raise e
        return word
    '''
    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["abc"].find(c) + 1)
        return seq
    '''


class TextDatasetRandomFont(Dataset):
    def __init__(self, data_path, lexicon, base_path=None, transform=None, fonts=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.lexicon = lexicon
        self.lexicon = lexicon
        self.fonts = fonts
        self.base_path = base_path
        with open(data_path, 'r') as f:
            lines = f.readlines()
        shuffle(lines)
        all_records = [line.split("   *   ") for line in lines if len(line.split("   *   ")) > 1]
        if base_path:
            all_records = [(os.path.join(base_path, rec[0]),rec[1]) for rec in all_records]

        self.all_records = all_records


    def abc_len(self):
        return len(self.lexicon)

    def get_lexicon(self):
        return {v: k for k, v in self.lexicon.items()}


    def __len__(self):
        return len(self.all_records)

    def __getitem__(self, idx):
        font_ids = np.random.permutation(len(self.fonts))
        for fid in font_ids:
            cur_font = self.fonts[fid]
            im_path = self.all_records[idx][0] + '_' + cur_font + '.png'
            if os.path.exists(im_path):
                break
        text = self.all_records[idx][1]

        # img = cv2.imread(os.path.join(self.data_path, "data", name))
        img = cv2.imread(im_path)
        if img is None:
            raise Exception("image: {}. Could not read image.".format(im_path))
        try:
            seq = self.text_to_seq(text)
        except Exception as e:
            print('exception proccessing line of image: {}'.format(im_path))
            print(e)
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "seq_text":text, 'im_path':im_path}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):

        #self.lexicon['།'] = self.lexicon['་']
        #self.lexicon['༎'] = self.lexicon['་']
        try:
            word = [self.lexicon[Consts.SOS_CHAR]] + [self.lexicon[c] for c in text if c not in ['\n', ' ']] + [
                self.lexicon[Consts.EOS_CHAR]]

        except Exception as e:
            print('text is: "{}"'.format(text))
            for c in text:
                if c not in self.lexicon:
                    print('letter "{}" appears, but is not in lexicon'.format(c))
            raise e
        return word
    '''
    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["abc"].find(c) + 1)
        return seq
    '''


class TextDatasetComparison(Dataset):
    def __init__(self, data_path, lexicon, base_path=None, transform=None, fonts=None):
        print('creating comparison data')
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.lexicon = lexicon
        self.lexicon = lexicon
        self.fonts = fonts
        self.base_path = base_path
        with open(data_path, 'r') as f:
            lines = f.readlines()
        shuffle(lines)
        all_records = [line.split("   *   ") for line in lines if len(line.split("   *   ")) > 1]
        if base_path:
            all_records = [(os.path.join(base_path, rec[0]),rec[1]) for rec in all_records]
        all_records = [(rec[0], rec[1]) for rec in all_records
                       if (len([font for font in fonts if os.path.exists(rec[0] + '_' + font + '.png')]) > 1)]
        print('finished creating comparison data')
        self.all_records = all_records


    def abc_len(self):
        return len(self.lexicon)

    def get_lexicon(self):
        return {v: k for k, v in self.lexicon.items()}


    def __len__(self):
        return len(self.all_records)

    def __getitem__(self, idx):
        font_ids = np.random.permutation(len(self.fonts))
        im_paths = []
        num_fonts = 0
        for fid in font_ids:
            cur_font = self.fonts[fid]
            im_path = os.path.join(self.base_path, self.all_records[idx][0] + '_' + cur_font + '.png')
            if os.path.exists(im_path):
                im_paths.append(im_path)
                num_fonts += 1
                if num_fonts >= 2:
                    break

        # img = cv2.imread(os.path.join(self.data_path, "data", name))
        imgs = []
        for im_path in im_paths:
            img = cv2.imread(im_path)
            if img is None:
                raise Exception("image: {}. Could not read image.".format(im_path))
            imgs.append(img)
        sample = {"img1": imgs[0], "img2": imgs[1], 'im_path1':im_paths[0], 'im_path2':im_paths[1]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):
        # 'a':97, '0':48
        #self.lexicon['།'] = self.lexicon['་']
        #self.lexicon['༎'] = self.lexicon['་']
        try:
            word = [self.lexicon[Consts.SOS_CHAR]] + [self.lexicon[c] for c in text if c not in ['\n', ' ']] + [
                self.lexicon[Consts.EOS_CHAR]]
        except Exception as e:
            print('text is: "{}"'.format(text))
            for c in text:
                if c not in self.lexicon:
                    print('letter "{}" appears, but is not in lexicon'.format(c))
            raise e
        return word
    '''
    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["abc"].find(c) + 1)
        return seq
    '''
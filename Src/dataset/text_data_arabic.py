from torch.utils.data import Dataset
import json
import os
import cv2
import pickle as pkl
from random import shuffle, sample
import numpy as np
from time import time
import traceback

class Consts(object):
    SOS_CHAR = 'sos'
    EOS_CHAR = 'eos'
    BLANK_CHAR = 'blank'

class TextDataset(Dataset):
    def __init__(self, data_path, lexicon, base_path=None, transform=None, fonts=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.lexicon = lexicon
        self.lexicon = lexicon

        with open(data_path, 'r') as f:
            lines = f.readlines()
        shuffle(lines)
        all_records = [line.split("   *   ") for line in lines if len(line.split("   *   ")) > 1]
        if base_path:
            all_records = [(os.path.join(base_path, rec[0]),rec[1]) for rec in all_records]
        if fonts is not None:
            self.all_records = []
            num_not_found = 0
            for record in all_records:
                if any([letter in record[1] for letter in ["{", "}"]]):
                    continue
                fount_font = False
                for font in fonts:
                    cur_path = os.path.join(base_path, record[0] + '_' + font + '.png')
                    if os.path.exists(cur_path):
                        self.all_records.append((record[0] + '_' + font + '.png', record[1]))
                        fount_font = True
                    else:
                        print(os.path.join(base_path, record[0]))
                if not fount_font:
                    num_not_found += 1
            if num_not_found > 0:
                print("Warning: {} lines had no images!!!".format(num_not_found))
            #for font in fonts:
            #    self.all_records = self.all_records + [(rec[0] + '_' + font + '.png', rec[1]) for rec in all_records
            #                                           if os.path.exists(os.path.join(base_path, rec[0] + '_' + font + '.png'))]

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
            print('image none: {}'.format(img))
            raise Exception("image: {}. Could not read image.".format(im_path))
        try:
            seq = self.text_to_seq(text)
        except Exception as e:
            print('exception proccessing line of image: {}'.format(im_path))
            traceback.print_exc()
        sample = {"img": img, "seq": seq, "seq_len": len(seq), "seq_text":text, 'im_path':im_path}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def text_to_seq(self, text):
        GO = 1
        EOS = 2
        # 'a':97, '0':48
        #self.lexicon['།'] = self.lexicon['་']
        #self.lexicon['༎'] = self.lexicon['་']
        try:
            word = [self.lexicon[c] for c in text if c not in ['\n', ' ']]
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
        GO = 1
        EOS = 2
        # 'a':97, '0':48
        self.lexicon['།'] = self.lexicon['་']
        self.lexicon['༎'] = self.lexicon['་']
        try:
            word = [self.lexicon[c] for c in text if c not in ['\n', ' ']]
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
        #all_records = [(rec[0], rec[1]) for rec in all_records
        #               if (len([font for font in fonts if os.path.exists(rec[0] + '_' + font + '.png')]) > 1)]
        all_records = [(rec[0], rec[1]) for rec in all_records]
        self.all_records_comp = all_records
        self.all_records_base = sample(all_records, len(all_records))

    def abc_len(self):
        return len(self.lexicon)

    def get_lexicon(self):
        return {v: k for k, v in self.lexicon.items()}


    def __len__(self):
        return len(self.all_records_comp)

    def __getitem__(self, idx):
        start_time = time()
        font_ids = np.random.permutation(len(self.fonts))
        im_paths = []
        num_fonts = 0
        for fid in font_ids:
            cur_font = self.fonts[fid]
            im_path = os.path.join(self.base_path, self.all_records_comp[idx][0] + '_' + cur_font + '.png')
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
        end_time = time()
        return sample

    def text_to_seq(self, text):
        GO = 1
        EOS = 2
        # 'a':97, '0':48
        self.lexicon['།'] = self.lexicon['་']
        self.lexicon['༎'] = self.lexicon['་']
        try:
            word = [self.lexicon[c] for c in text if c not in ['\n', ' ']]
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
# -*- coding: utf-8 -*-
import glob
import os
import pathlib
import pickle as pkl
import re
import subprocess
from random import shuffle
import numpy as np
import text_utils
from line_segmentation import im2lines
import traceback
from multiprocessing import Pool, Lock
import tqdm
from functools import partial
from numpy.random import choice
from skimage import io
import math
import argparse



ilegal_chars = ['&', '?', 'B', 'C', 'E', 'G', 'H', 'M', 'd', 'g', 'n', '{', '}', '\x81',
                '\x87', '«', '¹', '»', 'Ä', 'á', '¡', 'Ã', '+']

class SynthDataInfo:
    def __init__(self, is_long, use_spacing, multi_line):
        self.multi_line = multi_line
        self.font_names = ['Qomolangma-Drutsa', 'Qomolangma-Betsu', 'Shangshung Sgoba-KhraChen', 'Shangshung Sgoba-KhraChung']

        if is_long:
            self.min_len = 130
            self.max_len = 180
            self.min_allowed_len = 80
            self.max_allowed_len = 200
        else:
            # short lines:
            self.min_len = 10
            self.max_len = 50
            self.min_allowed_len = 5
            self.max_allowed_len = 60

        if use_spacing:
            self.has_rand_spaces = True
            self.has_initial_signs = True
            self.has_initial_space = True
        else:
            self.has_rand_spaces = False
            self.has_initial_signs = False
            self.has_initial_space = False

        self.num_rand_spaces_range = 3
        self.rand_space_range = [2, 6]
        self.initial_sign_space_range = [1, 8]
        self.initial_sing = '༄༄་'
        self.initial_sign_prob = 0.1
        self.initial_special_sing = '༄༄༄་་'
        self.initial_special_sign_prob = 0.01
        self.initial_space_range = [0, 4]


def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))


'''
should be: text, min_len = 130, max_len = 180, min_allowed_len = 80, max_allowed_len = 200
'''
def split_text(text, data_info_list, data_info_probs):
    text = text.replace('༑', '།')
    legal_line_end_chars = ['༎', '༑',  '།', '་']
    find_last = lambda x: np.max([x.rfind(char) for char in legal_line_end_chars])
    find_first = lambda x: np.min([x.find(char) for char in legal_line_end_chars if (x.find(char) > 0)])
    cur_start = 0
    output_texts = {}
    image_texts = {}
    while cur_start < len(text):
        data_info_i = choice(len(data_info_list), 1, p=data_info_probs)[0]
        if data_info_i not in output_texts:
            output_texts[data_info_i] = []
            image_texts[data_info_i] = []
        cur_data_info = data_info_list[data_info_i]
        size_left = len(text) - cur_start
        if size_left < cur_data_info.min_allowed_len:
            break
        if size_left <= cur_data_info.max_len:
            cur_size = size_left
        else:
            cur_size = np.random.randint(cur_data_info.min_len, cur_data_info.max_len)
            best_idx = find_last(text[cur_start:cur_start+cur_size])
            if best_idx == -1:
                best_idx = find_first(text[cur_start+cur_size:])
                cur_size = cur_size + best_idx
                if best_idx == -1 or cur_size > cur_data_info.max_allowed_len:
                    raise Exception('Cannot find legal ending charcter within a reasonalbe period in text starting from index {}, best_idx: {}, cur_size: {}, max_allowed_len:{}\n'.format(
                        cur_start, best_idx, cur_size, cur_data_info.max_allowed_len
                    ))
            cur_size = best_idx+1
        cur_text = text[cur_start:cur_start+cur_size]
        image_text = cur_text
        if cur_data_info.has_rand_spaces:
            image_text = add_random_space([image_text], cur_data_info)[0]
        if cur_data_info.has_initial_signs:
            image_text = add_initial_sign([image_text], cur_data_info)[0]
        if cur_data_info.has_initial_space:
            image_text = add_initial_space([image_text], cur_data_info)[0]

        output_texts[data_info_i].append(cur_text)
        image_texts[data_info_i].append(image_text)

        cur_start = cur_start+cur_size
    return output_texts, image_texts

def add_random_space(texts, data_info):
    if len(texts) == 0:
        raise Exception("No texts given")
    num_spaces_per_line = np.random.randint(0,data_info.num_rand_spaces_range,(len(texts),))
    spaces_sizes_all_lines = np.random.randint(data_info.rand_space_range[0],data_info.rand_space_range[1],
                                               (len(texts),data_info.num_rand_spaces_range))
    spaces_sizes_all_lines = np.split(spaces_sizes_all_lines, spaces_sizes_all_lines.shape[0], axis=0)
    legal_sep_strs = ['༎', '༑', '།', '།།']
    find_all_sep_strs = lambda string, char: [(m.end()) for m in re.finditer(char, string)]
    insert_spaces = lambda string, index, spaces: string[:index] + spaces + string[index:]

    def insert_seperators(string, num_sep, sep_sizes):
        sep_sizes = sep_sizes.reshape((-1,))
        if num_sep > 0:
            sep_lists = [find_all_sep_strs(string, char) for char in legal_sep_strs]
            all_sep_ids = np.array([item for sublist in sep_lists for item in sublist])
            if num_sep < len(all_sep_ids):
                sep_ids = all_sep_ids[np.random.permutation(len(all_sep_ids))][:num_sep]
            else:
                sep_ids = all_sep_ids
                '''
                if num_sep > len(all_sep_ids):
                    num_added_sep = num_sep-len(all_sep_ids)
                    added_sep = np.array(find_all_sep_strs(string, '་'))
                    num_added_sep = min(num_added_sep, len(added_sep))
                    if num_added_sep > 0:
                        added_sep = added_sep[np.random.permutation(len(added_sep))]
                        added_sep = added_sep[:num_added_sep]
                        sep_ids = np.concatenate((sep_ids.tolist(), added_sep))
                '''
            # reverse sorting is very important so that every sep insertion will not change the locations of other seps to insert
            sep_ids[::-1].sort()
            seps = [" " * sep_size for sep_size in sep_sizes][:num_sep]
            for idx, sep in zip(sep_ids, seps):
                try:
                    string = insert_spaces(string, (int(idx)), sep)
                except Exception as e:
                    raise e
        return string

    new_texts = [insert_seperators(text, num_sep, sep_sizes) for text, num_sep, sep_sizes
                 in zip(texts, num_spaces_per_line, spaces_sizes_all_lines)]
    return new_texts

def add_initial_space(texts, data_info):
    spaces_sizes_all_lines = np.random.randint(data_info.initial_space_range[0], data_info.initial_space_range[1], (len(texts),))
    texts = [(' '* sp_size) + txt for txt, sp_size in zip (texts, spaces_sizes_all_lines) ]
    return texts

def add_initial_sign(texts, data_info):
    spaces_sizes_all_lines = np.random.randint(data_info.initial_sign_space_range[0],
                                               data_info.initial_sign_space_range[1],
                                               (len(texts),))
    text_probs = np.random.rand(len(texts)).tolist()
    texts = [data_info.initial_special_sing + (' ' * sp_size) + txt
             if prob < data_info.initial_special_sign_prob else txt
             for txt, sp_size, prob in zip(texts, spaces_sizes_all_lines, text_probs)
    ]
    texts = [data_info.initial_sing + (' ' * sp_size) + txt
             if ((prob >= data_info.initial_special_sign_prob) and (prob < data_info.initial_sign_prob)) else txt
             for txt, sp_size, prob in zip(texts, spaces_sizes_all_lines, text_probs)]
    return texts

def split_text_to_files(text_path, num_lines_in_file, data_info_list, data_info_probs):

    # read file and convert to valid text of appropriate size
    file = pathlib.Path(text_path)
    file_base_name = ('.').join(str(file.name).split('.')[:-1])
    #text = file.read_text(encoding='iso-8859-1')

    text = file.read_text()
    text = text_utils.remove_parentheses(text)
    #texts = re.split('[༎ ༑ །]', text)
    #texts = [text for text in texts if len(text.replace(" ", "")) > 4]
    #texts = [list(chunkstring(text, max_len)) for text in texts]
    #texts = [item for sub in texts for item in sub]
    #texts = [text for text in texts if len(text) > 4]
    texts, image_texts = split_text(text, data_info_list, data_info_probs)
    texts = sum(texts.values(), [])
    image_texts = sum(image_texts.values(), [])

    image_texts = [image_text for text, image_text in zip(texts, image_texts)
                   if not any(x in text for x in ilegal_chars)]

    texts = [text for text in texts
                               if not any(x in text for x in ilegal_chars)]
    '''
    image_texts = texts
    if data_info.has_rand_spaces:
        image_texts = add_random_space(image_texts, data_info)
    if data_info.has_initial_signs:
        image_texts = add_initial_sign(image_texts, data_info)
    if data_info.has_initial_space:
        image_texts = add_initial_space(image_texts, data_info)
    '''
    if num_lines_in_file > 1:
        texts_new = []
        image_texts_new = []
        for i in range(math.floor(float(len(texts)-1)/num_lines_in_file)):
            end_i = min((i+1)*num_lines_in_file, len(texts))
            texts_new.append('\n'.join(texts[i*num_lines_in_file:end_i]))
            image_texts_new.append('&#x0a;'.join(image_texts[i * num_lines_in_file:end_i]))
            #texts[i] = zip_lines(texts[i*num_lines_in_file:end_i], num_lines_in_file, line_sep='\n')
            #image_texts[i] = zip_lines(image_texts[i*num_lines_in_file:end_i], num_lines_in_file, line_sep='&#x0a;')
        texts = texts_new
        image_texts = image_texts_new

    return texts, image_texts

def zip_lines(texts, num_lines_in_file, line_sep):
    zipped_texts = ()
    num_lines_in_file = min(num_lines_in_file, len(texts))
    for i in range(num_lines_in_file):
        zipped_texts = zipped_texts + (texts[i::num_lines_in_file],)
    zipped_texts = list(zip(*zipped_texts))
    if (len(texts) % num_lines_in_file) != 0:
        zipped_texts.append(tuple(texts[-(len(texts) % num_lines_in_file):]))

    #print('{}, {}'.format((len(texts) % 4), len(texts[-(len(texts) % 4):])))
    #print(len(texts[-(len(texts) % 4):]))
    texts = [line_sep.join(data) for data in zipped_texts]
    return texts

def create_images_per_path(orig_path, base_images_path, base_text_path, num_lines_in_file, font_dir,
                           data_info_list, data_info_probs, base_path_to_save=None, tmp_workplace='./tmp',
                           do_size_rand=True):
    outdata = []
    try:
        texts, image_texts = split_text_to_files(
            orig_path,
            num_lines_in_file,
            data_info_list, data_info_probs)

    except Exception as e:
        print('Error while parsing text in file {} to lines for synthesis.'.format(orig_path))
        traceback.print_exc()
        return []
    os.makedirs(tmp_workplace, exist_ok=True)
    file = pathlib.Path(orig_path)
    file_base_name = ('.').join(str(file.name).split('.')[:-1])

    for cur_im_num, (text, im_text) in enumerate(zip(texts, image_texts)):
        # save images
        rel_dir = pathlib.Path(file_base_name) / str(cur_im_num // 1000)
        rel_path = rel_dir / str(cur_im_num)
        path = pathlib.Path(base_images_path) / rel_path
        path.parents[0].mkdir(parents=True, exist_ok=True)
        for fid, font in enumerate(data_info_list[0].font_names):
            try:
                cur_path_no_font = str(path.absolute())
                if len(data_info_list[0].font_names) > 1:
                    cur_path = cur_path_no_font + '_' + str(font)
                if do_size_rand:
                    font_weight = str(np.random.randint(6))
                    font_stretch = str(np.random.randint(9))
                    letter_spacing = "'"+str(np.random.randint(3)) + "'"
                    font_size = "'"+ str(np.random.randint(4)) + "'"
                else:
                    font_weight = str(3)
                    font_stretch = str(4)
                    letter_spacing = "'" + str(1) + "'"
                    font_size = "'" + str(2) + "'"
                out_im_path = str(cur_path) + '.png'
                run_args = ['extra/TextRender/bin/main', im_text, out_im_path, font_dir, font, font_weight,
                            font_stretch, letter_spacing, font_size]
                # lock.acquire()
                subprocess.run(run_args, check=True)
                # lock.release()
                # save text
                if data_info_list[0].multi_line:
                    try:
                        line2im = im2lines(out_im_path, tmp_workplace=tmp_workplace, verbose=False,
                                           max_theta_diff=0.7, do_morphologic_cleaning=False)
                        os.remove(out_im_path)
                    except Exception as e:
                        print("exception on image: {}".format(out_im_path))
                        traceback.print_exc()
                        continue
                    if len(line2im) != len(text.split("\n")):
                        for i, text_line in enumerate(text.split("\n")):
                            line_im_path = cur_path_no_font + "_" + str(i) + "_" + str(font)+".png"
                            run_args = ['extra/TextRender/bin/main', text_line, line_im_path, font_dir, font, font_weight,
                                        font_stretch, letter_spacing, font_size]
                            subprocess.run(run_args, check=True)
                        # print("Image: {} - Found {} lines, but there are {} lines.".format(out_im_path, len(line2im), len(text.split("\n"))))
                    else:
                        for i in range(len(line2im)):
                            lines_texts = text.split("\n")
                            line_im_path = cur_path_no_font + "_" + str(i) + "_" + str(font)
                            line_im = line2im[i]
                            io.imsave(str(line_im_path) +".png", line_im)
                            outdata.append(str(rel_path) + "_" + str(i) + '   *   ' + lines_texts[i] + '\n')
                else:
                    if base_path_to_save is not None:
                        outdata.append(str(rel_path) + '   *   ' + text + '\n')
                    else:
                        outdata.append(str(rel_path) + '   *   ' + text + '\n')
            except Exception as e:
                print("Error while writing to path: {}".format(path))
                print('writing tibetan text:')
                print(text)
                print('original text path is: {}'.format(orig_path))
                traceback.print_exc()

    return outdata


def init_multi_p(in_lock):
    global lock
    lock = in_lock

def create_all_images(text_dir, outdir, data_info_list,
                      data_info_probs, data_info_name,
                      do_size_rand, num_parallel):
    num_lines_in_file = 5 if data_info_list[0].multi_line else 1
    font_dir = str(pathlib.Path('extra/Fonts').absolute())
    base_path = pathlib.Path(outdir)
    base_path.mkdir(parents=True, exist_ok=True)
    base_images_path = base_path / 'Images'
    base_images_path.mkdir(parents=False, exist_ok=True)
    base_text_path = base_path / 'Text'
    base_text_path.mkdir(parents=False, exist_ok=True)
    all_texts = glob.glob(os.path.join(text_dir, '*.txt'))
    all_texts = all_texts
    out_file = base_path / 'data.txt'

    out_file = str(out_file)

    if os.path.exists(out_file):
        raise Exception("Error: output file exists already. If you want to override, please delete it first:\n {}".format(
            out_file
        ))

    # save data creation info file
    for i, (data_info, name) in enumerate(zip(data_info_list,data_info_name)):
        with open(str(base_path / 'data_info_{}.pkl'.format(name)), 'wb') as data_inf_f:
            pkl.dump(data_info, data_inf_f)
    with open(str(base_path / 'data_info_probabilities.txt'), 'w') as f:
        f.writelines(["name - {} - prob: {}\n".format(name, prob) for name, prob in zip(data_info_name, data_info_probs)])

    create_images_partial = partial(create_images_per_path, base_images_path=str(base_images_path),
                                  base_text_path=str(base_text_path),
                                  num_lines_in_file=num_lines_in_file,
                                  font_dir=font_dir,
                                data_info_list=data_info_list, data_info_probs=data_info_probs,
                                    do_size_rand=do_size_rand)

    l = Lock()
    with Pool(processes=num_parallel,initializer=init_multi_p, initargs=(l,)) as p:
        max_ = len(all_texts)
        results = list(tqdm.tqdm(p.imap_unordered(create_images_partial, all_texts), total=max_))
    flatten = lambda l: [item for sublist in l for item in sublist]
    return flatten(results)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--text_dir',
                        type=str, default='../../Data/Synthetic/Text',
                        help=('tibetan_dir'))
    parser.add_argument('-l', '--line_length',
                        type=str,
                        help=('short|long|mixed'), default='long')
    parser.add_argument('-o', '--output_dir',
                        type=str, default='../../Data/Synthetic/Prepared',
                        help=('output directory'))
    parser.add_argument('-t', '--tmp_dir',
                       type=str, default='../../Data/Synthetic/tmp',
                       help=('temporary directory'))
    parser.add_argument('-n', '--num_parallel', type=int,
                        help='number of parallel threads to run', default=16)
    parser.add_argument('--remove_space_initial_sign', default=False, action='store_true')
    parser.add_argument('--remove_letter_sizes_rand', default=False, action='store_true')
    parser.add_argument('--no_multi_line', default=False, action='store_true')
    args = parser.parse_args()

    text_dir_path = args.text_dir
    out_dir = args.output_dir
    tmp_path = args.tmp_dir

    if args.remove_space_initial_sign:
        use_spacing = False
    else:
        use_spacing = True

    if args.remove_letter_sizes_rand:
        do_size_rand = False
    else:
        do_size_rand = True

    if args.no_multi_line:
        multi_line = False
    else:
        multi_line = True

    if (args.line_length == 'long'):
        data_info_list = [SynthDataInfo(is_long=True, use_spacing=use_spacing, multi_line=multi_line),
                          SynthDataInfo(is_long=True, use_spacing=use_spacing, multi_line=multi_line)]
        data_info_probs = [0.4, 0.6]
        data_info_name = ['long_space_1', 'long_space_2']
    elif (args.line_length == 'short'):
        data_info_list = [SynthDataInfo(is_long=False, use_spacing=use_spacing, multi_line=multi_line),
                          SynthDataInfo(is_long=False, use_spacing=use_spacing, multi_line=multi_line)]
        data_info_probs = [0.4, 0.6]
        data_info_name = ['short_space_1', 'short_space_2']
    elif (args.line_length == 'mixed'):
        data_info_list = [SynthDataInfo(is_long=True, use_spacing=use_spacing, multi_line=multi_line),
                          SynthDataInfo(is_long=False, use_spacing=use_spacing, multi_line=multi_line)]
        data_info_probs = [0.5, 0.5]
        data_info_name = ['long_space', 'short_space']
    else:
        raise Exception('should be one of the above')



    os.makedirs(out_dir, exist_ok=True)
    lines = create_all_images(text_dir_path, out_dir, data_info_list,
                              data_info_probs, data_info_name, do_size_rand, args.num_parallel)
    out_file = os.path.join(out_dir, 'data.txt')
    train_file = os.path.join(out_dir, 'data_train.txt')
    val_file = os.path.join(out_dir, 'data_val.txt')
    with open(out_file, 'w') as data_f:
        data_f.writelines(lines)
    idx = np.random.permutation(len(lines)).tolist()
    shuffle(lines)
    train_idx = int(0.9 * len(lines))
    val_idx = int(1 * len(lines))
    train_lines = lines[:train_idx]
    val_lines = lines[train_idx:val_idx]
    # test_lines = lines[val_idx:]

    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    with open(val_file, 'w') as f:
        f.writelines(val_lines)
    #pikle_to_text(out_dataset_file)

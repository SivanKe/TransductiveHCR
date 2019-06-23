import os
import cv2
import string
from tqdm import tqdm
import click
import numpy as np
import pickle as pkl
import operator

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.my_edit_distance import my_edit_distance_backpointer, Statistics

from models.crnn import CRNN
from models.model_loader import load_attn_model
from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from models.model_loader import load_model
from torchvision.transforms import Compose
from nltk.metrics import ConfusionMatrix
import editdistance
from models.Attention.Loss import NLLLoss

from dataset.data_transform import Resize, AddWidth, Normalize

from difflib import get_close_matches


# print_data_visuals(tb_writer, train_data.get_lexicon(), sample["img"], labels_flatten, label_lens, preds, ((epoch_count * len(data_loader)) + iter)))
def print_data_visuals(net, tb_writer, lexicon, images, labels_flatten, label_lens, preds, n_iter, initial_txt=""):
    preds_text = net.decode(preds, lexicon, num_decode=10)
    label_text = net.decode_flatten(labels_flatten, label_lens, lexicon, num_decode=10)
    tb_writer.show_images(images, label_text=label_text,
                          pred_text=preds_text,
                          n_iter=n_iter,
                          initial_title=initial_txt)

def text2image(text, font_file='data/extra/Fonts/Qomolangma-Betsu.ttf'): #out_path,
    font_size = 42
    background_color = 'rgb(100%,100%,100%)'
    fill_color = 'rgb(0%,0%,0%)'

    im_size = (50, 10) # width, hight
    im_offset = (random.randrange(3,11), random.randrange(3,11),
                 random.randrange(3, 11), random.randrange(3,11))
    desired_hight = 64

    im = Image.new("RGB", im_size, background_color)
    unicode_font = ImageFont.truetype(font_file, font_size)
    text_size = unicode_font.getsize(text)
    im = im.resize((text_size[0]+im_offset[0]+im_offset[1],text_size[1]+im_offset[2]+im_offset[3]))
    draw = ImageDraw.Draw(im)
    draw.text((im_offset[0], im_offset[2]), text, fill=fill_color, font=unicode_font)
    w,h = im.size
    desired_width = int(math.floor(float(w)*float(desired_hight)/float(h)))

    im = im.resize((desired_width, desired_hight), resample=Image.LANCZOS)
    #im.save(out_path)
    return im#(im.size[0])

def save_results(net, data, cuda, output_path):
    data_loader = DataLoader(data, batch_size=1, num_workers=2, shuffle=False, collate_fn=text_collate)
    stop_characters = ['-', '.', '༎', '༑', '།', '་']
    iterator = tqdm(data_loader)
    all_pred_text = []
    all_label_text = []
    all_im_pathes = []
    for i, sample in enumerate(iterator):
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        img_seq_lens = sample["im_seq_len"]
        out, orig_seq = net(imgs, img_seq_lens, decode=True)
        labels_flatten = Variable(sample["seq"]).view(-1)
        label_lens = Variable(sample["seq_len"].int())
        preds_text = net.decode(orig_seq, data.get_lexicon())
        all_pred_text.append(preds_text + '\n')
        label_text = net.decode_flatten(labels_flatten, label_lens, data.get_lexicon())
        all_label_text.append(label_text + '\n')
        all_im_pathes.append(sample["im_path"] + '\n')
    with open(output_path + '_pred.txt', 'w') as fp:
        fp.writelines(all_pred_text)
    with open(output_path + '_label.txt', 'w') as fp:
        fp.writelines(all_label_text)
    with open(output_path + '_im.txt', 'w') as fp:
        fp.writelines(all_im_pathes)

def test_attn(net, data, abc, cuda, visualize, batch_size=1,
         tb_writer=None, n_iter=0, initial_title="", is_trian=True, output_path=None):
    collate = lambda x: text_collate(x, do_mask=True)
    net.eval()
    data_loader = DataLoader(data, batch_size=1, num_workers=2, shuffle=False, collate_fn=collate)
    stop_characters = ['-', '.', '༎', '༑',  '།', '་']
    count = 0
    tp = 0
    avg_ed = 0
    avg_no_stop_ed = 0
    avg_loss = 0
    min_ed = 1000
    iterator = tqdm(data_loader)
    all_pred_text = all_label_text = all_im_pathes =[]
    test_letter_statistics = Statistics()
    with torch.no_grad():
        for i, sample in enumerate(iterator):
            if is_trian and (i > 1000):
                break
            imgs = Variable(sample["img"])
            mask = sample["mask"]
            padded_labels = sample["padded_seq"]
            if cuda:
                imgs = imgs.cuda()
                mask = mask.cuda()
                padded_labels = padded_labels.cuda()

            img_seq_lens = sample["im_seq_len"]



            # Forward propagation
            decoder_outputs, decoder_hidden, other = net(imgs, img_seq_lens, mask, None,
                                                         teacher_forcing_ratio=0)

            # Get loss
            loss = NLLLoss()
            loss.reset()
            zero_labels = torch.zeros_like(padded_labels[:, 1])
            max_label_size = padded_labels.size(1)
            for step, step_output in enumerate(decoder_outputs):
                batch_size = padded_labels.size(0)
                if (step + 1) < max_label_size:
                    loss.eval_batch(step_output.contiguous().view(batch_size, -1), padded_labels[:, step + 1])
                else:
                    loss.eval_batch(step_output.contiguous().view(batch_size, -1), zero_labels)
            # Backward propagation
            total_loss = loss.get_loss().data[0]
            avg_loss += total_loss
            labels_flatten = Variable(sample["seq"]).view(-1)
            label_lens = Variable(sample["seq_len"].int())
            preds_text = net.predict(other)
            padded_labels = (sample["padded_seq"].numpy()).tolist()
            lens = sample["seq_len"].numpy().tolist()
            label_text = net.padded_seq_to_txt(padded_labels, lens)
            if output_path is not None:

                all_pred_text = all_pred_text + [pd+'\n' for pd in preds_text]
                all_label_text = all_label_text + [lb + '\n' for lb in label_text]
                all_im_pathes.append(sample["im_path"] + '\n')#[imp +'\n' for imp in sample["im_path"]]

            if i == 0:
                if tb_writer is not None:
                    tb_writer.show_images(sample["img"], label_text=[lb + '\n' for lb in label_text],
                                          pred_text=[pd+'\n' for pd in preds_text],
                                          n_iter=n_iter,
                                          initial_title=initial_title)

            pos = 0
            key = ''
            for i in range(len(label_text)):
                cur_out_no_stops = ''.join(c for c in label_text[i] if not c in stop_characters)
                cur_gts_no_stops = ''.join(c for c in preds_text[i] if not c in stop_characters)
                cur_ed = editdistance.eval(preds_text[i], label_text[i]) / max(len(preds_text[i]), len(label_text[i]))
                errors, matches, bp = my_edit_distance_backpointer(cur_out_no_stops, cur_gts_no_stops)
                test_letter_statistics.add_data(bp)
                my_no_stop_ed = errors / max(len(cur_out_no_stops), len(cur_gts_no_stops))
                cur_no_stop_ed = editdistance.eval(cur_out_no_stops, cur_gts_no_stops) / max(len(cur_out_no_stops), len(cur_gts_no_stops))

                if my_no_stop_ed != cur_no_stop_ed:
                    print('old ed: {} , vs. new ed: {}\n'.format(my_no_stop_ed, cur_no_stop_ed))
                avg_no_stop_ed += cur_no_stop_ed
                avg_ed += cur_ed
                if cur_ed < min_ed: min_ed = cur_ed

                count += 1
                if visualize:
                    status = "pred: {}; gt: {}".format(preds_text[i], label_text[i])
                    iterator.set_description(status)
                    img = imgs[i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
                    cv2.imshow("img", img)
                    key = chr(cv2.waitKey() & 255)
                    if key == 'q':
                        break
            if key == 'q':
                break
            if not visualize:
                iterator.set_description("acc: {0:.4f}; avg_ed: {0:.4f}".format(
                    float(tp) / float(count), float(avg_ed) / float(count)))
    with open(output_path + '_{}_{}_statistics.pkl'.format(initial_title,n_iter), 'wb') as sf:

        pkl.dump(test_letter_statistics.total_actions_hists, sf)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        print('writing output')
        with open(output_path + '_{}_{}_pred.txt'.format(initial_title,n_iter), 'w') as fp:
            fp.writelines(all_pred_text)
        with open(output_path + '_{}_{}_label.txt'.format(initial_title,n_iter), 'w') as fp:
            fp.writelines(all_label_text)
        with open(output_path + '_{}_{}_im.txt'.format(initial_title, n_iter), 'w') as fp:
            fp.writelines(all_im_pathes)
        stop_characters = ['-', '.', '༎', '༑', '།', '་']

        all_pred_text = [''.join(c for c in line if not c in stop_characters) for line in all_pred_text]
        with open(output_path + '_{}_{}_pred_no_stopchars.txt'.format(initial_title,n_iter), 'w') as rf:
            rf.writelines(all_pred_text)
        all_label_text = [''.join(c for c in line if not c in stop_characters) for line in all_label_text]
        with open(output_path + '_{}_{}_label_no_stopchars.txt'.format(initial_title, n_iter), 'w') as rf:
            rf.writelines(all_label_text)
    acc = float(tp) / float(count)
    avg_ed = float(avg_ed) / float(count)
    avg_no_stop_ed = float(avg_no_stop_ed) / float(count)
    avg_loss = float(avg_loss) / float(count)
    return acc, avg_ed, avg_no_stop_ed, avg_loss

def test(net, data, abc, cuda, visualize, batch_size=1,
         tb_writer=None, n_iter=0, initial_title="", loss_function=None, is_trian=True, output_path=None,
         do_beam_search=False, do_results=False, word_lexicon=None):
    collate = lambda x: text_collate(x, do_mask=False)
    data_loader = DataLoader(data, batch_size=1, num_workers=2, shuffle=False, collate_fn=collate)
    stop_characters = ['-', '.', '༎', '༑',  '།', '་']
    garbage = '-'
    count = 0
    tp = 0
    avg_ed = 0
    avg_no_stop_ed = 0
    avg_accuracy = 0
    avg_loss = 0
    min_ed = 1000
    iterator = tqdm(data_loader)
    all_pred_text = all_label_text = all_im_pathes =[]
    test_letter_statistics = Statistics()
    im_by_error = {}

    for i, sample in enumerate(iterator):
        if is_trian and (i > 500):
            break
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        img_seq_lens = sample["im_seq_len"]
        out, orig_seq = net(imgs, img_seq_lens, decode=True, do_beam_search=do_beam_search)
        if loss_function is not None:
            labels_flatten = Variable(sample["seq"]).view(-1)
            label_lens = Variable(sample["seq_len"].int())
            loss = loss_function(orig_seq, labels_flatten,
                                 Variable(torch.IntTensor(np.array(img_seq_lens))), label_lens) / batch_size
            avg_loss += loss.data[0]
        gt = (sample["seq"].numpy()).tolist()
        lens = sample["seq_len"].numpy().tolist()
        labels_flatten = Variable(sample["seq"]).view(-1)
        label_lens = Variable(sample["seq_len"].int())
        if output_path is not None:
            preds_text = net.decode(orig_seq, data.get_lexicon())
            all_pred_text = all_pred_text + [''.join(c for c in pd if c != garbage)+'\n' for pd in preds_text]

            label_text = net.decode_flatten(labels_flatten, label_lens, data.get_lexicon())
            all_label_text = all_label_text + [lb + '\n' for lb in label_text]

            all_im_pathes.append(sample["im_path"] + '\n')#[imp +'\n' for imp in sample["im_path"]]

        if i == 0:
            if tb_writer is not None:
                print_data_visuals(net, tb_writer, data.get_lexicon(), sample["img"], labels_flatten, label_lens, orig_seq, n_iter,
                               initial_title)

        pos = 0
        key = ''
        for i in range(len(out)):
            gts = ''.join(abc[c] for c in gt[pos:pos+lens[i]])

            pos += lens[i]
            if gts == out[i]:
                tp += 1
            else:
                cur_out = ''.join(c for c in out[i] if c != garbage)
                cur_gts = ''.join(c for c in gts if c != garbage)
                cur_out_no_stops = ''.join(c for c in out[i] if not c in stop_characters)
                cur_gts_no_stops = ''.join(c for c in gts if not c in stop_characters)
                cur_ed = editdistance.eval(cur_out, cur_gts) / len(cur_gts)
                if word_lexicon is not None:
                    closest_word = get_close_matches(cur_out, word_lexicon, n=1, cutoff=0.2)
                else:
                    closest_word = cur_out

                if len(closest_word) > 0 and closest_word[0] == cur_gts:
                    avg_accuracy += 1

                errors, matches, bp = my_edit_distance_backpointer(cur_out_no_stops, cur_gts_no_stops)
                test_letter_statistics.add_data(bp)
                #my_no_stop_ed = errors / max(len(cur_out_no_stops), len(cur_gts_no_stops))
                #cur_no_stop_ed = editdistance.eval(cur_out_no_stops, cur_gts_no_stops) / max(len(cur_out_no_stops), len(cur_gts_no_stops))
                if do_results:
                    im_by_error[sample["im_path"]] =  cur_ed
                my_no_stop_ed = errors / len(cur_gts_no_stops)
                cur_no_stop_ed = editdistance.eval(cur_out_no_stops, cur_gts_no_stops) / len(cur_gts_no_stops)

                if my_no_stop_ed != cur_no_stop_ed:
                    print('old ed: {} , vs. new ed: {}\n'.format(my_no_stop_ed, cur_no_stop_ed))
                avg_no_stop_ed += cur_no_stop_ed
                avg_ed += cur_ed
                if cur_ed < min_ed: min_ed = cur_ed
            count += 1
            if visualize:
                status = "pred: {}; gt: {}".format(out[i], gts)
                iterator.set_description(status)
                img = imgs[i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
                cv2.imshow("img", img)
                key = chr(cv2.waitKey() & 255)
                if key == 'q':
                    break



        #if not visualize:
        #    iterator.set_description("acc: {0:.4f}; avg_ed: {0:.4f}".format(
        #        float(tp) / float(count), float(avg_ed) / float(count)))
    #with open(output_path + '_{}_{}_statistics.pkl'.format(initial_title,n_iter), 'wb') as sf:

    #    pkl.dump(test_letter_statistics.total_actions_hists, sf)

    if do_results and output_path is not None:
        print('printing results! :)')
        sorted_im_by_error = sorted(im_by_error.items(), key=operator.itemgetter(1))
        sorted_im = [key for (key, value) in sorted_im_by_error]
        all_im_pathes_no_new_line = [im.replace('\n','') for im in all_im_pathes]
        printed_res_best = ""
        printed_res_worst = ""
        for im in sorted_im[:20]:
            im_id = all_im_pathes_no_new_line.index(im)
            pred = all_pred_text[im_id]
            label = all_label_text[im_id]
            printed_res_best += im + '\n' + label + pred

        for im in list(reversed(sorted_im))[:20]:
            im_id = all_im_pathes_no_new_line.index(im)
            pred = all_pred_text[im_id]
            label = all_label_text[im_id]
            printed_res_worst += im + '\n' + label + pred

        with open(output_path + '_{}_{}_sorted_images_by_errors.txt'.format(initial_title, n_iter), 'w') as fp:
            fp.writelines([key+','+str(value)+'\n' for (key, value) in sorted_im_by_error])

        with open(output_path + '_{}_{}_res_on_best.txt'.format(initial_title, n_iter), 'w') as fp:
            fp.writelines([printed_res_best])
            with open(output_path + '_{}_{}_res_on_worst.txt'.format(initial_title, n_iter), 'w') as fp:
                fp.writelines([printed_res_worst])
        os.makedirs(output_path, exist_ok=True)
        with open(output_path + '_{}_{}_pred.txt'.format(initial_title,n_iter), 'w') as fp:
            fp.writelines(all_pred_text)
        with open(output_path + '_{}_{}_label.txt'.format(initial_title,n_iter), 'w') as fp:
            fp.writelines(all_label_text)
        with open(output_path + '_{}_{}_im.txt'.format(initial_title, n_iter), 'w') as fp:
            fp.writelines(all_im_pathes)
        stop_characters = ['-', '.', '༎', '༑', '།', '་']

        all_pred_text = [''.join(c for c in line if not c in stop_characters) for line in all_pred_text]
        with open(output_path + '_{}_{}_pred_no_stopchars.txt'.format(initial_title,n_iter), 'w') as rf:
            rf.writelines(all_pred_text)
        all_label_text = [''.join(c for c in line if not c in stop_characters) for line in all_label_text]
        with open(output_path + '_{}_{}_label_no_stopchars.txt'.format(initial_title, n_iter), 'w') as rf:
            rf.writelines(all_label_text)

    acc = float(avg_accuracy) / float(count)
    avg_ed = float(avg_ed) / float(count)
    avg_no_stop_ed = float(avg_no_stop_ed) / float(count)
    if loss_function is not None:
        avg_loss = float(avg_loss) / float(count)
        return acc, avg_ed, avg_no_stop_ed, avg_loss
    return acc, avg_ed, avg_no_stop_ed

@click.command()
@click.option('--data-path', type=str,
              default='../../Data/Test/Prepared/im2line.txt',
              help='Path to dataset')
@click.option('--base-data-dir', type=str,
              default='../../Data/Test/Prepared/LineImages',
              help='Path to dataset')
@click.option('--lexicon-path', type=str,
              default='../../Data/char_to_class.pkl', help='Alphabet')
@click.option('--output-path', type=str, default='../../Results', help='Path to dataset')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str,
              default='../../PreTrainedModels/trans_vat_model', help='Pre-trained weights'
              )
@click.option('--input-height', type=int, default=64, help='Input size')
@click.option('--visualize', type=bool, default=False, help='Visualize output')
@click.option('--do_beam_search', type=bool, default=True, help='Visualize output')
def main(data_path, base_data_dir, lexicon_path, output_path, seq_proj, backend, snapshot, input_height, visualize, do_beam_search):
    cuda = True
    with open(lexicon_path, 'rb') as f:
        lexicon = pkl.load(f)
        print(sorted(lexicon.items(), key=operator.itemgetter(1)))

    transform = Compose([
        Resize(hight=input_height),
        AddWidth(),
        Normalize()
    ])
    data = TextDataset(data_path=data_path, lexicon=lexicon,
                                 base_path=base_data_dir, transform=transform, fonts=None)

        # data = TextDataset(data_path=data_path, mode="test", transform=transform)
    #else:
    #    data = TestDataset(transform=transform, abc=abc)
    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(lexicon=data.get_lexicon(), seq_proj=seq_proj, backend=backend,
                     snapshot=snapshot, cuda=cuda, do_beam_search=do_beam_search).eval()
    acc, avg_ed, avg_no_stop_ed = test(net, data,
                                         data.get_lexicon(),
                                         cuda,
                                         batch_size=1,
                                         visualize=False,
                                         tb_writer=None,
                                         n_iter=0,
                                         initial_title='val_orig',
                                         loss_function=None,
                                         is_trian=False, output_path=output_path,
                                       do_beam_search=do_beam_search,
                                       do_results = True
                                         )
    print("Accuracy: {}".format(acc))
    print("Edit distance: {}".format(avg_ed))
    print("Edit distance without stop signs: {}".format(avg_no_stop_ed))

if __name__ == '__main__':
    main()

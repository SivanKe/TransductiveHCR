import os
import click
import string
import numpy as np
from tqdm import tqdm
from models.model_loader import load_model
from torchvision.transforms import Compose, Lambda
from dataset.data_transform import Resize, Rotation, ElasticAndSine, ColorGradGausNoise, AddWidth, Normalize, ToGray, OnlyElastic, OnlySine, ColorGrad, ColorGausNoise
from dataset.data_transform_semi_sup import ResizeDouble, RotationDouble, ElasticAndSineDouble, ColorGradGausNoiseDouble, AddWidthDouble, NormalizeDouble
from dataset.test_data import TestDataset
from dataset.text_data import TextDataset, TextDatasetRandomFont, TextDatasetComparison
from dataset.collate_fn import text_collate, collate_comp
from utils.data_visualization import TbSummary
from lr_policy import StepLR, DannLR
import pickle as pkl
import glob
import operator

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
from warpctc_pytorch import CTCLoss
from models.crnn import CRNN

from torchvision.utils import make_grid

from test import test, print_data_visuals
from models.vat import virtual_adversarial_loss, comp_loss
from models.new_vat import VATLoss, VATLossSign, LabeledATLoss, RandomLoss, LabeledAtAndUnlabeledTestVatLoss, VATonRnnSign, VATonRnnCnnSign, VATonCnnSign, PseudoLabel

@click.command()
@click.option('--base-data-dir', type=str,
              default=os.path.expandvars ('../Data/'),
              help='Path to base data directory (all other data paths are relative to this one).')
@click.option('--train-data-path', type=str,
              default=os.path.expandvars ('Synthetic/Prepared/data_train.txt'),
              help='Path to training dataset (image path to line text) text file (relative to base-data-dir)')
@click.option('--train-base-dir', type=str,
              default=os.path.expandvars(
                  'Synthetic/Prepared/Images'),
              help='Path to directory containing training images (relative to base-data-dir)')
@click.option('--orig-eval-data-path', type=str,
              default=os.path.expandvars(
                  'Test/Prepared/im2line.txt'),
              help='Path to original test dataset (image path to line text) text file (relative to base-data-dir)')
@click.option('--orig-eval-base-dir', type=str,
              default=os.path.expandvars(
                  'Test/Prepared/LineImages'),
              help='Path to directory containing original test images (relative to base-data-dir)')
@click.option('--synth-eval-data-path', type=str,
              default=os.path.expandvars ('Synthetic/Prepared/data_val.txt'),
              help='Path to synthetic evaluation dataset (image path to line text) text file (relative to base-data-dir)')
@click.option('--synth-eval-base-dir', type=str,
              default=os.path.expandvars(
                  'Synthetic/Prepared/Images'),
              help='Path to directory containing synthetic evaluation images (relative to base-data-dir)')
@click.option('--lexicon-path', type=str,
              default=os.path.expandvars('char_to_class.pkl'),
              help='Path to alphabet lexicon (letter to id), relative to base-data-dir.')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network to use (default is resnet18)')
@click.option('--snapshot', type=str, default=None, help='Path to pre-trained weights')
@click.option('--input-height', type=int, default=64, help='Height of input images to network')
@click.option('--base-lr', type=float, default=1e-3, help='Base learning rate.') # was e-3
#@click.option('--lr-decay', type=float, default=1e-4, help='Base learning rate') # was 0.0001
@click.option('--elastic-alpha', type=float, default=34, help='Elastic augmentation parameter alpha.')
@click.option('--elastic-sigma', type=float, default=3, help='Elastic augmentation parameter sigma.')
@click.option('--step-size', type=int, default=500, help='Step size for step lr change.')
@click.option('--max-iter', type=int, default=6000, help='Max iterations for taining')
@click.option('--batch-size', type=int, default=8, help='Batch size for training')
@click.option('--output-dir', type=str,
              default='../Output/exp1',
              help='Path to save output snapshot')
@click.option('--test-iter', type=int, default=1000, help='Number of iterations between test evaluation.')
@click.option('--show-iter', type=int, default=1000, help='Number of iterations between showing images in tensorboard.')
@click.option('--test-init', type=bool, default=False, help='Wether to test after network initialization initialization')
@click.option('--use-gpu', type=bool, default=True, help='Whether to use the gpu')
@click.option('--use-no-font-repeat-data', type=bool, default=True, help='Parameter to remove (always true) - whether to use random training data.')
@click.option('--do-vat', type=bool, default=False, help='Whether to do VAT on synthetic trainig data')
@click.option('--do-at', type=bool, default=False, help='Whether to do AT on synthetic trainig data')
@click.option('--vat-ratio', type=float, default=1, help='Ratio of vat on train data loss vs base loss')
@click.option('--test-vat-ratio', type=float, default=1, help='Ratio on vat on test data loss vs base loss')
@click.option('--vat-epsilon', type=float, default=2.5, help='VAT on train hyperparameter - epsilon')
@click.option('--vat-ip', type=int, default=1, help='VAT on train hyperparameter - number of power iterations')
@click.option('--vat-xi', type=float, default=10., help='VAT on train hyperparameter - xi')
@click.option('--vat-sign', type=bool, default=False, help='VAT on train hyperparameter - whether to do sign on vat loss')
@click.option('--do-remove-augs', type=bool, default=False, help='Whether to remove some of the augmentations (for ablation study)')
@click.option('--aug-to-remove', type=str,
              default='',
              help="with autmentation to remover out of ['elastic', 'sine', 'sine_rotate', 'rotation', 'color_aug', 'color_gaus', 'color_sine']")
@click.option('--do-beam-search', type=bool, default=False, help='whether to do beam search inference in evaluation')
@click.option('--dropout-conv', type=bool, default=False, help='Whether to do dropout between convolution and rnn.')
@click.option('--dropout-rnn', type=bool, default=False, help='Whether to do dropout in rnn.')
@click.option('--dropout-output', type=bool, default=False, help='Whether to do dropout after rnn.')
@click.option('--do-ema', type=bool, default=False, help='Whether to do exponential moving average on weights')
@click.option('--do-gray', type=bool, default=False, help='whether to use grayscale instread of rgb')
@click.option('--do-test-vat', type=bool, default=False, help='Whether to do VAT loss on original test data')
@click.option('--do-test-entropy', type=bool, default=False, help='Whether to do entropy loss on original test data')
@click.option('--do-test-vat-cnn', type=bool, default=False, help='Whether to do VAT loss on original test data only for cnn part')
@click.option('--do-test-vat-rnn', type=bool, default=False, help='Whether to do VAT loss on original test data only for rnn part')
@click.option('--ada-after-rnn', type=bool, default=False, help='Whether to do adversarial domain adaptaion on rnn part')
@click.option('--ada-before-rnn', type=bool, default=False, help='Whether to do adversarial domain adaptaion on cnn part')
@click.option('--do-ada-lr', type=bool, default=False, help='Whether to do lr rule suitable of adversarial domain adaptaion (from article)')
@click.option('--ada-ratio', type=float, default=1, help='Ratio of ADA loss vs base loss')
@click.option('--rnn-hidden-size', type=int, default=128, help='Size of rnn hidden layer')
@click.option('--do-lr-step', type=bool, default=False, help='Visualize output')

def main(base_data_dir, train_data_path, train_base_dir,
         orig_eval_data_path, orig_eval_base_dir,
         synth_eval_data_path, synth_eval_base_dir,
         lexicon_path, seq_proj, backend, snapshot, input_height, base_lr, elastic_alpha, elastic_sigma,
         step_size, max_iter,
         batch_size, output_dir, test_iter, show_iter, test_init, use_gpu, use_no_font_repeat_data,
         do_vat, do_at, vat_ratio, test_vat_ratio, vat_epsilon, vat_ip, vat_xi, vat_sign,
         do_remove_augs, aug_to_remove, do_beam_search,
         dropout_conv, dropout_rnn, dropout_output, do_ema, do_gray, do_test_vat, do_test_entropy, do_test_vat_cnn,
         do_test_vat_rnn,
         ada_after_rnn, ada_before_rnn, do_ada_lr, ada_ratio, rnn_hidden_size,
         do_lr_step
         ):
    if not do_lr_step and not do_ada_lr:
        raise NotImplementedError('learning rate should be either step or ada.')
    train_data_path = os.path.join(base_data_dir, train_data_path)
    train_base_dir = os.path.join(base_data_dir, train_base_dir)
    synth_eval_data_path = os.path.join(base_data_dir, synth_eval_data_path)
    synth_eval_base_dir = os.path.join(base_data_dir, synth_eval_base_dir)
    orig_eval_data_path = os.path.join(base_data_dir, orig_eval_data_path)
    orig_eval_base_dir = os.path.join(base_data_dir, orig_eval_base_dir)
    lexicon_path = os.path.join(base_data_dir, lexicon_path)

    all_parameters = locals()
    cuda = use_gpu
    #print(train_base_dir)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        tb_writer = TbSummary(output_dir)
        output_dir = os.path.join(output_dir, 'model')
        os.makedirs(output_dir, exist_ok=True)

    with open(lexicon_path, 'rb') as f:
        lexicon = pkl.load(f)
    #print(sorted(lexicon.items(), key=operator.itemgetter(1)))

    with open(os.path.join(output_dir, 'params.txt'),'w') as f:
        f.writelines(str(all_parameters))
    print(all_parameters)
    print('new vat')

    sin_magnitude = 4
    rotate_max_angle = 2
    train_fonts = ['Qomolangma-Betsu', 'Shangshung Sgoba-KhraChen', 'Shangshung Sgoba-KhraChung', 'Qomolangma-Drutsa']


    all_args = locals()


    allowed_removals = ['elastic', 'sine', 'sine_rotate', 'rotation', 'color_aug', 'color_gaus', 'color_sine']
    if do_remove_augs and aug_to_remove not in allowed_removals:
        raise Exception('augmentation removal value is not allowed.')


    if do_remove_augs:
        rand_trans = []
        if aug_to_remove == 'elastic':
            print('doing sine transform :)')
            rand_trans.append(OnlySine(sin_magnitude=sin_magnitude))
        elif aug_to_remove in ['sine', 'sine_rotate']:
            print('doing elastic transform :)')
            rand_trans.append(OnlyElastic(elastic_alpha=elastic_alpha, elastic_sigma=elastic_sigma))
        if aug_to_remove not in ['elastic', 'sine', 'sine_rotate']:
            print('doing elastic transform :)')
            print('doing sine transform :)')
            rand_trans.append(ElasticAndSine(elastic_alpha=elastic_alpha, elastic_sigma=elastic_sigma, sin_magnitude=sin_magnitude))
        if aug_to_remove not in ['rotation', 'sine_rotate']:
            print('doing rotation transform :)')
            rand_trans.append(Rotation(angle=rotate_max_angle, fill_value=255))
        if aug_to_remove not in ['color_aug', 'color_gaus', 'color_sine']:
            print('doing color_aug transform :)')
            rand_trans.append(ColorGradGausNoise())
        elif aug_to_remove == 'color_gaus':
            print('doing color_sine transform :)')
            rand_trans.append(ColorGrad())
        elif aug_to_remove == 'color_sine':
            print('doing color_gaus transform :)')
            rand_trans.append(ColorGausNoise())
    else:
        print('doing all transforms :)')
        rand_trans = [
            ElasticAndSine(elastic_alpha=elastic_alpha, elastic_sigma=elastic_sigma, sin_magnitude=sin_magnitude),
            Rotation(angle=rotate_max_angle, fill_value=255),
            ColorGradGausNoise()]
    if do_gray:
        rand_trans = rand_trans + [Resize(hight=input_height),
            AddWidth(),
            ToGray(),
            Normalize()]
    else:
        rand_trans = rand_trans + [Resize(hight=input_height),
                                   AddWidth(),
                                   Normalize()]

    transform_random = Compose(rand_trans)
    if do_gray:
        transform_simple = Compose([
            Resize(hight=input_height),
            AddWidth(),
            ToGray(),
            Normalize()
        ])
    else:
        transform_simple = Compose([
            Resize(hight=input_height),
            AddWidth(),
            Normalize()
        ])

    if use_no_font_repeat_data:
        print('create dataset')
        train_data = TextDatasetRandomFont(data_path=train_data_path, lexicon=lexicon,
                                 base_path=train_base_dir, transform=transform_random, fonts=train_fonts)
        print('finished creating dataset')
    else:
        print('train data path:\n{}'.format(train_data_path))
        print('train_base_dir:\n{}'.format(train_base_dir))
        train_data = TextDataset(data_path=train_data_path, lexicon=lexicon,
                                 base_path=train_base_dir, transform=transform_random, fonts=train_fonts)
    synth_eval_data = TextDataset(data_path=synth_eval_data_path, lexicon=lexicon,
                                  base_path=synth_eval_base_dir, transform=transform_random, fonts=train_fonts)
    orig_eval_data = TextDataset(data_path=orig_eval_data_path, lexicon=lexicon,
                                 base_path=orig_eval_base_dir, transform=transform_simple, fonts=None)
    if do_test_vat or do_test_vat_rnn or do_test_vat_cnn:
        orig_vat_data = TextDataset(data_path=orig_eval_data_path, lexicon=lexicon,
                                     base_path=orig_eval_base_dir, transform=transform_simple, fonts=None)

    if ada_after_rnn or ada_before_rnn:
        orig_ada_data = TextDataset(data_path=orig_eval_data_path, lexicon=lexicon,
                                    base_path=orig_eval_base_dir, transform=transform_simple, fonts=None)

    #else:
    #    train_data = TestDataset(transform=transform, abc=abc).set_mode("train")
    #    synth_eval_data = TestDataset(transform=transform, abc=abc).set_mode("test")
    #    orig_eval_data = TestDataset(transform=transform, abc=abc).set_mode("test")
    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(lexicon=train_data.get_lexicon(), seq_proj=seq_proj, backend=backend,
                     snapshot=snapshot, cuda=cuda, do_beam_search=do_beam_search,
                     dropout_conv=dropout_conv,
                     dropout_rnn=dropout_rnn,
                     dropout_output=dropout_output,
                     do_ema=do_ema,
                     ada_after_rnn=ada_after_rnn, ada_before_rnn=ada_before_rnn,
                     rnn_hidden_size=rnn_hidden_size
                     )
    optimizer = optim.Adam(net.parameters(), lr = base_lr, weight_decay=0.0001)
    if do_ada_lr:
        print('using ada lr')
        lr_scheduler = DannLR(optimizer, max_iter=max_iter)
    elif do_lr_step:
        print('using step lr')
        lr_scheduler = StepLR(optimizer, step_size=step_size, max_iter=max_iter)
    loss_function = CTCLoss()

    synth_avg_ed_best = float("inf")
    orig_avg_ed_best = float("inf")
    epoch_count = 0

    if do_test_vat or do_test_vat_rnn or do_test_vat_cnn:
        collate_vat = lambda x: text_collate(x, do_mask=True)
        vat_load = DataLoader(orig_vat_data, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate_vat)
        vat_len = len(vat_load)
        cur_vat = 0
        vat_iter = iter(vat_load)
    if ada_after_rnn or ada_before_rnn:
        collate_ada = lambda x: text_collate(x, do_mask=True)
        ada_load = DataLoader(orig_ada_data, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate_ada)
        ada_len = len(ada_load)
        cur_ada = 0
        ada_iter = iter(ada_load)

    loss_domain = torch.nn.NLLLoss()

    while True:
        collate = lambda x: text_collate(x, do_mask=(do_vat or ada_before_rnn or ada_after_rnn))
        data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate)

        loss_mean_ctc = []
        loss_mean_vat = []
        loss_mean_at = []
        loss_mean_comp = []
        loss_mean_total = []
        loss_mean_test_vat = []
        loss_mean_test_pseudo = []
        loss_mean_test_rand = []
        loss_mean_ada_rnn_s = []
        loss_mean_ada_rnn_t = []
        loss_mean_ada_cnn_s = []
        loss_mean_ada_cnn_t = []
        iterator = tqdm(data_loader)
        iter_count = 0
        for iter_num, sample in enumerate(iterator):
            total_iter = (epoch_count * len(data_loader)) + iter_num
            if ((total_iter > 1) and total_iter % test_iter == 0) or (test_init and total_iter == 0):
                # epoch_count != 0 and

                print("Test phase")
                net = net.eval()
                if do_ema:
                    net.start_test()

                synth_acc, synth_avg_ed, synth_avg_no_stop_ed, synth_avg_loss = test(net, synth_eval_data,
                                                                                     synth_eval_data.get_lexicon(),
                                                                                     cuda,
                                                                                     batch_size=batch_size,
                                                                                     visualize=False,
                                                                                     tb_writer=tb_writer,
                                                                                     n_iter=total_iter,
                                                                                     initial_title='val_synth',
                                                                                     loss_function=loss_function,
                                                                                     output_path=os.path.join(
                                                                                         output_dir, 'results'),
                                                                                     do_beam_search=False)


                orig_acc, orig_avg_ed, orig_avg_no_stop_ed, orig_avg_loss = test(net, orig_eval_data, orig_eval_data.get_lexicon(),
                                             cuda,
                                             batch_size=batch_size,
                                             visualize=False,
                                             tb_writer=tb_writer, n_iter=total_iter,
                                             initial_title='test_orig',
                                             loss_function=loss_function,
                                             output_path=os.path.join(output_dir, 'results'),
                                                                                 do_beam_search=do_beam_search)


                net = net.train()
                #save periodic
                if output_dir is not None and total_iter // 30000:
                    periodic_save = os.path.join(output_dir, 'periodic_save')
                    os.makedirs(periodic_save, exist_ok=True)
                    old_save = glob.glob(os.path.join(periodic_save,'*'))

                    torch.save(net.state_dict(), os.path.join(output_dir, "crnn_" + backend + "_" + str(total_iter)))

                if orig_avg_no_stop_ed < orig_avg_ed_best:
                    orig_avg_ed_best = orig_avg_no_stop_ed
                    if output_dir is not None:
                        torch.save(net.state_dict(), os.path.join(output_dir, "crnn_" + backend + "_best"))

                if synth_avg_no_stop_ed < synth_avg_ed_best:
                    synth_avg_ed_best = synth_avg_no_stop_ed
                if do_ema:
                    net.end_test()
                print("synth: avg_ed_best: {}\t avg_ed: {}; avg_nostop_ed: {}; acc: {}".format(synth_avg_ed_best,
                                                                                               synth_avg_ed,
                                                                                               synth_avg_no_stop_ed,
                                                                                               synth_acc))
                print("orig: avg_ed_best: {}\t avg_ed: {}; avg_nostop_ed: {}; acc: {}".format(orig_avg_ed_best,
                                                                           orig_avg_ed,
                                                                           orig_avg_no_stop_ed,
                                                                           orig_acc))
                tb_writer.get_writer().add_scalars('data/test',
                                                   {'synth_ed_total': synth_avg_ed,
                                                    'synth_ed_no_stop': synth_avg_no_stop_ed,
                                                    'synth_avg_loss': synth_avg_loss,
                                                    'orig_ed_total': orig_avg_ed,
                                                    'orig_ed_no_stop': orig_avg_no_stop_ed,
                                                    'orig_avg_loss': orig_avg_loss
                                                    }, total_iter)
                if len(loss_mean_ctc) > 0:
                    train_dict = {'mean_ctc_loss': np.mean(loss_mean_ctc)}
                    if do_vat:
                        train_dict = {**train_dict, **{'mean_vat_loss':np.mean(loss_mean_vat)}}
                    if do_at:
                        train_dict = {**train_dict, **{'mean_at_loss':np.mean(loss_mean_at)}}
                    if do_test_vat:
                        train_dict = {**train_dict, **{'mean_test_vat_loss': np.mean(loss_mean_test_vat)}}
                    if do_test_vat_rnn and do_test_vat_cnn:
                        train_dict = {**train_dict, **{'mean_test_vat_crnn_loss': np.mean(loss_mean_test_vat)}}
                    elif do_test_vat_rnn:
                        train_dict = {**train_dict, **{'mean_test_vat_rnn_loss': np.mean(loss_mean_test_vat)}}
                    elif do_test_vat_cnn:
                        train_dict = {**train_dict, **{'mean_test_vat_cnn_loss': np.mean(loss_mean_test_vat)}}
                    if ada_after_rnn:
                        train_dict = {**train_dict,
                                      **{'mean_ada_rnn_s_loss': np.mean(loss_mean_ada_rnn_s),
                                         'mean_ada_rnn_t_loss': np.mean(loss_mean_ada_rnn_t)}}
                    if ada_before_rnn:
                        train_dict = {**train_dict,
                                      **{'mean_ada_cnn_s_loss': np.mean(loss_mean_ada_cnn_s),
                                         'mean_ada_cnn_t_loss': np.mean(loss_mean_ada_cnn_t)}}
                    print(train_dict)
                    tb_writer.get_writer().add_scalars('data/train',
                                                       train_dict,
                                                       total_iter)
            '''
            # for multi-gpu support
            if sample["img"].size(0) % len(gpu.split(',')) != 0:
                continue
            '''
            optimizer.zero_grad()
            imgs = Variable(sample["img"])
            #print("images sizes are:")
            #print(sample["img"].shape)
            if do_vat or ada_after_rnn or ada_before_rnn:
                mask = sample['mask']
            labels_flatten = Variable(sample["seq"]).view(-1)
            label_lens = Variable(sample["seq_len"].int())
            #print("image sequence length is:")
            #print(sample["im_seq_len"])
            #print("label sequence length is:")
            #print(sample["seq_len"].view(1,-1))
            img_seq_lens = sample["im_seq_len"]
            if cuda:
                imgs = imgs.cuda()
                if do_vat or ada_after_rnn or ada_before_rnn:
                    mask = mask.cuda()

            if do_ada_lr:
                ada_p = float(iter_count) / max_iter
                lr_scheduler.update(ada_p)

            if ada_before_rnn or ada_after_rnn:
                if not do_ada_lr:
                    ada_p = float(iter_count) / max_iter
                ada_alpha = 2. / (1. + np.exp(-10. * ada_p)) - 1


                if cur_ada >= ada_len:
                    ada_load = DataLoader(orig_ada_data, batch_size=batch_size, num_workers=4, shuffle=True,
                                          collate_fn=collate_ada)
                    ada_len = len(ada_load)
                    cur_ada = 0
                    ada_iter = iter(ada_load)
                ada_batch = next(ada_iter)
                cur_ada += 1
                ada_imgs = Variable(ada_batch["img"])
                ada_img_seq_lens = ada_batch["im_seq_len"]
                ada_mask = ada_batch['mask'].byte()
                if cuda:
                    ada_imgs = ada_imgs.cuda()

                _, ada_cnn, ada_rnn = net(ada_imgs, ada_img_seq_lens,
                                          ada_alpha=ada_alpha, mask=ada_mask)
                if ada_before_rnn:
                    ada_num_features = ada_cnn.size(0)
                else:
                    ada_num_features = ada_rnn.size(0)
                domain_label = torch.zeros(ada_num_features)
                domain_label = domain_label.long()
                if cuda:
                    domain_label = domain_label.cuda()
                domain_label = Variable(domain_label)

                if ada_before_rnn:
                    err_ada_cnn_t = loss_domain(ada_cnn, domain_label)
                if ada_after_rnn:
                    err_ada_rnn_t = loss_domain(ada_rnn, domain_label)

            if do_test_vat and do_at:
                # test part!
                if cur_vat >= vat_len:
                    vat_load = DataLoader(orig_vat_data, batch_size=batch_size, num_workers=4, shuffle=True,
                                          collate_fn=collate_vat)
                    vat_len = len(vat_load)
                    cur_vat = 0
                    vat_iter = iter(vat_load)
                test_vat_batch = next(vat_iter)
                cur_vat += 1
                test_vat_mask = test_vat_batch['mask']
                test_vat_imgs = Variable(test_vat_batch["img"])
                test_vat_img_seq_lens = test_vat_batch["im_seq_len"]
                if cuda:
                    test_vat_imgs = test_vat_imgs.cuda()
                    test_vat_mask = test_vat_mask.cuda()
                # train part
                at_test_vat_loss = LabeledAtAndUnlabeledTestVatLoss(xi=vat_xi, eps=vat_epsilon, ip=vat_ip)

                at_loss, test_vat_loss = at_test_vat_loss(model=net, train_x=imgs, train_labels_flatten=labels_flatten,
                                 train_img_seq_lens=img_seq_lens, train_label_lens=label_lens, batch_size=batch_size,
                                test_x=test_vat_imgs, test_seq_len=test_vat_img_seq_lens, test_mask=test_vat_mask)
            elif do_test_vat or do_test_vat_rnn or do_test_vat_cnn:
                if cur_vat >= vat_len:
                    vat_load = DataLoader(orig_vat_data, batch_size=batch_size, num_workers=4, shuffle=True,
                                          collate_fn=collate_vat)
                    vat_len = len(vat_load)
                    cur_vat = 0
                    vat_iter = iter(vat_load)
                vat_batch = next(vat_iter)
                cur_vat += 1
                vat_mask = vat_batch['mask']
                vat_imgs = Variable(vat_batch["img"])
                vat_img_seq_lens = vat_batch["im_seq_len"]
                if cuda:
                    vat_imgs = vat_imgs.cuda()
                    vat_mask = vat_mask.cuda()
                if do_test_vat:
                    if do_test_vat_rnn or do_test_vat_cnn:
                        raise "can only do one of do_test_vat | (do_test_vat_rnn, do_test_vat_cnn)"
                    if vat_sign == True:
                        test_vat_loss = VATLossSign(do_test_entropy=do_test_entropy, xi=vat_xi, eps=vat_epsilon, ip=vat_ip)
                    else:
                        test_vat_loss = VATLoss(xi=vat_xi, eps=vat_epsilon, ip=vat_ip)
                elif do_test_vat_rnn and do_test_vat_cnn:
                    test_vat_loss = VATonRnnCnnSign(xi=vat_xi, eps=vat_epsilon, ip=vat_ip)
                elif do_test_vat_rnn:
                    test_vat_loss = VATonRnnSign(xi=vat_xi, eps=vat_epsilon, ip=vat_ip)
                elif do_test_vat_cnn:
                    test_vat_loss = VATonCnnSign(xi=vat_xi, eps=vat_epsilon, ip=vat_ip)
                if do_test_vat_cnn and do_test_vat_rnn:
                    test_vat_loss, cnn_lds, rnn_lds = test_vat_loss(net, vat_imgs, vat_img_seq_lens, vat_mask)
                elif do_test_vat:
                    test_vat_loss = test_vat_loss(net, vat_imgs, vat_img_seq_lens, vat_mask)
            elif do_vat:
                vat_loss = VATLoss(xi=vat_xi, eps=vat_epsilon, ip=vat_ip)
                vat_loss = vat_loss(net, imgs, img_seq_lens, mask)
            elif do_at:
                at_loss = LabeledATLoss(xi=vat_xi, eps=vat_epsilon, ip=vat_ip)
                at_loss = at_loss(net, imgs, labels_flatten, img_seq_lens, label_lens, batch_size)


            if ada_after_rnn or ada_before_rnn:
                preds, ada_cnn, ada_rnn = net(imgs, img_seq_lens, ada_alpha=ada_alpha, mask=mask)

                if ada_before_rnn:
                    ada_num_features = ada_cnn.size(0)
                else:
                    ada_num_features = ada_rnn.size(0)

                domain_label = torch.ones(ada_num_features)
                domain_label = domain_label.long()
                if cuda:
                    domain_label = domain_label.cuda()
                domain_label = Variable(domain_label)

                if ada_before_rnn:
                    err_ada_cnn_s = loss_domain(ada_cnn, domain_label)
                if ada_after_rnn:
                    err_ada_rnn_s = loss_domain(ada_rnn, domain_label)

            else:
                preds = net(imgs, img_seq_lens)

            '''
            if output_dir is not None:
                if (show_iter is not None and iter_num != 0 and iter_num % show_iter == 0):
                    print_data_visuals(net, tb_writer, train_data.get_lexicon(), sample["img"], labels_flatten, label_lens,
                                       preds, ((epoch_count * len(data_loader)) + iter_num))
            '''
            loss_ctc = loss_function(preds, labels_flatten,
                        Variable(torch.IntTensor(np.array(img_seq_lens))), label_lens) / batch_size

            if loss_ctc.data[0] in [float("inf"), -float("inf")]:
                print("warnning: loss should not be inf.")
                continue
            total_loss = loss_ctc


            if do_vat:
                #mask = sample['mask']
                #if cuda:
                #    mask = mask.cuda()
                #vat_loss = virtual_adversarial_loss(net, imgs, img_seq_lens, mask, is_training=True, do_entropy=False, epsilon=vat_epsilon, num_power_iterations=1,
                #             xi=1e-6, average_loss=True)
                total_loss = total_loss + vat_ratio * vat_loss.cpu()
            if do_test_vat or do_test_vat_rnn or do_test_vat_cnn:
                total_loss = total_loss + test_vat_ratio * test_vat_loss.cpu()

            if ada_before_rnn:
                total_loss = total_loss + ada_ratio * err_ada_cnn_s.cpu() + ada_ratio * err_ada_cnn_t.cpu()
            if ada_after_rnn:
                total_loss = total_loss + ada_ratio * err_ada_rnn_s.cpu() + ada_ratio * err_ada_rnn_t.cpu()

            total_loss.backward()
            nn.utils.clip_grad_norm(net.parameters(), 10.0)
            if -400 < loss_ctc.data[0] < 400:
                loss_mean_ctc.append(loss_ctc.data[0])
            if -1000 < total_loss.data[0] < 1000:
                loss_mean_total.append(total_loss.data[0])
            if len(loss_mean_total) > 100:
                loss_mean_total = loss_mean_total[-100:]
            status = "epoch: {0:5d}; iter_num: {1:5d}; lr: {2:.2E}; loss_mean: {3:.3f}; loss: {4:.3f}".format(epoch_count,
                                                                                   lr_scheduler.last_iter,
                                                                                   lr_scheduler.get_lr(),
                                                                                   np.mean(loss_mean_total),
                                                                                   loss_ctc.data[0])
            if ada_after_rnn:
                loss_mean_ada_rnn_s.append(err_ada_rnn_s.data[0])
                loss_mean_ada_rnn_t.append(err_ada_rnn_t.data[0])
                status += "; ladatrnns: {0:.3f}; ladatrnnt: {1:.3f}".format(
                    err_ada_rnn_s.data[0], err_ada_rnn_t.data[0]
                )
            if ada_before_rnn:
                loss_mean_ada_cnn_s.append(err_ada_cnn_s.data[0])
                loss_mean_ada_cnn_t.append(err_ada_cnn_t.data[0])
                status += "; ladatcnns: {0:.3f}; ladatcnnt: {1:.3f}".format(
                    err_ada_cnn_s.data[0], err_ada_cnn_t.data[0]
                )
            if do_vat:
                loss_mean_vat.append(vat_loss.data[0])
                status += "; lvat: {0:.3f}".format(
                   vat_loss.data[0]
                )
            if do_at:
                loss_mean_at.append(at_loss.data[0])
                status += "; lat: {0:.3f}".format(
                   at_loss.data[0]
                )
            if do_test_vat:
                loss_mean_test_vat.append(test_vat_loss.data[0])
                status += "; l_tvat: {0:.3f}".format(
                    test_vat_loss.data[0]
                )
            if do_test_vat_rnn or do_test_vat_cnn:
                loss_mean_test_vat.append(test_vat_loss.data[0])
                if do_test_vat_rnn and do_test_vat_cnn:
                    status += "; l_tvatc: {}".format(
                        cnn_lds.data[0]
                    )
                    status += "; l_tvatr: {}".format(
                        rnn_lds.data[0]
                    )
                else:
                    status += "; l_tvat: {}".format(
                        test_vat_loss.data[0]
                    )

            iterator.set_description(status)
            optimizer.step()
            if do_lr_step:
                lr_scheduler.step()
            if do_ema:
                net.udate_ema()
            iter_count += 1
        if output_dir is not None:
            torch.save(net.state_dict(), os.path.join(output_dir, "crnn_" + backend + "_last"))
        epoch_count += 1

    return

if __name__ == '__main__':
    main()

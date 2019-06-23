import os
import click
import string
import numpy as np
from scipy import stats
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
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
from warpctc_pytorch import CTCLoss
from models.crnn import CRNN

from torchvision.utils import make_grid

from test_multinet import test, print_data_visuals
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
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-height', type=int, default=64, help='Input size')
@click.option('--base-lr', type=float, default=1e-4, help='Base learning rate') # was e-3
#@click.option('--lr-decay', type=float, default=1e-4, help='Base learning rate') # was 0.0001
@click.option('--elastic-alpha', type=float, default=34, help='Base learning rate')
@click.option('--elastic-sigma', type=float, default=3, help='Base learning rate')
@click.option('--step-size', type=int, default=500, help='Step size')
@click.option('--max-iter', type=int, default=6000, help='Max iterations')
@click.option('--batch-size', type=int, default=8, help='Batch size')
@click.option('--output-dir', type=str,
              default='../Output/exp1',
              help='Path for snapshot')
@click.option('--test-iter', type=int, default=1000, help='Test epoch')
@click.option('--show-iter', type=int, default=1000, help='Test epoch')
@click.option('--test-init', type=bool, default=False, help='Test initialization')
@click.option('--use-gpu', type=bool, default=True, help='Whether to use the gpu')
@click.option('--use-no-font-repeat-data', type=bool, default=True, help='Whether to use the gpu')
@click.option('--do-vat', type=bool, default=False, help='Whether to do vat')
@click.option('--do-at', type=bool, default=False, help='Whether to do vat')
@click.option('--vat-ratio', type=float, default=1, help='ratio on vat loss in loss')
@click.option('--test-vat-ratio', type=float, default=1, help='ratio on vat loss in loss')
@click.option('--vat-epsilon', type=float, default=2.5, help='ratio on vat loss in loss')
@click.option('--vat-ip', type=int, default=1, help='ratio on vat loss in loss')
@click.option('--vat-xi', type=float, default=10., help='ratio on vat loss in loss')
@click.option('--vat-sign', type=bool, default=False, help='Whether to do sign in vat')
@click.option('--do-comp', type=bool, default=False, help='Whether to do comparison loss')
@click.option('--comp-ratio', type=float, default=1, help='ratio on comparison loss in loss')
@click.option('--do-remove-augs', type=bool, default=False, help='Whether to remove some of the augmentations')
@click.option('--aug-to-remove', type=str,
              default='',
              help="['elastic', 'sine', 'sine_rotate', 'rotation', 'color_aug', 'color_gaus', 'color_sine']")
@click.option('--do-beam-search', type=bool, default=False, help='Visualize output')
@click.option('--dropout-conv', type=bool, default=False, help='Visualize output')
@click.option('--dropout-rnn', type=bool, default=False, help='Visualize output')
@click.option('--dropout-output', type=bool, default=False, help='Visualize output')
@click.option('--do-ema', type=bool, default=False, help='Visualize output')
@click.option('--do-gray', type=bool, default=False, help='Visualize output')
@click.option('--do-test-vat', type=bool, default=False, help='Visualize output')
@click.option('--do-test-entropy', type=bool, default=False, help='Visualize output')
@click.option('--do-test-vat-cnn', type=bool, default=False, help='Visualize output')
@click.option('--do-test-vat-rnn', type=bool, default=False, help='Visualize output')
@click.option('--do-test-rand', type=bool, default=False, help='Visualize output')
@click.option('--ada-after-rnn', type=bool, default=False, help='Visualize output')
@click.option('--ada-before-rnn', type=bool, default=False, help='Visualize output')
@click.option('--do-ada-lr', type=bool, default=False, help='Visualize output')
@click.option('--ada-ratio', type=float, default=1, help='ratio on comparison loss in loss')
@click.option('--rnn-hidden-size', type=int, default=128, help='rnn_hidden_size')
@click.option('--do-test-pseudo', type=bool, default=False, help='Visualize output')
@click.option('--test-pseudo-ratio', type=float, default=10., help='ratio on vat loss in loss')
@click.option('--test-pseudo-thresh', type=float, default=0.9, help='ratio on vat loss in loss')
@click.option('--do-lr-step', type=bool, default=False, help='Visualize output')
@click.option('--do-test-ensemble', type=bool, default=False, help='Visualize output')
@click.option('--test-ensemble-ratio', type=float, default=10., help='ratio on vat loss in loss')
@click.option('--test-ensemble-thresh', type=float, default=0.9, help='ratio on vat loss in loss')
def main(base_data_dir, train_data_path, train_base_dir,
         orig_eval_data_path, orig_eval_base_dir,
         synth_eval_data_path, synth_eval_base_dir,
         lexicon_path, seq_proj, backend, snapshot, input_height, base_lr, elastic_alpha, elastic_sigma,
         step_size, max_iter,
         batch_size, output_dir, test_iter, show_iter, test_init, use_gpu, use_no_font_repeat_data,
         do_vat, do_at, vat_ratio, test_vat_ratio, vat_epsilon, vat_ip, vat_xi, vat_sign,
         do_comp, comp_ratio, do_remove_augs, aug_to_remove, do_beam_search,
         dropout_conv, dropout_rnn, dropout_output, do_ema, do_gray, do_test_vat, do_test_entropy, do_test_vat_cnn,
         do_test_vat_rnn, do_test_rand,
         ada_after_rnn, ada_before_rnn, do_ada_lr, ada_ratio, rnn_hidden_size, do_test_pseudo,
         test_pseudo_ratio, test_pseudo_thresh, do_lr_step, do_test_ensemble,
         test_ensemble_ratio, test_ensemble_thresh
         ):
    num_nets = 4

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
    if do_test_ensemble:
        orig_vat_data = TextDataset(data_path=orig_eval_data_path, lexicon=lexicon,
                                     base_path=orig_eval_base_dir, transform=transform_simple, fonts=None)

    #else:
    #    train_data = TestDataset(transform=transform, abc=abc).set_mode("train")
    #    synth_eval_data = TestDataset(transform=transform, abc=abc).set_mode("test")
    #    orig_eval_data = TestDataset(transform=transform, abc=abc).set_mode("test")
    seq_proj = [int(x) for x in seq_proj.split('x')]
    nets = []
    optimizers = []
    lr_schedulers = []
    for neti in range(num_nets):
        nets.append(load_model(lexicon=train_data.get_lexicon(), seq_proj=seq_proj, backend=backend,
                         snapshot=snapshot, cuda=cuda, do_beam_search=do_beam_search,
                         dropout_conv=dropout_conv,
                         dropout_rnn=dropout_rnn,
                         dropout_output=dropout_output,
                         do_ema=do_ema,
                         ada_after_rnn=ada_after_rnn, ada_before_rnn=ada_before_rnn,
                         rnn_hidden_size=rnn_hidden_size, gpu=neti
                         ))
        optimizers.append(optim.Adam(nets[neti].parameters(), lr = base_lr, weight_decay=0.0001))
        lr_schedulers.append(StepLR(optimizers[neti], step_size=step_size, max_iter=max_iter))
    loss_function = CTCLoss()

    synth_avg_ed_best = float("inf")
    orig_avg_ed_best = float("inf")
    epoch_count = 0

    if do_test_ensemble:
        collate_vat = lambda x: text_collate(x, do_mask=True)
        vat_load = DataLoader(orig_vat_data, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate_vat)
        vat_len = len(vat_load)
        cur_vat = 0
        vat_iter = iter(vat_load)

    loss_domain = torch.nn.NLLLoss()

    while True:
        collate = lambda x: text_collate(x, do_mask=(do_vat or ada_before_rnn or ada_after_rnn))
        data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate)
        if do_comp:
            data_loader_comp = DataLoader(train_data_comp, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=collate_comp)
            iter_comp = iter(data_loader_comp)

        loss_mean_ctc = []
        loss_mean_total = []
        loss_mean_test_ensemble = []
        num_labels_used_total = 0
        iterator = tqdm(data_loader)
        nll_loss = torch.nn.NLLLoss()
        iter_count = 0
        for iter_num, sample in enumerate(iterator):
            total_iter = (epoch_count * len(data_loader)) + iter_num
            if ((total_iter > 1) and total_iter % test_iter == 0) or (test_init and total_iter == 0):
                # epoch_count != 0 and

                print("Test phase")
                for net in nets:
                    net = net.eval()
                    if do_ema:
                        net.start_test()

                synth_acc, synth_avg_ed, synth_avg_no_stop_ed, synth_avg_loss = test(nets, synth_eval_data,
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


                orig_acc, orig_avg_ed, orig_avg_no_stop_ed, orig_avg_loss = test(nets, orig_eval_data, orig_eval_data.get_lexicon(),
                                             cuda,
                                             batch_size=batch_size,
                                             visualize=False,
                                             tb_writer=tb_writer, n_iter=total_iter,
                                             initial_title='test_orig',
                                             loss_function=loss_function,
                                             output_path=os.path.join(output_dir, 'results'),
                                             do_beam_search=do_beam_search)

                for net in nets:
                    net = net.train()
                #save periodic
                if output_dir is not None and total_iter // 30000:
                    periodic_save = os.path.join(output_dir, 'periodic_save')
                    os.makedirs(periodic_save, exist_ok=True)
                    old_save = glob.glob(os.path.join(periodic_save,'*'))
                    for neti, net in enumerate(nets):
                        torch.save(net.state_dict(), os.path.join(output_dir, "crnn_{}_".format(neti) + backend + "_" + str(total_iter)))

                if orig_avg_no_stop_ed < orig_avg_ed_best:
                    orig_avg_ed_best = orig_avg_no_stop_ed
                if output_dir is not None:
                    for neti, net in enumerate(nets):
                        torch.save(net.state_dict(), os.path.join(output_dir, "crnn_{}_".format(neti)
                                                                  + backend + "_iter_{}".format(total_iter)))

                if synth_avg_no_stop_ed < synth_avg_ed_best:
                    synth_avg_ed_best = synth_avg_no_stop_ed
                if do_ema:
                    for net in nets:
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
                    train_dict = {**train_dict, **{'mean_test_ensemble_loss': np.mean(loss_mean_test_ensemble)}}
                    train_dict = {**train_dict, **{'num_labels_used': num_labels_used_total}}
                    num_labels_used_total = 0
                    print(train_dict)
                    tb_writer.get_writer().add_scalars('data/train',
                                                       train_dict,
                                                       total_iter)
            '''
            # for multi-gpu support
            if sample["img"].size(0) % len(gpu.split(',')) != 0:
                continue
            '''
            for optimizer in optimizers:
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

            if do_test_ensemble:
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
                all_net_classes = []
                all_net_preds = []
                def run_net_get_classes(neti_net_pair, cur_vat_imgs,
                                        cur_vat_mask, cur_vat_img_seq_lens, cuda):
                    neti, net = neti_net_pair
                    if cuda:
                        cur_vat_imgs = cur_vat_imgs.cuda(neti)
                        cur_vat_mask = cur_vat_mask.cuda(neti)
                    vat_pred = net.vat_forward(cur_vat_imgs, cur_vat_img_seq_lens)
                    vat_pred = vat_pred * cur_vat_mask
                    vat_pred = F.softmax(vat_pred, dim=2).view(-1, vat_pred.size()[-1])
                    all_net_preds.append(vat_pred)
                    np_vat_preds = vat_pred.cpu().data.numpy()
                    classes_by_index = np.argmax(np_vat_preds, axis=1)
                    return classes_by_index
                for neti, net in enumerate(nets):
                    if cuda:
                        vat_imgs = vat_imgs.cuda(neti)
                        vat_mask = vat_mask.cuda(neti)
                    vat_pred = net.vat_forward(vat_imgs, vat_img_seq_lens)
                    vat_pred = vat_pred * vat_mask
                    vat_pred = F.softmax(vat_pred, dim=2).view(-1, vat_pred.size()[-1])
                    all_net_preds.append(vat_pred)
                    np_vat_preds = vat_pred.cpu().data.numpy()
                    classes_by_index = np.argmax(np_vat_preds, axis=1)
                    all_net_classes.append(classes_by_index)
                all_net_classes = np.stack(all_net_classes)
                all_net_classes, all_nets_count = stats.mode(all_net_classes, axis=0)
                all_net_classes = all_net_classes.reshape(-1)
                all_nets_count = all_nets_count.reshape(-1)
                ens_indices = np.argwhere(all_nets_count > test_ensemble_thresh)
                ens_indices = ens_indices.reshape(-1)
                ens_classes = all_net_classes[all_nets_count > test_ensemble_thresh]
                net_ens_losses = []
                num_labels_used = len(ens_indices)
                for neti, net in enumerate(nets):
                    indices = Variable(torch.from_numpy(ens_indices).cuda(neti))
                    labels = Variable(torch.from_numpy(ens_classes).cuda(neti))
                    net_preds_to_ens = all_net_preds[neti][indices]
                    loss = nll_loss(net_preds_to_ens, labels)
                    net_ens_losses.append(loss.cpu())
            nets_total_losses = []
            nets_ctc_losses = []
            loss_is_inf = False
            for neti, net in enumerate(nets):
                if cuda:
                    imgs = imgs.cuda(neti)
                preds = net(imgs, img_seq_lens)
                loss_ctc = loss_function(preds, labels_flatten,
                        Variable(torch.IntTensor(np.array(img_seq_lens))), label_lens) / batch_size

                if loss_ctc.data[0] in [float("inf"), -float("inf")]:
                    print("warnning: loss should not be inf.")
                    loss_is_inf = True
                    break
                total_loss = loss_ctc

                if do_test_ensemble:
                    total_loss = total_loss + test_ensemble_ratio * net_ens_losses[neti]
                    net_ens_losses[neti] = net_ens_losses[neti].data[0]
                total_loss.backward()
                nets_total_losses.append(total_loss.data[0])
                nets_ctc_losses.append(loss_ctc.data[0])
                nn.utils.clip_grad_norm(net.parameters(), 10.0)
            if loss_is_inf:
                continue
            if -400 < loss_ctc.data[0] < 400:
                loss_mean_ctc.append(np.mean(nets_ctc_losses))
            if -400 < total_loss.data[0] < 400:
                loss_mean_total.append(np.mean(nets_total_losses))
            status = "epoch: {0:5d}; iter_num: {1:5d}; lr: {2:.2E}; loss_mean: {3:.3f}; loss: {4:.3f}".format(epoch_count,
                                                                                   lr_schedulers[0].last_iter,
                                                                                   lr_schedulers[0].get_lr(),
                                                                                   np.mean(nets_total_losses),
                                                                                   np.mean(nets_ctc_losses))

            if do_test_ensemble:
                ens_loss = np.mean(net_ens_losses)
                if ens_loss != 0:
                    loss_mean_test_ensemble.append(ens_loss)
                    status += "; loss_ens: {0:.3f}".format(
                        ens_loss
                    )
                    status += "; num_ens_used {}".format(
                        num_labels_used
                    )
                else:
                    loss_mean_test_ensemble.append(0)
                    status += "; loss_ens: {}".format(
                        0
                    )
            iterator.set_description(status)
            for optimizer in optimizers:
                optimizer.step()
            if do_lr_step:
                for lr_scheduler in lr_schedulers:
                    lr_scheduler.step()
            iter_count += 1
        if output_dir is not None:
            for neti, net in enumerate(nets):
                torch.save(net.state_dict(), os.path.join(output_dir, "crnn_{}_".format(neti) + backend + "_last"))
        epoch_count += 1

    return

if __name__ == '__main__':
    main()

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from warpctc_pytorch import CTCLoss
from torch.autograd import Variable
import numpy as np

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def _entropy(logits, mask):
    p = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum((p * F.log_softmax(logits, dim=1)), dim=1) * mask)

def _kl_div(log_probs, probs, mask=None):
    # pytorch KLDLoss is averaged over all dim if size_average=True
    if mask is not None:
        kld = F.kl_div(log_probs, probs, size_average=False, reduce=False)
        kld = mask.view(-1) * kld.sum(1)
        kld = kld.sum() / mask.sum()
    else:
        kld = F.kl_div(log_probs, probs, size_average=False)
        kld = kld / log_probs.shape[0]
    return kld

class LabeledATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(LabeledATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, labels_flatten, img_seq_lens, label_lens, batch_size):
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            # prepare random unit tensor
            d = torch.rand(x.shape).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            d = _l2_normalize(d)
            for _ in range(self.ip):
                d.requires_grad_()
                loss_function = CTCLoss()
                preds = model.forward(x + self.xi * d, img_seq_lens)
                adv_loss_ctc = loss_function(preds, labels_flatten,
                                         Variable(torch.IntTensor(np.array(img_seq_lens))), label_lens) / batch_size

                adv_loss_ctc.backward()
                d = d.grad
                model.zero_grad()

            # calc LDS
            r_adv = torch.sign(d) * self.eps

            pred_hat = model.forward(x + r_adv, img_seq_lens)
            lds = loss_function(pred_hat, labels_flatten,
                                         Variable(torch.IntTensor(np.array(img_seq_lens))), label_lens) / batch_size

        return lds

class LabeledAtAndUnlabeledTestVatLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1, unlabeled_ratio=10.):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(LabeledAtAndUnlabeledTestVatLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.unlabeled_ratio = unlabeled_ratio

    def forward(self, model, train_x, train_labels_flatten, train_img_seq_lens, train_label_lens, batch_size,
                test_x, test_seq_len, test_mask
                ):
        with _disable_tracking_bn_stats(model):
            # TRAIN
            # calc adversarial direction
            # prepare random unit tensor
            train_d = torch.rand(train_x.shape).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            train_d = _l2_normalize(train_d)
            for _ in range(self.ip):
                train_d.requires_grad_()
                train_loss_function = CTCLoss()
                train_preds = model.forward(train_x + self.xi * train_d, train_img_seq_lens)
                train_adv_loss_ctc = train_loss_function(train_preds, train_labels_flatten,
                                         Variable(torch.IntTensor(np.array(train_img_seq_lens))), train_label_lens) / batch_size

                train_adv_loss_ctc.backward()
                train_d = train_d.grad
                model.zero_grad()

            #TEST
            with torch.no_grad():
                test_pred = model.vat_forward(test_x, test_seq_len)
                test_pred = test_pred * test_mask
                test_pred = F.softmax(test_pred, dim=2).view(-1, test_pred.size()[-1])
                # prepare random unit tensor
            test_d = torch.rand(test_x.shape).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            test_d = _l2_normalize(test_d)

            with _disable_tracking_bn_stats(model):
                # calc adversarial direction
                for _ in range(self.ip):
                    test_d.requires_grad_()

                    test_pred_hat = model.vat_forward(test_x + self.xi * test_d, test_seq_len)
                    test_pred_hat = test_pred_hat * test_mask
                    test_pred_hat = F.log_softmax(test_pred_hat, dim=2).view(-1, test_pred_hat.size()[-1])

                    # pred_hat = model(x + self.xi * d)
                    # adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                    test_adv_distance = _kl_div(test_pred_hat, test_pred)
                    test_adv_distance.backward()
                    test_d = _l2_normalize(test_d.grad)
                    model.zero_grad()

            #TRAIN
            # calc LDS
            train_r_adv = torch.sign(train_d) * self.eps

            train_pred_hat = model.forward(train_x + train_r_adv, train_img_seq_lens)
            train_lds = train_loss_function(train_pred_hat, train_labels_flatten,
                                         Variable(torch.IntTensor(np.array(train_img_seq_lens))), train_label_lens) / batch_size

            #TEST
            # calc LDS
            test_d = torch.sign(test_d)
            test_r_adv = test_d * self.eps

            test_pred_hat = model.vat_forward(test_x + test_r_adv, test_seq_len)
            test_pred_hat = test_pred_hat * test_mask
            test_pred_hat = F.log_softmax(test_pred_hat, dim=2).view(-1, test_pred_hat.size()[-1])

            #pred_hat = model(x + r_adv)
            #lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            test_lds = _kl_div(test_pred_hat, test_pred)

        return train_lds, test_lds

class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, seq_len, mask):
        with torch.no_grad():
            pred = model.vat_forward(x, seq_len)
            pred = pred * mask
            pred = F.softmax(pred, dim=2).view(-1, pred.size()[-1])
        # prepare random unit tensor
        d = torch.rand(x.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()

                pred_hat = model.vat_forward(x + self.xi * d, seq_len)
                pred_hat = pred_hat * mask
                pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

                #pred_hat = model(x + self.xi * d)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = _kl_div(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps

            pred_hat = model.vat_forward(x + r_adv, seq_len)
            pred_hat = pred_hat * mask
            pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

            #pred_hat = model(x + r_adv)
            #lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            lds = _kl_div(pred_hat, pred)

        return lds

class VATonRnnSign(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATonRnnSign, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, seq_len, mask):
        with torch.no_grad():
            x_features = model.vat_forward_cnn(x)
            pred = model.vat_forward_rnn(x_features, seq_len)
            pred = pred * mask
            pred = F.softmax(pred, dim=2).view(-1, pred.size()[-1])
        # prepare random unit tensor
        d = torch.rand(x_features.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()

                pred_hat = model.vat_forward_rnn(x_features + self.xi * d, seq_len)
                pred_hat = pred_hat * mask
                pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

                #pred_hat = model(x + self.xi * d)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = _kl_div(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            d = torch.sign(d)
            r_adv = d * self.eps

            pred_hat = model.vat_forward_rnn(x_features + r_adv, seq_len)
            pred_hat = pred_hat * mask
            pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

            #pred_hat = model(x + r_adv)
            #lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            lds = _kl_div(pred_hat, pred)

        return lds

class VATonCnnSign(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATonCnnSign, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, seq_len, mask):
        with torch.no_grad():
            x_features = model.vat_forward(x)
            pred = model.vat_forward_rnn(x, seq_len)
            pred = pred * mask
            pred = F.softmax(pred, dim=2).view(-1, pred.size()[-1])
        # prepare random unit tensor
        d = torch.rand(x.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()

                pred_hat = model.vat_forward(x + self.xi * d, seq_len)
                pred_hat = pred_hat * mask
                pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

                #pred_hat = model(x + self.xi * d)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = _kl_div(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            d = torch.sign(d)
            r_adv = d * self.eps

            pred_features_hat = model.vat_forward_cnn(x + r_adv, seq_len)
            pred_features_hat = pred_features_hat * mask
            l2_loss = torch.nn.MSELoss()
            lds = l2_loss(pred_features_hat, x_features)
        return lds

class VATonRnnCnnSign(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATonRnnCnnSign, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, seq_len, mask):
        with torch.no_grad():
            x_features = model.vat_forward_cnn(x)
            x_pred = model.vat_forward_rnn(x.size(0), x_features, seq_len)
            x_pred = x_pred * mask
            x_pred = F.softmax(x_pred, dim=2).view(-1, x_pred.size()[-1])

        # prepare random unit tensor
        d_rnn = torch.rand(x_features.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d_rnn = _l2_normalize(d_rnn)
        d_cnn = torch.rand(x.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d_cnn = _l2_normalize(d_cnn)

        with _disable_tracking_bn_stats(model):
            ### Calc rnn d
            for _ in range(self.ip):
                d_rnn.requires_grad_()

                pred_hat = model.vat_forward_rnn(x.size(0), x_features + self.xi * d_rnn, seq_len)
                pred_hat = pred_hat * mask
                pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

                #pred_hat = model(x + self.xi * d)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = _kl_div(pred_hat, x_pred)
                adv_distance.backward()
                d_rnn = _l2_normalize(d_rnn.grad)
                model.zero_grad()
            # calc LDS
            d_rnn = torch.sign(d_rnn)
            r_adv_rnn = d_rnn * self.eps
            ### Calc Cnn d
            for _ in range(self.ip):
                d_cnn.requires_grad_()

                pred_hat = model.vat_forward(x + self.xi * d_cnn, seq_len)
                pred_hat = pred_hat * mask
                pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

                #pred_hat = model(x + self.xi * d)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = _kl_div(pred_hat, x_pred)
                adv_distance.backward()
                d_cnn = _l2_normalize(d_cnn.grad)
                model.zero_grad()

            d_cnn = torch.sign(d_cnn)
            r_adv_cnn = d_cnn * self.eps

            #calc rnn lds
            pred_hat = model.vat_forward_rnn(x.size(0), x_features + r_adv_rnn, seq_len)
            pred_hat = pred_hat * mask
            pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])
            rnn_lds = _kl_div(pred_hat, x_pred)

            # calc cnn lds
            pred_features_hat = model.vat_forward_cnn(x + r_adv_cnn)
            pred_features_hat = pred_features_hat * mask
            l2_loss = torch.nn.L1Loss()
            cnn_lds = l2_loss(pred_features_hat, x_features)

            lds = cnn_lds + rnn_lds

        return lds, cnn_lds, rnn_lds

class VATLossSign(nn.Module):
    def __init__(self, do_test_entropy, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLossSign, self).__init__()
        self.do_test_entropy = do_test_entropy
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, seq_len, mask):
        with torch.no_grad():
            pred = model.vat_forward(x, seq_len)
            pred = pred * mask
            pred = F.softmax(pred, dim=2).view(-1, pred.size()[-1])
        # prepare random unit tensor
        d = torch.rand(x.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()

                pred_hat = model.vat_forward(x + self.xi * d, seq_len)
                pred_hat = pred_hat * mask
                pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

                #pred_hat = model(x + self.xi * d)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = _kl_div(pred_hat, pred, mask)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            d = torch.sign(d)
            r_adv = d * self.eps

            pred_hat = model.vat_forward(x + r_adv, seq_len)
            pred_hat = pred_hat * mask
            pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

            #pred_hat = model(x + r_adv)
            #lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            lds = _kl_div(pred_hat, pred, mask)
            if self.do_test_entropy:
                lds += _entropy(pred_hat, mask)

        return lds

class VATLossSignOld(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLossSign, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, seq_len, mask):
        with torch.no_grad():
            pred = model.vat_forward(x, seq_len)
            pred = pred * mask
            pred = F.softmax(pred, dim=2).view(-1, pred.size()[-1])
        # prepare random unit tensor
        d = torch.rand(x.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()

                pred_hat = model.vat_forward(x + self.xi * d, seq_len)
                pred_hat = pred_hat * mask
                pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

                #pred_hat = model(x + self.xi * d)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = _kl_div(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            d = torch.sign(d)
            r_adv = d * self.eps

            pred_hat = model.vat_forward(x + r_adv, seq_len)
            pred_hat = pred_hat * mask
            pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

            #pred_hat = model(x + r_adv)
            #lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            lds = _kl_div(pred_hat, pred)

        return lds

class RandomLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(RandomLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, seq_len, mask):
        with torch.no_grad():
            pred = model.vat_forward(x, seq_len)
            pred = pred * mask
            pred = F.softmax(pred, dim=2).view(-1, pred.size()[-1])
        # prepare random unit tensor
        d = torch.rand(x.shape).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # calc LDS
        d = torch.sign(d)
        r_adv = d * self.eps

        pred_hat = model.vat_forward(x + r_adv, seq_len)
        pred_hat = pred_hat * mask
        pred_hat = F.log_softmax(pred_hat, dim=2).view(-1, pred_hat.size()[-1])

        #pred_hat = model(x + r_adv)
        #lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
        lds = _kl_div(pred_hat, pred)

        return lds


class PseudoLabel(nn.Module):
    def __init__(self, confidence_thresh):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(PseudoLabel, self).__init__()
        self.confidence_thresh = confidence_thresh
    def forward(self, model, x, seq_len, mask):
        pred = model.vat_forward(x, seq_len)
        pred = pred * mask
        pred = F.softmax(pred, dim=2).view(-1, pred.size()[-1])
        np_preds = pred.cpu().data.numpy()
        indices, classes = np.where(np_preds > self.confidence_thresh)
        if len(indices) > 0:
            indices = Variable(torch.from_numpy(indices).cuda())
            labels = Variable(torch.from_numpy(classes).cuda())
            strong_preds = pred[indices]
            nll_loss = torch.nn.NLLLoss()
            loss = nll_loss(strong_preds, labels)
            return loss.cpu()
        else:
            return 0

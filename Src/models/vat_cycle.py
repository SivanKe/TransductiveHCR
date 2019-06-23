import torch
from torch.autograd import Variable

def get_normalized_vector(d, divide_by_max=True):

    if divide_by_max:
        max_d = torch.abs(d)
        for i in range(1, len(d.size())):
            max_d, _ = torch.max(max_d, i, keepdim=True)
        d = d / (1e-12 + max_d.expand_as(d))
    sum_strq_d = torch.pow(d, 2)
    for i in range(1, len(d.size())):
        sum_strq_d = torch.sum(sum_strq_d, i, keepdim=True)
    d = d / torch.sqrt(1e-6 + sum_strq_d).expand_as(d)
    return d

def generate_virtual_adversarial_perturbation(net, x, seq_len, logit_p, is_training=True,
                                              epsilon=2.5, num_power_iterations=1, xi = 1e-6):
    d = torch.randn(*x.size()).cuda()
    d = get_normalized_vector(d)

    for _ in range(num_power_iterations):
        d_var = Variable(xi * d, requires_grad=True)
        logit_m = net.forward(x + d_var, seq_len)
        dist = kl_categorical(logit_m, logit_p) # input, target
        dist.backward()
        d = get_normalized_vector(d_var.grad.data.clone(), divide_by_max=True)
    return epsilon * d

def kl_categorical(p_logit, q_logit, average_loss=True):
    softmax = torch.nn.Softmax()
    logsoftmax = torch.nn.LogSoftmax()
    _kl = torch.sum(softmax(q_logit) * (logsoftmax(q_logit) - logsoftmax(p_logit)), 1)
    if average_loss:
        return torch.sum(_kl) / _kl.nelement()
    else:
        #print(_kl.size())
        return _kl

def virtual_adversarial_loss(net, x, seq_len, mask, is_training=True, do_entropy=True, epsilon=2.5, num_power_iterations=1,
                             xi=1e-6, average_loss=True):
    mask = Variable(mask, requires_grad=False)
    net.eval()
    logit_p = net.forward(x, seq_len)
    logit_p = logit_p * mask
    logit_p.view(-1,logit_p.size()[-1])
    r_vadv = Variable(generate_virtual_adversarial_perturbation(net, x, seq_len, logit_p.detach(), is_training=is_training,
                                                                     epsilon=epsilon,
                                                                     num_power_iterations=num_power_iterations,
                                                                      xi=xi))
    if is_training:
        net.train()

    logit_m = net.forward(x + r_vadv, seq_len)
    logit_m = logit_m * mask
    logit_m.view(-1, logit_m.size()[-1])
    loss = kl_categorical(logit_m, logit_p.detach(), average_loss=average_loss)  # input, target

    if do_entropy:
        log_softmax_p = torch.nn.functional.logsoftmax(logit_p)
        softmax_p = torch.nn.functional.softmax(logit_p)
        loss = loss - torch.mean(torch.sum(softmax_p * log_softmax_p, 1))
    return loss


def comp_loss(net, b1, b2, len1, len2, mask1, mask2):
    mask1 = Variable(mask1, requires_grad=False)
    mask2 = Variable(mask2, requires_grad=False)
    logit1 = net.forward(b1, len1)
    logit2 = net.forward(b2, len2)
    logit1 = logit1 * mask1
    logit2 = logit2 * mask2
    logit1.view(-1, logit1.size()[-1])
    logit2.view(-1, logit2.size()[-1])
    loss = kl_categorical(logit1, logit2.detach(), average_loss=True)  # input, target
    return loss

import os

import pandas as pd

try:
    import pickle5 as pickle
except:
    import pickle

import shutil
from typing import Sequence

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score as f1_cal
# import pytorch_lightning
# from pytorch_lightning.metrics.classification import AUROC
# from pytorch_lightning.metrics.functional.classification import auroc
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import Sampler
from eval.utils import compute_calibration, compute_calibration_multilabel


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        # target_onehot = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AucMeter(object):
    """Store ground truth label and predict label, until epoch then calculate auc"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.count = 0

    def update(self, y_pred, y_true, n=1):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        self.count += n

    def auc(self, option='binary'):
        if option == 'multi_cls':
            self.y_pred = np.vstack(self.y_pred)
            self.y_true = np.hstack(self.y_true).astype(np.int64)
            self.avg = roc_auc_score(self.y_true, self.y_pred, multi_class='ovr', average='weighted')

            roc_multicls = {label: [] for label in np.unique(self.y_true)}
            for label in np.unique(self.y_true):
                binary_true = (self.y_true == label).astype(np.int)
                roc_multicls[label] = roc_auc_score(binary_true, self.y_pred[:, 1])

            self.cls_auc = [roc_multicls[i] for i in roc_multicls.keys()]
        elif option == 'binary':
            self.y_pred = np.vstack(self.y_pred)
            self.y_true = np.hstack(self.y_true).astype(np.int64)
            self.avg = roc_auc_score(self.y_true, self.y_pred[:, 1])
            self.cls_auc = [0]
        elif option == 'multi_label':
            self.y_pred = np.vstack(self.y_pred)
            self.y_true = np.vstack(self.y_true).astype(np.int64)
            self.avg = roc_auc_score(self.y_true, self.y_pred)
            self.cls_auc = roc_auc_score(self.y_true, self.y_pred, average=None)
        else:
            raise NotImplementedError


class ECEMeter(object):
    """Store ground truth label and predict label, until epoch then calculate auc"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.confidence = []
        self.count = 0

    def update(self, y_pred, y_true, confidence, n=1):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        self.confidence.append(confidence)
        self.count += n

    def ece(self, n_bin=10):
        g = np.hstack([i.cpu().numpy() for i in self.y_true])
        if g.ndim == 1:
            p = np.hstack([i.flatten().cpu().numpy() for i in self.y_pred])
            c = np.hstack([i.max(axis=1)[0].cpu().numpy() for i in self.confidence])
            c_norm = (c - np.min(c)) / (np.max(c) - np.min(c))

            # multi-class
            output = compute_calibration(g, p, c_norm, n_bin)
            self.ece_ = output['expected_calibration_error']
            self.acc = output['accuracies']
            self.conf = output['confidences']
        else:
            # multi-label
            g = np.vstack([i.detach().cpu() for i in self.y_true])
            p = np.vstack([i.detach().cpu() for i in self.y_pred])
            c = np.vstack([i.detach().cpu() for i in self.confidence])
            c_norm = np.zeros_like(c)
            num_clas = c.shape[1]
            for i in range(num_clas):
                c_norm[:, i] = (c[:, i] - np.min(c[:, i]) / (np.max(c[:, i]) - np.min(c[:, i])))

            output = compute_calibration_multilabel(g.T, p.T, c_norm.T, n_bins=10)
            self.ece_ = np.mean(output['class_wise_ece'])


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # number correctly predicted data under topk criterion
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def f1_score(output, target, n_class):
    with torch.no_grad():

        _, pred = output.topk(1)
        pred = pred.view(-1)
        if n_class == 2:
            score = f1_cal(target, pred, average='binary')
        else:
            score = f1_cal(target, pred, average='weighted')
        res = [score]

        return res


# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def multiclass_accuracy(output, target):
    """
    Accuracy of each class with threshold 0.5
    :param output: MxN matrix (number of data point * classes)
    :param target: 1xN matrix multi-class problem MxN matrix for multilabel problem
    :return: 1XN array
    """
    _acc = []
    _bs = output.size(0)
    _label_size = output.size(1)

    if torch.cuda.is_available():
        output = output.cpu()
        target = target.cpu()

    if target.dim() == 1:
        # convert to onehot encoding
        target = target.numpy()
        target = get_one_hot(target, _label_size)
        target = torch.tensor(target)
    elif target.dim() == 2:
        pass
    else:
        raise NotImplementedError
    _acti = torch.sigmoid(output)
    prediction = (_acti >= 0.5)
    for cls in range(_label_size):
        acc = 100 * (prediction[:, cls] == target[:, cls]).sum() / _bs
        _acc.append(acc.numpy())
    return np.array(_acc)


def cal_auc(output, target, list_correct, n_class):
    """
    calculate area under the curve
    :param output: torch.tensor
    :param target: torch.tensor
    :param list_correct: list
    :return: list
    """
    if n_class == 1:
        if len(np.unique(np.array(target.cpu()))) == 1:
            list_correct[0] += accuracy_score(
                np.array(target.detach().cpu().numpy()),
                np.rint(np.array(output.detach().cpu().numpy())))
        else:
            list_correct[0] += roc_auc_score(np.array(target.detach().cpu().numpy()),
                                             np.array(output.detach().cpu().numpy()))
    else:
        for classes_index in range(n_class):
            if len(np.unique(np.array(target[:, classes_index].cpu()))) == 1:
                list_correct[classes_index] += accuracy_score(
                    np.array(target[:, classes_index].detach().cpu().numpy()),
                    np.rint(np.array(output[:, classes_index].detach().cpu().numpy())))
            else:
                # print(np.array(target[:,classes_index].detach().cpu().numpy()))
                list_correct[classes_index] += roc_auc_score(np.array(target[:, classes_index].detach().cpu().numpy()),
                                                             np.array(output[:, classes_index].detach().cpu().numpy()))
        list_correct[classes_index + 1] += roc_auc_score(np.array(target.detach().cpu().numpy()),
                                                         np.array(output.detach().cpu().numpy()),
                                                         average='micro')  # over all AUC
    return list_correct


def save_data(args, data, file_name='none'):
    with open(os.path.join(args.tb_folder, file_name), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def load_data(dir):
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


class SubsetSampler(Sampler):
    r"""Samples elements sequantially from a given list of indices, without replacement.

        Arguments:
            indices (sequence): a sequence of indices
            generator (Generator): Generator used in sampling.
        """

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def remove_all_debug_files():
    for i in os.listdir('./save'):
        if 'debug_dataset' in i or 'debug_dataset_multicls' in i:
            shutil.rmtree(os.path.join('./save', i))
            print('remove ', os.path.join('./save', i), 'successful')

    for i in os.listdir('./saved_model'):
        if 'debug_dataset' in i or 'debug_dataset_multicls' in i:
            os.remove(os.path.join('./saved_model', i))
            print('remove ', os.path.join('./saved_model', i), 'successful')


def save_checkpoint(args, net, optimizer, scheduler, layer_index, epoch):
    state = {
        'epoch': epoch + 1,
        'args': args,
        'model': net.state_dict() if args.n_gpu <= 1 else net.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    save_file = os.path.join(args.save_folder,
                             '{}_{}layer_{}_last.pth'.format(args.dataset, layer_index, args.model_name))
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    torch.save(state, save_file)

    return None


def load_checkpoint(net, optimizer, scheduler, layer_index, args):
    model_name = args.dataset + '_' + f'{layer_index}layer_' + args.model_name + '_last.pth'
    if model_name in os.listdir(args.save_folder):
        saved = torch.load(os.path.join(args.save_folder, model_name))

        # loading network para, optimizer para, scheduler para
        # why optimizer para is needed:
        # https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
        net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler'])
        start_epoch = saved['epoch']

        print(f'--> load checkpoint from {os.path.join(args.save_folder, model_name)} successful')

        # overwriting parameters
        last_args = saved['args']
        last_args.save_folder = args.save_folder

        args = last_args
        args.restore_checkpoint = True

        # if 'CascadeNet' in args.model_name:
        #     args.tb_folder = os.path.join(args.tb_folder, '../')
        print(f'Continue from {args.tb_folder}')

        print('current epoch is:', start_epoch)
        if start_epoch == args.epochs:
            print('current epoch is the last')
            return 'jump_to_next'

    else:
        print('No module saved in ', os.path.join(args.save_folder, model_name)
              , 'retrain this layer from start')
        start_epoch = 1

    return net, optimizer, scheduler, layer_index, args, start_epoch


class FileReader:

    def __init__(self, summary_root):
        self.summary_root = summary_root
        self.summary_path = self.get_summary_path()

        self.df_all = self.get_param_summary()

    def get_param_data(self, index, p):
        para = load_data(p)
        a = vars(para.pop('args'))
        a.update(para)

        # seperate pretrain_layer_model
        a['pretrain'] = a['pretrain_layer_model'][0]
        a['layer'] = a['pretrain_layer_model'][1]
        a['model_name'] = a['pretrain_layer_model'][2]
        a.pop('pretrain_layer_model')

        param_df = pd.DataFrame(a, index=[index])

        return param_df

    def get_param_summary(self):
        lis = []
        for index, p in enumerate(self.summary_path):
            try:
                temp_summary = pd.read_csv(os.path.join(p, 'progress.csv'))
                assert list(temp_summary)[0] == 'AUC'

                temp_summary.index = [index] * len(temp_summary)

                temp_params = self.get_param_data(index, os.path.join(p, 'params.pkl'))
                lis.append(pd.concat([temp_summary, temp_params], axis=1))
            except Exception as e:
                print(e)
                continue

        return pd.concat(lis)

    def get_summary_path(self):
        """
        get matric/parammeter path
        target_name have two options: progress.csv/params.pkl
        """
        directory_list = []

        for root, dirs, files in os.walk(self.summary_root, topdown=True):
            for f in files:
                if ('progress.csv' in f):
                    directory_list.append(os.path.join(root))
        return directory_list

    @staticmethod
    def get_lm_arch(config):
        if config['model_name'] == 'CascadeNetPretrainDoris':
            return 'TCL (Conv)'
        elif config['model_name'] == 'E2EPretrainedDorisNet':
            return 'TE2E (Conv)'
        elif config['model_name'] == 'E2EDorisNet':
            return 'E2E (Conv)'
        elif (config['model_name'] == 'E2EResNet18') & (config['pretrain'] == True):
            return 'TE2E (ResNet)'
        elif (config['model_name'] == 'E2EResNet18') & (config['pretrain'] == False):
            return 'E2E (ResNet)'
        elif config['model_name'] == 'CL':
            return 'CL (Conv)'
        elif config['model_name'] == 'E2E':
            return 'E2E (Conv)'
        else:
            raise NotImplementedError


if __name__ == '__main__':
    summarys = '/local/jw7u18/ray_results/TCL_HAM10000'
    sum_ = FileReader(summarys)
    df = sum_.df_all
    df['Learning Method'] = df.apply(sum_.get_lm_arch, axis=1)
    df = df.reset_index().rename(columns={'index': 'trial'})
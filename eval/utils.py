import os

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model.CascadeNetPretrainDorisNet import load_dorisnet, E2E_intermedia
from model.resnet import resnet18

import sys

sys.path.append('./model/sourcemodel')
sys.path.append('../Cascade_Transfer_Learning/model/sourcemodel')
import Build_Network


class DummyArgs:
    def __init__(self, root, batch_size, cur_fold, n_wok, split, dataset):
        self.root_dir = root
        self.batch_size = batch_size
        self.currerent_fold = cur_fold
        self.num_worker = n_wok
        self.n_split = split
        self.dataset = dataset
        self.data_percentage = 1.0


class NewModule(nn.Module):
    def __init__(self, old_cascade_net):
        super(NewModule, self).__init__()
        modules = old_cascade_net
        lis = []
        for i in modules.children():
            if isinstance(i, Build_Network.non_first_layer_cascade_Net):
                for _, j in enumerate(i.children()):
                    lis.append(j)
                    if (isinstance(j, nn.BatchNorm2d)) | (isinstance(j, nn.Linear)):
                        lis.append(nn.ReLU())
                    continue
            else:
                lis.append(i)
            if (isinstance(i, nn.BatchNorm2d)) | (isinstance(i, nn.Linear)):
                lis.append(nn.ReLU())
        self.features = nn.Sequential(*lis[:-7])
        self.fc = nn.Sequential(*lis[-7:-1])

    #         print('NewModule========')
    #         print(self.features)
    #         print(self.fc)
    def forward(self, x):
        out = self.features(x)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# In[4]:


def load_model(args, **kwargs):
    dataset = kwargs['dataset']
    method = kwargs['method']
    arch = kwargs['arch']
    run = kwargs['run']
    percent = kwargs['percent']
    fold = kwargs['fold']
    layer = kwargs['layer']
    model_name = kwargs['model_name']
    model_dir = kwargs['model_dir']
    DorisnetAddress = kwargs['DorisnetAddress']
    pretrained_doris_address = kwargs['pretrained_doris_address']

    if (model_name == 'CascadeNetPretrainDoris'):
        args.DorisnetAddress = DorisnetAddress
        net = load_dorisnet(args, int(layer))
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
        net = NewModule(net)
        feature_module = net.features
        target_layer_names = [str((int(layer) + 1) * 4)]  # last conv index
    elif (model_name == 'E2EPretrainedDorisNet'):
        args.DorisnetAddress = DorisnetAddress
        args.pretrained_doris_address = pretrained_doris_address

        net = load_dorisnet(args, layer_index=11)
        net._modules['33']._modules['fc3'] = nn.Linear(256, 23)
        net.load_state_dict(torch.load(os.path.join(args.pretrained_doris_address
                                                    , 'imagenet23_11layer_E2EDorisNet_last.pth'), map_location='cpu')[
                                'model']
                            , strict=True)
        net = E2E_intermedia(net, args.n_class, int(layer) + 1)
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)

        feature_module = net.aux
        target_layer_names = ["0"]  # last conv index
    elif (model_name == 'E2EDorisNet'):
        args.DorisnetAddress = DorisnetAddress
        net = load_dorisnet(args, int(layer))
        #         reset_all_parameters_dorisnet(net)
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
        net = NewModule(net)
        print(net)
        feature_module = net.features
        target_layer_names = [str((int(layer) + 1) * 4)]  # last conv index
    elif model_name == 'E2EResNet18':
        net = resnet18(pretrained=args.pretrain)
        net.fc = nn.Linear(512, args.n_class)
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
        feature_module = net.layer4
        target_layer_names = ["1"]
    #     summary(net, (3, 224, 224), device='cpu')

    return net, feature_module, target_layer_names


def load_network(args, **kwargs):
    layer = kwargs['layer']
    model_name = kwargs['model_name']
    model_dir = kwargs['model_dir']
    DorisnetAddress = kwargs['DorisnetAddress']
    pretrained_doris_address = kwargs['pretrained_doris_address']
    pretrain = kwargs['pretrain']
    if (model_name == 'CascadeNetPretrainDoris'):
        args.DorisnetAddress = DorisnetAddress
        net = load_dorisnet(DorisnetAddress, args.n_class, int(layer))
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
    elif (model_name == 'E2EPretrainedDorisNet'):
        args.DorisnetAddress = DorisnetAddress
        args.pretrained_doris_address = pretrained_doris_address

        net = load_dorisnet(DorisnetAddress, args.n_class, layer_index=11)
        net._modules['33']._modules['fc3'] = nn.Linear(256, 23)
        net.load_state_dict(torch.load(os.path.join(args.pretrained_doris_address
                                                    , 'imagenet23_11layer_E2EDorisNet_last.pth'), map_location='cpu')[
                                'model']
                            , strict=True)
        net = E2E_intermedia(net, args.n_class, int(layer) + 1)
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
    elif (model_name == 'E2EDorisNet'):
        args.DorisnetAddress = DorisnetAddress
        net = load_dorisnet(DorisnetAddress, args.n_class, int(layer))
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
    elif model_name == 'E2EResNet18':
        net = resnet18(pretrained=pretrain)
        net.fc = nn.Linear(512, args.n_class)
        net.load_state_dict(torch.load(model_dir, map_location='cpu')['model'], strict=True)
    return net


def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return {"accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce}


def compute_calibration_auc(true_labels, pred_labels, confidences, n_bins=10):
    """Collects predictions into bins used to draw a reliability diagram using auc matric.
    Note that the input of true_label and pred_labels should be 1-d. Hence this function returns
    ece for individual class
    """
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert true_labels.ndim == 1
    assert pred_labels.ndim == 1
    assert (n_bins > 0)
    assert len(set(true_labels)) > 1

    bin_size = 1.0 / n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_auc = np.zeros(n_bins, dtype=np.float)
    bin_confidences = np.zeros(n_bins, dtype=np.float)
    bin_counts = np.zeros(n_bins, dtype=np.int)

    for b in range(n_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            print(true_labels[selected])
            bin_auc[b] = roc_auc_score(true_labels[selected], pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_auc = np.sum(bin_auc * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_auc - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return {"auc": bin_auc,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_auc": avg_auc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce}


def compute_calibration_multilabel(true_labels, pred_labels, confidences, **kwargs):
    """
    Computing calibration for multilabel. iterating for each class and take the average.
    :param true_labels: 2d np array, have shape n_class x n_data
    :param pred_labels: 2d np array, sigmoid activation of confidence, have shape n_class x n_data
    :param confidences: 2d np array, have shape n_class x n_data
    :param kwargs: num_bins
    :return: class-wise calibration error.
    """
    assert true_labels.ndim == 2
    assert pred_labels.ndim == 2
    assert confidences.ndim == 2
    assert (kwargs['n_bins'] > 0)

    ece_list = []
    for index, (c_gt_label, c_pred, c_conf) in enumerate(zip(true_labels, pred_labels, confidences)):
        # print(c_gt_label.shape, c_pred.shape, c_conf.shape)
        print(c_gt_label)
        results = compute_calibration_auc(c_gt_label, c_pred, c_conf, **kwargs)
        ece_list.append(results['expected_calibration_error'])
    return {'class_wise_ece': ece_list}


if __name__ == '__main__':
    true_ = np.array([[1,0,1,0,1],
                      [0,1,0,1,0]])
    pred_ = np.array([[0.02223,0.26485,0.47788,0.79723,0.94495]
                     , [0.09334,0.23154,0.99957,0.27893,0.08716]])
    conf_ = pred_.copy()

    result = compute_calibration_multilabel(true_.T, pred_.T, conf_.T, n_bins=1)
    print(result)


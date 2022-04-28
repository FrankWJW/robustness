import shutil

import os

import argparse

import numpy as np
import torchvision.datasets
from ray.tune.schedulers import ASHAScheduler
from ray.tune.trial import ExportFormat
from sklearn.metrics import confusion_matrix
from torch import optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.CascadeNet import build_cfg_list, cascade_net, build_e2e
from model.CascadeNetPretrainDorisNet import load_model
from train import train, train_multiclass
from ray import tune
import ray

from util import AverageMeter, accuracy, AucMeter

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
object_categories = {name: i for i, name in enumerate(object_categories)}


def multilabel(target):
    y = torch.zeros(20, dtype=torch.double)
    lis_positive_index = list(map(lambda x: object_categories[x['name']], target['annotation']['object']))
    for i in lis_positive_index:
        y[i] = torch.tensor(1, dtype=torch.double)
    return y


class Stopper:
    """
    forcing to stop the loop
    """

    def __init__(self, thres):
        self.counter = 0
        self.best = 0
        self.thres = thres

    def __call__(self, now):
        if self.counter >= self.thres:
            return True
        else:
            if self.best >= now:
                self.stop_flag('ADD')
            else:
                self.stop_flag('CLEAR')
                self.best = now
            return False

    def stop_flag(self, set_counter):
        if set_counter == 'ADD':
            self.counter += 1
        elif set_counter == 'CLEAR':
            self.counter = 0


def load_dataset(dataset, root_dir, batch_size, num_workers):
    if dataset == 'pascal':
        root = root_dir
        train_set = torchvision.datasets.VOCDetection(root=root, image_set='train', download=False
                                                      , transform=transform_train, target_transform=multilabel)
        val_set = torchvision.datasets.VOCDetection(root=root, image_set='val', download=False,
                                                    transform=transform_val, target_transform=multilabel)
        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        return train_loader, val_loader


def validate_aux(val_loader, model, criterion, print_freq):
    """One epoch validation for aux type classifier"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc_meter = AucMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        labels = []
        outputs = []
        logit_out = []
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            auc_meter.update(torch.sigmoid(output).detach().cpu().numpy()
                             , target.detach().cpu().numpy(), input.size(0))

            logit_out.append(output.cpu().numpy())

            if idx % print_freq == 0:
                print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses))

        auc_meter.auc(option='multi_label')

        print(' ** Avg Auc@ {auc:.3f} '.format(auc=auc_meter.avg))
        print(f' * Auc@ {auc_meter.cls_auc} ')

    return {'auc': auc_meter.avg,
            'loss': losses.avg,
            'auc_all': auc_meter.cls_auc,
            'predict_logit': logit_out,
            }


def train_aux(config, args, train_loader, val_loader, checkpoint_dir, net):
    """main training function for aux"""
    stop = Stopper(args.patience)  # after /patients/ iter not improve, stop the loop
    # push model to gpu first
    # https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783
    if torch.cuda.is_available():
        net = net.cuda()
        print("Running on GPU? --> ", all(p.is_cuda for p in net.parameters()))

    # optimizer
    lr_init = config['lr']
    # optimizer = optim.Adam(net.parameters(), lr=lr_init, weight_decay=config['weight_decay'])

    optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, weight_decay=config['weight_decay'])

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    # learning rate scheduler
    lrscheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter, eta_min=0)

    # load from checkpoint
    init_step = 1
    if checkpoint_dir:
        try:
            print("Loading from checkpoint.")
            path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(path)
            net.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if lrscheduler:
                lrscheduler.load_state_dict(checkpoint["LRscheduler"])
            init_step = checkpoint["step"]
        except Exception as x:
            print(x)
            return net

    # parallel
    if torch.cuda.is_available():
        if args.n_gpu > 1:
            net = nn.DataParallel(net)
        net = net.cuda()
        print("Running on GPU? --> ", all(p.is_cuda for p in net.parameters()))

    # start training
    for step in range(init_step, args.max_iter + 1):
        print("==> training...")
        net.train()

        # scheduler
        last_lr = config['lr']
        if step != 1:
            if lrscheduler:
                lrscheduler.step()
                last_lr = lrscheduler.get_last_lr()[0]
            else:
                last_lr = config['lr']

        train_auc, _, train_loss = train_multiclass(step, train_loader, net, criterion, optimizer, args)
        result_val = validate_aux(val_loader, net, criterion,
                                  args.print_freq)
        val_auc = result_val['auc']
        val_auc_all = {f'val_auc_{i}': x for i, x in enumerate(result_val['auc_all'])}
        val_loss = result_val['loss']
        predict_logit = result_val['predict_logit']
        layer = net.module.cur_layer if args.n_gpu > 1 else net.cur_layer

        # update current epoch
        if step % 5 == 0:
            # Every 5 steps, checkpoint our current state.
            # First get the checkpoint directory from tune.
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                # Then create a checkpoint file in this directory.
                path = os.path.join(checkpoint_dir, "checkpoint")
                # Save state to checkpoint file.
                # No need to save optimizer for SGD.
                torch.save({
                    "step": step + 1,
                    "model_state_dict": net.module.state_dict() if args.n_gpu > 1 else net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "LRscheduler": lrscheduler.state_dict() if lrscheduler is not None else None,
                    "score": val_auc,
                    "predict_logit": predict_logit,
                }, path)
        step += 1

        tune.report(AUC=val_auc, lr=last_lr, layer=layer + 1
                    , val_loss=val_loss, train_loss=train_loss, **val_auc_all)

        if stop(result_val['auc']) | (step == args.max_iter):
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                # Then create a checkpoint file in this directory.
                path = os.path.join(checkpoint_dir, "checkpoint")
                # Save state to checkpoint file.
                # No need to save optimizer for SGD.
                torch.save({
                    "step": step + 1,
                    "model_state_dict": net.module.state_dict() if args.n_gpu > 1 else net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "LRscheduler": lrscheduler.state_dict() if lrscheduler is not None else None,
                    "score": val_auc,
                    "predict_logit": predict_logit,
                }, path)

            return net.module if args.n_gpu > 1 else net


def save_gradient(data_loader, net, path):
    pass


def training(config, checkpoint_dir=None):
    """main training function"""
    args = config['args']
    config['pretrain'], config['layer_index'], config['model_name'] = config['pretrain_layer_model']

    # dataset
    train_loader, val_loader = load_dataset(args.dataset, args.root_dir, config['batch_size'],
                                            args.num_workers)

    # model
    net = None  # init network first
    layer_cfg = build_cfg_list(capacity='small_with_dropout')
    num_layers = len(layer_cfg)
    layer_list = [str(i) for i in range(num_layers)]

    if config['model_name'] == 'CL':
        for k in layer_list:
            net = cascade_net(net, freeze=True, cur_layerid=int(k), layer_config=layer_cfg,
                              num_classes=args.n_class, dropout_p=0, numconvaux=config['numconvaux'])
            net = train_aux(config, args, train_loader, val_loader, checkpoint_dir=checkpoint_dir, net=net)
    else:
        for k in layer_list:
            net = cascade_net(net, freeze=True, cur_layerid=int(k), layer_config=layer_cfg,
                              num_classes=args.n_class, dropout_p=0, numconvaux=config['numconvaux'])
        net = build_e2e(net)
        net = train_aux(config, args, train_loader, val_loader, checkpoint_dir=checkpoint_dir, net=net)


if __name__ == '__main__':
    os.environ['RAY_worker_register_timeout_seconds'] = '60'

    parser = argparse.ArgumentParser()

    # ray tune
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")

    # folder
    parser.add_argument('--root_dir', type=str, default='/local/jw7u18/pascalvoc')
    parser.add_argument('--save_dir', type=str, default='/local/jw7u18/ray_results')
    parser.add_argument('--name', type=str, default='test', help='ray result folder name')

    # name dataset
    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--n_class', type=int, default=20)
    parser.add_argument('--max_iter', type=int, default=600, help='max number of epoch')

    # others
    parser.add_argument('--print_freq', type=int, default=100, help='iterations')
    parser.add_argument('--n_gpu', default=1.0, type=float)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=7)

    # ray
    parser.add_argument(
        "--ray_address",
        help="Address of Ray cluster for seamless distributed execution.")

    args, _ = parser.parse_known_args()


    def model_layer_iter():
        for model in ['E2E']:
            if model in ['CL', 'E2E']:
                pretrain = False
                layer = 7
                yield pretrain, layer, model


    if args.ray_address:
        ray.init(address=args.ray_address, _redis_password='65884528')

    # try:
    #     shutil.rmtree(os.path.join('/local/jw7u18/ray_results', args.name))
    # except:
    #     pass

    # sche = ASHAScheduler(metric='AUC', max_t=args.max_iter, grace_period=15)

    analysis = tune.run(
        training,
        num_samples=1,
        resources_per_trial={"cpu": 4 * args.n_gpu, "gpu": args.n_gpu},
        mode="max",
        export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="AUC",
        keep_checkpoints_num=1,
        config={
            "args": args,
            "lr": 0.0053376,
            "batch_size": 12,
            "pretrain_layer_model": tune.grid_search(list(model_layer_iter())),
            "weight_decay": 1.6331e-05,
            "numconvaux": 0,
            "run": tune.grid_search([1,2,3])
        },
        # scheduler=sche,
        name=args.name,
        resume=False,  # if "ERRORED_ONLY", COMMENT line 343-346
        local_dir=args.save_dir
    )

    # config ={
    #         "args": args,
    #         "pretrain_layer_model": (False, 7, 'CL'),
    #     "lr": 1e-5,
    #     "batch_size": 32,
    #     "weight_decay": 1e-5,
    #     "run": 1
    #     }
    # training(config, checkpoint_dir="/local/jw7u18/ray_results/CL_pascal/training_bec46_00000_0_pretrain_layer_model=_False, 7, 'CL'_,run=1_2022-04-20_16-44-21/checkpoint_000140")

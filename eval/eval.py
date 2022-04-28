from __future__ import print_function

import torch
import numpy as np
import time
import sys
sys.path.append('../')

from util import AverageMeter, accuracy, cal_auc, AucMeter, f1_score, multiclass_accuracy, ECEMeter


def validate_multiclass(val_loader, model, criterion, **kwargs):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    auc_meter = AucMeter()

    option = 'multi_label'

    # for each label create an avg meter
    multilabel_acc_list = []
    for cls in range(kwargs['n_class']):
        multilabel_acc_list.append(AverageMeter())

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
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

            # multi-label accuracy
            muti_acc = multiclass_accuracy(output, target)
            for cls in range(kwargs['n_class']):
                multilabel_acc_list[cls].update(muti_acc[cls])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % kwargs['print_freq'] == 0:
                print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses))

        auc_meter.auc(option=option)  # calculate auc

        print(' ** Avg Auc@ {auc:.3f} '.format(auc=auc_meter.avg))
        print(f' * Auc@ {auc_meter.cls_auc} ')

        # for i, meter in enumerate(multilabel_acc_list):
        #     print(' * Avg Acc@ class {i}: {acc:.3f}'.format(i=i, acc=meter.avg))
        _multilabel_avg_acc = np.array([i.avg for i in multilabel_acc_list])

    return auc_meter.avg, auc_meter.cls_auc, losses.avg, _multilabel_avg_acc


def validate(val_loader, model, criterion, n_class, print_freq):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    f1 = AverageMeter()
    ece = ECEMeter()
    auc_meter = AucMeter()

    if n_class == 2:
        option = 'binary'
    elif n_class > 2:
        option = 'multi_cls'
    else:
        option = None

    # for each label create an avg meter
    multilabel_acc_list = []
    for cls in range(n_class):
        multilabel_acc_list.append(AverageMeter())

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1,_ = accuracy(output, target, topk=(1,2))

            if torch.cuda.is_available():
                output = output.cpu()
                target = target.cpu()
            # f1 score
            fi = f1_score(output, target, n_class)

            # auc update
            auc_meter.update(torch.softmax(output, dim=1).detach().cpu().numpy()
                             , target.detach().cpu().numpy(), input.size(0))

            # ece update
            ece.update(output.topk(1, 1)[1], target, output)

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            f1.update(fi[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      .format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, f1=f1))

        auc_meter.auc(option=option)  # calculate multicls auc
        ece.ece(n_bin=10)

        print(' * Acc@1 {top1.avg:.3f} F1 {f1.avg:.3f} AUC {auc.avg:.3f} ECE {ece.ece_}'
              .format(top1=top1, f1=f1, auc=auc_meter, ece=ece))

    # return top1.avg, losses.avg, f1.avg, auc_meter.avg, auc_meter.cls_auc, ece.ece_, ece.acc, ece.conf
    return {'top1 accuracy': top1.avg,
            'loss': losses.avg,
            'f1': f1.avg,
            'auc': auc_meter.avg,
            'auc_per_cls': auc_meter.cls_auc,
            'ece': ece.ece_}

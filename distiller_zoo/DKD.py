import torch
from ._base import Distiller

import torch.nn as nn
import torch.nn.functional as F



def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    #tckd_loss =(nn.KLDivLoss(reduction='none')(log_pred_student, pred_teacher) * (temperature ** 2)).sum(-1)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False, reduction='none')
            * (temperature**2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    #nckd_loss = (nn.KLDivLoss(reduction='none')(log_pred_student_part2, pred_teacher_part2) * (temperature**2)).sum(-1)
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False, reduction='none')
        * (temperature**2)
        / target.shape[0]
    )#.sum(-1) / target.shape[0]
    return alpha * tckd_loss + beta * nckd_loss, tckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self):
        super(DKD, self).__init__()
    def forward(self, y_s, y_t,target):
        logits_student = y_s
        logits_teacher = y_t
        loss_dkd, _ = dkd_loss(
            logits_student,
            logits_teacher,
            target,
            alpha=1.0,
            beta=8.0,
            temperature=4.0,
        )
        return loss_dkd


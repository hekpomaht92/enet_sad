import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import function
from torchvision import models
import numpy as np
from config import Config
np.random.seed(42)

cfg = Config()


class Metrics:
    '''Compute tpr, fpr, fpr, fnr and balanced accuracy'''
    @classmethod
    def compute_tpr(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @staticmethod
    def _compute_tpr(y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @classmethod
    def compute_tnr(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @staticmethod
    def _compute_tnr(y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @classmethod
    def compute_ppv(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tp / (tp + fp)

    @classmethod
    def compute_npv(cls, y_true, y_pred):
        
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_neg = 1 - y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tn / (tn + fn)

    @classmethod
    def balanced_accuracy(cls, y_true, y_pred):
        
        tpr = cls._compute_tpr(y_true, y_pred)
        tnr = cls._compute_tnr(y_true, y_pred)
        return (tpr+tnr)/2


# class DCLoss(torch.autograd.Function):

#     def __init__(self):
#         super().__init__()
    
#     @staticmethod
#     def forward(ctx, pred, label):
#         label = torch.cat((1. - torch.unsqueeze(label, 1), torch.unsqueeze(label, 1)), 1).type(torch.FloatTensor).to(cfg.device)
#         loss = 2. * torch.sum(torch.abs(pred * label)) / torch.sum(torch.abs(pred) + torch.abs(label))
#         ctx.save_for_backward(pred, label)
#         return loss
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         pred, label = ctx.saved_tensors
#         dDice = torch.add(torch.mul(label, 2), torch.mul(pred, -4))
#         # grad_input = torch.cat((torch.mul(torch.unsqueeze(dDice,1), grad_output.item()),\
#         #     torch.mul(torch.unsqueeze(dDice,1), -grad_output.item())), dim = 1)
#         grad_input = torch.mul(dDice, -grad_output.item())
#         return grad_input, None

class DCLoss(torch.autograd.Function):
    """Dice coeff for individual examples"""
    @staticmethod
    def forward(ctx, pred, target):
        target = target.type(torch.FloatTensor).to(cfg.device)
        pred = torch.abs(pred)
        eps = 0.0001
        #print('input into dice', input.view(-1).size())
        #print('target into dice', target.view(-1).size())
        # inter = torch.sum(torch.mul(pred, target))
        inter = torch.dot(pred.view(-1), target.view(-1))
        union = torch.sum(pred) + torch.sum(target) + eps
        ctx.save_for_backward(pred, target, inter, union)
        t = (2 * inter + eps) / union
        return t

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        pred, target, inter, union = ctx.saved_variables
        grad_input = grad_output * 2 * (target * union - inter)\
            / (union * union)
        return grad_input, None


# class DCLoss(nn.Module):

#     def __init__(self):
#         super().__init__()
    
#     def forward(self, pred, label):
#         label = torch.cat((1. - torch.unsqueeze(label, 1), torch.unsqueeze(label, 1)), 1).type(torch.FloatTensor).to(cfg.device)
#         loss = 2. * torch.sum(torch.abs(pred * label)) / torch.sum(torch.abs(pred) + torch.abs(label))
#         return loss

if __name__ == '__main__':
    test_value = torch.ones((2, 2, 64, 64), dtype=torch.float, requires_grad=True)
    test_value_ = torch.ones((2, 64, 64), dtype=torch.float)
    criterion = DCLoss()
    loss = criterion(test_value, test_value_)
    torch.autograd.gradcheck(criterion, (test_value, test_value_))
    # loss.backward()
    # layer = slice()
    # ans = layer(test_value)
    print()


import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import math    

class SMD_qnorm(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0, weight_decay = 0, dampening=0, q =3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 1.01 <= q:
            raise ValueError("Invalid q_norm value: {}".format(q))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, q = q)
        super(SMD_qnorm, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SMD_qnorm, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            lr = group['lr']
            q = group['q']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # if (d_p.isnan().any()):
                #     torch.set_printoptions(threshold=200)
                #     print("d_p", d_p)
                    
    #           q norm potential function
                update = torch.pow(torch.abs(p.data), q-1) * torch.sign(p.data) - lr * d_p
                p.data = torch.pow(torch.abs(update), 1/(q-1)) * torch.sign(update)

        return loss 

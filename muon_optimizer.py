"""
MUON Optimizer - Placeholder
GitHub: https://github.com/KellerJordan/modded-nanogpt/blob/master/muon.py

For now, this uses the Newton-Schulz preconditioning from COSMOS.
For full MUON implementation, clone: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
import math
from torch.optim import Optimizer


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


class MUON(Optimizer):
    """
    MUON Optimizer (Simplified version using Newton-Schulz preconditioning)
    
    For full implementation, see: https://github.com/KellerJordan/modded-nanogpt
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                
                # Apply Newton-Schulz preconditioning for 2D parameters
                if len(grad.shape) == 2:
                    grad = zeropower_via_newtonschulz5(grad, steps=ns_steps)
                    grad = grad * (grad.numel() ** 0.5)
                
                # Momentum update
                buf.mul_(momentum).add_(grad)
                
                # Parameter update
                p.add_(buf, alpha=-lr)
        
        return loss

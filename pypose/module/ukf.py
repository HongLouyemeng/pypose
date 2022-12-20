import torch
from torch import nn
from ..basics import bmv
from torch.linalg import pinv


class UKF(nn.Module):
    r'''
    Performs Batched Unscented Kalman Filter (UKF).
    '''

    def __init__(self, model, Q=None, R=None):
        super().__init__()
        self.set_uncertainty(Q=Q, R=R)
        self.model = model

    def compute_weight(self):
        return 1 / (2 * self.dim)

    def compute_sigma(self, x, u, P, C, D, c2):  # compute sigma point
        r'''
        compute sigma point
        '''
        x_sigma = []
        y_sigma = []
        for loop in range(1, self.loop_range):

            if loop <= self.dim:
                x_ = x + torch.sqrt_(self.weight * P).mT[loop-1]
            else:
                x_ = x - torch.sqrt_(self.weight * P).mT[loop - self.dim-1]
            y_ = bmv(C, x_) + bmv(D, u) + c2    # compute Observation
            x_sigma.append(x_)
            y_sigma.append(y_)

        return torch.cat(x_sigma, dim=0).reshape(-1,self.dim), torch.cat(y_sigma, dim=0).reshape(-1,self.dim)

    def compute_conv_mix(self, P, x_estimate, x_sigma, y_estimate, y_sigma):
        r'''
        compute mix covariance
        '''
        p_estimate = torch.zeros((P.shape), device=P.device, dtype=P.dtype)
        e_x = torch.sub(x_sigma,x_estimate).unsqueeze(1)
        e_y = torch.sub(y_sigma,y_estimate).unsqueeze(1)
        p_estimate+= torch.sum(torch.bmm(e_x,e_y.permute(0, 2, 1)))

        return self.weight * p_estimate

    def compute_conv(self, P, estimate, sigma, noise=0):
        r'''
        compute covariance
        '''
        P_estimate = torch.zeros((P.shape), device=P.device, dtype=P.dtype)
        e = torch.sub(sigma,estimate).unsqueeze(1)
        P_estimate += torch.sum(torch.bmm(e, e.permute(0, 2, 1)))

        return self.weight*P_estimate + noise

    def forward(self, x, y, u, P, Q=None, R=None):
        r'''
        Performs one step estimation.

        Args:
            x (:obj:`Tensor`): estimated system state of previous step
            y (:obj:`Tensor`): system observation at current step (measurement)
            u (:obj:`Tensor`): system input at current step
            P (:obj:`Tensor`): state estimation covariance of previous step
            Q (:obj:`Tensor`, optional): covariance of system transition model
            R (:obj:`Tensor`, optional): covariance of system observation model

        Return:
            list of :obj:`Tensor`: posteriori state and covariance estimation
        '''
        # Upper cases are matrices, lower cases are vectors
        self.model.set_refpoint(state=x, input=u)
        A, B = self.model.A, self.model.B
        C, D = self.model.C, self.model.D
        c1, c2 = self.model.c1, self.model.c2
        Q = Q if Q is not None else self.Q
        R = R if R is not None else self.R
        x = bmv(A, x) + bmv(B, u) + c1
        self.dim = x.shape[0]
        self.weight = self.compute_weight()
        self.loop_range = self.dim * 2 + 1

        # compute sigma point,mean,covariance
        x_sigma, y_sigma = self.compute_sigma(x, u, P, C, D, c2)
        x_estimate = self.weight * torch.sum(x_sigma, dim=0)
        Pk = self.compute_conv(P, x_estimate, x_sigma, Q)
        x_sigma, y_sigma = self.compute_sigma(x_estimate, u, Pk, C, D, c2)
        y_estimate = self.weight * torch.sum(y_sigma, dim=0)
        Py = self.compute_conv(P, y_estimate, y_sigma, R)
        Pxy = self.compute_conv_mix(Py, x_estimate, x_sigma, y_estimate, y_sigma)

        # Equation
        K = Pxy @ pinv(Py)
        e = y - y_estimate
        x = x_estimate + bmv(K, e)
        P = Pk - K @ Py @ K.mT

        return x, P

    @property
    def Q(self):
        r'''
        The covariance of system transition noise.
        '''
        if not hasattr(self, '_Q'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance Q.')
        return self._Q

    @property
    def R(self):
        r'''
        The covariance of system observation noise.
        '''
        if not hasattr(self, '_R'):
            raise NotImplementedError('Call set_uncertainty() to define\
                                        transition covariance R.')
        return self._R

    def set_uncertainty(self, Q=None, R=None):
        r'''
        Set the covariance matrices of transition noise and observation noise.

        Args:
            Q (:obj:`Tensor`): batched square covariance matrices of transition noise.
            R (:obj:`Tensor`): batched square covariance matrices of observation noise.
        '''
        if Q is not None:
            self.register_buffer("_Q", Q)
        if R is not None:
            self.register_buffer("_R", R)

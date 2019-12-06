import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNMF_Encoder_simplified(nn.Module):
    def __init__(self, m, n, q, lambda_star, lambdaO, eta, T, D):
        super(RNMF_Encoder_simplified, self).__init__()
        self.D = nn.Parameter(D, requires_grad = True)
        # normalization
        #self.W.data = NormDict(self.W.data)
        tmp1 = torch.mm(self.D.transpose(1,0), self.D) + lambda_star * torch.eye(q)
        tmp2 = (1 + lambda_star) * torch.eye(m)
        tmp3 = torch.cat((tmp1, self.D.transpose(1,0)), dim=1)
        tmp4 = torch.cat((self.D, tmp2), dim = 1)
        self.H = torch.eye(m+q) - eta * torch.cat((tmp3, tmp4), dim = 0)
        self.W = eta * torch.cat((self.D.transpose(1, 0), torch.eye(m)), dim = 0)
        self.t = lambdaO * eta * torch.cat((torch.zeros(q), torch.ones(m)), axis = 0)

        self.T = T

    def forward(self, X, q, lambda_star, lambdaO):
        
        #self.D = NormDict(self.D + eps)
        alpha = PowerMethod(self.D)
        eta = 1/alpha

        m, n = X.shape

        tmp1 = torch.mm(self.D.transpose(1,0), self.D) + lambda_star * torch.eye(q)
        tmp2 = (1 + lambda_star) * torch.eye(m)
        tmp3 = torch.cat((tmp1, self.D.transpose(1,0)), dim=1)
        tmp4 = torch.cat((self.D, tmp2), dim = 1)
        self.H = torch.eye(m+q) - eta * torch.cat((tmp3, tmp4), dim = 0)
        self.W = eta * torch.cat((self.D.transpose(1, 0), torch.eye(m)), dim = 0)
        self.t = lambdaO * eta * torch.cat((torch.zeros(q), torch.ones(m)), axis = 0)

        
        ## initialization
        Z = torch.zeros(m+q, n)
        b = torch.mm(self.W, X)
        for j in range(self.T):
            #residual = torch.max(torch.zeros(m+q,n), b - torch.ger(t, torch.ones(n))) - Z
            Zold = Z.clone()
            Z = torch.max(torch.zeros(m+q,n), b - torch.ger(self.t, torch.ones(n)))
            b += torch.matmul(self.H, Z - Zold)

        #S = Z[:q,:]
        #L = torch.mm(self.D, S)
        #O = Z[q:,:]

        return Z




#--------------------------------------------------------------

def NormDict(W):
    Wn = torch.norm(W, p=2, dim=0).detach()
    W = W.div(Wn.expand_as(W))
    return W

#--------------------------------------------------------------

def PowerMethod(W):
    ITER = 100
    m = W.shape[1]
    X = torch.randn(1, m).to(device)
    for i in range(ITER):
        Dgamma = torch.mm(X,W.transpose(1,0))
        X = torch.mm(Dgamma,W)
        nm = torch.norm(X,p=2)
        X = X/nm
    
    return nm


#--------------------------------------------------------------

def DictUpdate(W, X, Z, lambda_star, q, eps=2.3e-16):
    D = W * torch.mm(X - Z[q:,:], Z[:q,:].transpose(1,0)) / (torch.mm(W, torch.mm(Z[:q,:], Z[:q,:].transpose(1,0)) + lambda_star * torch.eye(q)) + eps)
    D.detach_()
    D = NormDict(D + eps)
    return D
    #tmp = NormDict(D)
    #if torch.sum(torch.isnan(tmp)) > 0:
    #    return D
    #else:
    #    return tmp




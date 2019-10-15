import torch
import torch.nn as nn
from time import time
import math
import numpy as np
from numpy import pi

def is_pos_def(x):
    """
    Check if the matrix is positive definite
    """
    x = x.detach().numpy()
    return np.all(np.linalg.eigvalsh(x) > 0)

def tril_mask(size):
    """
    Returns a lower triangular mask
    (Used to select lower triangular elements)
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.uint8)) # ByteTensor
    return mask

def vectorLowerCholesky(P):
    """
    Performs the lower cholesky decomposition and returns vectorized output
    P = L L.t()
    """
    L = torch.cholesky(P, upper=False)
    mask = tril_mask(P.size(0))
    return torch.masked_select(L, mask > 0)

def inverseCholesky(vecL):
    """
    Performs the inverse operation to lower cholesky decomposition
    and converts vectorized lower cholesky to matrix P
    P = L L.t()
    """
    size = int(np.sqrt(2 * vecL.size(0)))
    L = torch.zeros(size, size)
    mask = tril_mask(size)
    L[mask == 1] = vecL
    P = L.mm(L.t())
    return P

def range_angle(ang):
    """
    Adjusts the range of angle from -pi to pi
    """
    ang = torch.remainder(ang, 2*pi)
    ang = ang if ang < pi else (ang -2*pi)
    return ang

def pos_init(box=2.):
    """
    Initialize position (polar) coordinates and relative angle of monkey
    """
    #r = torch.rand(1) * box
    r = torch.sqrt(torch.rand(1)+0.25) * box
    ang = torch.zeros(1).uniform_(-pi, pi)
    #ang = torch.ones(1)*pi/2
    #pos = torch.cat((r, ang)) # polar coordinates
    rel_ang = torch.zeros(1).uniform_(-pi/4, pi/4)
    return (r, ang, rel_ang)

def ellipse(mu, cov, n=100, conf_int=5.991, torch=False):
    """
    Returns points on the confidence region boundary for 2D Gaussian
    distribution
    (Used to plot while rendering)
    """
    if torch:
        mu = mu.detach().numpy()
        cov = cov.detach().numpy()
    w, v = np.linalg.eigh(cov)
    max_eig = v[:, -1]
    phi = math.atan2(max_eig[1], max_eig[0])
    phi %= 2 * pi
    chi2_val = np.sqrt(conf_int) # 95% confidence interval
    a =  chi2_val * np.sqrt(w[1])
    b =  chi2_val * np.sqrt(w[0])
    #theta = np.arange(0, 2 * pi, 0.01)
    theta = np.linspace(0, 2 * pi, n)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    R = np.array([[np.cos(phi), np.sin(phi)],
                  [-np.sin(phi), np.cos(phi)]])
    X = np.array([x,y]).T
    X = X.dot(R)
    x = X[:,0] + mu[0]
    y = X[:,1] + mu[1]
    return x, y

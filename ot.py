"""
Sinkhorn Algorithm for Optimal Transport Distance 
"""

# Author: zchen
# Date: 2021-08-26
# Reference: Computational Optimal Transport(https://optimaltransport.github.io/)

import torch 


__all__ = ['sinkhorn']


def sinkhorn(A, B, C, eps=0.1):
    """Sinkhorn's algorithm for solving entropic regularization OT,
    for more detail, please reference chapter 4 of Computational OT
    
    Parameters:
    -----------
    A: torch.Tensor of shape (n_batches, n)
        Input data

    B: torch.Tensor of shape (n_batches, m)
        Input data

    C: torch.Tensor of shape (n, m)
        Cost matrix

    eps: positive float
        Regularization parameter

    Returns:
    --------
    dist: torch.Tensor of shape (n_batches)
        Sinkhorn distances for each pair of input

    P: torch.tensor of shape (n_batches, N, N)
        Coupling matrices for each pair of input
    """
    assert A.shape[0] == B.shape[0]
    n_batches = A.shape[0]
    n, m = A.shape[1], B.shape[1]
    A, B = A.T, B.T
    K = torch.exp(-C/eps) # Gibbs kernel 
    V = torch.ones(m, n_batches)
    max_niter = 100
    threshold = 1e-6
    actual_niter = 0
    for _ in range(max_niter):
        V1 = V
        U = A / torch.matmul(K, V)
        V = B / torch.matmul(K.T, U)
        err = torch.sum(torch.abs(V-V1))
        actual_niter += 1
        if err < threshold:
            break
    
    P = torch.unsqueeze(U.T, 2) * K * torch.unsqueeze(V.T, 1)
    dist = torch.sum(P * C, dim=(1, 2))

    return dist, P, actual_niter


def _normalize(U, method='square'):
    """Normalize wave signal to be a probability distribution(nonnegativity
    and unit total mass) with certain method

    Parameters:
    -----------
    U: torch.Tensor of shape (n_batches, nt)
       Wave signal

    method: str
        Method used to normalize sigmal, possible
        values:
         - 'suqare'
         - 'abs'
         - 'exp'

    Returns:
    --------
    U_normalized: torch.Tensor of shape (n_batches, nt)
        Normalized wave signal
    """

    def _unify(U):
        """ Unify X for each row
        """
        return U / torch.sum(U, dim=1, keepdim=True)

    if isinstance(method, str):
        if method == 'suqare':
            U = U ** 2
        elif method == 'abs':
            U = torch.abs(U)
        elif method == 'exp':
            raise NotImplementedError("method 'exp' is not implemented")
        else:
            raise ValueError('unknown method parameter')
    else:
        raise TypeError('method must be a string')
    
    U_normalized = _unify(U)

    assert torch.all(U_normalized >= 0)
    assert torch.allclose(U_normalized.sum(dim=1), torch.ones(U_normalized.shape[0]))

    return U_normalized


def sinkhorn_wave(U, V, t, method='suqare', p=2, eps=0.1):
    """ Computing mean sinkhorn distances for wave signals

    Parameters:
    ----------
    U: torch.Tensor of shape (n_batches, nt)
       Wave signal

    V: torch.Tensor of shape (n_batches, nt)
        Wave signal

    t: torch.Tensor of shape (nt)
        Timestamp for wave sigmal

    method: str
        Method used to normalize sigmal, possible
        values:
         - 'suqare'
         - 'abs'
         - 'exp'

    p: positive int

    eps: positive float
        Regularization parameter

    Returns:
    --------
    md: float
        Mean sinkhorn distance for wave signal
    """
    Un = _normalize(U, method)
    Vn = _normalize(V, method)
    t_row = torch.unsqueeze(t, 0)
    t_col = torch.unsqueeze(t, 1)
    C = (t_col - t_row) ** p
    dist, _ = sinkhorn(Un, Vn, C, eps)
    md = torch.mean(dist)
    return md
import torch
from .. import LieTensor
import numpy as np
import mpmath as mpm
import scipy.linalg
from scipy.special import ellipk, ellipj


def bmv(input, vec, *, out=None):
    r'''
    Performs batched matrix-vector product.

    Args:
        input (:obj:`Tensor`): matrices to be multiplied.
        vec (:obj:`Tensor`): vectors to be multiplied.

    Return:
        out (:obj:`Tensor`): the output tensor.

    Note:
        The ``input`` has to be a (:math:`\cdots\times n \times m`) tensor,
        the ``vec`` has to be a (:math:`\cdots\times m`) tensor,
        and ``out`` will be a (:math:`\cdots\times n`) tensor.
        Different from ``torch.mv``, which is not broadcast, this function
        is broadcast and supports batched product.

    Example:
        >>> matrix = torch.randn(2, 1, 3, 2)
        >>> vec = torch.randn(1, 2, 2)
        >>> out = pp.bmv(matrix, vec)
        >>> out.shape
        torch.Size([2, 2, 3])
    '''
    assert input.ndim >= 2 and vec.ndim >= 1, 'Input arguments invalid'
    assert input.shape[-1] == vec.shape[-1], 'matrix-vector shape invalid'
    input = input.tensor() if isinstance(input, LieTensor) else input
    vec = vec.tensor() if isinstance(vec, LieTensor) else vec
    return torch.matmul(input, vec.unsqueeze(-1), out=out).squeeze(-1)


def msqrt(input, method='mpa'):
    r'''
    Compute batched matrix square root

    Args:
        input (:obj:`Tensor`): matrices to be square rooted.
        method(:obj:`Str`, optional): Calculation method of matrix square root. Default: ``mpa``

    Return:
        input_sqrt (:obj:`Tensor`): the matrix square root of input.

    Note:
        - ``input`` has to be a (:math:`b\times n \times n`) tensor,and ``out`` will be a (
        :math:`b\times n \times n`) tensor.

    Example1:
        >>> matrix = torch.randn(10, 10).unsqueeze(0)
        >>> out = pp.msqrt(matrix)
        >>> out.shape
        torch.Size([1, 10, 10])

    Example2:
        >>> matrix = torch.randn(2, 10, 10)
        >>> out = pp.msqrt(matrix)
        >>> out.shape
        torch.Size([2, 10, 10])

    '''
    assert input.ndim == 3, 'Invalid Input dim'

    if method == 'mpa':
        # matrix pade approximant
        input_norm = torch.linalg.norm(input, dim=[1, 2]).reshape(input.size(0), 1, 1)
        I = torch.eye(input.size(1), device=input.device).reshape(1, input.size(1),
                                                                  input.size(
                                                                      1)).repeat(
            input.size(0), 1, 1)
        return Msqrt().matrix_pade_approximant(input, input_norm, I)
    elif method == 'gcholesky':
        return Msqrt().generalized_cholesky(input)
    elif method == 'contour_integral':
        return Msqrt().contour_integral(input)
    else:
        ValueError('The model_name parameters are incorrect')


class Msqrt:
    r'''
     Compute batched Matrix Square root
    '''

    def matrix_pade_approximant(self, input, input_norm, I):
        r'''
        compute batched Padé approximant matrix square root

        Args:
            input (:obj:`Tensor`): matrices to be square rooted.
            input_norm(:obj:`Tensor`): Frobenius Norm of input.
            I (:obj:`Tensor`): Identity matrix.
        Return:
            input_sqrt (:obj:`Tensor`): the matrix square root of input.

        Matrix Pade approximants can be described as the following  equations:

        1.Consider the :math:`\mathbf{A}` power series:

            .. math::
                \left ( 1-z \right ) =1-\sum_{k = 0}^{\infty } \left |\begin{pmatrix}1/2\\
                    k\end{pmatrix} \right | z^{k}

           where :math:`\begin{pmatrix}1\\2k\end{pmatrix}` denotes the binomial coefficients
           that involve fractions,and the series converges when the :math:`z<1` according to
           the Cauchy root test.


        2.The power series of matrix can be similarly defined by:

            .. math::
                \mathbf{A} ^{\frac{1}{2} } = \mathbf{I} - \sum_{k=0}^{\infty } \begin{
                pmatrix}1/2\\
                k\end{pmatrix}\left ( \mathbf{I} -\mathbf{A}  \right ) ^{k}
            where :math:`\mathbf{I}`  is the Identity  matrix

        3.matrix Taylor Polynomial matrix square root

            .. math::
                \mathbf{A} ^{\frac{1}{2} } = \sqrt{\left   \|\mathbf{A}   \right \| _{F}  }
                \left ( \mathbf{I} - \sum_{k=0}^{\infty }
                \begin{pmatrix}1/2\\k\end{pmatrix}\left ( \mathbf{I} -\frac{\mathbf{A} }
                {\sqrt{\left   \|\mathbf{A}   \right \| _{F}  }}  \right ) ^{k}  \right )
            where :math:`k` is the degree of Truncated the series and the :math:`\sqrt{\left
            \|\mathbf{A}   \right \| _{F}  }`  is the Square root of Frobenius Norm

        4.Padé approximant

            .. math::
                \frac{1-\sum_{m=1}^{M} p_mz^m}{1-\sum_{n=1}^{N}q_nz^n } = 1-\sum_{k=1}^{M+N}
                \left | \begin{pmatrix}
                 1/2 \\ k
                \end{pmatrix} \right | z^k

        5.Padé approximant  matrix square root

            .. math::

                \mathbf{A^{\frac{1}{2} } } = \sqrt{\left \| \mathbf{A}  \right \|_{F}  }
                \mathbf{Q} _{-1}^{N} \mathbf{P}_m\\
            where,

            .. math::
                \left\{\begin{matrix}
                    \mathbf{P}_M = \mathbf{I}  -\sum_{m=1}^{M} p_m\left ( \mathbf{I} -\frac{
                    \mathbf{A} }{\sqrt{\left \| \mathbf{A}  \right \|_{F}  }}  \right )^m
                     \\
                    \mathbf{Q}_N = \mathbf{I}  -\sum_{n=1}^{N} q_n\left ( \mathbf{I} -\frac{
                    \mathbf{A} }{\sqrt{\left \| \mathbf{A}  \right \|_{F}  }}  \right ) ^n
                     \\
                      M =N=\frac{K-1}{2}
                      \\
                    \end{matrix}\right.
        6.solving the linear system

            .. math::
                \mathbf{Q} _N \mathbf{A} \frac{1}2{} ={\sqrt{\left \| \mathbf{A}  \right \|_{
                F}  }}\mathbf{ P_m}
        Example:
            >>> m_sqrt = Msqrt()
            >>> input = torch.randn(2,10,10)
            >>> input_norm = torch.linalg.norm(input, dim=[1, 2]).reshape(input.size(0), 1, 1)
            >>> I = torch.eye(input.size(1), requires_grad=False,
            device=input.device).reshape(1, input.size(1),input.size(1)).repeat(input.size(0),
            1, 1)
            >>> input_sqrt = m_sqrt.matrix_pade_approximant(input,input_norm,I)
            >>> input_sqrt.shape
            torch.Size([2, 10, 10])
        Note:
            - The initial values of P and Q in the code are different from those in the paper,
            so the code and formula are different. `Link
            <https://github.com/KingJamesSong/FastDifferentiableMatSqrt/issues/2#issuecomment
            -1364681194>`_

        Refrence:
            [1] Yue Song, Nicu Sebe, and Wei Wang. Fast differentiable matrix square root.
            ICLR. 2022.

        '''
        assert input.ndim == input_norm.ndim and input.ndim == I.ndim, 'Input arguments invalid'
        assert input.shape == I.shape, 'Input shape not equal to Identity matrix shape.'

        mpm.mp.dps = 32  # Set precision
        mpm.mp.pretty = True
        taylor_degree = 5
        taylor_approximation = mpm.taylor(lambda x: mpm.sqrt(mpm.mpf(1) - x), 0,
                                          taylor_degree * 2)
        pade_p, pade_q = mpm.pade(taylor_approximation, taylor_degree, taylor_degree)
        pade_p = torch.from_numpy(np.array(pade_p).astype(float))
        pade_q = torch.from_numpy(np.array(pade_q).astype(float))

        # compute P and Q
        p_sqrt = pade_p[0] * I
        q_sqrt = pade_q[0] * I
        input_prenormalize = input / input_norm  # pre-normalize the input matrices
        p_norm = I - input_prenormalize
        p_norm_k = p_norm
        for i in range(taylor_degree):
            p_sqrt += pade_p[i + 1] * p_norm_k
            q_sqrt += pade_q[i + 1] * p_norm_k
            p_norm_k = p_norm_k.bmm(p_norm)

        # solving the linear system
        input_sqrt = torch.linalg.solve(q_sqrt, torch.sqrt(input_norm) * p_sqrt)
        return input_sqrt

    def generalized_cholesky(self, input):
        n = input.shape[1]
        input_sqrt = []
        matrix = torch.eye(n)
        row, col = np.diag_indices_from(matrix)
        for f in input:
            p = torch.zeros(n, dtype=input.dtype, device=input.device)
            d = torch.zeros(n, dtype=input.dtype, device=input.device)
            D = torch.eye(n, dtype=input.dtype, device=input.device)
            for i in range(n):
                d[i] = 1.0
                for j in range(i, n):
                    sum = 0.0
                    for k in range(i):
                        sum += d[k] * f[i][k] * f[j][k]
                    aux = f[i][j] - sum
                    if i == j:
                        if aux <= 0.0:
                            d[i] = -1.0
                        p[i] = torch.sqrt(d[i] * aux)
                    else:
                        f[j][i] = d[i] * aux / p[i]

            D[row, col] = d
            f[row, col] = p
            # input_sqrt.append(torch.tensor(np.dot(np.tril(f),np.sqrt(np.abs(D))),
            # dtype = input.dtype,device = input.device))
            input_sqrt.append(torch.as_tensor(torch.tril(f), dtype=input.dtype,
                                              device=input.device))

        return torch.cat(input_sqrt)

    def contour_integral(self, input):

        n = input.shape[1]
        input_sqrt = []
        for A in input:

            A = A.cpu().numpy()
            I = np.eye(n)
            e, ev = np.linalg.eig(A)
            m = e.min()
            M = e.max()
            k2 = m / M
            Kp = ellipk(1 - k2)
            for N in range(n, 30, n):
                t = 1j * np.arange(n / 10, N) * Kp / N
                sn, cn, dn, pn = ellipj(np.imag(t), 1 - k2)
                cn = 1.0 / cn
                dn = dn * cn
                sn = 1j * sn * cn
                w = np.sqrt(m) * sn
                dzdt = cn * dn
                S = np.zeros(np.size(A)).reshape(n, n)
                for l in range(N):
                    S = S - np.linalg.pinv(A - w[l] ** 2 * I) * dzdt[l]
                S = (-2 * Kp * np.sqrt(m) / (np.pi * N)) * np.dot(A, S)
            input_sqrt.append(torch.tensor(S.real, dtype=input.dtype,
                                           device=input.device))
        return torch.cat(input_sqrt)

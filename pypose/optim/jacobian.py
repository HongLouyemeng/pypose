import torch, functorch
import sys, math, warnings
from torch import nn, Tensor
from torch.autograd.functional import jacobian


def modjac(model, inputs, flatten=False, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode'):
    r'''
    Compute the model Jacobian with respect to the model parameters.

    Args:
        model (torch.nn.Module): a PyTorch model that takes Tensor or LieTensor
            inputs and returns a tuple of Tensor/LieTensor or a Tensor/LieTensor.
        inputs (tuple of Tensors/LieTensor or Tensor/LieTensor): inputs to the
            module ``model``.
        flatten (bool, optional): If ``True``, all module parameters and outputs
            are flattened and concatenated to form a single vector. The Jacobian
            will be computed with respect to this single flattened vectors, thus
            a single Tensor will be returned.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we perform only a single ``autograd.grad`` call with
            ``batched_grad=True`` which uses the vmap prototype feature.
            Though this should lead to performance improvements in many cases,
            because this feature is still experimental, there may be performance
            cliffs. See :func:`torch.autograd.grad`'s ``batched_grad`` parameter for
            more information.
        strategy (str, optional): Set to ``"forward-mode"`` or ``"reverse-mode"`` to
            determine whether the Jacobian will be computed with forward or reverse
            mode AD. Currently, ``"forward-mode"`` requires ``vectorized=True``.
            Defaults to ``"reverse-mode"``. If ``func`` has more outputs than
            inputs, ``"forward-mode"`` tends to be more performant. Otherwise,
            prefer to use ``"reverse-mode"``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        parameter and output, this will be a single Tensor containing the
        Jacobian for the linearized parameter and output. If there are more
        than one parameters, then the Jacobian will be a tuple of Tensors.
        If there are more than one outputs (even if there is only one parameter),
        then the Jacobian will be a tuple of tuple of Tensors where ``Jacobian[i][j]``
        will contain the Jacobian of the ``i``\th output and ``j``\th parameter
        and will have as size the concatenation of the sizes of the corresponding
        output and the corresponding parameter and will have same dtype and device as the
        corresponding parameter. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the parameters.

    Warning:
        The function :obj:`modjac` calculate Jacobian of model parameters.
        This is in contrast to PyTorch's function `jacobian
        <https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html>`_,
        which computes the Jacobian of a given Python function.

    Example:

        Calculates Jacobian with respect to all model parameters.

        >>> model = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        >>> inputs = torch.randn(1, 1, 1)
        >>> J = pp.optim.modjac(model, inputs)
        (tensor([[[[[[[0.3365]]]]]]]), tensor([[[[1.]]]]))
        >>> [j.shape for j in J]
        [torch.Size([1, 1, 1, 1, 1, 1, 1]), torch.Size([1, 1, 1, 1])]

        Function with flattened parameters returns a combined Jacobian.

        >>> inputs = torch.randn(2, 2, 2)
        >>> model = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        >>> J = pp.optim.modjac(model, inputs, flatten=True)
        tensor([[-0.4162,  0.0968,  0.0000,  0.0000,  1.0000,  0.0000],
                [-0.6042,  1.1886,  0.0000,  0.0000,  1.0000,  0.0000],
                [ 1.4623,  0.7389,  0.0000,  0.0000,  1.0000,  0.0000],
                [ 1.0716,  2.4293,  0.0000,  0.0000,  1.0000,  0.0000],
                [ 0.0000,  0.0000, -0.4162,  0.0968,  0.0000,  1.0000],
                [ 0.0000,  0.0000, -0.6042,  1.1886,  0.0000,  1.0000],
                [ 0.0000,  0.0000,  1.4623,  0.7389,  0.0000,  1.0000],
                [ 0.0000,  0.0000,  1.0716,  2.4293,  0.0000,  1.0000]])
        >>> J.shape
        torch.Size([8, 6])

        Calculate Jacobian with respect to parameter of :obj:`pypose.LieTensor`.

        >>> class PoseTransform(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.p = pp.Parameter(pp.randn_so3(2))
        ...
        ...     def forward(self, x):
        ...         return self.p.Exp() * x
        ...
        >>> model, inputs = PoseTransform(), pp.randn_SO3()
        >>> J = pp.optim.modjac(model, inputs, flatten=True)
        tensor([[ 0.4670,  0.7041,  0.0029,  0.0000,  0.0000,  0.0000],
                [-0.6591,  0.4554, -0.2566,  0.0000,  0.0000,  0.0000],
                [-0.2477,  0.0670,  0.9535,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.8593,  0.2672,  0.3446],
                [ 0.0000,  0.0000,  0.0000, -0.2417,  0.9503, -0.1154],
                [ 0.0000,  0.0000,  0.0000, -0.3630, -0.0179,  0.9055],
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
        >>> J.shape
        torch.Size([8, 6])
    '''
    func, params = functorch.make_functional(model)
    func_of_param = lambda *param: func(param, inputs)
    J = jacobian(func_of_param, params, create_graph, strict, vectorize, strategy)

    if flatten and isinstance(J, tuple):
        if any(isinstance(j, tuple) for j in J):
            J = torch.cat([torch.cat([j.view(-1, p.numel()) \
                    for j, p in zip(Jr, params)], dim=1) for Jr in J])
        else:
            J = torch.cat([j.view(-1, p.numel()) for j, p in zip(J, params)], dim=1)

    return J


def modjacrev(model, inputs, argnums=0, *, has_aux=False):
    func, params = functorch.make_functional(model)
    jacrev = functorch.jacrev(func, argnums=argnums, has_aux=has_aux)
    return jacrev(params, inputs)


def modjacfwd(model, inputs, argnums=0, *, has_aux=False):
    func, params = functorch.make_functional(model)
    jacfwd = functorch.jacfwd(func, argnums=argnums, has_aux=has_aux)
    return jacfwd(params, inputs)

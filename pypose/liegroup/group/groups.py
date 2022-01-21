import torch
from torch import nn
from .broadcasting import broadcast_inputs
from .group_ops import exp, log, inv, mul, adj
from .group_ops import adjT, jinv, act3, act4, toMatrix


HANDLED_FUNCTIONS = ['view', 'view_as', 'squeeze', 'unsqueeze', 'cat', 'stack',
                     'split', 'hsplit', 'dsplit', 'vsplit', 'tensor_split']


class GroupType:
    '''Lie Group Type Base Class'''
    def __init__(self, groud, dimension, embedding, manifold):
        self._group     = groud     # Group ID
        self._dimension = dimension # Data dimension
        self._embedding = embedding # Embedding dimension
        self._manifold  = manifold  # Manifold dimension

    @property
    def group(self):
        return self._group

    @property
    def dimension(self):
        return self._dimension

    @property
    def embedding(self):
        return self._embedding

    @property
    def manifold(self):
        return self._manifold

    @property
    def on_manifold(self):
        return self.dimension == self.manifold

    def Log(self, X):
        if self.on_manifold:
            raise AttributeError("Manifold Type has no Log attribute")
        raise NotImplementedError("Instance has no Log attribute.")

    def Exp(self, x):
        if not self.on_manifold:
            raise AttributeError("Embedding Type has no Exp attribute")
        raise NotImplementedError("Instance has no Exp attribute.")

    def Inv(self, x):
        if self.on_manifold:
            return LieGroup(-x, gtype=x.gtype, requires_grad=x.requires_grad)
        out = self.__op__(self.group, inv, x)
        return LieGroup(out, gtype=x.gtype, requires_grad=x.requires_grad)

    def Act(self, x, p):
        """ action on a points tensor(*, 3[4]) (homogeneous)"""
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        act = act3 if p.shape[-1]==3 else act4
        return self.__op__(self.group, act, x, p)

    def Mul(self, x, y):
        # Transform on transform
        if not self.on_manifold and isinstance(y, LieGroup) and not y.gtype.on_manifold:
            out = self.__op__(self.group, mul, x, y)
            return LieGroup(out, gtype=x.gtype, requires_grad=x.requires_grad)
        # Transform on points
        if not self.on_manifold and isinstance(y, torch.Tensor):
            return self.Act(x, y)
        # scalar * manifold
        if self.on_manifold:
            if isinstance(y, torch.Tensor):
                assert y.dim()==0 or y.shape[-1]==1, "Tensor Dimension Invalid"
            return torch.mul(x, y)
        raise NotImplementedError('Invalid __mul__ operation')

    def Retr(self, X, a):
        if self.on_manifold:
            raise AttributeError("Gtype has no Retr attribute")
        return a.Exp() * X

    def Adj(self, X, a):
        ''' X * Exp(a) = Exp(Adj) * X '''
        if self.on_manifold:
            raise AttributeError("Gtype has no Adj attribute")
        assert not X.gtype.on_manifold and a.gtype.on_manifold
        assert X.gtype.group == a.gtype.group
        out = self.__op__(self.group, adj, X, a)
        return LieGroup(out, gtype=a.gtype, requires_grad=a.requires_grad or X.requires_grad)

    def AdjT(self, X, a): # It seems that only works for SO3, but not SE3
        ''' Exp(a) * X = X * Exp(AdjT) '''
        if self.on_manifold:
            raise AttributeError("Gtype has no AdjT attribute")
        assert not X.gtype.on_manifold and a.gtype.on_manifold, "Gtype Invalid"
        assert X.gtype.group == a.gtype.group, "Gtype Invalid"
        out = self.__op__(self.group, adjT, X, a)
        return LieGroup(out, gtype=a.gtype, requires_grad=a.requires_grad or X.requires_grad)

    def Jinv(self, X, a):
        if self.on_manifold:
            raise AttributeError("Gtype has no Jinv attribute")
        return self.__op__(self.group, jinv, X, a)

    def matrix(self, gtensor):
        """ To 4x4 matrix """
        X = gtensor.Exp() if self.on_manifold else gtensor
        I = torch.eye(4, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [4, 4])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def translation(self, gtensor):
        """ To translation """
        X = gtensor.Exp() if self.on_manifold else gtensor
        p = torch.tensor([0., 0., 0.], dtype=X.dtype, device=X.device)
        return X.Act(p.view([1] * (X.dim() - 1) + [3,]))

    def quaternion(self, gtensor):
        raise NotImplementedError('quaternion not implemented yet')

    @classmethod
    def identity(cls, *args, **kwargs):
        raise NotImplementedError("Instance has no identity.")

    @classmethod
    def identity_like(cls, *args, **kwargs):
        return cls.identity(*args, **kwargs)

    def randn_like(self, *args, sigma=1, **kwargs):
        return self.randn(*args, sigma=1, **kwargs)

    def randn(self, *args, sigma=1., **kwargs):
        return sigma * torch.randn(*(list(args)+[self.manifold]), **kwargs)

    @classmethod
    def __op__(cls, group, op, x, y=None):
        inputs, out_shape = broadcast_inputs(x, y)
        out = op.apply(group, *inputs)
        return out.view(out_shape + (-1,))


class SO3Type(GroupType):
    def __init__(self):
        super().__init__(1, 4, 4, 3)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=so3_type, requires_grad=X.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=SO3_type, requires_grad=data.requires_grad)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = so3_type.Exp(so3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=SO3_type).requires_grad_(requires_grad)


class so3Type(GroupType):
    def __init__(self):
        super().__init__(1, 3, 4, 3)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=SO3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SO3_type.Log(SO3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=so3_type).requires_grad_(requires_grad)


class SE3Type(GroupType):
    def __init__(self):
        super().__init__(3, 7, 7, 6)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=se3_type, requires_grad=X.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=SE3_type, requires_grad=data.requires_grad)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = se3_type.Exp(se3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=SE3_type).requires_grad_(requires_grad)


class se3Type(GroupType):
    def __init__(self):
        super().__init__(3, 6, 7, 6)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=SE3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SE3_type.Log(SE3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=se3_type).requires_grad_(requires_grad)


class Sim3Type(GroupType):
    def __init__(self):
        super().__init__(4, 8, 8, 7)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=sim3_type, requires_grad=X.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=Sim3_type, requires_grad=data.requires_grad)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = sim3_type.Exp(sim3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=Sim3_type).requires_grad_(requires_grad)


class sim3Type(GroupType):
    def __init__(self):
        super().__init__(4, 7, 8, 7)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=Sim3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return Sim3_type.Log(Sim3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=sim3_type).requires_grad_(requires_grad)


class RxSO3Type(GroupType):
    def __init__(self):
        super().__init__(2, 5, 5, 4)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=rxso3_type, requires_grad=X.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 1., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=rxso3_type, requires_grad=data.requires_grad)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = rxso3_type.Exp(rxso3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=RxSO3_type).requires_grad_(requires_grad)


class rxso3Type(GroupType):
    def __init__(self):
        super().__init__(2, 4, 5, 4)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=RxSO3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return RxSO3_type.Log(RxSO3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=rxso3_type).requires_grad_(requires_grad)


SO3_type, so3_type = SO3Type(), so3Type()
SE3_type, se3_type = SE3Type(), se3Type()
Sim3_type, sim3_type = Sim3Type(), sim3Type()
RxSO3_type, rxso3_type = RxSO3Type(), rxso3Type()


class LieGroup(torch.Tensor):
    """ Lie Group """
    def __init__(self, data, gtype, **kwargs):
        assert data.shape[-1] == gtype.dimension, 'Dimension Invalid.'
        self.gtype = gtype

    def __new__(cls, data=None, **kwargs):
        if data is None:
            data = torch.tensor([])
        return torch.Tensor.as_subclass(data, LieGroup) 

    def __repr__(self):
        if hasattr(self, 'gtype'):
            return self.gtype.__class__.__name__ + " Group:\n" + super().__repr__()
        else:
            return super().__repr__()

    @classmethod
    def __torch_function__(cls, func, types, *args, **kwargs):
        if func.__name__ in HANDLED_FUNCTIONS:
            data = super().__torch_function__(func, types, *args, **kwargs)
            liegroup = args
            while not isinstance(liegroup, LieGroup):
                liegroup = liegroup[0]
            if isinstance(data, tuple):
                return (cls(item, gtype=liegroup.gtype) for item in data)
            return cls(data, gtype=liegroup.gtype)
        return super().__torch_function__(func, types, *args, **kwargs)

    @property
    def gshape(self):
        return self.shape[:-1]

    def gview(self, *shape):
        return self.view(*shape+(self.gtype.dimension,))

    def Exp(self):
        return self.gtype.Exp(self)

    def Log(self):
        return self.gtype.Log(self)

    def Inv(self):
        return self.gtype.Inv(self)

    def Act(self, p):
        return self.gtype.Act(self, p)

    def __mul__(self, other):
        return self.gtype.Mul(self, other)

    def Retr(self, a):
        return self.gtype.Retr(self, a)

    def Adj(self, a):
        return self.gtype.Adj(self, a)

    def AdjT(self, a):
        return self.gtype.AdjT(self, a)

    def Jinv(self, a):
        return self.gtype.Jinv(self, a)

    def tensor(self):
        return self.data

    def matrix(self):
        return self.gtype.matrix(self)

    def translation(self):
        return self.gtype.translation(self)

    def quaternion(self):
        return self.gtype.quaternion(self)


class Parameter(LieGroup, nn.Parameter):
    def __new__(cls, data=None, gtype=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return LieGroup._make_subclass(cls, data, requires_grad)

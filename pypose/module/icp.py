import torch
import pypose as pp
from torch import nn
from sklearn.neighbors import NearestNeighbors
from torch.linalg import svd,det
class ICP(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.get_trans = self.icp(A, B)

    def best_fit_transform(self, A, B):
        assert A.shape == B.shape
        m = A.shape[1]
        centroid_A = torch.mean(A, axis=0)
        centroid_B = torch.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = AA.mT @ BB
        U, S, Vt = svd(H)
        R = Vt.T @ U.T
        if det(R) < 0:
           Vt[m-1, :] *= -1
           R = Vt.T @ U.T
        t = centroid_B.T - R @ centroid_A.T
        T = torch.eye(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def nearest_neighbor(self, src, dst):
        assert src.shape == dst.shape
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)

        return distances.ravel(), indices.ravel()

    def icp(self, A, B, init_pose=None, max_iterations=20, tolerance=0.001):

        assert A.shape == B.shape
        m = A.shape[1]
        src = torch.ones((m+1, A.shape[0]))
        dst = torch.ones((m+1, B.shape[0]))
        src[:m, :] = torch.as_tensor(A.T)
        dst[:m, :] = torch.as_tensor(B.T)
        if init_pose is not None:
            src = init_pose @ src
        prev_error = 0
        for i in range(max_iterations):
            distances, indices = self.best_fit_transform(src[:m, :].T, dst[:m, :].T)
            T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)
            src = T @ src
            mean_error = torch.mean(distances)
            if torch.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
        T, _, _ = self.best_fit_transform(A, src[:m, :].T)

        return
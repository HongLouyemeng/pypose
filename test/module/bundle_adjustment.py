import pypose as pp
from torch import nn
import torch
import bz2


def read_bal_data(file_name):
    # https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = torch.zeros(n_observations, dtype=torch.int)
        point_indices = torch.zeros(n_observations, dtype=torch.int)
        points_2d = torch.zeros((n_observations, 2), dtype=torch.float)

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = torch.tensor([float(x), float(y)], dtype=torch.float)

        camera_params = torch.zeros(n_cameras * 9, dtype=torch.float)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = torch.zeros(n_points * 3, dtype=torch.float)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


class BA(nn.Module):
    def __init__(self, n, camera_param):
        super().__init__()
        self.pose = pp.Parameter(pp.randn_SE3(n))
        self.camera_param = camera_param

    def forward(self, x3d):
        return self.reprojection(x3d)

    def reprojection(self, points3d):
        # https://scipy - cookbook.readthedocs.io / items / bundle_adjustment.html
        points_proj = self.pose @ points3d
        points_proj += self.camera_param[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2].unsqueeze(1)
        f = self.camera_param[:, 6]
        k1 = self.camera_param[:, 7]
        k2 = self.camera_param[:, 8]
        n = torch.sum(points_proj ** 2, dim=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f).unsqueeze(1)

        return points_proj


if __name__ == '__main__':

    # n, epoch = 1, 10
    # inputs = pp.randn_SE3(n)
    # target = pp.randn_SE3(n)
    # net = BA(n)
    # optimizer = pp.optim.LM(net)
    # for idx in range(epoch):
    #     optimizer.zero_grad()
    #
    #     loss = optimizer.step(inputs, target)
    #     print('Pose Inversion loss %.7f @ %d it' % (loss, idx))
    #
    #     if loss < 1e-5:
    #         print('Early Stopping with loss:', loss.item())
    #         break
    # print(net.pose)
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(
        'data/problem-49-7776-pre.txt.bz2')
    points_3d = torch.from_numpy(points_3d.numpy()[point_indices])
    camera_params = torch.from_numpy(camera_params.numpy()[camera_indices])
    net = BA(len(camera_params), camera_params)
    optimizer = pp.optim.LM(net)
    epoch = 10
    for idx in range(epoch):
        optimizer.zero_grad()
        loss = optimizer.step(points_3d, points_2d)
        print('Pose Inversion loss %.7f @ %d it' % (loss, idx))

        if loss < 1e-5:
            print('Early Stopping with loss:', loss.item())
            break

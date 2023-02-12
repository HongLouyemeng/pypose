import torch, argparse
from pypose.module import UKF
from bicycle import Bicycle, bicycle_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UKF Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--k", type=int, default=3, help='A integer parameter for \
        weighting the sigma points.')
    parser.add_argument("--num_sensors", type=int, default=3, help='A integer parameter for \
            number of sensor.')
    args = parser.parse_args()

    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation
    num_sensors = args.num_sensors
    input = torch.randn(T, num_sensors, M, device=args.device) * 0.1 + \
            torch.tensor([1, 0], device=args.device)
    state = torch.zeros(T, num_sensors, N, device=args.device)  # true states
    est = torch.randn(T, num_sensors, N, device=args.device) * p  # estimation
    obs = torch.zeros(T, num_sensors, N, device=args.device)  # observation
    P = torch.eye(N, device=args.device).repeat(T, num_sensors, 1,
                                                1) * p ** 2  # estimation covariance
    Q = torch.eye(N, device=args.device) * q ** 2  # covariance of transition
    R = torch.eye(N, device=args.device) * r ** 2  # covariance of observation

    bicycle = Bicycle()
    filter = UKF(bicycle, Q, R).to(args.device)

    for i in range(T - 1):
        w = q * torch.randn(num_sensors, N, device=args.device)
        v = r * torch.randn(num_sensors, N, device=args.device)
        state[i + 1], obs[i] = bicycle(state[i] + w, input[i])  # model measurement
        est[i + 1], P[i + 1] = filter(est[i], obs[i] + v, input[i], P[i], k=args.k)
    est = est.mean(dim=1)
    P = P.mean(dim=1)
    state = state.mean(dim=1)
    error = (state - est).norm(dim=-1)
    bicycle_plot('UKF', state, est, P)

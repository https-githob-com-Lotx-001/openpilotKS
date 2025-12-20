import numpy as np
from selfdrive.controls.lib.lat_kalman import KalmanFilter
from selfdrive.controls.lib.lqr import lqr

class LatControlLQG:
    def __init__(self):
        self.A = np.array([[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, -1]])
        self.B = np.array([[0], [0], [0], [1]])
        self.C = np.eye(4)

        self.Q = np.diag([1.0, 1.0, 0.5, 0.1])
        self.R = np.array([[1.0]])

        self.K = lqr(self.A, self.B, self.Q, self.R)

        self.kalman = KalmanFilter(
            self.A, self.B, self.C,
            Q=np.eye(4) * 0.01,
            R=np.eye(4) * 0.1
        )

    def update(self, y, steer_prev):
        x_hat = self.kalman.update(y, np.array([[steer_prev]]))
        steer = -self.K @ x_hat
        return float(steer)

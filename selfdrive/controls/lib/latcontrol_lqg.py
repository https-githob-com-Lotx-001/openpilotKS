import numpy as np
from selfdrive.controls.lib.lat_kalman import KalmanFilter
from selfdrive.controls.lib.lqr import lqr

class LatControlLQG:
    def __init__(self, tune):
    self.Q = np.diag([
        tune.qLat,
        tune.qHead,
        tune.qYaw,
        tune.qSteer
    ])
    self.R = np.array([[tune.rSteer]])

        # ===== TUNING (كادينزا 2018) =====
        self.Q = np.diag([3.0, 2.0, 0.5, 0.2])
        self.R = np.array([[2.5]])

        self.K = lqr(self.A, self.B, self.Q, self.R)

        self.kalman = KalmanFilter(
            self.A, self.B, self.C,
            Q=np.eye(4) * 0.01,
            R=np.eye(4) * 0.1
        )

        self.last_steer = 0.0

    def update(self, y):
        x_hat = self.kalman.update(y, np.array([[self.last_steer]]))
        steer = float(-self.K @ x_hat)

        # limits (مهم لمنع الرعشة)
        steer = np.clip(steer, -1.0, 1.0)

        self.last_steer = steer
        return steer

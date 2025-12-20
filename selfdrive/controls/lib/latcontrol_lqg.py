import numpy as np
from openpilot.common.params import Params
from selfdrive.controls.lib.lat_kalman import KalmanFilter
from selfdrive.controls.lib.lqr import lqr


class LatControlLQG:
  def __init__(self, tune):
    # ===== State-space model =====
    self.A = np.array([
      [0., 1., 0., 0.],
      [0., 0., 1., 0.],
      [0., 0., 0., 1.],
      [0., 0., 0., -1.]
    ])

    self.B = np.array([
      [0.],
      [0.],
      [0.],
      [1.]
    ])

    self.C = np.eye(4)

    # ===== LQG tuning from Params =====
    self.Q = np.diag([
      float(Params().get("LqgQLat")),
      float(Params().get("LqgQHead")),
      float(Params().get("LqgQYaw")),
      float(Params().get("LqgQSteer")),
    ])

    self.R = np.array([[float(Params().get("LqgRSteer"))]])

    # ===== LQR gain =====
    self.K = lqr(self.A, self.B, self.Q, self.R)

    # ===== Kalman Filter =====
    self.kalman = KalmanFilter(
      self.A, self.B, self.C,
      Q=np.eye(4) * 0.01,
      R=np.eye(4) * 0.1
    )

    self.last_steer = 0.0

  def update(self, y):
    # State estimation
    x_hat = self.kalman.update(
      y, np.array([[self.last_steer]])
    )

    # LQG control
    steer = float(-self.K @ x_hat)

    # Safety limits
    steer = np.clip(steer, -1.0, 1.0)

    self.last_steer = steer
    return steer

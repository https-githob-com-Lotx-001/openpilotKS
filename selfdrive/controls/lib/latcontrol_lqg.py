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

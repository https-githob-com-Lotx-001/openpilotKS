import numpy as np
from scipy.linalg import solve_discrete_are

class KalmanFilter:
    def __init__(self, A, B, Qk, Rk):
        self.A = A
        self.B = B
        self.Qk = Qk
        self.Rk = Rk

        self.x_hat = np.zeros((A.shape[0], 1))
        self.P = np.eye(A.shape[0])

    def update(self, y, u):
        # Prediction
        x_pred = self.A @ self.x_hat + self.B * u
        P_pred = self.A @ self.P @ self.A.T + self.Qk

        # Update
        S = P_pred + self.Rk
        K = P_pred @ np.linalg.inv(S)

        self.x_hat = x_pred + K @ (y - x_pred)
        self.P = (np.eye(len(self.P)) - K) @ P_pred

        return self.x_hat


class LQG:
    def __init__(self, A, B, Q, R, Qk, Rk):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

        self.kf = KalmanFilter(A, B, Qk, Rk)

    def control(self, y, u_prev):
        x_hat = self.kf.update(y, u_prev)
        u = -self.K @ x_hat
        return float(u)

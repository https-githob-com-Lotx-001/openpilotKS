import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are

class RLS:
    def __init__(self, n, lam=0.995):
        self.theta = np.zeros((n, 1))
        self.P = np.eye(n) * 1000.0
        self.lam = lam

    def update(self, phi, y):
        phi = phi.reshape(-1, 1)
        y = np.array([[y]])

        K = self.P @ phi / (self.lam + phi.T @ self.P @ phi)
        self.theta += K @ (y - phi.T @ self.theta)
        self.P = (self.P - K @ phi.T @ self.P) / self.lam
        return self.theta


class AdaptiveLQG:
    def __init__(self, dt):
        self.dt = dt
        self.nx = 4

        self.A = np.eye(self.nx)
        self.B = np.zeros((self.nx, 1))
        self.B[1, 0] = dt
        self.B[3, 0] = dt

        self.Q = np.diag([1.0, 0.3, 1.2, 0.2])
        self.R = np.array([[0.8]])

        self.Qk = np.diag([0.01, 0.05, 0.01, 0.05])
        self.Rk = np.diag([0.2, 0.4, 0.2, 0.4])

        self.x_hat = np.zeros((self.nx, 1))
        self.P = np.eye(self.nx)

        self.rls = RLS(self.nx + 1)
        self.last_u = 0.0
        self.K = np.zeros((1, self.nx))

    def update_model(self, x, u):
        phi = np.hstack([x.flatten(), [u]])
        for i in range(self.nx):
            theta = self.rls.update(phi, x[i])
            self.A[i, :] = theta[:self.nx, 0]
            self.B[i, 0] = theta[self.nx, 0]

    def kalman(self, y, u):
        x_pred = self.A @ self.x_hat + self.B * u
        P_pred = self.A @ self.P @ self.A.T + self.Qk

        S = P_pred + self.Rk
        Kk = P_pred @ inv(S)

        self.x_hat = x_pred + Kk @ (y - x_pred)
        self.P = (np.eye(self.nx) - Kk) @ P_pred
        return self.x_hat

    def solve_lqr(self):
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)

    def control(self, y):
        x_hat = self.kalman(y, self.last_u)
        self.solve_lqr()
        u = -self.K @ x_hat
        self.last_u = float(u)
        return float(u)

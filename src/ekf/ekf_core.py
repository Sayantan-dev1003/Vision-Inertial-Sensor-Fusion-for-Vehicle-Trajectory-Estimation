from typing import Sequence
import numpy as np

from .motion_model import PlanarVehicleMotionModel


class ExtendedKalmanFilter:
    """Numerically stable Extended Kalman Filter using Joseph covariance updates & gating."""

    def __init__(
        self,
        motion_model: PlanarVehicleMotionModel,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.model = motion_model
        self.x = np.asarray(initial_state, dtype=float).reshape(motion_model.STATE_DIM)
        self.P = np.asarray(initial_covariance, dtype=float).reshape(
            motion_model.STATE_DIM, motion_model.STATE_DIM
        )

    def predict(self, u: np.ndarray, dt: float) -> None:
        if dt <= 0.0:
            return
        F = self.model.jacobian_F(self.x, dt)
        Q = self.model.process_noise_Q(self.x, dt)
        self.x = self.model.predict_state(self.x, u, dt)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)  # Enforce symmetry

    def update(
        self,
        z: np.ndarray,
        h_x: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        angle_indices: Sequence[int] = (),
        mahalanobis_gate: float | None = None,
    ) -> bool:
        """Apply linear/linearized measurement update. Returns True if accepted."""
        y = np.asarray(z, dtype=float) - np.asarray(h_x, dtype=float)
        for idx in angle_indices:
            y[idx] = self.model.wrap_angle(float(y[idx]))

        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        if mahalanobis_gate is not None:
            nis = float(y.T @ S_inv @ y)
            if nis > mahalanobis_gate:
                return False

        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[2] = max(0.0, float(self.x[2]))
        self.x[3] = self.model.wrap_angle(float(self.x[3]))

        # Joseph-form covariance update for numerical stability
        I = np.eye(self.model.STATE_DIM, dtype=float)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True
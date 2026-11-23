import numpy as np


class PlanarVehicleMotionModel:
    """
    6-DOF planar vehicle kinematic model with Gauss-Markov bias damping:
      state x = [px, py, v, psi, b_a, b_g]^T
      input u = [a_m, omega_m]^T
    """

    STATE_DIM: int = 6

    def __init__(
        self,
        pos_noise_std: float = 0.85,
        accel_noise_std: float = 0.90,
        gyro_noise_std: float = 0.045,
        acc_bias_walk_std: float = 0.04,
        gyro_bias_walk_std: float = 0.002,
        acc_bias_tau_s: float = 30.0,
        max_speed_m_s: float = 28.0,
        max_dt_s: float = 0.50,
    ) -> None:
        self.pos_noise_std = float(pos_noise_std)
        self.accel_noise_std = float(accel_noise_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.acc_bias_walk_std = float(acc_bias_walk_std)
        self.gyro_bias_walk_std = float(gyro_bias_walk_std)
        self.acc_bias_tau_s = float(acc_bias_tau_s)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_dt_s = float(max_dt_s)

    @staticmethod
    def wrap_angle(angle_rad: float) -> float:
        return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

    def predict_state(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 0.0, self.max_dt_s))
        if dt <= 0.0:
            return x.copy()

        px, py, v, psi, b_a, b_g = x
        a_m, omega_m = u

        a_true = a_m - b_a
        omega_true = omega_m - b_g

        px_next = px + v * np.cos(psi) * dt
        py_next = py + v * np.sin(psi) * dt
        v_next = float(np.clip(v + a_true * dt, 0.0, self.max_speed_m_s))
        psi_next = self.wrap_angle(psi + omega_true * dt)
        ba_next = float(np.clip(b_a * np.exp(-dt / self.acc_bias_tau_s), -1.2, 1.2))
        bg_next = float(np.clip(b_g, -0.15, 0.15))

        return np.array([px_next, py_next, v_next, psi_next, ba_next, bg_next], dtype=float)

    def jacobian_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 0.0, self.max_dt_s))
        _, _, v, psi, _, _ = x

        F = np.eye(self.STATE_DIM, dtype=float)
        F[0, 2] = np.cos(psi) * dt
        F[0, 3] = -v * np.sin(psi) * dt
        F[1, 2] = np.sin(psi) * dt
        F[1, 3] = v * np.cos(psi) * dt
        F[2, 4] = -dt
        F[3, 5] = -dt
        F[4, 4] = float(np.exp(-dt / self.acc_bias_tau_s))
        return F

    def process_noise_Q(self, x: np.ndarray, dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 1e-4, self.max_dt_s))

        q_pos = (self.pos_noise_std * dt) ** 2
        q_v = (self.accel_noise_std * np.sqrt(dt)) ** 2
        q_psi = (self.gyro_noise_std * np.sqrt(dt)) ** 2
        q_ba = (self.acc_bias_walk_std * np.sqrt(dt)) ** 2
        q_bg = (self.gyro_bias_walk_std * np.sqrt(dt)) ** 2

        return np.diag([q_pos, q_pos, q_v, q_psi, q_ba, q_bg])
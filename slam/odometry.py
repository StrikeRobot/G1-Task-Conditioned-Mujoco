"""Noisy odometry: takes true velocity from sim, returns delta pose (body frame) with noise+bias."""
import numpy as np


class NoisyOdometry:
    def __init__(self, noise_v=0.03, noise_w=0.02, bias_w=0.002, seed=0):
        self.rng = np.random.default_rng(seed)
        self.noise_v, self.noise_w = noise_v, noise_w
        self.bias_w = self.rng.normal(0, bias_w)

    def step(self, v_body_xy, wz, dt):
        """Return (dx, dy, dyaw) in body frame, with noise applied."""
        v = np.asarray(v_body_xy) * (1 + self.rng.normal(0, self.noise_v, 2))
        w = wz + self.rng.normal(0, self.noise_w) + self.bias_w
        return np.array([v[0] * dt, v[1] * dt, w * dt])

import math


class OneEuroFilter2D:
    """
    1Euro Filter for 2D points.

    Dynamically adjusts its cutoff frequency based on the speed of the point.
    - mincutoff: Minimum cutoff frequency (Hz). Decreasing this reduces slow-speed jitter.
    - beta: Speed coefficient. Increasing this reduces high-speed lag.
    - dcutoff: Cutoff frequency for the derivative (Hz). Typically 1.0.
    """

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0) -> None:
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: tuple[float, float] | None = None
        self.dx_prev: tuple[float, float] | None = None
        self.t_prev: float | None = None

    def _alpha(self, dt: float, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: tuple[float, float], t: float) -> tuple[float, float]:
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = (0.0, 0.0)
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0.0:
            return x

        ad = self._alpha(dt, self.dcutoff)
        dx = ((x[0] - self.x_prev[0]) / dt, (x[1] - self.x_prev[1]) / dt)
        dx_hat = (ad * dx[0] + (1 - ad) * self.dx_prev[0], ad * dx[1] + (1 - ad) * self.dx_prev[1])

        speed = math.hypot(dx_hat[0], dx_hat[1])
        cutoff = self.mincutoff + self.beta * speed
        a = self._alpha(dt, cutoff)

        x_hat = (a * x[0] + (1 - a) * self.x_prev[0], a * x[1] + (1 - a) * self.x_prev[1])

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def snapshot(self) -> dict:
        return {"x_prev": self.x_prev, "dx_prev": self.dx_prev, "t_prev": self.t_prev}

    @property
    def dx(self) -> float:
        """Smoothed X velocity (pixels/s) from the derivative filter."""
        return self.dx_prev[0] if self.dx_prev is not None else 0.0

    @property
    def dy(self) -> float:
        """Smoothed Y velocity (pixels/s) from the derivative filter."""
        return self.dx_prev[1] if self.dx_prev is not None else 0.0

    def restore(self, state: dict) -> None:
        self.x_prev = state["x_prev"]
        self.dx_prev = state["dx_prev"]
        self.t_prev = state["t_prev"]

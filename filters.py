"""
filters.py
----------------
Two-stage smoothing for noisy, jittery gaze/cursor signals.

MedianFilter: a short rolling-window median, used to reject single-frame
outlier spikes (a momentary landmark mis-detection, a half-blink, a flash
of glare) before anything adaptive touches the signal. A low-pass filter
alone doesn't reject outliers well -- it treats a spike as "the signal
suddenly moved fast" and, if adaptive, opens its cutoff to track it, which
is exactly the wrong response. A median filter throws single-frame spikes
out entirely instead of blending them in.

OneEuroFilter: smooths a stream of N-D points while staying responsive
during genuine fast movement.
Reference: Casiez, Roussel, Vogel (2012), "1-Euro Filter: A Simple
Speed-based Low-pass Filter for Noisy Input in Interactive Systems".

Why this instead of fixed-alpha exponential smoothing: a fixed alpha is a
tradeoff you can't win -- low alpha kills jitter but adds lag on fast
glances, high alpha tracks fast glances but lets jitter through. The
One-Euro filter adapts its cutoff based on how fast the signal is currently
moving, so it's smooth when your gaze is roughly still and responsive when
it's moving quickly. The important tuning knob for "moves when my eyes are
still" is `beta`: too high, and ordinary noise gets misread as movement,
the cutoff opens up, and more of that noise gets passed through -- a small
feedback loop of jitter. Keeping beta conservative (see main.py) means the
signal has to be moving a real amount before the filter loosens up.
"""
import time
import numpy as np


class MedianFilter:
    """Rolling-window median over the last `window` samples. Cheap outlier
    rejection to run before OneEuroFilter."""

    def __init__(self, window=3):
        self.window = max(1, window)
        self.buf = []

    def filter(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return np.median(np.array(self.buf), axis=0)

    def reset(self):
        self.buf = []


class _LowPassFilter:
    def __init__(self):
        self.y = None
        self.initialized = False

    def filter(self, x, alpha):
        if not self.initialized:
            self.y = x
            self.initialized = True
        else:
            self.y = alpha * x + (1 - alpha) * self.y
        return self.y

    def reset(self):
        self.y = None
        self.initialized = False


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """
    Smooths a stream of N-D points.

    min_cutoff : lower = more smoothing / more lag at low speed.
    beta       : higher = more responsive during fast movement (less lag,
                 but more jitter is let through while "moving fast" --
                 including apparent movement that's actually just noise).
    """
    def __init__(self, min_cutoff=1.0, beta=0.03, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = _LowPassFilter()
        self.dx_filter = _LowPassFilter()
        self.last_time = None

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

    def filter(self, x, timestamp=None):
        x = np.asarray(x, dtype=np.float64)
        t = timestamp if timestamp is not None else time.time()

        if self.last_time is None:
            dt = 1.0 / 30.0
        else:
            dt = max(t - self.last_time, 1e-6)
        self.last_time = t

        if not self.x_filter.initialized:
            dx = np.zeros_like(x)
        else:
            dx = (x - self.x_filter.y) / dt

        edx = self.dx_filter.filter(dx, _alpha(self.d_cutoff, dt))
        speed = float(np.linalg.norm(edx))
        cutoff = self.min_cutoff + self.beta * speed
        return self.x_filter.filter(x, _alpha(cutoff, dt))

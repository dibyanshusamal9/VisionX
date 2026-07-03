"""
calibration.py
----------------
Maps gaze features (h_ratio, v_ratio, yaw, pitch) to dashboard screen
coordinates using a 2nd-degree polynomial in the eye-ratio terms plus a
linear head-pose correction, fit by ridge-regularized least squares from a
short calibration sequence.

Why ridge regression, and why standardize first: with as few as ~10-13
calibration points and 8 regression coefficients, plain least squares fits
the calibration points almost exactly but extrapolates wildly anywhere the
squared terms weren't trained on -- which is most of the screen. That's the
single biggest cause of "the cursor jumps to a corner for no reason".
Regularizing keeps the fit closer to sane behavior past the calibrated
points instead of it flying toward infinity. Standardizing the feature
columns first makes the regularization penalty apply evenly across terms of
very different natural scale (h^2 vs yaw, say).

Why the ridge strength is cross-validated instead of a fixed constant: one
hardcoded lambda is either too weak for some people's calibration geometry
(still overfits, still extrapolates badly) or too strong for others (fit
doesn't track the calibration points well, cursor feels sluggish/offset).
Leave-one-out CV over a small lambda grid picks the value that actually
generalizes best for *this* calibration session, at negligible cost (a
handful of 8x8 solves).
"""
import numpy as np

LAMBDA_GRID = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]


def _poly_features(feat):
    """(h_ratio, v_ratio, yaw, pitch) -> [1, h, v, h^2, v^2, h*v, yaw, pitch]"""
    h, v, yaw, pitch = feat
    return np.array([1.0, h, v, h * h, v * v, h * v, yaw, pitch], dtype=np.float64)


def _standardize(A):
    mean = A[:, 1:].mean(axis=0)
    std = A[:, 1:].std(axis=0)
    std[std < 1e-8] = 1.0
    A_std = np.hstack([A[:, :1], (A[:, 1:] - mean) / std])
    return A_std, mean, std


def _ridge_solve(A_std, y, lam):
    n_features = A_std.shape[1]
    reg = lam * np.eye(n_features)
    reg[0, 0] = 0.0  # don't penalize the bias term
    return np.linalg.solve(A_std.T @ A_std + reg, A_std.T @ y)


def _loo_cv_sq_error(A, y, lam):
    """Leave-one-out cross-validated mean squared error for a given lambda."""
    n = A.shape[0]
    sq_errs = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        A_train, mean, std = _standardize(A[mask])
        w = _ridge_solve(A_train, y[mask], lam)

        a = A[i]
        a_std = np.concatenate([a[:1], (a[1:] - mean) / std])
        pred = a_std @ w
        sq_errs[i] = (pred - y[i]) ** 2
    return float(np.mean(sq_errs))


class GazeCalibrator:
    def __init__(self, ridge_lambda=None):
        """ridge_lambda: fixed value to use, or None (default) to auto-tune
        via leave-one-out CV over LAMBDA_GRID inside fit()."""
        self.samples_x = []
        self.samples_y = []
        self.wx = None
        self.wy = None
        self.calibrated = False
        self.ridge_lambda = ridge_lambda        # resolved value, set in fit() if auto-tuning
        self._auto_tune = ridge_lambda is None

        # Set by fit(): used to standardize features at predict time and to
        # clip inputs into the range the model was actually trained on.
        self._feat_mean = None
        self._feat_std = None
        self._raw_bounds = None  # (min, max) per raw feature
        self.fit_rmse = None     # training-error diagnostic, in pixels (optimistic)
        self.cv_rmse = None      # leave-one-out diagnostic, more honest than fit_rmse

    def add_sample(self, gaze_feat, screen_xy):
        self.samples_x.append(np.asarray(gaze_feat, dtype=np.float64))
        self.samples_y.append(screen_xy)

    def _pick_lambda(self, A, targets):
        best_lam, best_err = LAMBDA_GRID[0], np.inf
        for lam in LAMBDA_GRID:
            err = _loo_cv_sq_error(A, targets[:, 0], lam) + _loo_cv_sq_error(A, targets[:, 1], lam)
            if err < best_err:
                best_err, best_lam = err, lam
        return best_lam, float(np.sqrt(best_err))

    def fit(self):
        # Need a few points beyond the 8 coefficients or both the fit and
        # the CV lambda search are meaningless (and accuracy suffers).
        min_required = 8
        if len(self.samples_x) < min_required:
            return False

        raw = np.array(self.samples_x)
        A = np.array([_poly_features(f) for f in raw])
        targets = np.array(self.samples_y, dtype=np.float64)

        if self._auto_tune:
            self.ridge_lambda, self.cv_rmse = self._pick_lambda(A, targets)
        else:
            self.cv_rmse = float(np.sqrt(
                _loo_cv_sq_error(A, targets[:, 0], self.ridge_lambda)
                + _loo_cv_sq_error(A, targets[:, 1], self.ridge_lambda)
            ))

        A_std, mean, std = _standardize(A)
        self.wx = _ridge_solve(A_std, targets[:, 0], self.ridge_lambda)
        self.wy = _ridge_solve(A_std, targets[:, 1], self.ridge_lambda)
        self._feat_mean, self._feat_std = mean, std

        # Keep the raw (h, v, yaw, pitch) range seen during calibration so
        # predict() can clip into it (with a little slack) -- this stops the
        # quadratic terms from blowing up when gaze/head goes somewhere far
        # outside what was actually calibrated.
        margin = 0.15
        raw_min = raw.min(axis=0)
        raw_max = raw.max(axis=0)
        span = raw_max - raw_min
        self._raw_bounds = (raw_min - margin * span, raw_max + margin * span)

        # Training RMSE as a rough diagnostic -- note it's optimistic (fit
        # and scored on the same points). cv_rmse above is the honest one.
        pred_x = A_std @ self.wx
        pred_y = A_std @ self.wy
        err = np.sqrt(np.mean((pred_x - targets[:, 0]) ** 2 + (pred_y - targets[:, 1]) ** 2))
        self.fit_rmse = float(err)

        self.calibrated = True
        return True

    def predict(self, gaze_feat):
        if not self.calibrated:
            return None
        raw = np.asarray(gaze_feat, dtype=np.float64)
        lo, hi = self._raw_bounds
        raw = np.clip(raw, lo, hi)

        a = _poly_features(raw)
        a_std = np.concatenate([a[:1], (a[1:] - self._feat_mean) / self._feat_std])
        x = float(a_std @ self.wx)
        y = float(a_std @ self.wy)
        return x, y

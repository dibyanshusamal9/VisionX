"""
main.py
--------
Gaze + Steering-Wheel-Click Dashboard POC

Run:
    python main.py

Controls:
    Calibration phase : look at the highlighted dot and hold steady -- it
                         auto-captures once your gaze settles (a bar fills
                         in as it detects steadiness). 'C' captures manually
                         any time, 'S' skips the point, 'Q' aborts.
    Normal operation  : SPACE = confirm selection (simulated steering-wheel
                         thumb button -- your webcam-only "click")
                         R     = recalibrate
                         Q     = quit

Note on calibration technique: move your head naturally to each dot the way
you actually would while driving (don't lock your head still and only move
your eyes). The tracker uses head pose as part of the model, so it needs to
see some natural head+eye coordination during calibration to compensate for
head movement afterward.

What changed in this pass (see the individual files for the why in detail):
  - Screen size is now auto-detected (detect_screen_size, below) and used
    for the dashboard window, the calibration point layout, and the
    cursor's valid range, instead of an assumed fixed 1280x720.
  - Every frame is quality-gated (gaze_tracker.py's `trusted` flag) before
    it's allowed to update the cursor -- blinks, occluded/disagreeing eyes,
    and extreme head angles now hold the last cursor instead of injecting a
    bad sample into the filters.
  - A median pre-filter rejects single-frame spikes before the adaptive
    One-Euro filters smooth what's left, and the filters themselves are
    tuned more conservatively -- together this is the fix for "the cursor
    moves even when my eyes are still".
  - Every confirmed click now also feeds a weak passive-recalibration
    sample back into the model (the driver was presumably looking at what
    they clicked), so accuracy drifts up with use instead of only down.
"""
import time
import cv2
import numpy as np

from gaze_tracker import GazeTracker
from calibration import GazeCalibrator
from dashboard_ui import Dashboard
from filters import OneEuroFilter, MedianFilter

CAM_INDEX = 0

# Fallback if the real monitor resolution can't be detected (e.g. no
# display, or tkinter isn't installed -- on some Linux distros you may need
# `sudo apt-get install python3-tk`).
DEFAULT_DASH_W, DEFAULT_DASH_H = 1280, 720
MAX_DASH_W = 1600  # cap the window on very large/4K screens so it stays a manageable size

# 3x3 grid plus 4 in-between points, as fractions of the dashboard size --
# more coverage than a plain 9, which matters more now that the model also
# has head-pose terms to fit.
CALIB_POINTS = [
    (0.08, 0.15), (0.5, 0.15), (0.92, 0.15),
    (0.08, 0.5),  (0.5, 0.5),  (0.92, 0.5),
    (0.08, 0.85), (0.5, 0.85), (0.92, 0.85),
    (0.29, 0.325), (0.71, 0.325),
    (0.29, 0.675), (0.71, 0.675),
]

# Auto-capture kicks in once (h_ratio, v_ratio) hold steady across this many
# consecutive trusted (non-blink, non-occluded, non-extreme-angle) frames.
STABILITY_FRAMES = 5
STABILITY_STD_THRESH = 0.08


def detect_screen_size():
    """Real monitor resolution. Calibrating and rendering against a
    hardcoded 1280x720 while the actual screen is a different size means
    the gaze->pixel mapping range never matches what's actually on screen
    -- this fixes that at the source rather than compensating for it later."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return None


def compute_dash_size():
    detected = detect_screen_size()
    if detected is None:
        print(f"Could not detect screen size -- using default {DEFAULT_DASH_W}x{DEFAULT_DASH_H}.")
        return DEFAULT_DASH_W, DEFAULT_DASH_H
    w, h = detected
    if w > MAX_DASH_W:
        scale = MAX_DASH_W / w
        w, h = int(w * scale), int(h * scale)
    print(f"Detected screen resolution -- using {w}x{h} for the dashboard and gaze mapping range.")
    return w, h


def run_calibration(cap, tracker, dash_w, dash_h):
    calibrator = GazeCalibrator()
    for (nx, ny) in CALIB_POINTS:
        target = (int(nx * dash_w), int(ny * dash_h))
        captured = False
        window = []  # recent full feature vectors [h, v, yaw, pitch]

        while not captured:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            feats, _, found, blink, quality = tracker.process(frame)
            trusted = found and quality is not None and quality["trusted"]

            canvas = np.full((dash_h, dash_w, 3), (20, 20, 24), dtype="uint8")
            cv2.putText(canvas, "CALIBRATION - hold your gaze on the dot",
                        (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Point {len(calibrator.samples_x) + 1}/{len(CALIB_POINTS)}   (C = capture now, S = skip, Q = quit)",
                        (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.circle(canvas, target, 18, (0, 200, 255), -1)
            cv2.circle(canvas, target, 26, (0, 200, 255), 2)

            if not found:
                cv2.putText(canvas, "Face not detected", (40, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                window.clear()
            elif blink:
                cv2.putText(canvas, "Blink detected - hold eyes open on the dot", (40, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
                window.clear()
            elif not trusted:
                cv2.putText(canvas, "Hold face steady / more square to the camera", (40, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
                window.clear()
            else:
                window.append(feats)
                if len(window) > STABILITY_FRAMES:
                    window.pop(0)

                if len(window) == STABILITY_FRAMES:
                    hv = np.array(window)[:, :2]
                    stability = float(np.std(hv, axis=0).mean())
                    bar_w = int(200 * min(STABILITY_STD_THRESH / max(stability, 1e-6), 1.0))
                    cv2.rectangle(canvas, (40, 130), (240, 150), (80, 80, 80), 1)
                    cv2.rectangle(canvas, (40, 130), (40 + bar_w, 150), (0, 220, 120), -1)
                    cv2.putText(canvas, "holding steady...", (250, 147),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                    if stability <= STABILITY_STD_THRESH:
                        avg_feat = np.mean(window, axis=0)
                        calibrator.add_sample(avg_feat, target)
                        captured = True

            small = cv2.resize(frame, (240, 180))
            canvas[dash_h - 190:dash_h - 10, dash_w - 250:dash_w - 10] = small

            cv2.imshow("Gaze Dashboard POC", canvas)
            key = cv2.waitKey(1) & 0xFF

            if not captured and key == ord('c') and found and not blink and window:
                avg_feat = np.mean(window, axis=0)
                calibrator.add_sample(avg_feat, target)
                captured = True
            elif key == ord('s'):
                captured = True  # skip this point
            elif key == ord('q'):
                return None

    ok = calibrator.fit()
    if ok:
        quality = "good" if calibrator.cv_rmse < 70 else "rough - consider pressing R to recalibrate"
        print(f"Calibration fit RMSE: {calibrator.fit_rmse:.1f}px | cross-validated RMSE: {calibrator.cv_rmse:.1f}px "
              f"| ridge_lambda={calibrator.ridge_lambda:g} ({quality})")
    elif len(calibrator.samples_x) > 0:
        print(f"Only {len(calibrator.samples_x)} usable point(s) captured -- need at least 10. "
              f"Recalibrate and try to avoid skipping points / keep your face trusted-quality on each dot.")
    return calibrator if ok else None


def main():
    dash_w, dash_h = compute_dash_size()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Try a different CAM_INDEX in main.py.")
        return

    # Higher capture resolution = more pixels across the eye/iris = finer
    # sub-pixel precision on h_ratio/v_ratio. 640x480 was a real source of
    # avoidable feature noise -- jitter in this signal gets amplified by the
    # polynomial model downstream, which is a big part of "the cursor moves
    # when my eyes are still". Falls back gracefully if the webcam doesn't
    # support this resolution. If FPS drops too much on slower hardware,
    # dial this back to e.g. 960x540.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = GazeTracker()
    dashboard = Dashboard(dash_w, dash_h)

    calibrator = run_calibration(cap, tracker, dash_w, dash_h)
    if calibrator is None:
        print("Calibration cancelled or failed (need a clear view of your face/eyes).")
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        return

    # feat_median rejects single-frame outlier spikes (a landmark
    # mis-detection, a half-blink) before smoothing -- without this, a spike
    # looks like "fast movement" to the adaptive filter below, which opens
    # its cutoff to track it and the cursor visibly twitches.
    #
    # feat_filter then smooths the de-spiked gaze features before they hit
    # the polynomial model. beta is deliberately conservative -- a high beta
    # means ordinary noise gets misread as real movement and the filter
    # opens up, which is the feedback loop that causes drift on a still gaze.
    #
    # cursor_filter does a second, lighter smoothing pass on the final
    # predicted screen point. Both stages adapt to movement speed instead of
    # a single fixed alpha, so fast glances aren't laggy but a steady gaze
    # stays jitter-free.
    feat_median = MedianFilter(window=3)
    feat_filter = OneEuroFilter(min_cutoff=0.4, beta=0.12)
    cursor_filter = OneEuroFilter(min_cutoff=0.5, beta=0.1)

    last_cursor_xy = None
    prev_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        feats, _, found, blink, quality = tracker.process(frame)
        trusted = found and quality is not None and quality["trusted"]
        cursor_xy = None
        hovered_btn = None
        smoothed_feat = None  # only set this frame if we actually computed a fresh one

        if trusted and calibrator.calibrated:
            despiked = feat_median.filter(feats)
            smoothed_feat = feat_filter.filter(despiked)
            pred = calibrator.predict(smoothed_feat)
            if pred is not None:
                px = float(np.clip(pred[0], 0, dash_w))
                py = float(np.clip(pred[1], 0, dash_h))
                smoothed_xy = cursor_filter.filter([px, py])
                cursor_xy = (float(smoothed_xy[0]), float(smoothed_xy[1]))
                last_cursor_xy = cursor_xy
                hovered_btn = dashboard.hit_test(cursor_xy[0], cursor_xy[1])
        elif last_cursor_xy is not None:
            # Hold the cursor in place through a blink / untrusted frame
            # instead of letting it snap to a garbage position -- avoids
            # both visible jumps and misfires on SPACE right after.
            cursor_xy = last_cursor_xy
            hovered_btn = dashboard.hit_test(cursor_xy[0], cursor_xy[1])

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        canvas = dashboard.draw(cursor_xy=cursor_xy, hovered_btn=hovered_btn,
                                 fps=fps, calibrated=calibrator.calibrated)

        if not found:
            cv2.putText(canvas, "Face/eyes not detected", (dash_w // 2 - 150, dash_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Gaze Dashboard POC", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            new_cal = run_calibration(cap, tracker, dash_w, dash_h)
            if new_cal is None:
                break
            calibrator = new_cal
            feat_median.reset()
            feat_filter.reset()
            cursor_filter.reset()
            last_cursor_xy = None
        elif key == 32:  # SPACE == the "steering-wheel thumb-click" for this laptop demo
            if hovered_btn is not None:
                hovered_btn.trigger(dashboard.state)
                # Passive recalibration: the driver was presumably looking
                # at what they just clicked, so feed this as a weak extra
                # calibration sample. Accuracy improves the more the
                # dashboard actually gets used instead of only ever
                # degrading from the initial calibration.
                if smoothed_feat is not None:
                    center = (hovered_btn.x + hovered_btn.w / 2, hovered_btn.y + hovered_btn.h / 2)
                    calibrator.add_sample(smoothed_feat, center)
                    calibrator.fit()

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

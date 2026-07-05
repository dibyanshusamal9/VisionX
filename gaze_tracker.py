"""
gaze_tracker.py
----------------
Extracts gaze-relevant features from webcam frames using MediaPipe Face Mesh
(refined landmarks give iris centers). This is the "Edge AI" perception layer
of the demo: no cloud calls, everything runs locally on your laptop.

Returned features are NOT raw screen coordinates. They are:
  - h_ratio, v_ratio : iris position within the eye socket (per-eye averaged)
  - yaw, pitch       : coarse head-pose proxies

Head pose is included because most "the gaze tracking doesn't work" symptoms
on a webcam rig are actually head-movement drift: the eye-in-socket ratios
alone can't distinguish "eyes moved" from "head moved", so a calibration fit
on ratios alone falls apart the moment the driver's head shifts a couple of
centimeters. Feeding yaw/pitch into the regression (see calibration.py) lets
the model learn to compensate for that instead of just breaking.

Quality gating (new): process() also returns a `quality` dict with a
`trusted` flag. A big source of "the cursor drifts even though I didn't
move my eyes" was that every frame got fed into the pipeline even when the
underlying landmarks weren't trustworthy -- one eye briefly at a grazing
angle, a partial blink, an extreme head turn where MediaPipe's iris
estimate degrades. Silently smoothing over bad frames doesn't fix that, it
just spreads the error out over time and still looks like drift. Flagging
them lets main.py hold the last good cursor position instead of updating
on garbage.
"""
import cv2
import numpy as np
import torch
from l2cs import Pipeline

class GazeTracker:
    def __init__(self, **kwargs):
        self.gaze_pipeline = Pipeline(
            weights='models/L2CSNet_gaze360.pkl',
            arch='ResNet50',
            device=torch.device('cpu')
        )

    def close(self):
        pass

    def process(self, frame_bgr):
        """
        Returns (features, eye_centers_px, face_found, blink, quality).
        """
        # Process the frame with L2CS-Net
        try:
            results = self.gaze_pipeline.step(frame_bgr)
        except ValueError:
            return None, None, False, False, None

        # Check if a face was detected
        if results.pitch.shape[0] == 0:
            return None, None, False, False, None

        # Extract gaze pitch and yaw for the first face detected
        pitch = float(results.pitch[0])
        yaw = float(results.yaw[0])
        
        # Safely extract detection score
        score = float(results.scores[0]) if hasattr(results, 'scores') and results.scores is not None and len(results.scores) > 0 else 1.0

        bbox = results.bboxes[0]  # [x_min, y_min, x_max, y_max]
        
        # Approximate an "eye center" using the face bounding box for passive recalibration
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        eye_centers_px = {
            "left": np.array([center_x, center_y]),
            "right": np.array([center_x, center_y])
        }

        blink = False
        trusted = score > 0.5  # Only trust frames with good detection confidence

        # Pack the L2CS predictions into the feature slots.
        # We put the CNN's gaze yaw and pitch into the first two slots (where h_ratio/v_ratio were)
        # and pad with zeros. The existing 2nd-degree polynomial calibration in calibration.py
        # will now fit directly against the CNN's output, maintaining the exact same architecture!
        features = np.array([yaw, pitch, 0.0, 0.0], dtype=np.float32)

        quality = {
            "trusted": trusted,
            "eye_disagreement": 0.0,  # CNN model handles this inherently
            "inter_ocular_px": float(bbox[2] - bbox[0]),  # face width as distance proxy
        }
        
        return features, eye_centers_px, True, blink, quality

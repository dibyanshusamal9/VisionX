"""
dashboard_ui.py
----------------
A mock in-car infotainment dashboard rendered with OpenCV. Stands in for the
"digital dashboard" in the pitch: climate, media, navigation, and call controls
that the driver operates via gaze + steering-wheel click instead of touch.

Button layout is defined as fractions of (width, height) rather than fixed
pixel offsets, so it scales to whatever resolution main.py detects for the
actual screen (see detect_screen_size() there) instead of assuming a fixed
1280x720 canvas. This matters for accuracy, not just looks: the calibration
points, the cursor's clipping range, and the button hit-boxes all need to
live in the same coordinate system, or "the range of movement" the model
was calibrated against doesn't match what's on screen.
"""
import time
import numpy as np
import cv2


class Button:
    def __init__(self, name, x, y, w, h, action=None):
        self.name = name
        self.x, self.y, self.w, self.h = x, y, w, h
        self.action = action or (lambda state: None)
        self.flash_until = 0.0

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def trigger(self, state):
        self.flash_until = time.time() + 0.35
        self.action(state)


class Dashboard:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.state = {
            "temperature": 21,
            "volume": 40,
            "media": "Bluetooth Audio",
            "nav": "No destination set",
            "call": "Idle",
        }
        self.buttons = self._build_buttons()
        self.last_action_text = ""
        self.last_action_time = 0.0

    def _build_buttons(self):
        # (name, frac_x, frac_y, frac_w, frac_h, action) -- fractions taken
        # from the original 1280x720 layout, kept as fractions so they scale
        # with self.width/self.height.
        specs = [
            ("Temp -",      0.0469, 0.6944, 0.1094, 0.1250, lambda s: self._adjust_temp(-1)),
            ("Temp +",      0.1719, 0.6944, 0.1094, 0.1250, lambda s: self._adjust_temp(1)),
            ("Vol -",       0.3125, 0.6944, 0.1094, 0.1250, lambda s: self._adjust_vol(-5)),
            ("Vol +",       0.4375, 0.6944, 0.1094, 0.1250, lambda s: self._adjust_vol(5)),
            ("Media",       0.5781, 0.6944, 0.1094, 0.1250, lambda s: self._toggle_media()),
            ("Nav Home",    0.0469, 0.5278, 0.1719, 0.1250, lambda s: self._set_nav("Home")),
            ("Nav Work",    0.2344, 0.5278, 0.1719, 0.1250, lambda s: self._set_nav("Work")),
            ("Answer Call", 0.7188, 0.6944, 0.1719, 0.1250, lambda s: self._answer_call()),
        ]
        buttons = []
        for name, fx, fy, fw, fh, action in specs:
            x = int(fx * self.width)
            y = int(fy * self.height)
            w = int(fw * self.width)
            h = int(fh * self.height)
            buttons.append(Button(name, x, y, w, h, action))
        return buttons

    def _adjust_temp(self, delta):
        self.state["temperature"] = max(16, min(28, self.state["temperature"] + delta))
        self._note(f"Temperature set to {self.state['temperature']}C")

    def _adjust_vol(self, delta):
        self.state["volume"] = max(0, min(100, self.state["volume"] + delta))
        self._note(f"Volume set to {self.state['volume']}%")

    def _toggle_media(self):
        self.state["media"] = "Radio FM 98.3" if self.state["media"] != "Radio FM 98.3" else "Bluetooth Audio"
        self._note(f"Media source: {self.state['media']}")

    def _set_nav(self, dest):
        self.state["nav"] = f"Navigating to {dest}"
        self._note(self.state["nav"])

    def _answer_call(self):
        self.state["call"] = "Call connected"
        self._note("Call answered hands-free")

    def _note(self, text):
        self.last_action_text = text
        self.last_action_time = time.time()

    def hit_test(self, px, py):
        for btn in self.buttons:
            if btn.contains(px, py):
                return btn
        return None

    def draw(self, cursor_xy=None, hovered_btn=None, fps=0.0, calibrated=True):
        canvas = np.full((self.height, self.width, 3), (30, 24, 20), dtype="uint8")

        cv2.putText(canvas, "GAZE + STEERING-CLICK DASHBOARD DEMO", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2, cv2.LINE_AA)

        status = f"Temp: {self.state['temperature']}C   Vol: {self.state['volume']}%   {self.state['media']}"
        cv2.putText(canvas, status, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, self.state["nav"], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Call: {self.state['call']}", (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 1, cv2.LINE_AA)

        now = time.time()
        for btn in self.buttons:
            is_hover = hovered_btn is btn
            is_flash = now < btn.flash_until
            if is_flash:
                color = (80, 220, 80)
            elif is_hover:
                color = (60, 140, 230)
            else:
                color = (70, 70, 70)
            cv2.rectangle(canvas, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), color, -1)
            cv2.rectangle(canvas, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), (255, 255, 255), 2)
            text_size = cv2.getTextSize(btn.name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            tx = btn.x + (btn.w - text_size[0]) // 2
            ty = btn.y + (btn.h + text_size[1]) // 2
            cv2.putText(canvas, btn.name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        if self.last_action_text and now - self.last_action_time < 2.0:
            cv2.putText(canvas, f"> {self.last_action_text}", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 180), 2, cv2.LINE_AA)

        if cursor_xy is not None:
            cx, cy = int(cursor_xy[0]), int(cursor_xy[1])
            cv2.circle(canvas, (cx, cy), 14, (0, 255, 255), 2)
            cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)

        cv2.putText(canvas, f"FPS: {fps:.1f}", (self.width - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        cal_text = "Calibrated" if calibrated else "NOT calibrated - press R"
        cv2.putText(canvas, cal_text, (self.width - 300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if calibrated else (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, "SPACE = steering-wheel click   |   R = recalibrate   |   Q = quit",
                    (50, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        return canvas

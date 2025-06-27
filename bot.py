import json
import time
import logging
import abc
import random
import sys
from typing import Dict, Any, List

import cv2
import numpy as np
from mss import mss
from pynput.keyboard import Controller, Key, Listener
import serial

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class ConfigLoader:
    def __init__(self, path: str):
        self._path = path

    def load(self) -> Dict[str, Any]:
        with open(self._path, "r") as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {self._path}")
        return config


class ScreenCapture:
    def __init__(self, region: Dict[str, int]):
        self._sct = mss()
        self.region = region

    def next_frame(self) -> np.ndarray:
        return np.array(self._sct.grab(self.region))


class Detector(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str: ...
    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> Any: ...


class PositionDetector(Detector):
    def __init__(self, templates: List[str], threshold: float):
        self.templates = []
        for template_path in templates:
            try:
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates.append(template)
                    logging.info(f"Loaded template: {template_path}")
                else:
                    logging.warning(f"Failed to load template: {template_path}")
            except Exception as e:
                logging.error(f"Error loading template {template_path}: {e}")

        if not self.templates:
            logging.warning("No templates loaded! Using fallback position detection.")

        self.threshold = threshold

    def name(self) -> str:
        return "position"

    def detect(self, frame: np.ndarray) -> Dict[str, int]:
        if not self.templates:
            h, w = frame.shape[:2]
            return {"x": w // 2, "y": h // 2}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        best = {"x": -1, "y": -1, "score": 0.0}
        h, w = gray.shape

        for tpl in self.templates:
            if tpl is None:
                continue
            th, tw = tpl.shape
            if th > h or tw > w:
                continue
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score >= self.threshold and score > best["score"]:
                best = {"x": loc[0] + tw // 2, "y": loc[1] + th // 2, "score": score}

        if best["x"] == -1:
            h, w = gray.shape
            return {"x": w // 2, "y": h // 2}

        return {"x": best["x"], "y": best["y"]}


class RedDotDetector(Detector):
    def __init__(
        self, hsv_lower: List[int], hsv_upper: List[int], min_area: float = 20.0
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area

    def name(self) -> str:
        return "red_dot"

    def detect(self, frame: np.ndarray) -> Dict[str, int]:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = {"x": -1, "y": -1}
        max_area = self.min_area
        for c in contours:
            area = cv2.contourArea(c)
            if area >= max_area:
                M = cv2.moments(c)
                if M["m00"]:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best = {"x": cx, "y": cy}
                    max_area = area
        return best


class ColorDotDetector(Detector):
    def __init__(
        self,
        name: str,
        hsv_lower: List[int],
        hsv_upper: List[int],
        min_area: float = 20.0,
    ):
        self._name = name
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area

    def name(self) -> str:
        return self._name

    def detect(self, frame: np.ndarray) -> List[Dict[str, int]]:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                M = cv2.moments(c)
                if M["m00"]:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    results.append({"x": cx, "y": cy, "area": area})
        return results


class IconDetector(Detector):
    def __init__(self, templates: List[str], threshold: float, region: Dict[str, int]):
        self.templates = []
        for template_path in templates:
            try:
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates.append(template)
                    logging.info(f"Loaded puzzle icon template: {template_path}")
                else:
                    logging.warning(f"Failed to load icon template: {template_path}")
            except Exception as e:
                logging.error(f"Error loading icon template {template_path}: {e}")

        self.threshold = threshold
        self.region = region
        self._sct = mss()

    def name(self) -> str:
        return "puzzle_icon"

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True if puzzle icon is detected"""
        if not self.templates:
            return False

        try:
            # Capture the icon region (separate from minimap)
            icon_frame = np.array(self._sct.grab(self.region))
            gray = cv2.cvtColor(icon_frame, cv2.COLOR_BGRA2GRAY)

            for template in self.templates:
                if template is None:
                    continue

                th, tw = template.shape
                h, w = gray.shape

                if th > h or tw > w:
                    continue

                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)

                if score >= self.threshold:
                    logging.info(f"🎯 Puzzle icon detected! (confidence: {score:.2f})")
                    return True

            return False
        except Exception as e:
            logging.error(f"Error detecting puzzle icon: {e}")
            return False


class Strategy(abc.ABC):
    @abc.abstractmethod
    def should_run(self, context: Dict[str, Any]) -> bool: ...
    @abc.abstractmethod
    def get_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]: ...


class TraversalStrategy(Strategy):
    def __init__(
        self,
        start: Dict[str, int],
        end: Dict[str, int],
        mode: str,
        blink_key: str,
        jump_key: str,
        blink_interval_range: List[float],
        jump_interval_range: List[float],
        switch_threshold: int = 5,
    ):
        self.start = start
        self.end = end
        self.mode = mode
        self.blink_key = blink_key
        self.jump_key = jump_key
        self.blink_interval_range = blink_interval_range
        self.jump_interval_range = jump_interval_range
        self.switch_threshold = switch_threshold

        self.target = end
        self.last_direction = None

        # Navigation state
        self.navigation_target = None
        self.navigation_active = False

        now = time.perf_counter()
        self.next_blink = now + random.uniform(*self.blink_interval_range)
        self.next_jump = now + random.uniform(*self.jump_interval_range)

    def should_run(self, context: Dict[str, Any]) -> bool:
        return True

    def get_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        now = time.perf_counter()
        pos = context.get("position", {"x": -1, "y": -1})

        if pos["x"] < 0:
            return []

        # Check for puzzle trigger
        puzzle_icon_detected = context.get("puzzle_icon", False)
        if puzzle_icon_detected and not self.navigation_active:
            purple_dots = context.get("purple_dots", [])
            if purple_dots:
                self.navigation_target = purple_dots[0]
                self.navigation_active = True
                logging.info(
                    f"🎯 Puzzle detected! Navigating to {self.navigation_target}"
                )

        # Navigation mode (2D movement to purple dot)
        if self.navigation_active and self.navigation_target:
            target_x = self.navigation_target["x"]
            target_y = self.navigation_target["y"]
            current_x = pos["x"]
            current_y = pos["y"]

            distance_x = abs(current_x - target_x)
            distance_y = abs(current_y - target_y)

            # Check if reached
            if distance_x <= 8 and distance_y <= 8:
                logging.info("🎯 Reached puzzle location!")
                self.navigation_active = False
                self.navigation_target = None
                # Fall through to normal patrol
            else:
                # Move in direction with largest distance
                if distance_x > distance_y and distance_x > 5:
                    direction = "right" if target_x > current_x else "left"
                elif distance_y > 5:
                    direction = "down" if target_y > current_y else "up"
                else:
                    direction = "right" if target_x > current_x else "left"

                actions = [{"type": "traverse", "dir": direction}]

                if self.last_direction != direction:
                    logging.info(
                        f"🧭 Navigating {direction} to ({target_x}, {target_y})"
                    )
                    self.last_direction = direction

                return actions

        # Normal patrol mode
        tx = self.target["x"]
        current_x = pos["x"]

        if abs(current_x - tx) <= self.switch_threshold:
            self.target = self.start if self.target == self.end else self.end
            logging.info(f"🎯 Reached target! Switching to {self.target}")

        if self.target["x"] > current_x:
            direction = "right"
        elif self.target["x"] < current_x:
            direction = "left"
        else:
            direction = "right" if self.target == self.end else "left"

        actions = [{"type": "traverse", "dir": direction}]

        if self.last_direction != direction:
            logging.info(
                f"🏃 Moving {direction} toward target x={self.target['x']} (current x={current_x})"
            )
            self.last_direction = direction

        if self.mode == "blink" and now >= self.next_blink:
            actions.append({"type": "blink", "key": self.blink_key})
            self.next_blink = now + random.uniform(*self.blink_interval_range)
            logging.info("✨ Adding blink action")

        if self.mode == "jump" and now >= self.next_jump:
            actions.append({"type": "jump", "key": self.jump_key})
            self.next_jump = now + random.uniform(*self.jump_interval_range)
            logging.info("🦘 Adding jump action")

        return actions


class CombatStrategy(Strategy):
    def __init__(self, attack_key: str, attack_interval_range: List[float]):
        self.attack_key = attack_key
        self.attack_interval_range = attack_interval_range
        self.next_attack = time.perf_counter() + random.uniform(*attack_interval_range)

    def should_run(self, context: Dict[str, Any]) -> bool:
        return time.perf_counter() >= self.next_attack

    def get_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.next_attack = time.perf_counter() + random.uniform(
            *self.attack_interval_range
        )
        return [{"type": "attack", "key": self.attack_key}]


class JumpComboStrategy(Strategy):
    def __init__(
        self, jump_key: str, attack_key: str, combo_interval_range: List[float]
    ):
        self.jump_key = jump_key
        self.attack_key = attack_key
        self.combo_interval_range = combo_interval_range
        self.last_combo_time = 0
        self.combo_in_progress = False
        self.combo_stage = 0
        self.stage_start_time = 0
        self.stage_counter = 0

    def should_run(self, context: Dict[str, Any]) -> bool:
        now = time.perf_counter()
        if not self.combo_in_progress:
            next_combo_interval = random.uniform(*self.combo_interval_range)
            return now - self.last_combo_time >= next_combo_interval
        return self.combo_in_progress

    def get_actions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        now = time.perf_counter()
        actions = []

        if not self.combo_in_progress:
            self.combo_in_progress = True
            self.combo_stage = 1
            self.stage_start_time = now
            self.stage_counter = 0
            logging.info("🦘 Starting jump combo")

        if self.combo_in_progress:
            stage_elapsed = now - self.stage_start_time

            if self.combo_stage <= 4:
                actions.append({"type": "jump", "key": self.jump_key})

                if stage_elapsed >= 0.04:
                    self.combo_stage += 1
                    self.stage_start_time = now
                    self.stage_counter += 1
                    logging.info(f"🦘 Jump {self.stage_counter}")

            elif self.combo_stage == 5:
                actions.append({"type": "attack", "key": self.attack_key})

                if stage_elapsed >= 0.08:
                    self.combo_in_progress = False
                    self.combo_stage = 0
                    self.last_combo_time = now
                    logging.info("⚔️ Jump combo complete")

        return actions


class HardwareExecutor:
    def __init__(self, port: str, baudrate: int, delay_min: float, delay_max: float):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(3)
            logging.info(f"Connected to Arduino on {port}")
        except Exception as e:
            logging.error(f"Failed to connect to Arduino: {e}")
            raise

        self.delay_min = delay_min
        self.delay_max = delay_max
        self.current_direction = None
        self.last_movement_command = 0
        self.last_action_command = 0

        self.movement_interval = 0.2
        self.action_interval = 0.2

        self.last_attack_interval = 0.15
        self.last_blink_interval = 0.3

    def execute(self, actions: List[Dict[str, Any]]) -> None:
        now = time.perf_counter()

        # Handle movement
        trav_action = next((a for a in actions if a["type"] == "traverse"), None)
        if trav_action:
            direction = trav_action["dir"]

            should_send = (
                self.current_direction != direction
                or now - self.last_movement_command >= self.movement_interval
            )

            if should_send:
                if self._send_command({"type": "traverse", "dir": direction}):
                    if self.current_direction != direction:
                        logging.info(f"🏃 Direction: {direction}")
                    self.current_direction = direction
                    self.last_movement_command = now

        # Handle action commands
        action_commands = [
            a for a in actions if a["type"] in ["attack", "blink", "jump"]
        ]

        if action_commands and now - self.last_action_command >= self.action_interval:
            for action in action_commands:
                if self._send_command(action):
                    logging.info(f"⚔️ {action['type']} ({action.get('key', 'unknown')})")
            self.last_action_command = now

    def _send_command(self, command_dict: dict) -> bool:
        """Send command using Arduino-compatible format (no spaces)"""
        try:
            if command_dict["type"] == "traverse":
                cmd = f'{{"type":"traverse","dir":"{command_dict["dir"]}"}}'
            elif command_dict["type"] == "release":
                cmd = f'{{"type":"release","dir":"{command_dict["dir"]}"}}'
            elif command_dict["type"] in ["attack", "blink", "jump"]:
                cmd = (
                    f'{{"type":"{command_dict["type"]}","key":"{command_dict["key"]}"}}'
                )
            else:
                cmd = json.dumps(command_dict)

            self.ser.write(f"{cmd}\n".encode())
            self.ser.flush()
            time.sleep(0.05)
            return True

        except Exception as e:
            logging.error(f"Send failed: {e}")
            return False

    def stop_movement(self):
        if self.current_direction:
            self._send_command({"type": "release", "dir": self.current_direction})
            self.current_direction = None
            logging.info("⏹️ Stopped movement")

    def close(self):
        if hasattr(self, "ser") and self.ser.is_open:
            self.stop_movement()
            time.sleep(0.1)
            self.ser.close()


class BotEngine:
    def __init__(self, config_path: str):
        cfg = ConfigLoader(config_path).load()

        self.capture = ScreenCapture(cfg["minimap_region"])

        # Always-on detector (every frame)
        self.position_detector = PositionDetector(
            cfg["dot_templates"], cfg["dot_threshold"]
        )

        # Purple dot detector for navigation
        self.purple_detector = ColorDotDetector(
            "purple_dots",
            cfg.get("purple_hsv_lower", [120, 50, 50]),
            cfg.get("purple_hsv_upper", [160, 255, 255]),
        )

        # Puzzle icon detector
        if cfg.get("puzzle_icon_region"):
            self.icon_detector = IconDetector(
                cfg.get("puzzle_icon_templates", ["puzzle_icon_template.png"]),
                cfg.get("puzzle_icon_threshold", 0.8),
                cfg["puzzle_icon_region"],
            )
        else:
            self.icon_detector = None
            logging.warning("No puzzle icon region configured - navigation disabled")

        # Periodic detectors (every 3 frames)
        self.color_detectors = [
            RedDotDetector(cfg["red_hsv_lower"], cfg["red_hsv_upper"]),
            ColorDotDetector(
                "blue_dots",
                cfg.get("blue_hsv_lower", [100, 50, 50]),
                cfg.get("blue_hsv_upper", [130, 255, 255]),
            ),
            ColorDotDetector(
                "green_dots",
                cfg.get("green_hsv_lower", [40, 50, 50]),
                cfg.get("green_hsv_upper", [80, 255, 255]),
            ),
        ]

        self.frame_count = 0
        self.cached_color_results = {}

        # Strategy setup based on traversal mode
        traversal_mode = cfg.get("traversal_mode", "blink")

        if traversal_mode == "jump":
            self.strategies: List[Strategy] = [
                TraversalStrategy(
                    cfg["patrol_start"],
                    cfg["patrol_end"],
                    "jump",
                    cfg.get("blink_key", "4"),
                    cfg.get("jump_key", " "),
                    cfg.get("blink_interval_range", [5, 10]),
                    [0.2, 0.3],
                    cfg.get("switch_threshold", 5),
                ),
                JumpComboStrategy(
                    cfg.get("jump_key", " "),
                    cfg.get("attack_key", "a"),
                    [0.2, 0.3],
                ),
            ]
        else:
            self.strategies: List[Strategy] = [
                TraversalStrategy(
                    cfg["patrol_start"],
                    cfg["patrol_end"],
                    "blink",
                    cfg.get("blink_key", "4"),
                    cfg.get("jump_key", " "),
                    cfg.get("blink_interval_range", [5, 10]),
                    cfg.get("jump_interval_range", [2, 5]),
                    cfg.get("switch_threshold", 5),
                ),
                CombatStrategy(
                    cfg.get("attack_key", "a"),
                    cfg.get("attack_interval_range", [1.0, 3.0]),
                ),
            ]

        if cfg["use_hardware"]:
            self.executor = HardwareExecutor(
                cfg["serial_port"], cfg["baudrate"], cfg["delay_min"], cfg["delay_max"]
            )

        self.display_window = cfg.get("display_window", False)
        self.display_skip = cfg.get("display_skip", 50)

        self.running = False
        self.should_exit = False

    def toggle(self):
        self.running = not self.running
        if self.running:
            logging.info("🚀 Bot started")
        else:
            logging.info("⏸️ Bot paused")
            if hasattr(self.executor, "stop_movement"):
                self.executor.stop_movement()

    def run(self):
        def on_press(key):
            if key == Key.f8:
                self.toggle()
            elif key == Key.esc:
                logging.info("Exit requested")
                self.should_exit = True
                if hasattr(self.executor, "stop_movement"):
                    self.executor.stop_movement()
                return False

        listener = Listener(on_press=on_press, daemon=True)
        listener.start()

        logging.info("Bot ready. Press F8 to start/stop, ESC to exit.")

        try:
            while not self.should_exit:
                frame = self.capture.next_frame()

                context = {}

                # Always detect player position (every frame)
                position = self.position_detector.detect(frame)
                context["position"] = position

                # Always detect purple dots (needed for navigation)
                context["purple_dots"] = self.purple_detector.detect(frame)

                # Puzzle icon detection
                if self.icon_detector:
                    puzzle_icon_detected = self.icon_detector.detect(frame)
                    context["puzzle_icon"] = puzzle_icon_detected
                else:
                    context["puzzle_icon"] = False

                # Detect colors every 3rd frame
                if self.frame_count % 3 == 0:
                    for detector in self.color_detectors:
                        result = detector.detect(frame)
                        detector_name = detector.name()
                        self.cached_color_results[detector_name] = result
                        context[detector_name] = result
                else:
                    context.update(self.cached_color_results)

                self.frame_count += 1

                if self.display_window and self.frame_count % self.display_skip == 0:
                    self._show_debug_window(frame, context)

                if self.running:
                    actions: List[Dict[str, Any]] = []
                    for strat in self.strategies:
                        if strat.should_run(context):
                            actions.extend(strat.get_actions(context))

                    if self.frame_count % 200 == 0:
                        pos = context.get("position", {"x": -1, "y": -1})
                        logging.info(f"📍 Position: ({pos['x']}, {pos['y']})")

                    if actions and hasattr(self, "executor"):
                        self.executor.execute(actions)

                time.sleep(0.05)

        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        finally:
            if hasattr(self.executor, "stop_movement"):
                self.executor.stop_movement()
            if hasattr(self.executor, "close"):
                self.executor.close()
            cv2.destroyAllWindows()

    def _show_debug_window(self, frame: np.ndarray, context: Dict[str, Any]):
        display_frame = frame.copy()

        pos = context.get("position", {"x": -1, "y": -1})
        if pos["x"] >= 0:
            cv2.circle(display_frame, (pos["x"], pos["y"]), 5, (0, 255, 0), 2)

        # Draw purple dots
        purple_dots = context.get("purple_dots", [])
        for dot in purple_dots:
            cv2.circle(display_frame, (dot["x"], dot["y"]), 3, (255, 0, 255), -1)

        status = "RUNNING" if self.running else "PAUSED"
        color = (0, 255, 0) if self.running else (0, 255, 255)
        cv2.putText(
            display_frame,
            f"Status: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        # Show puzzle detection status
        puzzle_detected = context.get("puzzle_icon", False)
        if puzzle_detected:
            cv2.putText(
                display_frame,
                "PUZZLE DETECTED!",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.namedWindow("Bot Debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bot Debug", 380, 178)
        cv2.imshow("Bot Debug", display_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    BotEngine(path).run()

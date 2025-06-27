import json
import time
import logging
from typing import Dict, Any, List
from pynput.keyboard import Key, Listener
import serial
import numpy as np
from mss import mss
import cv2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class PositionDetectionBot:
    def __init__(self):
        with open("config.json", "r") as f:
            cfg = json.load(f)

        self.sct = mss()
        self.region = cfg["minimap_region"]

        # Add position detection
        self.templates = []
        for template_path in cfg["dot_templates"]:
            try:
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates.append(template)
                    print(f"Loaded template: {template_path}")
            except Exception as e:
                print(f"Failed to load template {template_path}: {e}")

        self.threshold = cfg["dot_threshold"]

        self.ser = serial.Serial(cfg["serial_port"], cfg["baudrate"], timeout=1)
        time.sleep(3)
        logging.info("Connected to Arduino")

        self.attack_key = cfg["attack_key"]
        self.patrol_start = cfg["patrol_start"]
        self.patrol_end = cfg["patrol_end"]

        self.running = False
        self.should_exit = False
        self.last_attack = 0
        self.last_move = 0

        self.target = self.patrol_end
        self.current_direction = None

    def detect_position(self, frame):
        """Simple position detection"""
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

    def send_command(self, command_dict: dict):
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

            print(f"Sending: {cmd}")
            self.ser.write(f"{cmd}\n".encode())
            self.ser.flush()
            time.sleep(0.05)
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            return False

    def run(self):
        def on_press(key):
            if key == Key.f8:
                self.running = not self.running
                print(f"Bot {'STARTED' if self.running else 'PAUSED'}")
            elif key == Key.esc:
                self.should_exit = True
                return False

        listener = Listener(on_press=on_press, daemon=True)
        listener.start()

        print("Position detection bot ready. Press F8 to start/stop, ESC to exit.")

        try:
            while not self.should_exit:
                frame = np.array(self.sct.grab(self.region))

                # Detect position every frame
                pos = self.detect_position(frame)

                if self.running:
                    now = time.time()

                    # Attack every 3 seconds
                    if now - self.last_attack > 3.0:
                        self.send_command({"type": "attack", "key": self.attack_key})
                        self.last_attack = now

                    # Simple movement logic every 2 seconds
                    if now - self.last_move > 2.0:
                        current_x = pos["x"]
                        target_x = self.target["x"]

                        if abs(current_x - target_x) <= 5:
                            # Switch target
                            self.target = (
                                self.patrol_start
                                if self.target == self.patrol_end
                                else self.patrol_end
                            )
                            print(f"Switching to target: {self.target}")

                        # Move toward target
                        if target_x > current_x:
                            direction = "right"
                        else:
                            direction = "left"

                        if direction != self.current_direction:
                            self.send_command({"type": "traverse", "dir": direction})
                            self.current_direction = direction
                            print(
                                f"Moving {direction} (pos: {current_x}, target: {target_x})"
                            )

                        self.last_move = now

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            if self.current_direction:
                self.send_command({"type": "release", "dir": self.current_direction})
            self.ser.close()


if __name__ == "__main__":
    PositionDetectionBot().run()

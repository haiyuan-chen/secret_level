import json
import cv2
import numpy as np
from mss import mss


def setup_minimap():
    print("=== Minimap Setup Tool ===")
    print()
    print("INSTRUCTIONS:")
    print("1. Position your game window so the minimap is clearly visible")
    print("2. Press Enter to take a screenshot")
    print("3. In the screenshot window:")
    print("   - Click and drag to select your minimap area")
    print("   - Watch the coordinates update as you move your mouse")
    print("   - Press 'S' to save your selection")
    print("   - Press 'ESC' to cancel and exit")
    print("   - Press 'R' to retake screenshot")
    print()

    input("Press Enter when your game is ready...")

    # Take full screen screenshot
    sct = mss()
    monitor = sct.monitors[1]
    screenshot = np.array(sct.grab(monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    print(f"Screenshot captured: {screenshot.shape[1]}x{screenshot.shape[0]}")

    # Scale down screenshot if it's too large
    height, width = screenshot.shape[:2]
    max_display_size = 1200

    if width > max_display_size or height > max_display_size:
        scale = min(max_display_size / width, max_display_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_screenshot = cv2.resize(screenshot, (new_width, new_height))
        print(f"Scaled for display: {new_width}x{new_height}")
    else:
        display_screenshot = screenshot.copy()
        scale = 1.0

    # Selection state
    selecting = False
    start_pos = None
    current_pos = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_pos, current_pos

        # Convert display coordinates back to actual coordinates
        actual_x = int(x / scale)
        actual_y = int(y / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_pos = (actual_x, actual_y)
            current_pos = (actual_x, actual_y)
            print(f"Selection started at: ({actual_x}, {actual_y})")

        elif event == cv2.EVENT_MOUSEMOVE:
            current_pos = (actual_x, actual_y)

        elif event == cv2.EVENT_LBUTTONUP and selecting:
            selecting = False
            current_pos = (actual_x, actual_y)
            print(f"Selection ended at: ({actual_x}, {actual_y})")

            # Calculate size
            if start_pos:
                width = abs(current_pos[0] - start_pos[0])
                height = abs(current_pos[1] - start_pos[1])
                print(f"Selected area: {width}x{height} pixels")

    # Create window
    window_name = "Minimap Selection Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Position window
    cv2.resizeWindow(
        window_name, display_screenshot.shape[1], display_screenshot.shape[0]
    )
    cv2.moveWindow(window_name, 50, 50)

    print("\n📸 Screenshot window opened!")
    print("💡 TIP: Move your mouse to see coordinates, then click and drag to select")

    while True:
        # Copy image for display
        display = display_screenshot.copy()

        # Show current mouse coordinates
        if current_pos:
            x, y = current_pos
            coord_text = f"Cursor: ({x}, {y})"
            cv2.putText(
                display,
                coord_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Draw selection rectangle
        if start_pos and current_pos:
            # Convert actual coordinates to display coordinates for drawing
            x1_display = int(start_pos[0] * scale)
            y1_display = int(start_pos[1] * scale)
            x2_display = int(current_pos[0] * scale)
            y2_display = int(current_pos[1] * scale)

            cv2.rectangle(
                display,
                (x1_display, y1_display),
                (x2_display, y2_display),
                (0, 255, 0),
                2,
            )

            # Show selection info
            width = abs(current_pos[0] - start_pos[0])
            height = abs(current_pos[1] - start_pos[1])
            size_text = f"Selection: {width}x{height}"
            cv2.putText(
                display,
                size_text,
                (x1_display, y1_display - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Instructions overlay
        instructions = [
            "CONTROLS:",
            "Click & Drag = Select area",
            "S = Save selection",
            "R = Retake screenshot",
            "ESC = Cancel & exit",
        ]

        for i, instruction in enumerate(instructions):
            y_pos = display.shape[0] - 100 + (i * 20)
            cv2.putText(
                display,
                instruction,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC - Exit
            print("❌ Setup cancelled by user")
            cv2.destroyAllWindows()
            return None

        elif key == ord("r") or key == ord("R"):  # R - Retake screenshot
            print("🔄 Retaking screenshot...")
            cv2.destroyAllWindows()
            return setup_minimap()  # Restart

        elif key == ord("s") or key == ord("S"):  # S - Save
            if start_pos and current_pos:
                print("💾 Saving selection...")
                cv2.destroyAllWindows()
                return save_selection(start_pos, current_pos, sct)
            else:
                print("⚠️ No selection made! Click and drag first.")

        elif key == ord("q") or key == ord("Q"):  # Q - Alternative exit
            print("❌ Setup cancelled")
            cv2.destroyAllWindows()
            return None


def save_selection(start_pos, current_pos, sct):
    # Calculate final region
    x1, y1 = start_pos
    x2, y2 = current_pos
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    if width < 10 or height < 10:
        print("⚠️ Selection too small (minimum 10x10 pixels)")
        return setup_minimap()

    region = {"top": top, "left": left, "width": width, "height": height}

    print(f"📏 Final selection:")
    print(f"   Position: ({left}, {top})")
    print(f"   Size: {width}x{height}")

    # Test capture
    try:
        minimap_frame = np.array(sct.grab(region))
        cv2.imwrite("minimap_reference.png", minimap_frame)
        print("✅ Saved minimap_reference.png")

        # Show preview with better focus handling
        print("👀 Showing preview...")
        preview_window = "Preview: Y=confirm, N=retry, ESC=cancel"

        # Create window with proper settings
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.imshow(preview_window, minimap_frame)

        # Force window to front and give it focus
        cv2.setWindowProperty(preview_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)  # Process window events

        print("Press Y to confirm, N to retry, ESC to cancel")
        print("(Make sure the preview window is selected/clicked)")

        reminder_counter = 0  # Add counter to control reminder frequency

        while True:
            key = cv2.waitKey(100) & 0xFF  # 100ms wait

            if key == ord("y") or key == ord("Y"):
                print("✅ Confirmed! Saving config...")
                cv2.destroyAllWindows()
                save_config(region)
                return region

            elif key == ord("n") or key == ord("N"):
                print("🔄 Let's try again...")
                cv2.destroyAllWindows()
                return setup_minimap()

            elif key == 27:  # ESC
                print("❌ Setup cancelled")
                cv2.destroyAllWindows()
                return None

            elif key == ord("q") or key == ord("Q"):  # Alternative exit
                print("❌ Setup cancelled")
                cv2.destroyAllWindows()
                return None

            # Show reminder only every 50 iterations (5 seconds)
            elif key == 255:  # No key pressed (timeout)
                reminder_counter += 1
                if reminder_counter >= 50:  # Only show every 5 seconds
                    print("💡 Click on the preview window, then press Y/N/ESC")
                    reminder_counter = 0  # Reset counter

    except Exception as e:
        print(f"❌ Error capturing region: {e}")
        return None


def save_config(region):
    width = region["width"]
    height = region["height"]

    # Calculate patrol boundaries
    patrol_start = {"x": int(width * 0.2), "y": int(height * 0.5)}
    patrol_end = {"x": int(width * 0.8), "y": int(height * 0.5)}

    config = {
        "minimap_region": region,
        "patrol_start": patrol_start,
        "patrol_end": patrol_end,
        "switch_threshold": max(5, int(width * 0.05)),
        "dot_templates": ["yellow_static.png", "yellow_walk.png", "yellow_edge.png"],
        "dot_threshold": 0.68,
        "red_hsv_lower": [0, 50, 50],
        "red_hsv_upper": [10, 255, 255],
        "use_hardware": True,
        "serial_port": "/dev/cu.usbmodemHIDPC1",
        "baudrate": 9600,
        "delay_min": 0.05,
        "delay_max": 0.23,
        "traversal_mode": "blink",
        "attack_key": "a",
        "blink_key": "4",
        "blink_interval_range": [5.0, 10.0],
        "jump_key": " ",
        "jump_interval_range": [2.0, 5.0],
        "attack_interval_range": [1.0, 3.0],
        "display_window": True,
        "display_skip": 10,
    }

    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Configuration saved!")
    print(
        f"📍 Minimap coordinates: ({region['left']}, {region['top']}) - {width}x{height}"
    )
    print(
        f"🎯 Patrol boundaries: ({patrol_start['x']}, {patrol_start['y']}) to ({patrol_end['x']}, {patrol_end['y']})"
    )
    print()
    print("🎉 Setup complete! You can now run your bot.")


if __name__ == "__main__":
    setup_minimap()

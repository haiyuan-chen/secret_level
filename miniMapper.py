import json
import cv2
import numpy as np
from mss import mss


def show_menu():
    print("🎮 Bot Setup Tool")
    print("=================")
    print()
    print("Choose what to set up:")
    print("1. Minimap region")
    print("2. Puzzle icon region")
    print("3. View current configuration")
    print("4. Exit")
    print()

    while True:
        choice = input("Enter your choice (1-4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            return choice
        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


def setup_region(region_name: str, region_key: str, template_name: str = None):
    print(f"=== {region_name.title()} Setup ===")
    print()
    print("INSTRUCTIONS:")
    print(f"1. Position your game window so the {region_name} is clearly visible")
    print("2. Press Enter to take a screenshot")
    print("3. In the screenshot window:")
    print(f"   - Click and drag to select the {region_name} area")
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
        print(f"Scaled for display: {new_width}x{new_height} (scale: {scale:.2f})")
    else:
        display_screenshot = screenshot.copy()
        scale = 1.0

    # Selection state
    selecting = False
    start_pos = None
    current_pos = None
    mouse_x, mouse_y = 0, 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_pos, current_pos, mouse_x, mouse_y

        # Always update mouse position for cursor display
        mouse_x, mouse_y = x, y

        # Convert display coordinates back to actual coordinates
        actual_x = int(x / scale)
        actual_y = int(y / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_pos = (actual_x, actual_y)
            current_pos = (actual_x, actual_y)
            print(f"🎯 Selection started at: ({actual_x}, {actual_y})")

        elif event == cv2.EVENT_MOUSEMOVE:
            if selecting:
                current_pos = (actual_x, actual_y)

        elif event == cv2.EVENT_LBUTTONUP:
            if selecting:
                selecting = False
                current_pos = (actual_x, actual_y)
                print(
                    f"🎯 Selection completed: ({start_pos[0]}, {start_pos[1]}) to ({actual_x}, {actual_y})"
                )

                # Calculate size
                if start_pos:
                    width = abs(current_pos[0] - start_pos[0])
                    height = abs(current_pos[1] - start_pos[1])
                    print(f"📏 Selected area: {width}x{height} pixels")
                    print("💾 Press 'S' to save this selection")

    # Create window with proper flags
    window_name = f"{region_name.title()} Selection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Resize and position window
    cv2.resizeWindow(
        window_name, display_screenshot.shape[1], display_screenshot.shape[0]
    )
    cv2.moveWindow(window_name, 50, 50)

    print(f"\n📸 Screenshot window opened!")
    print("💡 Click and drag in the window to select the area")

    while True:
        # Copy image for display
        display = display_screenshot.copy()

        # Show current mouse coordinates (in actual image coordinates)
        actual_mouse_x = int(mouse_x / scale)
        actual_mouse_y = int(mouse_y / scale)
        coord_text = f"Mouse: ({actual_mouse_x}, {actual_mouse_y})"
        cv2.putText(
            display,
            coord_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # Draw selection rectangle if we have both points
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
            size_text = f"Selection: {width}x{height}px"

            # Position text above the rectangle
            text_x = min(x1_display, x2_display)
            text_y = min(y1_display, y2_display) - 10
            if text_y < 20:
                text_y = max(y1_display, y2_display) + 25

            cv2.putText(
                display,
                size_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Show selection status
        if selecting:
            status_text = "SELECTING... (drag to resize)"
            color = (0, 255, 255)  # Yellow
        elif start_pos and current_pos:
            status_text = "READY TO SAVE (press S)"
            color = (0, 255, 0)  # Green
        else:
            status_text = "Click and drag to select area"
            color = (255, 255, 255)  # White

        cv2.putText(
            display,
            status_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
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
            y_pos = display.shape[0] - 120 + (i * 20)
            cv2.putText(
                display,
                instruction,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF  # Increased refresh rate

        if key == 27:  # ESC - Exit
            print("❌ Setup cancelled by user")
            cv2.destroyAllWindows()
            return None

        elif key == ord("r") or key == ord("R"):  # R - Retake screenshot
            print("🔄 Retaking screenshot...")
            cv2.destroyAllWindows()
            return setup_region(region_name, region_key, template_name)

        elif key == ord("s") or key == ord("S"):  # S - Save
            if start_pos and current_pos:
                print("💾 Saving selection...")
                cv2.destroyAllWindows()
                return save_selection(
                    start_pos, current_pos, sct, region_name, region_key, template_name
                )
            else:
                print("⚠️ No selection made! Click and drag to select an area first.")

        elif key == ord("q") or key == ord("Q"):  # Q - Alternative exit
            print("❌ Setup cancelled")
            cv2.destroyAllWindows()
            return None

        # Debug: Print if any key is pressed
        if key != 255:
            print(
                f"🔧 Key pressed: {key} ({chr(key) if 32 <= key <= 126 else 'special'})"
            )


def save_selection(start_pos, current_pos, sct, region_name, region_key, template_name):
    # Calculate final region
    x1, y1 = start_pos
    x2, y2 = current_pos
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    if width < 10 or height < 10:
        print("⚠️ Selection too small (minimum 10x10 pixels)")
        return setup_region(region_name, region_key, template_name)

    region = {"top": top, "left": left, "width": width, "height": height}

    print(f"📏 Final {region_name} selection:")
    print(f"   Position: ({left}, {top})")
    print(f"   Size: {width}x{height}")

    # Test capture
    try:
        captured_frame = np.array(sct.grab(region))

        # Save reference image
        reference_filename = f"{region_name.replace(' ', '_')}_reference.png"
        cv2.imwrite(reference_filename, captured_frame)
        print(f"✅ Saved {reference_filename}")

        # If this is puzzle icon, also save template
        if template_name:
            gray_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGRA2GRAY)
            cv2.imwrite(template_name, gray_frame)
            print(f"✅ Saved {template_name}")

        # Show preview
        print("👀 Showing preview...")
        preview_window = "Preview: Y=confirm, N=retry, ESC=cancel"

        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.imshow(preview_window, captured_frame)
        cv2.setWindowProperty(preview_window, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)

        print("Press Y to confirm, N to retry, ESC to cancel")
        print("(Make sure the preview window is selected/clicked)")

        reminder_counter = 0

        while True:
            key = cv2.waitKey(100) & 0xFF

            if key == ord("y") or key == ord("Y"):
                print("✅ Confirmed! Saving config...")
                cv2.destroyAllWindows()
                update_config(region_key, region, template_name)
                return region

            elif key == ord("n") or key == ord("N"):
                print("🔄 Let's try again...")
                cv2.destroyAllWindows()
                return setup_region(region_name, region_key, template_name)

            elif key == 27:  # ESC
                print("❌ Setup cancelled")
                cv2.destroyAllWindows()
                return None

            elif key == 255:  # No key pressed (timeout)
                reminder_counter += 1
                if reminder_counter >= 50:
                    print("💡 Click on the preview window, then press Y/N/ESC")
                    reminder_counter = 0

    except Exception as e:
        print(f"❌ Error capturing region: {e}")
        return None


def update_config(region_key, region, template_name):
    """Update config.json with the new region"""
    try:
        # Load existing config
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}

        # Update the specific region
        config[region_key] = region

        # Add specific settings based on region type
        if region_key == "minimap_region":
            width = region["width"]
            height = region["height"]

            # Calculate patrol boundaries
            config["patrol_start"] = {"x": int(width * 0.2), "y": int(height * 0.5)}
            config["patrol_end"] = {"x": int(width * 0.8), "y": int(height * 0.5)}
            config["switch_threshold"] = max(5, int(width * 0.05))

            # Add default settings for minimap
            defaults = {
                "dot_templates": [
                    "yellow_static.png",
                    "yellow_walk.png",
                    "yellow_edge.png",
                ],
                "dot_threshold": 0.68,
                "red_hsv_lower": [0, 50, 50],
                "red_hsv_upper": [10, 255, 255],
                "blue_hsv_lower": [100, 50, 50],
                "blue_hsv_upper": [130, 255, 255],
                "green_hsv_lower": [40, 50, 50],
                "green_hsv_upper": [80, 255, 255],
                "purple_hsv_lower": [120, 50, 50],
                "purple_hsv_upper": [160, 255, 255],
                "use_hardware": True,
                "serial_port": "/dev/cu.usbmodemHIDPC1",
                "baudrate": 9600,
                "delay_min": 0.05,
                "delay_max": 0.23,
                "traversal_mode": "jump",
                "attack_key": "a",
                "blink_key": "4",
                "blink_interval_range": [5.0, 10.0],
                "jump_key": " ",
                "jump_interval_range": [2.0, 5.0],
                "jump_combo_interval_range": [0.2, 0.3],
                "attack_interval_range": [1.0, 3.0],
                "display_window": True,
                "display_skip": 10,
            }

            for key, value in defaults.items():
                if key not in config:
                    config[key] = value

        elif region_key == "puzzle_icon_region":
            # Add puzzle icon specific settings
            config["puzzle_icon_templates"] = (
                [template_name] if template_name else ["puzzle_icon_template.png"]
            )
            config["puzzle_icon_threshold"] = 0.8

        # Save updated config
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ {region_key} configuration saved!")

    except Exception as e:
        print(f"❌ Error saving config: {e}")


def view_config():
    """Display current configuration"""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)

        print("=== Current Configuration ===")
        print()

        if "minimap_region" in config:
            mr = config["minimap_region"]
            print(
                f"📍 Minimap: ({mr['left']}, {mr['top']}) - {mr['width']}x{mr['height']}"
            )
        else:
            print("❌ No minimap region configured")

        if "puzzle_icon_region" in config:
            pr = config["puzzle_icon_region"]
            print(
                f"🎯 Puzzle icon: ({pr['left']}, {pr['top']}) - {pr['width']}x{pr['height']}"
            )
            print(f"   Templates: {config.get('puzzle_icon_templates', 'None')}")
        else:
            print("❌ No puzzle icon region configured")

        print()

    except FileNotFoundError:
        print("❌ No config.json found. Please set up regions first.")
    except Exception as e:
        print(f"❌ Error reading config: {e}")


if __name__ == "__main__":
    while True:
        choice = show_menu()

        if choice == "1":
            print()
            setup_region("minimap", "minimap_region")
        elif choice == "2":
            print()
            setup_region(
                "puzzle icon", "puzzle_icon_region", "puzzle_icon_template.png"
            )
        elif choice == "3":
            print()
            view_config()
        elif choice == "4":
            print("👋 Goodbye!")
            break

        print()
        input("Press Enter to continue...")

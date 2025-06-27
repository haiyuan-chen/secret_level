import serial
import time

ser = serial.Serial("/dev/cu.usbmodemHIDPC1", 9600, timeout=1)
time.sleep(3)

print("Focus game in 5 seconds...")
time.sleep(5)

# Use string formatting (no spaces)
cmd = f'{{"type":"attack","key":"a"}}'
print(f"Sending: {cmd}")
ser.write(f"{cmd}\n".encode())
ser.flush()
time.sleep(2)

ser.close()
print("Did that work?")

import serial
import time

ser = serial.Serial('COM5', 115200, timeout=1)

try:
    # Send pulse command
    ser.write(b'H')
    ser.flush()  # force immediate send
    print("Pulse sent")
    
    # Optional: read Arduino response
    while True:
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            if line:
                print(f"Arduino says: {line}")
            if "Pulse ended" in line:
                break
        time.sleep(0.001)  # tiny sleep to avoid busy wait
finally:
    ser.close()

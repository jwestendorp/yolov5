# https://gist.github.com/mpilosov/92c6e7aea32345248504084c72a96926

import serial
import time

# Open grbl serial port ==> CHANGE THIS BELOW TO MATCH YOUR USB LOCATION
# GRBL operates at 115200 baud. Leave that part alone.
s = serial.Serial('COM4', 250000)

gcode = [
    "G28 ",  # Home extruder
    "G1 Z15 F4800",
    "M107 ",  # Turn off fan
    "G90",  # Absolute positioning
    "M82 ",  # Extruder in absolute mode
    "M190 S50",
    #  Activate all used extruder
    "M104 T0 S210",
    "G92 E0"  # Reset extruder position
]

# Wake up grbl
s.write("\r\n\r\n")
time.sleep(2)   # Wait for grbl to initialize
s.flushInput()  # Flush startup text in serial input

# Stream g-code to grbl
for line in gcode:
    l = line.strip()  # Strip all EOL characters for consistency
    print('Sending: ' + l)
    s.write(l + '\n')  # Send g-code block to grbl
    grbl_out = s.readline()  # Wait for grbl response with carriage return
    print(' : ' + grbl_out.strip())

# Wait here until grbl is finished to close serial port and file.
raw_input("  Press <Enter> to exit and disable grbl.")

# Close  serial port
s.close()

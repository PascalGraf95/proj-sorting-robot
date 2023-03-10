from modules.misc import serial_ports
import serial
import time


class ConveyorBelt:
    def __init__(self):
        # Establish a serial connection to the arduino
        available_ports = serial_ports()
        arduino_port = ""
        for port in available_ports:
            if 'Arduino Mega 2560' in port[1]:
                arduino_port = port[0]
                break
        if arduino_port == "":
            raise ConnectionError("[ERROR] Conveyor port could not be found!")
        self.serial_connection = serial.Serial(arduino_port, 115200)
        self.running = False
        time.sleep(2)  # Seems necessary for serial connection to establish correctly

    def start(self):
        self.running = True
        # Set direction to forward and start the conveyor
        self.serial_connection.write(b'r\n')
        self.serial_connection.write(b'1\n')
        for i in range(10):
            time.sleep(0.05)
            self.serial_connection.write(b'+\n')

    def start_reversed(self):
        self.running = True
        # Set direction to backward and start the conveyor
        self.serial_connection.write(b'l\n')
        self.serial_connection.write(b'1\n')

    def is_running(self):
        return self.running

    def speed_up(self):
        self.serial_connection.write(b'+\n')

    def stop(self):
        self.running = False
        self.serial_connection.write(b'0\n')

    def disconnect(self):
        self.running = False
        self.serial_connection.close()


if __name__ == '__main__':
    conveyor = ConveyorBelt()
    conveyor.start()
    time.sleep(5)
    conveyor.stop()
    conveyor.disconnect()

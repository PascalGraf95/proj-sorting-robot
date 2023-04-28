from modules.misc import serial_ports
import serial
import time

verbose = False

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
        self.serial_connection = serial.Serial(arduino_port, 9600)
        self.running = False
        time.sleep(2)  # Seems necessary for serial connection to establish correctly

    def start(self):
        if verbose:
            print("[INFO] ConveyorBelt Start")
        self.running = True
        # Set direction to forward and start the conveyor
        self.serial_connection.write(b'+')

    def start_reversed(self):
        if verbose:
            print("[INFO] ConveyorBelt reverse Start")
        # Set direction to backward and start the conveyor
        self.serial_connection.write(b'-')
        self.running = True

    def is_running(self):
        return self.running

    def stop(self):
        if verbose:
            print("[INFO] ConveyorBelt Stop")
        self.serial_connection.write(b'0\n')
        self.running = False

    def activate_manual_control(self):
        if verbose:
            print("[INFO] ConveyorBelt manual control activated")
        self.serial_connection.write(b'a\n')
        self.running = False

    def disconnect(self):
        self.stop()
        self.serial_connection.close()


if __name__ == '__main__':
    conveyor = ConveyorBelt()
    conveyor.start()
    time.sleep(5)
    conveyor.stop()
    conveyor.disconnect()

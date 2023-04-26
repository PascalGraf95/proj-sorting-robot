from modules.misc import serial_ports
import serial
import time

class Seperator:
    def __init__(self):
        # Establish a serial connection to the arduino
        available_ports = serial_ports()
        arduino_port = ""
        for port in available_ports:
            if 'Arduino Uno' in port[1]:
                arduino_port = port[0]
                break
        if arduino_port == "":
            raise ConnectionError("[ERROR] Seperator port could not be found!")

        self.serial_connection = serial.Serial(arduino_port, 9600)
        self.running = False
        time.sleep(2)  # Seems necessary for serial connection to establish correctly

    def start(self):
        print("Start Seperator")
        self.running = True
        self.serial_connection.write(0x01)

    def stop(self):
        print("Stop Seperator")
        self.running = False
        self.serial_connection.write(0x02)

    def disconnect(self):
        self.running = False
        self.serial_connection.close()


if __name__ == '__main__':
    seperator = Seperator()
    seperator.start()
    time.sleep(5)
    seperator.stop()
    time.sleep(5)
    seperator.start()
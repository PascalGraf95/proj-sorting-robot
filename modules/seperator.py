from modules.misc import serial_ports
import serial
import time

# verbose to Debug and get mor information in the Terminal Output
verbose = False

class Seperator:
    def __init__(self):
        # Establish a serial connection to the arduino
        available_ports = serial_ports()
        arduino_port = ""
        for port in available_ports:
            if 'Arduino Uno' in port[1]:
                arduino_port = port[0]
                if verbose:
                    print("[INFO] Arduino Port set to {}".format(arduino_port))
                break
        if arduino_port == "":
            raise ConnectionError("[ERROR] Seperator port could not be found!")

        self.serial_connection = serial.Serial(arduino_port, 9600)
        self.running = False
        time.sleep(2)  # Seems necessary for serial connection to establish correctly

    def start(self):
        if verbose:
            print("[INFO] Seperator Start")
        self.serial_connection.write(b'\x01')
        self.running = True

    def stop(self):
        if verbose:
            print("[INFO] Seperator Stop")
        self.serial_connection.write(b'\x02')
        self.running = False

    def disconnect(self):
        self.stop()
        self.serial_connection.close()
        print("[INFO] Seperator disconnected")

    def seperate_cycle(self, time_of_cycle=5):
        self.start()
        time.sleep(time_of_cycle)
        self.stop()

if __name__ == '__main__':
    # Testing the if the seperator starts moving
    seperator = Seperator()
    seperator.seperate_cycle(30)
    seperator.disconnect()
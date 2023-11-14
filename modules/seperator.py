from modules.misc import serial_ports
import serial
import time
import struct

# verbose to Debug and get mor information in the Terminal Output
verbose = True


class Seperator:
    def __init__(self):
        """Initialize the Seperator"""
        # Establish a serial connection to the arduino
        available_ports = serial_ports()
        if verbose:
            print("[INFO] Available Ports are {}".format(available_ports))
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

    def forward(self):
        """Move the Seperator forward"""
        if verbose:
            print("[INFO] Seperator forward")
        command = b'\x01' + b'\x00'
        self.serial_connection.write(command)
        self.running = True

    def backward(self):
        """Move the Seperator backward."""
        if verbose:
            print("[INFO] Seperator backward")
        command = b'\x02' + b'\x00'
        self.serial_connection.write(command)
        self.running = True

    def stop(self):
        """Stop the Seperator"""
        if verbose:
            print("[INFO] Seperator Stop")
        command = b'\xFF' + b'\x00'
        self.serial_connection.write(command)
        self.running = False

    def jiggle(self):
        """Jiggle the Seperator (3Steps forward/One Step Backward)"""
        if verbose:
            print("[INFO] Seperator jiggle")
        command = b'\x04' + b'\x00'
        self.serial_connection.write(command)
        self.running = False

    def disconnect(self):
        self.stop()
        self.serial_connection.close()
        print("[INFO] Seperator disconnected")

    def seperate_cycle(self, time_of_cycle=5):
        self.start()
        time.sleep(time_of_cycle)
        self.stop()

    def do_stepps(self, n_steps):
        if verbose:
            print("[INFO] Seperator make {} Steps". format(n_steps))
        byte_representation = n_steps.to_bytes(1, byteorder='big')
        print(byte_representation)
        command = b'\x03' + byte_representation
        self.serial_connection.write(command)

    def setacceleration(self, acceleration):
        if verbose:
            print("[INFO] Acceleration set to: {}". format(acceleration))
        byte_representation = acceleration.to_bytes(1, byteorder='big')
        print(byte_representation)
        command = b'\x05' + byte_representation
        self.serial_connection.write(command)

    def setVMax(self, vmax):
        if verbose:
            print("[INFO] Maximal Velocity set to: {} ". format(vmax))
        byte_representation = vmax.to_bytes(1, byteorder='big')
        print(byte_representation)
        command = b'\x06' + byte_representation
        self.serial_connection.write(command)

    def test_seperator(self):
        """Test the Seperator by providing Python Console Input for control."""
        print("Select the Mode:")
        print("[1] Spin the Seperator Forward")
        print("[2] Spin the Seperator Backward")
        print("[3] Jiggle The Seperator")
        print("[4] STOP")
        print("[5] Choose amount of Forward Steps")
        print("[6] Set acceleration")
        print("[7] Set maximal velocity")
        print("[9] END")
        while True:
            mode = input()
            if mode == "1":
                self.forward()
            elif mode == "2":
                self.backward()
            elif mode == "3":
                self.jiggle()
            elif mode == "4":
                self.stop()
            elif mode == "5":
                n_steps = int(input("How many steps should the Stepper do?"))
                self.do_stepps(n_steps)
            elif mode == "6":
                acceleration = int(input("Set the acceleration to: "))
                self.setacceleration(acceleration)
            elif mode == "7":
                vmax = int(input("Set maximal velocity to: "))
                self.setVMax(vmax)
            elif mode == "9":
                self.stop()
                break
            else:
                print("[Warning] Input unknown")

            separator_info = self.serial_connection.read(2)
            print("[INFO] n_steps low/high set to {}", separator_info)
        return


def main():
    # Testing the if the seperator starts moving
    seperator = Seperator()
    seperator.test_seperator()
    seperator.disconnect()
    return


if __name__ == '__main__':
    main()

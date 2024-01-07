import cv2
import socket
import tkinter as tk
from tkinter import filedialog


def select_image():
    root = tk.Tk()
    root.withdraw()  # Verstecke das Hauptfenster

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif")])

    if file_path:
        print("Choosed Image:", file_path)
        # Hier kannst du den ausgewählten Dateipfad verwenden, z.B. weiterverarbeiten oder anzeigen
    return file_path

def main():
    image_path=select_image()
    # Load Image from Path
    image = cv2.imread(image_path)

    # Connect to Socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Define the Socket
    client.connect(('127.0.0.1', 12346))

    # Split Data into packages
    encoded_image = cv2.imencode('.png', image)[1].tobytes()
    chunk_size = 2048
    offset = 0

    while offset < len(encoded_image):
        client.send(encoded_image[offset:offset+chunk_size])
        offset += chunk_size

    # Close Connection
    client.close()

if __name__ == '__main__':
    main()
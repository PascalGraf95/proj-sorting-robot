import cv2
import socket
from pathlib import Path


def rec_img_by_TCP(displayRecImg=True, port=12346):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', port))
    server.listen()

    print('[INFO] Waiting for Image')

    client_socket, client_address = server.accept()
    file_path = Path('../Datasets').resolve()
    image_path = str(file_path)+'/REC_TCP_Image/rec_image.png'

    # Opens a File to write Bytes
    file = open(image_path, "wb")
    image_chunk = client_socket.recv(2048)

    print('ready to Receive Image')
    # Loop over the received Data packages
    while image_chunk:
        file.write(image_chunk)
        image_chunk = client_socket.recv(2048)

    # CLose all
    file.close()
    #client_address.close()
    client_socket.close()
    server.close()

    if displayRecImg:
        # Load Received Image and display it
        received_image = cv2.imread(image_path)
        cv2.imshow('Received Image', received_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image_path


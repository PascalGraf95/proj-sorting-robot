import socket


def start_tcp_server():
    # Define address for connection
    address = 12345

    # Define Connection
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', address))
    server_socket.listen(1)

    # Connect to Client
    print("Waiting for connection...")
    client_socket, address = server_socket.accept()
    print("Connected to {}!".format(address))

    # Receive Data
    received_data = client_socket.recv(1024)
    print(f"Received data: {received_data}")

    # Close Connection
    client_socket.close()
    server_socket.close()


if __name__ == '__main__':
    start_tcp_server()


import socket
import random


def next_free_port( port=1994, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')


if __name__=='__main__':
    start_port = random.choice(list(range(1994, 2994)))
    port = next_free_port(port=start_port)
    print(port)
    exit(port)
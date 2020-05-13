import subprocess
import argparse
from torchvision import datasets
from torchvision import transforms

import signal
import sys


# Downloads MNIST dataset
mnist_trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

# Parse args
parser = argparse.ArgumentParser(description="Start websocket remote server worker.")

parser.add_argument(
    "--localworkers",
    action="store_true",
    help="if set, (localhost) websocket server workers will created, otherwize, only start a (remote) worker server."
)

parser.add_argument(
    "--host", type=str, 
    default="0.0.0.0", help="host for the connection, e.g. --host 192.168.0.12"
)

parser.add_argument(
    "--port",
    "-p",
    type=str,
    default="8777",
    help="port number of the websocket server worker, e.g. --port 8777",
)

parser.add_argument(
    "--id", type=str, 
    default="alice", 
    help="name (id) of the websocket server worker, e.g. --id alice"
)
parser.add_argument(
    "--testing",
    action="store_true",
    help="if set, websocket server worker will load the test dataset instead of the training dataset",
)

args = parser.parse_args()

process_workers = []

# given local or remote mode, create workers
if(args.localworkers):  
    call_alice = [
        "python3",
        "run_websocket_server.py",
        "--port",
        "8777",
        "--id",
        "alice",
        "--host",
        "0.0.0.0",
    ]

    call_bob = [
        "python3",
        "run_websocket_server.py",
        "--port",
        "8778",
        "--id",
        "bob",
        "--host",
        "0.0.0.0",
    ]

    call_charlie = [
        "python3",
        "run_websocket_server.py",
        "--port",
        "8779",
        "--id",
        "charlie",
        "--host",
        "0.0.0.0",
    ]

    call_testing = [
        "python3",
        "run_websocket_server.py",
        "--port",
        "8780",
        "--id",
        "testing",
        "--testing",
        "--host",
        "0.0.0.0",
    ]  

    print("Starting server for Alice")
    worker_alice = subprocess.Popen(call_alice)

    print("Starting server for Bob")
    worker_bob = subprocess.Popen(call_bob)

    print("Starting server for Charlie")
    worker_charlie = subprocess.Popen(call_charlie)

    print("Starting server for Testing")
    worker_testing = subprocess.Popen(call_testing)

    process_workers=[worker_alice, worker_bob, worker_charlie, worker_testing]

else:
    if(args.testing): 
        call_worker = [
            "python3",
            "run_websocket_server.py",
            "--port",
            args.port,
            "--id",
            args.id,
            "--host",
            args.host,
            "--testing"
        ]
    else:
        call_worker = [
            "python3",
            "run_websocket_server.py",
            "--port",
            args.port,
            "--id",
            args.id,
            "--host",
            args.host
        ]

    print("Starting server for", args.id)
    worker_remote = subprocess.Popen(call_worker) 

    process_workers=[worker_remote]   


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    for p in process_workers:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

signal.pause()

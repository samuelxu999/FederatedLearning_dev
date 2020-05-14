import subprocess
import sys
import signal
import os
import argparse

if os.name == "nt":
    python = "python"
else:
    python = "python" + sys.version[0:3]

# Parse args
parser = argparse.ArgumentParser(description="Start websocket remote server worker.")

parser.add_argument(
    "--localworkers", action="store_true", 
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

args = parser.parse_args()

process_workers = []


# given local or remote mode, create workers
if(args.localworkers):  
	call_alice = [python, "run_websocket_server.py", "--port", "8777", "--id", "alice"]

	call_bob = [python, "run_websocket_server.py", "--port", "8778", "--id", "bob"]

	call_charlie = [python, "run_websocket_server.py", "--port", "8779", "--id", "charlie"]

	print("Starting local worker server for Alice")
	worker_alice = subprocess.Popen(call_alice)

	print("Starting local worker server for Bob")
	worker_bob = subprocess.Popen(call_bob)

	print("Starting local worker server for Charlie")
	worker_charlie = subprocess.Popen(call_charlie)

	process_workers=[worker_alice, worker_bob, worker_charlie]

else:
	call_worker = [
		python,
		"run_websocket_server.py",
		"--port",
		args.port,
		"--id",
		args.id,
		"--host",
		args.host
	]

	print("Starting websocket server for", args.id)
	worker_socket = subprocess.Popen(call_worker) 

	process_workers=[worker_socket] 

def signal_handler(sig, frame):
	print("You pressed Ctrl+C!")
	for p in process_workers:
		p.terminate()
	sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

signal.pause()

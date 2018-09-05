import numpy as np
import rpyc
import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_service_pool(ports):
	import subprocess
	return [subprocess.Popen(
		['python', 'utils.py', str(port)]).pid for port in ports]

def stop_service_pool(pids):
	import os, signal
	for pid in pids:
		os.kill(pid, signal.SIGKILL)

import subprocess
import threading
import time
import sys
import io
import os
import psutil
import socket

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

# Function to get pipt IPv4 address - integrated from ip.py
def get_ethernet_ipv4():
    # Try ethernet interfaces in order: ethernet, ethernet 2, ethernet 3, ethernet 4, ethernet 5
    ethernet_interfaces = ["ethernet 2", "ethernet 3", "ethernet 4", "ethernet 5"]
    
    for target_interface in ethernet_interfaces:
        for interface, addrs in psutil.net_if_addrs().items():
            if interface.lower() == target_interface:
                for addr in addrs:
                    if addr.family == socket.AF_INET:  # IPv4
                        print(f"Found IP address {addr.address} on interface {interface}")
                        return addr.address
    
    print("No ethernet interfaces found, falling back to localhost")
    return None

def start_backend():
    # Change to the src directory so uvicorn can find the api module
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get dynamic IP address
    host_ip = get_ethernet_ipv4()
    if host_ip is None:
        print("Could not find any Ethernet interface, falling back to localhost")
        host_ip = "127.0.0.1"
    else:
        print(f"Using IP address: {host_ip}")
    
    subprocess.Popen([
        "uvicorn", 
        "src.api.backend_fastAPI:app",
        "--host", host_ip, 
        "--port", "5000"
    ], cwd=src_dir)

def launch_gradio():
    try:
        from src.UI import frontend as FRONTEND
        # Check what attributes the frontend module has
        if hasattr(FRONTEND, 'launch'):
            FRONTEND.launch()
        elif hasattr(FRONTEND, 'app'):
            # If it has an app attribute, try to launch it
            FRONTEND.app.launch()
        else:
            print("Available attributes in frontend module:")
            print([attr for attr in dir(FRONTEND) if not attr.startswith('_')])
            # Try common gradio launch patterns
            if hasattr(FRONTEND, 'demo'):
                FRONTEND.demo.launch()
            elif hasattr(FRONTEND, 'interface'):
                FRONTEND.interface.launch()
            else:
                print("Could not find a suitable launch method")
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")

if __name__ == "__main__":
    threading.Thread(target=start_backend, daemon=True).start()
    print("Waiting for backend to start...")
    print(os.getcwd())
    time.sleep(10)
    print("Launching Gradio interface...")
    launch_gradio()
import os
import sys
import psutil
import signal

def kill_my_python_processes():
    """Kill all Python processes owned by the current user."""
    my_username = os.environ.get('USER')
    killed = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        try:
            if (proc.info['name'] == 'python3' or proc.info['name'] == 'python') and \
               proc.info['username'] == my_username and \
               proc.pid != os.getpid():  # Don't kill self
                print(f"Killing Python process {proc.pid}")
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                    killed += 1
                except PermissionError:
                    print(f"  Permission denied for PID {proc.pid}")
                except ProcessLookupError:
                    print(f"  Process {proc.pid} no longer exists")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    print(f"Killed {killed} Python processes")

def kill_processes_using_over_memory(min_gb=10):
    """Kill my processes using more than specified GB of memory."""
    my_username = os.environ.get('USER')
    killed = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        try:
            if proc.info['username'] == my_username and proc.pid != os.getpid():
                memory_gb = proc.memory_info().rss / (1024**3)
                if memory_gb > min_gb:
                    print(f"Killing process {proc.pid} ({proc.name()}) using {memory_gb:.2f} GB")
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                        killed += 1
                    except (PermissionError, ProcessLookupError):
                        print(f"  Failed to kill PID {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    print(f"Killed {killed} high-memory processes")

if __name__ == "__main__":
    print("=== Process Cleanup Utility ===")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all-python":
        kill_my_python_processes()
    elif len(sys.argv) > 1 and sys.argv[1] == "--high-memory":
        threshold = 10
        if len(sys.argv) > 2:
            try:
                threshold = float(sys.argv[2])
            except ValueError:
                pass
        kill_processes_using_over_memory(threshold)
    else:
        print("Usage:")
        print("  python cleanup.py --all-python    # Kill all your Python processes")
        print("  python cleanup.py --high-memory [GB_threshold]  # Kill processes using more than GB_threshold")

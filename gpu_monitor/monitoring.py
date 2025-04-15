import os
import json
import subprocess
import psutil
import sys
import time
from datetime import datetime
from multiprocessing import Process, Manager
from .utils import (
    generate_log_filename,
    log_gpu_metrics,
    report_summary_metrics,
)
from .gpu_metrics import (
    initialize_nvml, 
    shutdown_nvml, 
    collect_gpu_metrics_nvml,
    get_gpu_model,
    get_gpu_count,
    find_vllm_server_pid  # Import this directly
)

# Monitoring interval in seconds
MONITORING_INTERVAL = 0.5  # Change from 1.0 to 0.5 seconds

LOG_DIR = "LOGS"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def monitor_resource_usage(shared_metrics, pids, log_file_path, llm_model_name=None):
    """
    Monitors system's CPU, memory, and GPU usage for a specific process and its children.
    Logs detailed instantaneous metrics to a CSV file.
    """
    try:
        initialize_nvml()
        headers_written = False
        start_time = None  # Initialize but don't set yet
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_time = time.time()
            
            # Set the start time at first successful metric collection
            if start_time is None:
                start_time = current_time
            
            elapsed_time = current_time - start_time
            
            # Monitor each PID in the process tree
            for pid in list(pids):  # Make a copy of the list to avoid modification during iteration
                try:
                    # Collect CPU and memory usage
                    process = psutil.Process(pid)
                    cpu_usage = process.cpu_percent(interval=MONITORING_INTERVAL)
                    memory_info = process.memory_info()
                    memory_usage_bytes = memory_info.rss
                    memory_usage_percent = memory_usage_bytes / psutil.virtual_memory().total * 100
                    memory_usage_mb = memory_usage_bytes / (1024 * 1024)

                    # Collect GPU metrics for this PID
                    gpu_metrics = collect_gpu_metrics_nvml(pid, elapsed_time, llm_model_name=llm_model_name)
                    
                    # Only log if we have valid GPU metrics
                    process_metrics = gpu_metrics.get("process_gpu_metrics", [])
                    if process_metrics and not any('message' in metric for metric in process_metrics):
                        # If this is our first valid metric, reset the start time to now
                        if not shared_metrics:
                            start_time = current_time
                            elapsed_time = 0  # First entry should be at time 0
                            
                        data_entry = {
                            "timestamp": timestamp,
                            "elapsed_time": round(elapsed_time, 2),
                            "pid": pid,
                            "process_name": process.name(),
                            "total_cpu_usage": round(cpu_usage, 2),
                            "memory_usage_bytes": memory_usage_bytes,
                            "memory_usage_mb": round(memory_usage_mb, 2),
                            "memory_usage_percent": round(memory_usage_percent, 2),
                            "gpu_metrics": gpu_metrics,
                        }
                        if llm_model_name:
                            data_entry["llm_model_name"] = llm_model_name
                        shared_metrics.append(data_entry)

                        if shared_metrics:
                            log_gpu_metrics(log_file_path, [data_entry], headers=not headers_written)
                            headers_written = True

                except psutil.NoSuchProcess:
                    # Remove non-existent PIDs from the list
                    if pid in pids:
                        pids.remove(pid)
                    continue

            elapsed_time = time.time() - start_time
            time.sleep(MONITORING_INTERVAL)
            
            # Exit if no PIDs left to monitor
            if not pids:
                print("No processes left to monitor, exiting monitoring loop.")
                break

    except Exception as e:
        print(f"Error in monitor_resource_usage: {e}")
    finally:
        shutdown_nvml()

def get_process_tree_pids(pid):
    """Get all child process PIDs including the parent."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        return [pid] + [child.pid for child in children]
    except psutil.NoSuchProcess:
        return [pid]

def find_ollama_server_pid():
    """Find the PID of the ollama_llama_server process and extract the LLM model name."""
    try:
        # Run the ollama ps command to get the model name
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        if result.returncode != 0:
            return None, None
        
        # Extract the first model name from the output
        lines = result.stdout.splitlines()
        if len(lines) > 1:
            model_name = lines[1].split()[0]
        else:
            model_name = None

        # Find the PID of the ollama_llama_server process
        for proc in psutil.process_iter(['pid', 'name']):
            if 'ollama_llama_server' in proc.info['name']:
                return proc.info['pid'], model_name
    except Exception as e:
        print(f"Error finding ollama_llama_server PID: {e}")
    return None, None

def is_query_in_file(file_path, query_term):
    """Check if the Python file contains the specified query term."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return query_term in content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

def measure_metrics(command, *args, **kwargs):
    """
    Monitors CPU, memory, and GPU metrics during the execution of an external command.
    """
    metrics = {
        "execution_started_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "host_name": os.uname().nodename,
    }

    with Manager() as manager:
        shared_metrics = manager.list()
        command_pids = manager.list()  # Store PIDs of the command and its children

        # Convert command list to string if necessary
        if isinstance(command, list):
            command_str = " ".join(command)
            parent_process_name = command[0]
        else:
            command_str = command
            parent_process_name = command.split()[0]

        # Start the command process
        process = subprocess.Popen(
            command_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Give the process a moment to spawn children
        time.sleep(1)
        
        # Get all related PIDs
        pids = get_process_tree_pids(process.pid)
        command_pids.extend(pids)

        llm_model_name = None

        # Check if the command involves running the "ollama" query
        script_path = command_str.split()[-1] if command_str.endswith(".py") else None
        if "ollama" in command_str or (script_path and is_query_in_file(script_path, "ollama")):
            # Find the ollama_llama_server PID and add it to the list
            ollama_pid, model_name = find_ollama_server_pid()
            if ollama_pid:
                command_pids.append(ollama_pid)
                print(f"Tracking ollama_llama_server PID: {ollama_pid} with model: {model_name}")
                llm_model_name = model_name

        # Check if the command involves running the "vllm" query
        if "vllm" in command_str or (script_path and is_query_in_file(script_path, "vllm")):
            # Find the vLLM server PID and add it to the list
            vllm_pid, model_name = find_vllm_server_pid()  # This now calls the imported function
            if vllm_pid:
                command_pids.append(vllm_pid)
                print(f"Tracking vLLM server PID: {vllm_pid} with model: {model_name}")
                llm_model_name = model_name

        # Use llm_model_name if available, otherwise use the parent process name
        process_name = llm_model_name if llm_model_name else parent_process_name

        # Generate log file name - use the centralized GPU functions
        initialize_nvml()
        if get_gpu_count() > 0:
            gpu_model = get_gpu_model(0)
        else:
            gpu_model = "No_GPU"
        shutdown_nvml()
        
        log_file_name = generate_log_filename(process_name, gpu_model, 0)
        log_file_path = os.path.join(LOG_DIR, log_file_name)
        json_file_path = log_file_path.replace("_gpu_metrics.csv", "_summary_metrics.json")

        # Start monitoring with all related PIDs
        monitor_proc = Process(
            target=monitor_resource_usage,
            args=(shared_metrics, command_pids, log_file_path, llm_model_name)
        )
        monitor_proc.start()

        try:
            # Monitor stdout in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Print command output with a marker to distinguish it from metrics
                    print(f"[COMMAND OUTPUT] {output.strip()}")
                    
            # Get the final output and return code
            stdout, stderr = process.communicate()
            
            result = subprocess.CompletedProcess(
                args=command_str,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )

        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user.")
            process.terminate()
            result = None
        finally:
            # Stop monitoring
            monitor_proc.terminate()
            monitor_proc.join()
            # Print a newline to separate from metrics output
            print("\n")

        if shared_metrics:
            metrics["hosts"] = list(shared_metrics)
            metrics["execution_ended_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            summary = report_summary_metrics(LOG_DIR, metrics["host_name"], metrics["hosts"], json_file_path)
            metrics["summary"] = summary
        else:
            print("No metrics recorded during task execution.")
            metrics["execution_ended_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            summary = {}

    elapsed_time = (datetime.strptime(metrics["execution_ended_at"], "%d/%m/%Y %H:%M:%S") - 
                    datetime.strptime(metrics["execution_started_at"], "%d/%m/%Y %H:%M:%S")).total_seconds()

    return metrics, result, elapsed_time

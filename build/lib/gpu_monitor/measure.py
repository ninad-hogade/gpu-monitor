# measure.py
import os
import json
import subprocess
import psutil
import sys  # Add missing import
import time
from datetime import datetime
from multiprocessing import Process, Manager
from .utils import (
    monitor_resource_usage,
    generate_log_filename,
    report_summary_metrics,
)
from .gpu_metrics import initialize_nvml, shutdown_nvml
import pynvml

LOG_DIR = "LOGS"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

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

def find_vllm_server_pid():
    """Find the PID of the vLLM server process running on GPU and extract the model name."""
    try:
        # Initialize NVML
        initialize_nvml()
        
        # Iterate over all GPUs
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Get all processes running on the GPU
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            all_procs = compute_procs + graphics_procs
            
            for proc in all_procs:
                try:
                    p = psutil.Process(proc.pid)
                    if 'vllm_env/bin/python' in ' '.join(p.cmdline()):
                        # Get the model name by querying the vLLM server
                        try:
                            result = subprocess.run(
                                ["curl", "-s", "http://localhost:8000/v1/models"],
                                capture_output=True,
                                text=True,
                                timeout=5  # Timeout of 5 seconds
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                data = json.loads(result.stdout)
                                model_id = data.get("data", [{}])[0].get("id", "")
                                if not model_id:
                                    raise RuntimeError("Error: vLLM server did not return a model name.")
                                model_name = "vllm-" + model_id.split("/")[-1]
                                return proc.pid, model_name
                            else:
                                raise RuntimeError("Error: vLLM server is running but did not return any model information.")
                        except subprocess.TimeoutExpired:
                            raise RuntimeError("Timeout expired while querying vLLM server for model name.")
                        except json.JSONDecodeError:
                            raise RuntimeError("Error: Failed to parse JSON response from vLLM server.")
                        except Exception as e:
                            raise RuntimeError(f"Unexpected error querying vLLM server: {e}")
                except psutil.NoSuchProcess:
                    continue
    except Exception as e:
        print(f"Error finding vLLM server PID: {e}")
        return None, None
    finally:
        # Shutdown NVML
        shutdown_nvml()
    
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
            vllm_pid, model_name = find_vllm_server_pid()
            if vllm_pid:
                command_pids.append(vllm_pid)
                print(f"Tracking vLLM server PID: {vllm_pid} with model: {model_name}")
                llm_model_name = model_name

        # Use llm_model_name if available, otherwise use the parent process name
        process_name = llm_model_name if llm_model_name else parent_process_name

        # Generate log file name
        initialize_nvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_model = pynvml.nvmlDeviceGetName(handle)
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
                    print(output.strip())
                    
            # Get the final output and return code
            stdout, stderr = process.communicate()
            
            result = subprocess.CompletedProcess(
                args=command_str,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )

        except KeyboardInterrupt:
            print("Monitoring interrupted by user.")
            process.terminate()
            result = None
        finally:
            # Stop monitoring
            monitor_proc.terminate()
            monitor_proc.join()

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

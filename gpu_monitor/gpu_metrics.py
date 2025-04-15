import pynvml
import psutil
import subprocess
import json

# Track NVML initialization status
_nvml_initialized = False

def initialize_nvml():
    """
    Initialize NVIDIA Management Library (NVML).
    Returns True if successful, False otherwise.
    """
    global _nvml_initialized
    try:
        if not _nvml_initialized:
            pynvml.nvmlInit()
            _nvml_initialized = True
        return True
    except pynvml.NVMLError as e:
        print(f"Error initializing NVML: {str(e)}")
        return False

def shutdown_nvml():
    """
    Shutdown NVML.
    """
    global _nvml_initialized
    try:
        if _nvml_initialized:
            pynvml.nvmlShutdown()
            _nvml_initialized = False
    except pynvml.NVMLError as e:
        print(f"Error shutting down NVML: {str(e)}")

def get_gpu_count():
    """
    Get the number of available GPUs.
    """
    if not initialize_nvml():
        return 0
    
    try:
        return pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError as e:
        print(f"Error getting GPU count: {str(e)}")
        return 0

def get_gpu_model(gpu_id):
    """
    Get GPU model name for the given GPU ID.
    """
    if not initialize_nvml():
        return "Unknown GPU"
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return pynvml.nvmlDeviceGetName(handle)
    except pynvml.NVMLError as e:
        print(f"Error getting GPU model: {str(e)}")
        return "Unknown GPU"

def get_gpu_handle(gpu_id):
    """
    Get the handle for a specific GPU.
    """
    if not initialize_nvml():
        return None
        
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    except pynvml.NVMLError as e:
        print(f"Error getting GPU handle: {str(e)}")
        return None

def get_processes_on_gpu(gpu_id):
    """
    Get all processes running on a specific GPU.
    """
    handle = get_gpu_handle(gpu_id)
    if not handle:
        return []
        
    try:
        compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        return compute_procs + graphics_procs
    except pynvml.NVMLError as e:
        print(f"Error getting GPU processes: {str(e)}")
        return []

def get_theoretical_gflops(gpu_model):
    """
    Retrieve theoretical GFLOPS for specific GPU models.
    """
    gpu_gflops = {
        "Tesla V100-SXM2-16GB": 31200.0,
        "NVIDIA A100X": 39000.0,
        "NVIDIA H100 PCIe": 62400.0,
        "NVIDIA A100 80GB PCIe": 39000.0
    }
    for key in gpu_gflops:
        if key in gpu_model:
            return gpu_gflops[key]
    return None

def find_vllm_server_pid():
    """Find the PID of the vLLM server process running on GPU and extract the model name."""
    if not initialize_nvml():
        return None, None
        
    try:
        # Check each GPU for vLLM processes
        for i in range(get_gpu_count()):
            gpu_processes = get_processes_on_gpu(i)
            
            for proc in gpu_processes:
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

def collect_gpu_metrics_nvml(target_pid, elapsed_time, log_once=[False], llm_model_name=None):
    """
    Collect GPU metrics using NVML
    """
    if not initialize_nvml():
        return {"error": "Failed to initialize NVML"}

    try:
        gpu_count = get_gpu_count()
        if not log_once[0]:
            print(f"Number of GPUs detected: {gpu_count}")
            print(f"Monitoring target PID: {target_pid}")
            log_once[0] = True

        process_gpu_metrics = []
        gpu_system_metrics = []

        for i in range(gpu_count):
            handle = get_gpu_handle(i)
            if not handle:
                continue
                
            # System-wide metrics
            gpu_name = get_gpu_model(i)
            
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError as e:
                print(f"Error getting GPU metrics: {e}")
                continue

            # Get all processes on this GPU
            gpu_processes = get_processes_on_gpu(i)
            
            # Check if target process is using this GPU
            for proc in gpu_processes:
                if proc.pid == target_pid:
                    theoretical_gflops = get_theoretical_gflops(gpu_name)
                    inst_gflops = (
                        theoretical_gflops * (utilization.gpu / 100.0)
                        if theoretical_gflops else None
                    )
                    
                    try:
                        process_name = psutil.Process(proc.pid).name()
                    except psutil.NoSuchProcess:
                        process_name = "Unknown"
                        
                    if llm_model_name:
                        process_name = llm_model_name
                        
                    process_gpu_metrics.append({
                        "pid": proc.pid,
                        "process_name": process_name,
                        "gpu_index": i,
                        "gpu_name": gpu_name,
                        "gpu_memory_used_MB": proc.usedGpuMemory / 1024**2,
                        "gpu_utilization_percent": utilization.gpu,
                        "gpu_memory_utilization_percent": (memory_info.used / memory_info.total) * 100,
                        "power_usage_W": power_usage,
                        "inst_gflops": inst_gflops
                    })

            # System metrics even if process not found
            gpu_system_metrics.append({
                "gpu_index": i,
                "gpu_name": gpu_name,
                "gpu_utilization": utilization.gpu,
                "gpu_memory_used_MB": memory_info.used / 1024**2,
                "gpu_memory_total_MB": memory_info.total / 1024**2,
                "power_usage_W": power_usage
            })

        if not process_gpu_metrics:
            try:
                parent = psutil.Process(target_pid)
                process_gpu_metrics.append({
                    "pid": target_pid,
                    "process_name": parent.name(),
                    "message": f"No GPU usage detected for PID {target_pid}"
                })
            except psutil.NoSuchProcess:
                process_gpu_metrics.append({
                    "pid": target_pid,
                    "process_name": "Unknown",
                    "message": f"Process {target_pid} not found"
                })

        return {
            "process_gpu_metrics": process_gpu_metrics,
            "gpu_system_metrics": gpu_system_metrics,
        }

    except Exception as e:
        print(f"Error collecting GPU metrics: {str(e)}")
        return {"error": f"Error collecting GPU metrics via NVML: {str(e)}"}



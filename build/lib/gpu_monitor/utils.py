import os
import json
import csv
from datetime import datetime
import socket

# Monitoring interval in seconds (adjusted from 0.1 to 0.5 for better performance)
MONITORING_INTERVAL = 0.5

def generate_log_filename(process_name, gpu_name, gpu_index):
    """
    Generate a unique log file name based on timestamp, hostname, process name, GPU name, and GPU index.
    """
    hostname = os.uname().nodename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_process_name = ''.join(c if c.isalnum() else '_' for c in process_name)
    return f"{hostname}_{gpu_name}_{gpu_index}_{safe_process_name}_{timestamp}_gpu_metrics.csv"

def log_gpu_metrics(file_path, data, headers=False):
    """
    Log GPU metrics to a CSV file.
    """
    hostname = socket.gethostname()
    try:
        with open(file_path, "a") as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow([
                    "Timestamp", "Elapsed_Time(s)", "Hostname", "PID", "Process_Name", "GPU_Index",
                    "GPU_Name", "Inst_GFLOPS", "Inst_GPU_Memory_Used(MB)",
                    "Inst_%_GPU_Mem_Utilization", "Inst_%_GPU_Utilization", "Power_Usage(W)",
                    "CPU_Usage_Percent", "Memory_Usage_MB", "Memory_Usage_Percent"  # Added CPU and memory columns
                ])
            for entry in data:
                for gpu_metric in entry["gpu_metrics"]["process_gpu_metrics"]:
                    if gpu_metric.get("message"):  # Skip invalid metrics
                        continue
                    process_name = gpu_metric.get("process_name", "N/A")
                    if "llm_model_name" in entry:
                        process_name = entry['llm_model_name']
                    
                    # Get CPU and memory metrics from the data entry
                    cpu_usage = entry.get("total_cpu_usage", "N/A")
                    memory_usage_percent = entry.get("memory_usage_percent", "N/A")
                    
                    # Calculate memory usage in MB if available
                    memory_usage_mb = "N/A"
                    if "memory_usage_bytes" in entry and entry["memory_usage_bytes"] is not None:
                        memory_usage_mb = entry["memory_usage_bytes"] / (1024 * 1024)
                    
                    # Prepare row data
                    row = [
                        entry["timestamp"],
                        entry.get("elapsed_time", "N/A"),
                        hostname,
                        gpu_metric.get("pid", "N/A"),
                        process_name,
                        gpu_metric.get("gpu_index", "N/A"),
                        gpu_metric.get("gpu_name", "N/A"),
                        gpu_metric.get("inst_gflops", "N/A"),
                        gpu_metric.get("gpu_memory_used_MB", "N/A"),
                        gpu_metric.get("gpu_memory_utilization_percent", "N/A"),
                        gpu_metric.get("gpu_utilization_percent", "N/A"),
                        gpu_metric.get("power_usage_W", "N/A"),
                        cpu_usage,                        # Add CPU usage
                        memory_usage_mb,                  # Add memory usage in MB
                        memory_usage_percent              # Add memory usage percentage
                    ]
                    
                    # Write to CSV
                    writer.writerow(row)
                    
                    # Print metrics with a distinctive marker and on a new line
                    print(f"\n[GPU METRICS] {entry['timestamp']} | Elapsed: {entry.get('elapsed_time', 'N/A')}s | " 
                          f"CPU: {cpu_usage}% | RAM: {memory_usage_percent}% | "
                          f"GPU: {gpu_metric.get('gpu_utilization_percent', 'N/A')}% | "
                          f"Memory: {gpu_metric.get('gpu_memory_used_MB', 'N/A'):.1f}MB ({gpu_metric.get('gpu_memory_utilization_percent', 'N/A'):.1f}%) | "
                          f"Power: {gpu_metric.get('power_usage_W', 'N/A'):.1f}W")
                    
    except Exception as e:
        print(f"Error logging GPU metrics: {e}")

def calculate_avg(values):
    """Calculate average of a list of values."""
    return sum(values) / len(values) if values else 0.0

def report_summary_metrics(log_folder, host_name, data, json_file_path):
    """
    Reports average and total metrics from the collected data.
    """
    if not data:
        print("No metrics collected during the task.")
        return {}

    elapsed_time = data[-1]["elapsed_time"]

    # Extract metrics from the data
    valid_gflops = [
        gpu.get("inst_gflops", 0) for entry in data
        for gpu in entry["gpu_metrics"].get("process_gpu_metrics", []) if gpu.get("inst_gflops") is not None
    ]
    
    valid_gpu_memory = [
        gpu.get("gpu_memory_used_MB", 0) for entry in data
        for gpu in entry["gpu_metrics"].get("process_gpu_metrics", []) if gpu.get("gpu_memory_used_MB") is not None
    ]
    
    valid_gpu_utilization = [
        gpu.get("gpu_utilization_percent", 0) for entry in data
        for gpu in entry["gpu_metrics"].get("process_gpu_metrics", []) if gpu.get("gpu_utilization_percent") is not None
    ]
    
    valid_gpu_memory_utilization = [
        gpu.get("gpu_memory_utilization_percent", 0) for entry in data
        for gpu in entry["gpu_metrics"].get("process_gpu_metrics", []) if gpu.get("gpu_memory_utilization_percent") is not None
    ]
    
    valid_cpu_usage = [entry.get("total_cpu_usage", 0) for entry in data]
    valid_memory_usage = [entry.get("memory_usage_percent", 0) for entry in data]

    # Calculate energy consumption
    total_energy = sum([
        gpu.get("power_usage_W", 0) * MONITORING_INTERVAL / 3600.0 for entry in data
        for gpu in entry["gpu_metrics"].get("process_gpu_metrics", []) if gpu.get("power_usage_W") is not None
    ])

    peak_power = max([
        gpu.get("power_usage_W", 0) for entry in data
        for gpu in entry["gpu_metrics"].get("process_gpu_metrics", []) if gpu.get("power_usage_W") is not None
    ], default=0.0)

    process_name = data[-1]["process_name"]
    if "llm_model_name" in data[-1]:
        process_name = data[-1]["llm_model_name"]

    summary = {
        "host_name": host_name,
        "elapsed_time_s": elapsed_time,
        "avg_gflops": calculate_avg(valid_gflops),
        "avg_gpu_memory_used_MB": calculate_avg(valid_gpu_memory),
        "average_gpu_utilization_percent": calculate_avg(valid_gpu_utilization),
        "average_gpu_memory_utilization_percent": calculate_avg(valid_gpu_memory_utilization),
        "average_cpu_usage_percent": calculate_avg(valid_cpu_usage),
        "average_memory_usage_percent": calculate_avg(valid_memory_usage),
        "total_energy_Wh": total_energy,
        "peak_power_W": peak_power,
        "pid": data[-1]["pid"],
        "process_name": process_name
    }

    try:
        with open(json_file_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        # Print a clear separator before summary
        print("\n\n" + "="*80)
        print("MONITORING SUMMARY")
        print("="*80)
        print(f"Host: {summary['host_name']}")
        print(f"Process: {summary['process_name']} (PID: {summary['pid']})")
        print(f"Elapsed time: {summary['elapsed_time_s']:.2f} seconds")
        print("\nPerformance Metrics:")
        print(f"  Average GPU Utilization: {summary['average_gpu_utilization_percent']:.2f}%")
        print(f"  Average GPU Memory Usage: {summary['avg_gpu_memory_used_MB']:.2f} MB")
        print(f"  Average GPU Memory Utilization: {summary['average_gpu_memory_utilization_percent']:.2f}%")
        print(f"  Average CPU Usage: {summary['average_cpu_usage_percent']:.2f}%")
        print(f"  Average System Memory Usage: {summary['average_memory_usage_percent']:.2f}%")
        print(f"  Average GFLOPS: {summary['avg_gflops']:.2f}")
        print("\nEnergy Metrics:")
        print(f"  Total Energy Consumption: {summary['total_energy_Wh']:.4f} Wh")
        print(f"  Peak Power Usage: {summary['peak_power_W']:.2f} W")
        print(f"\nDetailed metrics saved to: {json_file_path}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Error writing summary metrics: {e}")

    return summary

def monitor_resource_usage(shared_metrics, pids, log_file_path, llm_model_name=None):
    """
    Monitors system's CPU, memory, and GPU usage for a specific process and its children.
    Logs detailed instantaneous metrics to a CSV file.
    """
    try:
        initialize_nvml()
        elapsed_time = 0
        headers_written = False
        start_time = time.time()
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Monitor each PID in the process tree
            for pid in list(pids):  # Make a copy of the list to avoid modification during iteration
                try:
                    # Collect CPU and memory usage
                    process = psutil.Process(pid)
                    cpu_usage = process.cpu_percent(interval=MONITORING_INTERVAL)
                    memory_info = process.memory_info()
                    memory_usage = memory_info.rss / psutil.virtual_memory().total * 100

                    # Collect GPU metrics for this PID
                    gpu_metrics = collect_gpu_metrics_nvml(pid, elapsed_time, llm_model_name=llm_model_name)
                    
                    # Only log if we have valid GPU metrics
                    process_metrics = gpu_metrics.get("process_gpu_metrics", [])
                    if process_metrics and not any('message' in metric for metric in process_metrics):
                        data_entry = {
                            "timestamp": timestamp,
                            "elapsed_time": round(elapsed_time, 2),
                            "pid": pid,
                            "process_name": process.name(),
                            "total_cpu_usage": round(cpu_usage, 2),
                            "memory_usage_percent": round(memory_usage, 2),
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
# GPU Monitor Tool

A comprehensive tool for monitoring GPU, CPU, and memory usage during computation tasks, with a focus on AI/ML workloads.

## Installation

### Clone the Repository

```bash
git clone link
cd gpu-monitor
```

### Set Up Environment

1. Install UV package manager (a faster alternative to pip)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment

```bash
uv venv gpu-mon --python 3.12 --seed
source ~/gpu-mon/bin/activate
```

3. Install dependencies

```bash
# Install vLLM for inference tasks
uv pip install vllm

# Install the GPU monitoring tool
pip install .
```

## Basic Usage

### Matrix Multiplication Benchmark

Test GPU performance and monitoring with the built-in benchmarking utility:

```bash
# Run basic GPU benchmark
gpu-monitor run matrix_bench.py

# Run benchmark with additional GPU memory allocation
gpu-monitor run matrix_bench.py --extra-memory 10 --duration 20
```

### vLLM Inference Testing

1. Check available models (after starting vLLM server)

```bash
curl http://localhost:8000/v1/models
```

2. Start vLLM server with a selected model

```bash
# Start a vLLM server with a specified model
python -m vllm.entrypoints.openai.api_server --model path/to/model --port 8000

# Or use the provided script
python gpu-monitor/inference_vllm/vllm_ex/server_vllm.py
```

3. Run inference experiment

```bash
cd gpu-monitor/inference_vllm/
python vllm_inference.py
```

## Advanced Features

### Monitoring Options

- `--duration`: Specify test duration in seconds
- `--size`: Matrix size for benchmark testing
- `--device`: Choose between GPU and CPU
- `--dtype`: Select data type (float32/float64)
- `--extra-memory`: Additional memory allocation in GB

### Cleanup Utilities

If you encounter stuck processes, use the included cleanup script:

```bash
# Kill all Python processes
python cleanup.py --all-python

# Kill processes using more than 5GB memory
python cleanup.py --high-memory 5
```

## Output

The tool provides:

1. Real-time monitoring output in the terminal
2. CSV logs with detailed metrics
3. Summary statistics after test completion
4. JSON summary files for further analysis

All logs are saved in the `LOGS` directory.

## Log Files

GPU Monitor creates detailed log files to help you analyze resource usage patterns. These logs are saved in the `LOGS` directory.

### Log File Structure

When you run the GPU Monitor, it creates two types of log files:

1. **GPU Metrics CSV** - Detailed timestamped metrics collected at regular intervals
2. **Summary Metrics JSON** - Aggregated statistics from the monitoring session

#### GPU Metrics CSV

The GPU metrics are stored in CSV files with naming format:
`<hostname>_<GPU_Name>_<GPU_Index>_<process_name>_<timestamp>_gpu_metrics.csv`

Example: `hsc-11_NVIDIA A100X_0_python3_20250314_134655_gpu_metrics.csv`

This file contains the following columns:
- `Timestamp`: Exact time when metrics were collected
- `Elapsed_Time(s)`: Time elapsed since monitoring began
- `Hostname`: Name of the host machine
- `PID`: Process ID being monitored
- `Process_Name`: Name of the process
- `GPU_Index`: Index of the GPU
- `GPU_Name`: Model name of the GPU
- `Inst_GFLOPS`: Instantaneous GFLOPS utilization
- `Inst_GPU_Memory_Used(MB)`: GPU memory used at that moment in MB
- `Inst_%_GPU_Mem_Utilization`: Percentage of total GPU memory utilized
- `Inst_%_GPU_Utilization`: Percentage of GPU compute resources utilized
- `Power_Usage(W)`: Power consumption in watts

#### Summary Metrics JSON

The summary metrics are stored in JSON files with naming format:
`<hostname>_<GPU_Name>_<GPU_Index>_<process_name>_<timestamp>_summary_metrics.json`

Example: `hsc-11_NVIDIA A100X_0_python3_20250314_134655_summary_metrics.json`

This file contains aggregated statistics for the entire monitoring session:
- `host_name`: Name of the host machine
- `elapsed_time_s`: Total duration of monitoring
- `avg_gflops`: Average GFLOPS utilization over the session
- `avg_gpu_memory_used_MB`: Average GPU memory used in MB
- `average_gpu_utilization_percent`: Average GPU compute utilization percentage
- `average_gpu_memory_utilization_percent`: Average GPU memory utilization percentage
- `average_cpu_usage_percent`: Average CPU usage by the process
- `average_memory_usage_percent`: Average system memory usage by the process
- `total_energy_Wh`: Total energy consumed in watt-hours
- `peak_power_W`: Maximum power consumption observed in watts
- `pid`: Process ID that was monitored
- `process_name`: Name of the process that was monitored

## vLLM Integration

GPU Monitor provides special support for monitoring vLLM server processes with automatic model detection.

### Model Setup

Before running vLLM inference experiments, you need to download the required models:

1. Create a models directory (if it doesn't exist, change path):
```bash
mkdir -p path-to/huggingface_models/
```

2. Download models from Hugging Face using git lfs:
```bash
# Install git-lfs if not already installed
git lfs install

# Clone a model
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base path-to/huggingface_models/DeepSeek-R1-Distill-Llama-7B
```

3. Update model paths in the inference scripts:
   - Open scripts in the `inference_vllm` folder
   - Modify the model path to point to your downloaded models
   - For example, edit `vllm_inference.py` to use your model path

### Example Output

When monitoring a vLLM inference server, you might see output similar to:

```
Using model: path-to/huggingface_models/DeepSeek-R1-Distill-Llama-8B
NVML initialized successfully.
Tracking vLLM server PID: 575846 with model: vllm-Unknown
Number of GPUs detected: 1
Monitoring target PID: 584315
...
```

## License

MIT License

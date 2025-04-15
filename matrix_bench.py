import argparse
import time
import torch
import numpy as np
import psutil
import os
import signal
import sys

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\nStopping benchmark...")
    running = False

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

def parse_args():
    parser = argparse.ArgumentParser(description='Matrix Multiplication Benchmark')
    parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds')
    parser.add_argument('--size', type=int, default=4096, help='Matrix size (N for NxN matrices)')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float64', help='Data type')
    parser.add_argument('--report-interval', type=int, default=1, help='Status report interval in seconds')
    # Add new parameter for additional memory consumption
    parser.add_argument('--extra-memory', type=float, default=0.0, 
                       help='Additional memory to allocate in GB (beyond matrices)')
    return parser.parse_args()

def print_system_info():
    """Print information about the system"""
    print("=== System Information ===")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("No GPU detected")
    print("=======================")

def run_benchmark(args):
    """Run the matrix multiplication benchmark with the specified settings"""
    # Set up parameters
    size = args.size
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    device_name = args.device
    duration = args.duration
    device = torch.device("cuda" if device_name == "gpu" and torch.cuda.is_available() else "cpu")
    
    print(f"\n=== Matrix Multiplication Benchmark ===")
    print(f"Matrix size: {size}x{size}")
    print(f"Data type: {args.dtype}")
    print(f"Device: {device_name} {'(available)' if device == torch.device('cuda') or device_name == 'cpu' else '(not available, using CPU)'}")
    print(f"Duration: {duration} seconds")
    
    # Add clear marker for monitoring tool
    print("\n==== BENCHMARK PREPARATION PHASE ====")
    
    # Create matrices
    print("Creating matrices...")
    try:
        # Pre-allocate memory with fixed seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        if device == torch.device("cuda"):
            # GPU matrices - directly create on GPU
            a = torch.rand((size, size), dtype=dtype, device=device)
            b = torch.rand((size, size), dtype=dtype, device=device)
            
            # Allocate additional memory if requested
            extra_tensors = []
            if args.extra_memory > 0:
                print(f"Allocating {args.extra_memory:.2f} GB additional GPU memory...")
                bytes_per_element = 4 if dtype == torch.float32 else 8
                elements_per_gb = (1024**3) // bytes_per_element
                chunk_size = 1_000_000  # Allocate in 1M element chunks to avoid OOM
                remaining_elements = int(args.extra_memory * elements_per_gb)
                
                allocated_gb = 0
                last_reported_gb = 0  # Track last reported GB for threshold printing
                
                while remaining_elements > 0:
                    current_chunk = min(chunk_size, remaining_elements)
                    try:
                        tensor = torch.ones(current_chunk, dtype=dtype, device=device)
                        extra_tensors.append(tensor)
                        remaining_elements -= current_chunk
                        allocated_gb += (current_chunk * bytes_per_element) / (1024**3)
                        
                        # Only print status every 0.25 GB
                        if int(allocated_gb / 0.25) > int(last_reported_gb / 0.25):
                            print(f"  Allocated {allocated_gb:.2f} GB of {args.extra_memory:.2f} GB requested")
                            last_reported_gb = allocated_gb
                            
                    except torch.cuda.OutOfMemoryError:
                        print("  Cannot allocate more GPU memory, stopping")
                        break
                
                # Final allocation report
                print(f"  Total allocated: {allocated_gb:.2f} GB of {args.extra_memory:.2f} GB requested")
        else:
            # CPU matrices - use NumPy for better memory management
            a = torch.tensor(np.random.rand(size, size), dtype=dtype)
            b = torch.tensor(np.random.rand(size, size), dtype=dtype)
            
            # Allocate additional memory if requested
            extra_arrays = []
            if args.extra_memory > 0:
                print(f"Allocating {args.extra_memory:.2f} GB additional system memory...")
                bytes_per_element = 4 if dtype == torch.float32 else 8
                elements_per_gb = (1024**3) // bytes_per_element
                chunk_size = 10_000_000  # Allocate in 10M element chunks
                remaining_elements = int(args.extra_memory * elements_per_gb)
                
                allocated_gb = 0
                last_reported_gb = 0  # Track last reported GB for threshold printing
                
                while remaining_elements > 0:
                    current_chunk = min(chunk_size, remaining_elements)
                    try:
                        array = np.ones(current_chunk, dtype=np.float32 if dtype == torch.float32 else np.float64)
                        extra_arrays.append(array)
                        remaining_elements -= current_chunk
                        allocated_gb += (current_chunk * bytes_per_element) / (1024**3)
                        
                        # Only print status every 0.25 GB
                        if int(allocated_gb / 0.25) > int(last_reported_gb / 0.25):
                            print(f"  Allocated {allocated_gb:.2f} GB of {args.extra_memory:.2f} GB requested")
                            last_reported_gb = allocated_gb
                            
                    except MemoryError:
                        print("  Cannot allocate more system memory, stopping")
                        break
                
                # Final allocation report
                print(f"  Total allocated: {allocated_gb:.2f} GB of {args.extra_memory:.2f} GB requested")
        
        memory_usage = 2 * size * size * (4 if args.dtype == 'float32' else 8) / (1024**3)
        total_memory = memory_usage + args.extra_memory
        print(f"Memory usage for matrices: {memory_usage:.2f} GB")
        print(f"Total memory allocated: {total_memory:.2f} GB")
        
        # Add a pre-benchmark delay to ensure monitoring is ready
        print("Ensuring monitoring is active...")
        time.sleep(2)  # Wait 2 seconds before starting benchmark
        
        # Warm up
        print("Warming up...")
        _ = torch.matmul(a, b)
        torch.cuda.synchronize() if device == torch.device("cuda") else None
        
        # Send clear marker that benchmark is starting
        print("\n==== BENCHMARK EXECUTION START ====")
        start_time = time.time()
        iterations = 0
        last_report_time = start_time
        
        # Run the benchmark
        print(f"\nStarting benchmark for {duration} seconds...")
        while running and (time.time() - start_time) < duration:
            _ = torch.matmul(a, b)
            if device == torch.device("cuda"):
                torch.cuda.synchronize()  # Wait for GPU completion
            
            iterations += 1
            
            # Report status periodically
            current_time = time.time()
            if (current_time - last_report_time) >= args.report_interval:
                elapsed = current_time - start_time
                ops_per_second = iterations / elapsed
                
                # Get system stats
                cpu_percent = psutil.cpu_percent()
                mem_percent = psutil.virtual_memory().percent
                
                # Get GPU stats if applicable
                gpu_stats = ""
                if device == torch.device("cuda"):
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                    gpu_stats = f" | GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
                
                print(f"Time: {elapsed:.2f}s | Iterations: {iterations} | Rate: {ops_per_second:.2f} ops/s | CPU: {cpu_percent}%{gpu_stats}")
                last_report_time = current_time
        
        # Send clear marker that benchmark has ended
        actual_duration = time.time() - start_time
        print(f"\n==== BENCHMARK EXECUTION END ({actual_duration:.2f}s) ====")
        
        # Add a post-benchmark delay to ensure monitoring captures everything
        print("Ensuring all metrics are captured...")
        time.sleep(2)  # Wait 2 seconds after benchmark
        
        # Final report
        total_time = time.time() - start_time
        ops_per_second = iterations / total_time if total_time > 0 else 0
        
        print("\n=== Benchmark Results ===")
        print(f"Total iterations: {iterations}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Performance: {ops_per_second:.2f} matrix multiplications/second")
        
        # Calculate compute throughput
        flops_per_op = 2 * size**3  # Approximate FLOPs for matrix multiply
        gflops = (flops_per_op * iterations) / (total_time * 1e9)
        print(f"Compute throughput: {gflops:.2f} GFLOPS")
        
        # Release memory explicitly
        del a, b
        if device == torch.device("cuda"):
            del extra_tensors  # Clean up extra GPU memory
            torch.cuda.empty_cache()
        else:
            del extra_arrays  # Clean up extra system memory
        
    except torch.cuda.OutOfMemoryError:
        print(f"Error: Not enough GPU memory for {size}x{size} matrices. Try a smaller size.")
    except MemoryError:
        print(f"Error: Not enough system memory for {size}x{size} matrices. Try a smaller size.")
    except Exception as e:
        print(f"Error during benchmark: {e}")

if __name__ == "__main__":
    args = parse_args()
    print_system_info()
    run_benchmark(args)
    print("\nBenchmark completed")

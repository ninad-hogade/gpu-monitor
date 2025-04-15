def package_monitor():
    import os
    import torch
    import time
    import socket

    def simple_task(duration):
        # Check if PyTorch can detect GPUs
        if not torch.cuda.is_available():
            print("No GPU detected by PyTorch. Exiting.")
            return {"error": "No GPU detected."}, None
        else:
            print(f"GPUs detected: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  - {torch.cuda.get_device_name(i)}")

        # Initialize PyTorch tensors on the GPU
        device = torch.device("cuda:0")
        matrix_size = 2048
        matrix_a = torch.rand((matrix_size, matrix_size), device=device)
        matrix_b = torch.rand((matrix_size, matrix_size), device=device)

        print("Starting GPU stress test. Monitor GPU utilization using nvidia-smi.")
        start_time = time.time()
        iterations = 0

        # Perform matrix multiplications to stress the GPU
        while time.time() - start_time < duration:
            _ = torch.mm(matrix_a, matrix_b)
            iterations += 1
            if iterations % 100 == 0:
                print(f"Completed {iterations} matrix multiplications.")

        return {"message": f"Task executed on host: {socket.gethostname()}"}, None

    # Execute the simple_task for a specified duration
    return simple_task(10)

# Execute the package_monitor function
if __name__ == "__main__":
    import json
    
    package_monitor()

    #metrics, result = package_monitor()
    #metrics_json = json.dumps(metrics, indent=4)
    #print(metrics_json)
    #print("Simple task result:", result)
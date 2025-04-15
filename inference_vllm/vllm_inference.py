import os
import json
import time
import requests
from gpu_monitor.measure import measure_metrics

PORT = 8000

def get_served_models(port=PORT):
    """Queries the local vLLM server for the list of served models."""
    url = f"http://localhost:{port}/v1/models"
    try:
        time.sleep(5)  # Wait for 5 seconds
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if not models:
                raise ValueError("No models are currently being served.")
            return models
        else:
            raise ValueError(f"Error checking server: {resp.status_code}")
    except Exception as e:
        raise RuntimeError(f"Error checking server: {e}")

def monitor_and_infer():
    # Get the list of served models.
    try:
        models = get_served_models()
    except RuntimeError as e:
        print(e)
        return

    # Use the first served model.
    model_name = models[0].get("id")
    print(f"Using model: {model_name}")

    script_path = os.path.join(os.path.dirname(__file__), "vllm_inference_task.py")
    command = ["python3", script_path, model_name]

    metrics, result, elapsed_time = measure_metrics(command)

    if metrics:
        metrics_json = json.dumps(metrics, indent=4)
        # Print summary metrics
        #print("Summary metrics:")
        #print(metrics_json)
        # Print inference output
        if result.stdout:
            print("Inference output:")
            print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

if __name__ == "__main__":
    for i in range(10):
        monitor_and_infer()

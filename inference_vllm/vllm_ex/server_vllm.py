#!/usr/bin/env python
import subprocess
import time
import requests
import argparse
import os
import sys
import logging

# Hardcoded port and base directory for models.
PORT = 8000
BASE_MODEL_DIR = "/HSC/users/hogade/huggingface_models"

# Set up logging: overwrite log file on each run and log to console.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers.
if logger.hasHandlers():
    logger.handlers.clear()

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the log file path relative to the script's directory
log_file = os.path.join(script_dir, "vllm_servers.log")

# File handler: overwrite file (mode "w")
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def is_server_running(port):
    """Check if a vLLM server is running on the specified port."""
    url = f"http://localhost:{port}/v1/models"
    try:
        resp = requests.get(url, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.info("Server check on port %d failed: %s", port, e)
    return False

def start_server(model_path, port):
    """Start the vLLM server as a subprocess using the given model path."""
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port)
    ]
    logger.info("Starting server for model at '%s' on port %d with command: %s",
                model_path, port, " ".join(command))
    # Pass current environment variables to the subprocess.
    env = os.environ.copy()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    # Monitor the process and output everything to console and log file.
    try:
        while True:
            output = proc.stdout.readline()
            error = proc.stderr.readline()
            if output:
                logger.info(output.strip())
                print(output.strip())
            if error:
                logger.error(error.strip())
                print(error.strip())
                if "RuntimeError" in error:
                    logger.error("RuntimeError detected. Stopping the script.")
                    proc.terminate()
                    sys.exit(1)
            # Check if the process has terminated.
            if proc.poll() is not None:
                break
    except KeyboardInterrupt:
        logger.info("Terminating server on port %d...", port)
        proc.terminate()
        print("Server terminated.")
    
    return proc

def wait_for_server(port, timeout=600):
    """Wait until the server on the specified port becomes available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_server_running(port):
            logger.info("Server on port %d is now running.", port)
            return True
        time.sleep(2)
    logger.error("Server on port %d did not start within %d seconds.", port, timeout)
    return False

def list_available_models():
    """List all directories under BASE_MODEL_DIR."""
    if not os.path.isdir(BASE_MODEL_DIR):
        print(f"Base model directory {BASE_MODEL_DIR} does not exist.")
        sys.exit(1)
    models = [d for d in os.listdir(BASE_MODEL_DIR) if os.path.isdir(os.path.join(BASE_MODEL_DIR, d))]
    return models

def choose_model_interactively(models):
    """Prompt the user to select a model from the list."""
    print("Available models:")
    for idx, model in enumerate(models):
        print(f"{idx+1}. {model}")
    try:
        choice = int(input("Select a model by number: "))
        if 1 <= choice <= len(models):
            return os.path.join(BASE_MODEL_DIR, models[choice-1])
    except Exception as e:
        print("Invalid input:", e)
        sys.exit(1)
    print("Invalid choice. Exiting.")
    sys.exit(1)

def ensure_environment():
    """Ensure the 'vllm_env' Python environment is active."""
    # Check if VIRTUAL_ENV is set and contains 'vllm_env'
    if not os.getenv('VIRTUAL_ENV') or 'vllm_env' not in os.getenv('VIRTUAL_ENV'):
        # Expand the user's home directory
        activate_script = os.path.expanduser("~/vllm_env/bin/activate")
        if os.path.exists(activate_script):
            logger.info("Activating 'vllm_env' environment...")
            # Build the command to source the activation script and re-exec the current command
            command = f". {activate_script} && exec python {' '.join(sys.argv)}"
            os.execvp("bash", ["bash", "-c", command])
        else:
            logger.error("Activation script for 'vllm_env' not found.")
            sys.exit(1)

def main():
    ensure_environment()
    
    parser = argparse.ArgumentParser(
        description="Start a local vLLM server using a model from a local directory."
    )
    parser.add_argument("--model", type=str,
                        help="Local model path. If not provided, available models under " + BASE_MODEL_DIR + " will be listed.")
    args = parser.parse_args()

    # Determine model path.
    if args.model:
        if not os.path.isabs(args.model):
            model_path = os.path.join(BASE_MODEL_DIR, args.model)
        else:
            model_path = args.model
    else:
        models = list_available_models()
        if not models:
            print(f"No models found in {BASE_MODEL_DIR}.")
            sys.exit(1)
        model_path = choose_model_interactively(models)

    if not os.path.isdir(model_path):
        print(f"Specified model path '{model_path}' does not exist or is not a directory.")
        sys.exit(1)

    logger.info("Starting vLLM server for model at '%s' on port %d...", model_path, PORT)
    print(f"Starting vLLM server for model at '{model_path}' on port {PORT}...")

    if is_server_running(PORT):
        logger.info("Server already running on port %d using model at '%s'.", PORT, model_path)
        print("Server is already running.")
        sys.exit(0)

    proc = start_server(model_path, PORT)
    if wait_for_server(PORT):
        logger.info("vLLM server started successfully for model at '%s' on port %d.", model_path, PORT)
        print(f"vLLM server started successfully for model at '{model_path}' on port {PORT}.")
        # Print ongoing output from the server.
        try:
            while True:
                out = proc.stdout.readline()
                if out:
                    print(out.strip())
                # Sleep briefly to avoid busy waiting.
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Terminating server on port %d...", PORT)
            proc.terminate()
            print("Server terminated.")
    else:
        logger.error("Failed to start server for model at '%s' on port %d.", model_path, PORT)
        try:
            stdout, stderr = proc.communicate(timeout=10)
            logger.error("Server stdout: %s", stdout)
            logger.error("Server stderr: %s", stderr)
            print("Server failed to start. Check vllm_servers.log for details.")
        except Exception as e:
            logger.error("Error retrieving server output: %s", e)
            print("Failed to start server.")

if __name__ == "__main__":
    main()

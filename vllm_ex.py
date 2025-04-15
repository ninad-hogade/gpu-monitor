#!/usr/bin/env python
import argparse
import json
import time
import socket
import requests
from transformers import AutoTokenizer

DEFAULT_QUERIES = [
    "Explain quantum mechanics. Write a 5 page story.",
    "What is the theory of relativity? Write a 3 page essay.",
    "Describe the process of photosynthesis. Write a 4 page thesis."
]
PORT = 8000

def get_served_models(port=PORT):
    """Queries the local vLLM server for the list of served models."""
    url = f"http://localhost:{port}/v1/models"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
    except Exception as e:
        print("Error checking server:", e)
    return []

def query_server(queries, model_name, port=PORT):
    """Sends multiple queries to the local vLLM server and returns a single response."""
    endpoint = f"http://localhost:{port}/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"  # Adjust if needed.
    }
    payload = {
        "model": model_name,
        "prompt": queries,
        "max_tokens": 500,
        "temperature": 0.8,
        "top_p": 0.95,
        "logprobs": 5  # This requests the top 5 log probabilities per token.
    }
    start_time = time.time()
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
    except Exception as e:
        print("Error during request:", e)
        return None, "", str(e)
    elapsed_time = time.time() - start_time  # Elapsed time for the request
    if response.status_code == 200:
        res_json = response.json()
        response_text = res_json.get("choices", [{}])[0].get("text", "")
        error = None
    else:
        response_text = f"Error {response.status_code}: {response.text}"
        error = response_text

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Encode the text to get token count
    tokens = tokenizer.encode(response_text, add_special_tokens=False)
    token_count = len(tokens)
    
    summary = {
        "host_name": socket.gethostname(),
        "elapsed_time_s": elapsed_time,  # Elapsed time for the request
        "token_count": token_count,
        "response_preview": " ".join(response_text.split()[:50])
    }
    
    return summary, response_text, error

def llm_test():
    print(f"Starting benchmark...")

    # Get the list of served models.
    models = get_served_models()
    if not models:
        print("No models are currently being served. Please start a model server first.")
        return None, None, "No models are currently being served."

    # Use the first served model.
    model_name = models[0].get("id")
    print(f"Using model: {model_name}")

    summary, response_text, error = query_server(DEFAULT_QUERIES, model_name, port=PORT)

    return summary, response_text, error

if __name__ == "__main__":
    metrics, stdout, stderr = llm_test()
    metrics_json = json.dumps(metrics, indent=4)
    print(metrics_json)
    if stderr:
        print("Errors:", stderr)
